"""
Patched two-stage pipeline:

1. generate_qa_pairs()  — instruction/response pairs from doc chunks.
   - Exponential backoff + jitter on 429s (fixes the "dies at 25" bug,
     which was the Gemini free-tier per-minute rate limit, not quota).
   - ~6s between calls (free tier = ~10 RPM / ~1500 req/day).
   - Resume support: skips docs already present in azure_qa_pairs.jsonl
     and appends, so you can run it across multiple nights safely.
   - Groq fallback (Llama 3.3 70B, free tier) if Gemini keeps failing.

2. finetune()  — QLoRA fine-tune a small instruct model on those pairs.

3. push_to_hub() — upload the adapter (tiny, ~30-60MB) to the HF Hub.

Env vars needed:
    GEMINI_API_KEY   (google ai studio, free)
    GROQ_API_KEY     (optional fallback)
    HF_TOKEN         (only for push_to_hub)
    HF_MODEL_REPO    (e.g. "yourname/azure-slm-lora", only for push)
"""

import json
import os
import random
import time
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

CORPUS_FILE = Path("./azure_docs_corpus.jsonl")
QA_FILE = Path("./azure_qa_pairs.jsonl")
BASE_MODEL = "Qwen/Qwen2.5-3B-Instruct"  # swap for Phi-3-mini, Llama-3.2-3B, etc.
OUTPUT_DIR = "./azure-slm-lora"
HF_MODEL_REPO = os.environ.get("HF_MODEL_REPO", "")  # e.g. "yourname/azure-slm-lora"

GEMINI_MODEL = "gemini-2.5-flash"
GROQ_MODEL = "llama-3.3-70b-versatile"   # free tier on console.groq.com

REQUEST_DELAY = 6.0        # ~10 RPM courtesy for the Gemini free tier
MAX_RETRIES = 6            # per provider, per chunk
FALLBACK_AFTER = 3         # consecutive gemini failures -> cooldown + groq
GEMINI_COOLDOWN = 900      # 15 min

QA_GEN_PROMPT = """You are generating training data for a technical support model.
Given the Azure documentation excerpt below, write 3 question/answer pairs a
developer or admin would realistically ask. Answers must be fully grounded in
the excerpt — no outside knowledge, no invented CLI flags or numbers not
present in the text. If the excerpt doesn't support a good question, return
fewer than 3 pairs, or none.

Return ONLY a JSON array like:
[{{"question": "...", "answer": "..."}}, ...]

Excerpt (service: {service}, title: {title}):
---
{content}}
---
"""


def chunk_content(content: str, max_words: int = 400) -> list[str]:
    words = content.split()
    return [" ".join(words[i:i + max_words]) for i in range(0, len(words), max_words)]


def parse_pairs(text: str) -> list[dict]:
    """Strip markdown fences if present, then parse the JSON array."""
    text = text.strip()
    if text.startswith("```json"):
        text = text[7:]
    if text.startswith("```"):
        text = text[3:]
    if text.endswith("```"):
        text = text[:-3]
    pairs = json.loads(text.strip())
    if not isinstance(pairs, list):
        return []
    return [p for p in pairs if isinstance(p, dict) and "question" in p and "answer" in p]


class QAProvider:
    """
    Tries Gemini first (with exponential backoff). After FALLBACK_AFTER
    consecutive failures it puts Gemini in cooldown for GEMINI_COOLDOWN
    seconds and falls back to Groq, also with backoff. Returns None only
    if both providers give up on a chunk.
    """

    def __init__(self):
        self.gemini_client = None
        self.groq_client = None

        gemini_key = os.environ.get("GEMINI_API_KEY")
        if gemini_key:
            from google import genai
            self.gemini_client = genai.Client(api_key=gemini_key)

        groq_key = os.environ.get("GROQ_API_KEY")
        if groq_key:
            from groq import Groq
            self.groq_client = Groq(api_key=groq_key)

        if not (self.gemini_client or self.groq_client):
            raise RuntimeError("Set GEMINI_API_KEY and/or GROQ_API_KEY in .env")

        self._gemini_failures = 0
        self._gemini_cooldown_until = 0.0

    def _gemini(self, prompt: str) -> str:
        resp = self.gemini_client.models.generate_content(model=GEMINI_MODEL, contents=prompt)
        return resp.text.strip()

    def _groq(self, prompt: str) -> str:
        resp = self.groq_client.chat.completions.create(
            model=GROQ_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
        )
        return resp.choices[0].message.content.strip()

    def _with_backoff(self, fn, provider_name: str, prompt: str) -> str | None:
        for attempt in range(MAX_RETRIES):
            try:
                return fn(prompt)
            except Exception as e:
                wait = min(2 ** attempt, 120) + random.random()
                print(f"[qa-gen] {provider_name} error ({e}); backoff {wait:.0f}s")
                time.sleep(wait)
        return None

    def generate(self, prompt: str) -> str | None:
        # --- Gemini (unless cooling down) ---
        if self.gemini_client and time.time() >= self._gemini_cooldown_until:
            text = self._with_backoff(self._gemini, "gemini", prompt)
            if text is not None:
                self._gemini_failures = 0
                return text
            self._gemini_failures += 1
            if self._gemini_failures >= FALLBACK_AFTER:
                self._gemini_cooldown_until = time.time() + GEMINI_COOLDOWN
                self._gemini_failures = 0
                print(f"[qa-gen] gemini failing repeatedly -> {GEMINI_COOLDOWN}s cooldown, using groq")

        # --- Groq fallback ---
        if self.groq_client:
            return self._with_backoff(self._groq, "groq", prompt)

        return None


def load_processed_docs() -> set[str]:
    """Docs that already have pairs in QA_FILE (for resume)."""
    done = set()
    if QA_FILE.exists():
        with QA_FILE.open() as f:
            for line in f:
                try:
                    done.add(json.loads(line).get("source_doc", ""))
                except json.JSONDecodeError:
                    continue
    return done


def generate_qa_pairs(limit_docs: int | None = None,
                      max_words_per_chunk: int = 400,
                      resume: bool = True):
    """
    Cost/time warning: this makes one API call per chunk. A full azure-docs
    pull is tens of thousands of articles -> can be 50k+ chunks. Start with
    limit_docs=200 to sanity check output quality, then resume across runs.
    Free-tier pace (~6s/call) means ~600 docs/hour. 1,500 Gemini requests/day
    is plenty when split across a few nights — resume makes that painless.
    """
    provider = QAProvider()
    processed = load_processed_docs() if resume else set()
    if processed:
        print(f"[qa-gen] resuming: {len(processed)} docs already done, skipping them")

    written, docs_done, docs_seen = 0, 0, 0

    with CORPUS_FILE.open() as fin, QA_FILE.open("a") as fout:  # append = resumable
        for i, line in enumerate(fin):
            doc = json.loads(line)
            doc_id = doc.get("path", f"line-{i}")

            if limit_docs and docs_seen >= limit_docs:
                break
            docs_seen += 1

            if resume and doc_id in processed:
                continue

            for chunk in chunk_content(doc["content"], max_words_per_chunk):
                prompt = QA_GEN_PROMPT.format(
                    service=doc.get("service", ""),
                    title=doc.get("title", ""),
                    content=chunk,
                )
                text = provider.generate(prompt)
                if text is None:
                    print(f"[qa-gen] both providers failed, skipping chunk (doc {i})")
                    time.sleep(REQUEST_DELAY)
                    continue

                try:
                    pairs = parse_pairs(text)
                except json.JSONDecodeError as e:
                    print(f"[qa-gen] bad JSON, skipping chunk (doc {i}): {e}")
                    time.sleep(REQUEST_DELAY)
                    continue

                for p in pairs:
                    fout.write(json.dumps({
                        "instruction": p["question"],
                        "response": p["answer"],
                        "source_doc": doc_id,
                    }) + "\n")
                    written += 1

                time.sleep(REQUEST_DELAY)  # ~10 RPM courtesy

            docs_done += 1
            processed.add(doc_id)
            if docs_done % 50 == 0:
                print(f"[qa-gen] {docs_done} new docs, {written} new pairs so far")

    print(f"[qa-gen] done: {written} new QA pairs -> {QA_FILE}")


def format_example(example, tokenizer):
    prompt = (
        f"<|im_start|>system\nYou are an assistant answering questions about "
        f"Microsoft Azure. Answer only from what you know to be accurate; say "
        f"so if you're unsure.<|im_end|>\n"
        f"<|im_start|>user\n{example['instruction']}<|im_end|>\n"
        f"<|im_start|>assistant\n{example['response']}<|im_end|>"
    )
    tokenized = tokenizer(prompt, truncation=True, max_length=1024, padding="max_length")
    tokenized["labels"] = tokenized["input_ids"].copy()
    return tokenized


def finetune():
    """
    QLoRA fine-tune. Needs a GPU with >=16GB VRAM for a 3B model in 4-bit.
    Teaches response format and grounding behavior on top of the base model's
    existing knowledge — it does not replace RAG for keeping facts current.
    """
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL, quantization_config=bnb_config, device_map="auto"
    )
    model = prepare_model_for_kbit_training(model)

    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    dataset = load_dataset("json", data_files=str(QA_FILE), split="train")
    dataset = dataset.train_test_split(test_size=0.1, seed=42)  # keep a real eval split
    train_ds = dataset["train"].map(lambda ex: format_example(ex, tokenizer))
    eval_ds = dataset["test"].map(lambda ex: format_example(ex, tokenizer))

    args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=4,
        num_train_epochs=3,
        learning_rate=2e-4,
        fp16=False,
        bf16=True,
        logging_steps=20,
        eval_strategy="steps",
        eval_steps=200,
        save_strategy="steps",
        save_steps=200,
        save_total_limit=2,
        load_best_model_at_end=True,
        report_to="none",
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False),
    )

    trainer.train()
    model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    print(f"[finetune] adapter saved to {OUTPUT_DIR}")


def push_to_hub():
    """
    Uploads the LoRA adapter (~30-60MB, free) to the HF Hub. The repo must
    exist or be auto-created (needs HF_TOKEN with write access). Users load it
    with PeftModel.from_pretrained(BASE_MODEL, repo_id). If you want the
    merged full model (~6GB), do the merge in Colab free tier and push there.
    """
    if not HF_MODEL_REPO:
        print("[push] HF_MODEL_REPO not set — skipping. Set it in .env, e.g. "
              "HF_MODEL_REPO=yourname/azure-slm-lora")
        return

    from huggingface_hub import HfApi

    api = HfApi()
    api.create_repo(repo_id=HF_MODEL_REPO, repo_type="model", exist_ok=True)
    api.upload_folder(folder_path=OUTPUT_DIR, repo_id=HF_MODEL_REPO, repo_type="model")
    print(f"[push] adapter uploaded to https://huggingface.co/{HF_MODEL_REPO}")


if __name__ == "__main__":
    # Step 1 — sanity check on a small slice first. Re-run to resume.
    generate_qa_pairs(limit_docs=200, resume=True)

    # Step 2 — only after you've eyeballed azure_qa_pairs.jsonl for quality.
    # finetune()

    # Step 3 — after training.
    # push_to_hub()
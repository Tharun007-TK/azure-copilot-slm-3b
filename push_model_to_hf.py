"""
Run this AFTER train.ipynb produces an adapter in ./azure-slm-lora.

Merges the LoRA adapter into the base model weights, producing a standalone
model, then pushes it to the HF Hub. This merged fp16/bf16 model is what you
later quantize with AutoAWQ for vLLM serving — that's a separate step, not
done here, because AWQ quantization needs its own calibration pass.
"""

from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from huggingface_hub import login, HfApi

BASE_MODEL = "Qwen/Qwen2.5-3B-Instruct"
ADAPTER_DIR = Path("./azure-slm-lora")
MERGED_DIR = Path("./azure-slm-merged")
HF_MODEL_REPO = "your-username/azure-slm-qwen2.5-3b"   # <-- change this
PUSH_PUBLIC = False                                     # <-- flip deliberately


MODEL_CARD = f"""---
license: apache-2.0
base_model: {BASE_MODEL}
tags:
- azure
- lora
- fine-tuned
- chrome-extension
---

# Azure SLM ({BASE_MODEL} + LoRA)

Fine-tuned on Q/A pairs generated from Microsoft's official Azure
documentation, for use as the generator in a RAG pipeline behind a Chrome
extension. **This model is not a standalone source of truth for Azure
facts** — it's tuned for answer format and grounding behavior on top of
retrieved context, not for parametric recall of Azure specifics. Serve it
with a retrieval step (Azure AI Search over the azure-docs corpus); don't
query it without retrieved context and expect current, correct answers.

- Base model: `{BASE_MODEL}`
- Method: QLoRA (4-bit NF4), merged into base weights for this repo
- Training data: [your dataset repo link here]
- Intended serving: quantized further (AWQ) and served via vLLM

## Known limitations
- Small model, small dataset — expect gaps on edge-case Azure services or
  recently released features not covered in the training corpus.
- Not evaluated against a formal benchmark; validated only on a held-out
  split of the same generated-QA distribution, which likely overstates
  real-world accuracy versus genuinely novel questions.
"""


def merge_adapter():
    print(f"[merge] loading base model {BASE_MODEL}")
    base = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL, torch_dtype=torch.bfloat16, device_map="auto"
    )
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)

    print(f"[merge] loading adapter from {ADAPTER_DIR}")
    model = PeftModel.from_pretrained(base, str(ADAPTER_DIR))

    print("[merge] merging LoRA weights into base model")
    model = model.merge_and_unload()

    MERGED_DIR.mkdir(exist_ok=True)
    model.save_pretrained(str(MERGED_DIR))
    tokenizer.save_pretrained(str(MERGED_DIR))
    print(f"[merge] merged model saved to {MERGED_DIR}")
    return model, tokenizer


def sanity_check(model, tokenizer, question="How do I create an Azure Storage account using the CLI?"):
    """
    Don't push a merged model you haven't sanity-checked. This is a raw
    generation test with NO retrieval context — expect it to be shakier
    than your eventual RAG-backed answers. It's here to catch outright
    broken merges (garbled output, wrong EOS handling), not to validate
    factual accuracy.
    """
    prompt = (
        f"<|im_start|>system\nYou are an assistant answering questions about "
        f"Microsoft Azure.<|im_end|>\n<|im_start|>user\n{question}<|im_end|>\n"
        f"<|im_start|>assistant\n"
    )
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    output = model.generate(**inputs, max_new_tokens=200, do_sample=False)
    text = tokenizer.decode(output[0], skip_special_tokens=True)
    print("[sanity-check] sample output:\n", text)
    confirm = input("\nDoes this look like coherent, on-topic output? (y/n): ")
    if confirm.strip().lower() != "y":
        raise RuntimeError("Merge looks broken — stop and debug before pushing to the Hub.")


def push_to_hub():
    login()
    api = HfApi()
    api.create_repo(HF_MODEL_REPO, private=not PUSH_PUBLIC, exist_ok=True)

    api.upload_folder(
        folder_path=str(MERGED_DIR),
        repo_id=HF_MODEL_REPO,
    )
    api.upload_file(
        path_or_fileobj=MODEL_CARD.encode(),
        path_in_repo="README.md",
        repo_id=HF_MODEL_REPO,
    )
    print(f"[push] pushed to https://huggingface.co/{HF_MODEL_REPO}")


if __name__ == "__main__":
    model, tokenizer = merge_adapter()
    sanity_check(model, tokenizer)
    push_to_hub()

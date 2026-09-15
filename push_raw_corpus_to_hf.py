import json
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()
from datasets import Dataset
from huggingface_hub import login

CORPUS_FILE = Path("./azure_docs_corpus.jsonl")
HF_DATASET_REPO = "Tharun007/azure-docs-raw"  # Change this to your preferred repo
PUSH_PUBLIC = False

def push_raw_corpus():
    print("Loading raw corpus...")
    data = []
    with CORPUS_FILE.open("r", encoding="utf-8") as f:
        for line in f:
            data.append(json.loads(line))
    
    print(f"Loaded {len(data)} documents. Building dataset...")
    ds = Dataset.from_list(data)
    
    # Login to HF (will use cached token or prompt)
    login()
    
    print(f"Pushing to Hugging Face Hub: {HF_DATASET_REPO}...")
    ds.push_to_hub(HF_DATASET_REPO, private=not PUSH_PUBLIC)
    print("Done!")

if __name__ == "__main__":
    push_raw_corpus()

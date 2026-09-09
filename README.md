# Azure Docs Scrapper

A small Python pipeline for collecting Microsoft Azure documentation, generating grounded question-and-answer data, and preparing datasets and adapters for Hugging Face workflows.

## What it does

1. Clones or updates `MicrosoftDocs/azure-docs` and extracts usable Markdown from `articles/`.
2. Writes cleaned documentation records to `azure_docs_corpus.jsonl`.
3. Uses an Anthropic model to generate instruction/response pairs from document chunks.
4. Filters, deduplicates, splits, and optionally uploads the QA dataset to Hugging Face.
5. Provides a training notebook and a script for merging and publishing a LoRA adapter.

The generated model is intended to answer using retrieved documentation. It is not a replacement for retrieval and should not be treated as a current or authoritative source of Azure facts on its own.

## Requirements

- Python 3.10 or newer
- Git
- An Anthropic API key for QA generation
- A Hugging Face account and token for dataset/model uploads
- A CUDA-capable GPU with sufficient VRAM for the training notebook

Install the Python dependencies:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## Pipeline

Run the stages in this order:

```powershell
python azure_docs_collector.py
```

Set `ANTHROPIC_API_KEY` in the environment, then generate QA pairs. The generator defaults to processing the whole corpus; start with a small `limit_docs` value while checking quality and API cost.

```powershell
python generate_qa_and_finetune.py
```

Before preparing a Hugging Face dataset, update `HF_DATASET_REPO` in `prepare_hf_dataset.py`, choose the intended visibility, and authenticate with Hugging Face.

```powershell
python prepare_hf_dataset.py
```

Open `train.ipynb` to train the adapter. After training, update `HF_MODEL_REPO` in `push_model_to_hf.py`, review the model configuration, and run:

```powershell
python push_model_to_hf.py
```

## Important configuration

- `azure_docs_collector.py`: source repository, corpus output path, and minimum document length.
- `generate_qa_and_finetune.py`: base model, QA output path, chunk size, and Anthropic model.
- `prepare_hf_dataset.py`: dataset repository and public/private setting.
- `push_model_to_hf.py`: base model, adapter paths, model repository, and public/private setting.

Do not commit API keys, access tokens, local model weights, virtual environments, or generated datasets unless you deliberately intend to publish them.

## Data and licensing

The project code is licensed under the MIT License; see [LICENSE](LICENSE).

The Azure documentation collected by this project comes from [MicrosoftDocs/azure-docs](https://github.com/MicrosoftDocs/azure-docs) and is available under the [Creative Commons Attribution 4.0 International license](https://creativecommons.org/licenses/by/4.0/). Generated corpora, QA pairs, and models may contain or derive from that documentation and should retain appropriate attribution and license information. Microsoft Azure and Microsoft Learn are trademarks of Microsoft Corporation.

The generated QA data is synthetic and only lightly quality-checked. Review it before using it for training or evaluation.

## Project status

This is an experimental research pipeline. Reproducibility and factual quality depend on the source documentation revision, generation model, prompts, training configuration, and retrieval setup.
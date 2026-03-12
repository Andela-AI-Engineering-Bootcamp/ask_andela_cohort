# Ask Andela

[![Python](https://img.shields.io/badge/python-3.12-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![ChromaDB](https://img.shields.io/badge/ChromaDB-1.5-E85D04?logo=databricks&logoColor=white)](https://www.trychroma.com/)
[![Gradio](https://img.shields.io/badge/Gradio-UI-FF7C00?logo=gradio&logoColor=white)](https://gradio.app/)
[![OpenRouter](https://img.shields.io/badge/OpenRouter-API-6E40C9?logo=openai&logoColor=white)](https://openrouter.ai/)
[![sentence-transformers](https://img.shields.io/badge/Embeddings-MiniLM--L6--v2-4CAF50?logo=huggingface&logoColor=white)](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2)
[![uv](https://img.shields.io/badge/uv-package%20manager-DE5FE9?logo=astral&logoColor=white)](https://docs.astral.sh/uv/)

> **An AI study buddy that knows your specific course.**

Ask Andela is a local RAG-powered Q&A assistant that answers student questions using the A3 AI Engineering Bootcamp's own course materials, channel discussions, and programme resources — not generic internet knowledge. It answers in the cohort's teaching style, cites its sources, and runs entirely on your laptop.

---

## Architecture

```
Student Question (Gradio UI)
         │
         ▼
   Query Embedding
   (all-MiniLM-L6-v2)
         │
         ▼
  ChromaDB Vector Store ──► Top-5 Relevant Chunks
  (local, persistent)        (with source metadata)
         │
         ▼
   Prompt Assembly
   [System Prompt + Context + Question]
         │
         ▼
  LLM via OpenRouter
  (Llama 3.1 8B · free tier)
         │
         ▼
  Answer + Source Citations (Gradio)
```

---

## Features

- **Curriculum-grounded answers** — retrieves from the cohort's own documents before generating, never hallucinates generic content
- **Source citations** — every answer shows which document(s) it was drawn from
- **Local & free to run** — ChromaDB runs on SQLite, embeddings run on CPU, LLM served via OpenRouter free tier
- **Gradio UI** — single-page interface ready for live demo; supports `--share` for a public link
- **Evaluation harness** — compares Baseline vs RAG vs Fine-tuned+RAG on 12 held-out questions
- **Fine-tuning ready** — stub in `notebooks/` for QLoRA adapter integration (Day 2)

---

## Corpus

13 documents across two categories, totalling ~62k characters:

| File | Category |
|------|----------|
| `channel_general_chat.txt` | Discourse channel |
| `channel_igniters_information.txt` | Discourse channel |
| `channel_protocol_daily_updates.txt` | Discourse channel |
| `channel_solidroad_behavioral_assignments.txt` | Discourse channel |
| `channel_staff_announcements.txt` | Discourse channel |
| `resource_additional_learning_material.txt` | Course resource |
| `resource_community_map.txt` | Course resource |
| `resource_deliverables.txt` | Course resource |
| `resource_group_project_guidelines.txt` | Course resource |
| `resource_program_expectations.txt` | Course resource |
| `resource_solidroad.txt` | Course resource |
| `resource_toolkit.txt` | Course resource |
| `resource_welcome_page.txt` | Course resource |

To add more documents (PDFs, additional `.txt` exports), drop them into `data/` and re-run `python ingest.py`.

---

## Project Structure

```
ask_andela_cohort/
│
├── rag/                          # Core RAG package
│   ├── config.py                 # All constants: paths, models, chunking params
│   ├── loader.py                 # Load .txt / .pdf files from data/
│   ├── chunker.py                # Token-based chunking (400 tok / 50 overlap)
│   ├── vectorstore.py            # Embedding model + ChromaDB build/load
│   ├── retrieval.py              # Embed query → top-K chunks
│   ├── llm.py                    # OpenRouter client, prompts, generation
│   ├── pipeline.py               # ask_andela() — full end-to-end RAG call
│   └── __init__.py
│
├── data/                         # Source documents (txt + optional PDFs)
│
├── notebooks/
│   └── ask_andela_rag_pipeline.ipynb   # Exploratory notebook (all sections)
│
├── scripts/                      # Reserved for fine-tuning scripts (Day 2)
│
├── app.py                        # Gradio UI  →  python app.py
├── ingest.py                     # Build vectorstore  →  python ingest.py
├── evaluate.py                   # Run evaluation  →  python evaluate.py
│
├── .env                          # API keys (never commit)
├── pyproject.toml
└── requirements.txt
```

---

## Quickstart

### Prerequisites

- Python 3.12
- [uv](https://docs.astral.sh/uv/) (recommended) or pip
- An [OpenRouter](https://openrouter.ai/) API key — free tier is sufficient

### 1. Clone & install

```bash
git clone <repository-url>
cd ask_andela_cohort

# With uv (recommended)
uv venv && uv sync

# Or with pip
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Set your API key

```bash
cp .env.example .env   # or create .env manually
```

Add to `.env`:

```
OPENROUTER_API_KEY=sk-or-...
```

### 3. Build the vector store

Run once. Re-run whenever you add documents to `data/`.

```bash
python ingest.py
```

Expected output:
```
[1/3] Loading documents from .../data ...
      13 documents loaded
[2/3] Chunking ...
      44 chunks created
[3/3] Building vector store ...
✓  Done. Run `python app.py` to start the UI.
```

### 4. Launch the UI

```bash
python app.py           # local only  →  http://127.0.0.1:7860
python app.py --share   # public Gradio link (for live demo)
```

---

## Changing the LLM

All model configuration lives in `rag/config.py`. Swap one line:

```python
# Free tier (default)
LLM_MODEL = "meta-llama/llama-3.1-8b-instruct:free"

# Better quality (costs ~$0.001 / query)
LLM_MODEL = "openai/gpt-4o-mini"
LLM_MODEL = "anthropic/claude-3-haiku"

# Other free options
LLM_MODEL = "qwen/qwen-2.5-7b-instruct:free"
LLM_MODEL = "mistralai/mistral-7b-instruct:free"
```

---

## Evaluation

Runs 12 held-out questions comparing Baseline (no RAG) vs RAG pipeline. Results saved to `eval_results.json`.

```bash
python evaluate.py              # baseline + RAG  (~24 API calls, ~2 min)
python evaluate.py --no-baseline  # RAG only  (~12 calls, saves API spend)
```

Results schema per question:

| Column | Description |
|--------|-------------|
| `baseline_answer` | LLM answer with no context |
| `rag_answer` | LLM answer with retrieved context |
| `finetuned_rag_answer` | *(Day 2)* Fine-tuned model + RAG |
| `*_accuracy` / `*_specificity` / `*_conciseness` | Manual scores 1–3 |

---

## Roadmap

- [x] RAG pipeline (ChromaDB + OpenRouter)
- [x] Gradio UI with source citations
- [x] Evaluation harness (Baseline vs RAG)
- [ ] QLoRA fine-tuning on cohort Q&A pairs (Qwen2.5-0.5B / Phi-3-mini, Google Colab T4)
- [ ] Fine-tuned adapter integration (`scripts/finetune.py`)
- [ ] Three-way eval comparison for demo (Baseline → RAG → Fine-tuned+RAG)

---

## Team

| Name | Role |
|------|------|
| Eben | Product, PRD |
| Amit | Data collection & Q&A pair generation |
| Phillip | RAG pipeline |
| Kayode | Fine-tuning (QLoRA) |
| Tunde | Gradio UI |
| Mugao | Evaluation |

---

## Tech Stack

| Component | Tool |
|-----------|------|
| Vector store | ChromaDB (local SQLite) |
| Embeddings | `sentence-transformers/all-MiniLM-L6-v2` |
| LLM gateway | OpenRouter |
| Default model | `meta-llama/llama-3.1-8b-instruct:free` |
| Fine-tuning | PEFT + QLoRA (`trl`, `transformers`) |
| UI | Gradio |
| Doc parsing | Plain text + PyMuPDF (PDFs) |
| Package manager | uv |

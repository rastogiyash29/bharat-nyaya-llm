# BharatNyayaLLM

**A fine-tuned LLM for Indian criminal law — mapping crime descriptions to Bharatiya Nyaya Sanhita (BNS) sections**

India replaced its 163-year-old criminal law (Indian Penal Code, 1860) with the Bharatiya Nyaya Sanhita (BNS) on 1 July 2024. Every lawyer, judge, police officer, and law student now needs to translate between the old law and the new. This project fine-tunes Mistral 7B to do that automatically.

---

## What Does This Project Do?

**Input (user types):**
> "A person stole a mobile phone from someone's pocket in a crowded market"

**Output (model returns):**
> **BNS Section 305(a)** — Theft
> Previously IPC Section 379.
> Punishment: Imprisonment up to 3 years, or fine, or both.
> Key change from IPC: BNS adds subsections for theft with specific aggravating factors.

**Input:**
> "What is the BNS equivalent of IPC Section 302?"

**Output:**
> IPC Section 302 (Murder) is now **BNS Section 103(1)**.
> Punishment: Death or imprisonment for life, and fine.
> BNS retains the same punishment but reorganizes murder under Chapter VI (Of Offences Affecting the Human Body).

---

## What You'll Learn by Building This

| Skill | What Exactly | Resume Keyword |
|-------|-------------|----------------|
| **LLM Fine-Tuning** | Take a general-purpose model and teach it Indian criminal law | QLoRA, LoRA, PEFT, SFT |
| **Data Engineering** | Collect, clean, merge, and format legal datasets from 5+ sources | Dataset Curation, Data Pipeline |
| **Synthetic Data Generation** | Use GPT-4o/Claude to create training examples programmatically | Synthetic Data, Prompt Engineering |
| **Quantization** | Compress a 14GB model to 4GB while keeping 92% quality | QLoRA, GGUF, 4-bit Quantization |
| **Experiment Tracking** | Log every training run, compare results, manage model versions | MLflow, Model Registry |
| **LLM Evaluation** | Measure if the model's answers are faithful, relevant, and correct | DeepEval, RAGAS, LLM-as-Judge |
| **Model Deployment** | Serve the model as a chat app anyone can use | GGUF, Gradio, HuggingFace Spaces |
| **Domain Expertise** | Understand IPC→BNS transition, Indian legal NLP landscape | Indian Legal AI, BNS, NLP |

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                        DATA PIPELINE                                │
│                                                                     │
│  ┌──────────────┐  ┌──────────────┐  ┌───────────────────────────┐ │
│  │ Aalap Dataset │  │ BNS CSV      │  │ IPC-BNS Mapping JSON      │ │
│  │ (22K instrs)  │  │ (358 sections)│  │ (511 IPC → 358 BNS)      │ │
│  └──────┬───────┘  └──────┬───────┘  └───────────┬───────────────┘ │
│         │                 │                       │                  │
│         │    ┌────────────┴───────────────────────┘                  │
│         │    │                                                       │
│         ▼    ▼                                                       │
│  ┌─────────────────────┐     ┌──────────────────────────────┐       │
│  │ Merge & Format      │     │ Synthetic QA Generation      │       │
│  │ (Alpaca JSON)       │     │ (Claude/GPT-4o API)          │       │
│  └─────────┬───────────┘     └──────────────┬───────────────┘       │
│            │                                 │                       │
│            └────────────┬────────────────────┘                       │
│                         ▼                                            │
│              ┌─────────────────────┐                                 │
│              │ Combined Dataset    │                                 │
│              │ ~25K-30K pairs      │                                 │
│              │ (train/val/test)    │                                 │
│              └─────────┬───────────┘                                 │
└────────────────────────┼────────────────────────────────────────────┘
                         │
┌────────────────────────┼────────────────────────────────────────────┐
│                   TRAINING PIPELINE                                  │
│                        ▼                                             │
│  ┌─────────────────────────────────────┐                            │
│  │ Mistral 7B v0.3 (4-bit quantized)  │                            │
│  │ + QLoRA adapters (r=16, alpha=32)   │                            │
│  │ Unsloth + TRL SFTTrainer            │                            │
│  │ RTX 3070 Ti (8GB VRAM)              │                            │
│  └─────────────────┬───────────────────┘                            │
│                    │                                                 │
│         ┌──────────┼──────────┐                                     │
│         ▼          ▼          ▼                                     │
│  ┌──────────┐ ┌──────────┐ ┌──────────┐                            │
│  │ Run 1    │ │ Run 2    │ │ Run 3    │  ← MLflow tracks all       │
│  │ r=16     │ │ r=32     │ │ r=16     │                            │
│  │ lr=2e-4  │ │ lr=2e-4  │ │ lr=1e-4  │                            │
│  └──────────┘ └──────────┘ └──────────┘                            │
│         │          │          │                                     │
│         └──────────┼──────────┘                                     │
│                    ▼                                                 │
│         ┌─────────────────────┐                                     │
│         │ MLflow Model        │                                     │
│         │ Registry            │                                     │
│         │ → @champion model   │                                     │
│         └─────────┬───────────┘                                     │
└───────────────────┼─────────────────────────────────────────────────┘
                    │
┌───────────────────┼─────────────────────────────────────────────────┐
│              EVALUATION                                              │
│                   ▼                                                  │
│  ┌────────────────────────────────────────────────────┐             │
│  │ 100-question test set                              │             │
│  │ (50 BNS mapping + 50 legal QA)                     │             │
│  │                                                    │             │
│  │ Metrics:                                           │             │
│  │  • DeepEval: faithfulness, relevancy, hallucination│             │
│  │  • LLM-as-Judge: GPT-4o rates outputs 1-5         │             │
│  │  • Manual: are BNS section numbers correct?        │             │
│  │                                                    │             │
│  │ Compare: Base Mistral vs Fine-tuned vs Aalap       │             │
│  └────────────────────────┬───────────────────────────┘             │
└───────────────────────────┼─────────────────────────────────────────┘
                            │
┌───────────────────────────┼─────────────────────────────────────────┐
│                    DEPLOYMENT                                        │
│                           ▼                                          │
│  ┌───────────────────────────────────┐                              │
│  │ GGUF Conversion (Q4_K_M, ~4GB)   │                              │
│  └───────────────┬───────────────────┘                              │
│                  │                                                   │
│       ┌──────────┼──────────────┐                                   │
│       ▼          ▼              ▼                                   │
│  ┌─────────┐ ┌──────────┐ ┌───────────────┐                       │
│  │ HF Hub  │ │ Gradio   │ │ Ollama        │                       │
│  │ (model) │ │ (HF      │ │ (local on     │                       │
│  │         │ │  Spaces)  │ │  any machine) │                       │
│  └─────────┘ └──────────┘ └───────────────┘                       │
└─────────────────────────────────────────────────────────────────────┘
```

---

## Tech Stack

### Core Training

| Library | What It Does | Why We Need It |
|---------|-------------|----------------|
| **[Unsloth](https://github.com/unslothai/unsloth)** | Makes fine-tuning 2-5x faster, uses 60-80% less VRAM | Without it, Mistral 7B QLoRA wouldn't fit on 8GB VRAM comfortably |
| **[transformers](https://github.com/huggingface/transformers)** | HuggingFace's model loading, tokenization, inference | The foundation everything else builds on |
| **[trl](https://github.com/huggingface/trl)** | SFTTrainer for supervised fine-tuning | Clean API for instruction-tuning with LoRA |
| **[peft](https://github.com/huggingface/peft)** | LoRA/QLoRA adapter implementation | Trains only 1-5% of parameters instead of all 7 billion |
| **[bitsandbytes](https://github.com/TimDettmers/bitsandbytes)** | 4-bit quantization (NF4) | Compresses model from 14GB to ~4GB in VRAM |
| **[datasets](https://github.com/huggingface/datasets)** | Load and process datasets from HuggingFace Hub | Standard way to handle training data |
| **[accelerate](https://github.com/huggingface/accelerate)** | Distributed training, mixed precision | Handles GPU memory management and training optimization |

### Experiment Tracking

| Library | What It Does | Why We Need It |
|---------|-------------|----------------|
| **[mlflow](https://mlflow.org/)** | Track experiments, compare runs, manage model versions | The #1 MLOps tool — appears in 40%+ of Indian AI job postings |

### Evaluation

| Library | What It Does | Why We Need It |
|---------|-------------|----------------|
| **[deepeval](https://github.com/confident-ai/deepeval)** | 50+ LLM evaluation metrics | Automated faithfulness, relevancy, hallucination scoring |

### Deployment

| Library | What It Does | Why We Need It |
|---------|-------------|----------------|
| **[gradio](https://gradio.app/)** | Build chat UI in Python | Fastest way to create a demo anyone can try |
| **[llama-cpp-python](https://github.com/abetlen/llama-cpp-python)** | Run GGUF models on CPU/GPU | Inference engine for quantized models |
| **[huggingface_hub](https://github.com/huggingface/huggingface-cli)** | Push models and apps to HuggingFace | Free hosting for model + demo |

### Data Processing

| Library | What It Does | Why We Need It |
|---------|-------------|----------------|
| **[pandas](https://pandas.pydata.org/)** | Tabular data manipulation | Merging CSV datasets, data exploration |
| **[anthropic](https://github.com/anthropics/anthropic-sdk-python)** or **[openai](https://github.com/openai/openai-python)** | API clients for synthetic data generation | Generate BNS-specific QA pairs |

---

## Project Structure

```
bharatnyaya-llm/
│
├── README.md                          ← You are here
│
├── data/
│   ├── raw/                           ← Downloaded datasets (gitignored)
│   │   ├── aalap_instruction_dataset/ ← 22K legal instruction pairs
│   │   ├── bns_sections.csv           ← All 358 BNS sections
│   │   ├── ipc_to_bns_mapping.json    ← IPC→BNS section mapping
│   │   ├── ipc_complete.csv           ← All 511 IPC sections
│   │   └── constitution_3300.json     ← 3,300 Constitution QA pairs
│   │
│   ├── processed/                     ← Cleaned, merged, formatted
│   │   ├── bns_mapping_qa.json        ← Generated IPC↔BNS QA pairs
│   │   ├── synthetic_qa.json          ← GPT-4o/Claude generated QA
│   │   └── combined_dataset.json      ← Final merged training data
│   │
│   └── splits/                        ← Train/val/test splits
│       ├── train.json                 ← 85% (~21K-25K pairs)
│       ├── val.json                   ← 10% (~2.5K-3K pairs)
│       └── test.json                  ← 5% (~1.2K-1.5K pairs)
│
├── notebooks/
│   ├── 01_data_exploration.ipynb      ← Stage 1: Explore all datasets
│   ├── 02_data_preparation.ipynb      ← Stage 2: Merge, format, split
│   ├── 03_synthetic_generation.ipynb  ← Stage 2: Generate synthetic QA
│   ├── 04_fine_tuning.ipynb           ← Stage 3: Unsloth QLoRA training
│   ├── 05_mlflow_experiments.ipynb    ← Stage 4: Multi-run experiments
│   └── 06_evaluation.ipynb           ← Stage 5: DeepEval + manual eval
│
├── src/
│   ├── data/
│   │   ├── download.py                ← Download all datasets
│   │   ├── prepare.py                 ← Merge and format datasets
│   │   └── synthetic.py               ← Synthetic QA generation
│   │
│   ├── training/
│   │   ├── train.py                   ← Fine-tuning script (CLI)
│   │   └── config.py                  ← Training hyperparameters
│   │
│   ├── evaluation/
│   │   ├── evaluate.py                ← Run DeepEval metrics
│   │   ├── judge.py                   ← LLM-as-judge evaluation
│   │   └── test_questions.json        ← 100 held-out test questions
│   │
│   └── deployment/
│       ├── convert_gguf.py            ← GGUF conversion script
│       └── app.py                     ← Gradio chat interface
│
├── models/                            ← Saved models (gitignored)
│   ├── checkpoints/                   ← Training checkpoints
│   ├── lora_adapter/                  ← Trained LoRA weights
│   └── gguf/                          ← Quantized GGUF files
│
├── mlruns/                            ← MLflow experiment data (gitignored)
│
├── requirements.txt                   ← Python dependencies
├── .gitignore
└── .env.example                       ← API keys template
```

---

## Hardware Requirements

| Component | Minimum | What We're Using |
|-----------|---------|-----------------|
| **GPU** | 8GB VRAM (NVIDIA, CUDA capable) | RTX 3070 Ti (8GB GDDR6X) |
| **System RAM** | 16GB | 32GB |
| **Disk** | 30GB free | Enough |
| **CUDA** | 12.1+ | Latest |
| **Python** | 3.10 or 3.11 | 3.11 |

Fine-tuning Mistral 7B with QLoRA uses ~6.5GB VRAM. The RTX 3070 Ti's 8GB gives ~1.5GB headroom.

---

## Datasets

### Primary Training Data

| Dataset | Source | Size | Format | What It Contains |
|---------|--------|------|--------|-----------------|
| **Aalap Instruction Dataset** | [HuggingFace](https://huggingface.co/datasets/opennyaiorg/aalap_instruction_dataset) | 22,272 pairs | Instruction JSON | Legal QA, argument generation, timeline creation from SC/HC judgments. **Already formatted for SFT.** |
| **BNS All 358 Sections** | [Kaggle](https://www.kaggle.com/datasets/nandr39/bharatiya-nyaya-sanhita-dataset-bns) | 358 rows | CSV | Every BNS section with text, chapter, description |
| **IPC-BNS Mapping JSON** | [GitHub](https://github.com/Pranshu321/IndLegal) | 511 mappings | JSON | Structured mapping: IPC section number → BNS section number |
| **IPC Complete Dataset** | [Kaggle](https://www.kaggle.com/datasets/omdabral/indian-penal-code-complete-dataset) | 511 rows | CSV | All IPC sections with descriptions and punishments |
| **Constitution Instructions** | [HuggingFace](https://huggingface.co/datasets/nisaar/Articles_Constitution_3300_Instruction_Set) | 3,300 pairs | Instruction JSON | QA pairs on Indian Constitution articles |

### Supplementary (for enrichment)

| Dataset | Source | Use |
|---------|--------|-----|
| **Indian Legal Summaries (Alpaca)** | [HuggingFace](https://huggingface.co/datasets/andrewmos/indian-legal-summaries-alpaca-format) | Already in Alpaca format, direct merge |
| **Indian Legal Texts Fine-Tuning** | [Kaggle](https://www.kaggle.com/datasets/akshatgupta7/llm-fine-tuning-dataset-of-indian-legal-texts) | IPC/CrPC QA pairs |
| **IL-TUR Benchmark** | [HuggingFace](https://huggingface.co/datasets/Exploration-Lab/IL-TUR) | Evaluation only (8 legal tasks, English + Hindi) |

### IPC-BNS Mapping Reference (for synthetic data generation)

| Source | Format | URL |
|--------|--------|-----|
| UP Police Comparative Table | PDF | [uppolice.gov.in](https://uppolice.gov.in/site/writereaddata/siteContent/Three%20New%20Major%20Acts/202406281710564823BNS_IPC_Comparative.pdf) |
| BPRD Comparison Summary | PDF | [bprd.nic.in](https://bprd.nic.in/uploads/pdf/COMPARISON%20SUMMARY%20BNS%20to%20IPC%20.pdf) |
| NLS Bangalore Table | PDF | [nls.ac.in](https://www.nls.ac.in/wp-content/uploads/2023/12/IPC-BNS_Table.pdf) |
| Devgan.in BNS sections | HTML (scrapeable) | [devgan.in/all_sections_bns.php](https://devgan.in/all_sections_bns.php) |
| TrueLawyer BNS with IPC mapping | Web | [truelawyer.in/bharatiya-nyaya-sanhita](https://truelawyer.in/bharatiya-nyaya-sanhita) |

---

## Base Model: Why Mistral 7B v0.3

| Factor | Mistral 7B v0.3 | Llama 3.1 8B | Qwen 2.5 7B |
|--------|-----------------|--------------|-------------|
| **Proven for Indian Legal?** | **Yes** — Aalap model built on Mistral, matches GPT-3.5-turbo on legal tasks | No | No |
| **Fits 8GB VRAM (QLoRA)?** | **Yes** (~6.5GB) | Tight (~7GB) | Yes (~6.5GB) |
| **License** | Apache 2.0 | Llama Community | Apache 2.0 |
| **Fine-tuning docs** | Most tutorials available | Good | Good |
| **Context window** | 32K | 128K | 128K |

**Decision:** Mistral 7B v0.3 — proven track record (Aalap), comfortable VRAM fit, Apache 2.0 license, most fine-tuning tutorials available.

**HuggingFace link:** [mistralai/Mistral-7B-Instruct-v0.3](https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.3)

**Existing Indian legal model to benchmark against:** [opennyaiorg/Aalap-Mistral-7B-v0.1-bf16](https://huggingface.co/opennyaiorg/Aalap-Mistral-7B-v0.1-bf16)

---

# Stage-by-Stage Roadmap

## Stage 0 — Environment Setup

**Objective:** Get your RTX 3070 Ti PC ready for training. Set up remote access from your Mac.

**Time:** ~1 hour

### Step 0.1: Set up VS Code Tunnel (on your Windows/Linux PC)

```bash
# Option A: VS Code CLI Tunnel (simplest)
# Download VS Code on your PC, then from terminal:
code tunnel

# This gives you a URL you can open from your Mac browser
# Full VS Code with terminal access to your GPU machine
```

```bash
# Option B: SSH + Tailscale (works from anywhere)
# Install Tailscale on both Mac and PC
# Both devices get a stable IP that works from any network
# Then use VS Code Remote-SSH extension on your Mac
```

### Step 0.2: Install CUDA and Python (on your PC)

```bash
# 1. Verify NVIDIA driver is installed
nvidia-smi
# Should show: RTX 3070 Ti, Driver Version, CUDA Version

# 2. Install Miniconda (if not already installed)
# Download from: https://docs.conda.io/en/latest/miniconda.html

# 3. Create project environment
conda create -n bharatnyaya python=3.11 -y
conda activate bharatnyaya

# 4. Install PyTorch with CUDA
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# 5. Verify GPU access
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'GPU: {torch.cuda.get_device_name(0)}'); print(f'VRAM: {torch.cuda.get_device_properties(0).total_mem / 1e9:.1f} GB')"
# Expected output:
# CUDA available: True
# GPU: NVIDIA GeForce RTX 3070 Ti
# VRAM: 8.0 GB
```

### Step 0.3: Install Project Dependencies

```bash
# Core training stack
pip install "unsloth[cu121-torch250] @ git+https://github.com/unslothai/unsloth.git"
pip install transformers trl peft accelerate bitsandbytes datasets

# Experiment tracking
pip install mlflow

# Evaluation
pip install deepeval

# Deployment
pip install gradio llama-cpp-python huggingface_hub

# Data processing
pip install pandas openpyxl anthropic openai

# Jupyter
pip install jupyterlab ipywidgets
```

### Step 0.4: Set Up Project Directory

```bash
mkdir -p bharatnyaya-llm/{data/{raw,processed,splits},notebooks,src/{data,training,evaluation,deployment},models/{checkpoints,lora_adapter,gguf},mlruns}
cd bharatnyaya-llm
git init
```

### Step 0.5: Create .gitignore

```
# Models (too large for git)
models/
mlruns/
data/raw/

# Environment
.env
__pycache__/
*.pyc
.ipynb_checkpoints/

# OS
.DS_Store
Thumbs.db
```

### Step 0.6: Create .env.example

```
# For synthetic data generation (Stage 2)
# You only need ONE of these
ANTHROPIC_API_KEY=sk-ant-...
OPENAI_API_KEY=sk-...

# For pushing to HuggingFace (Stage 6)
HF_TOKEN=hf_...
```

### Checkpoint: How to Verify Stage 0

```bash
# Run this script — all 4 checks should pass
python -c "
import torch
assert torch.cuda.is_available(), 'CUDA not available'
print(f'[PASS] GPU: {torch.cuda.get_device_name(0)}')

import unsloth
print(f'[PASS] Unsloth installed')

import mlflow
print(f'[PASS] MLflow installed')

import deepeval
print(f'[PASS] DeepEval installed')

print('\nAll checks passed. Ready for Stage 1.')
"
```

---

## Stage 1 — Data Collection & Exploration

**Objective:** Download all datasets and understand what we're working with before touching anything.

**Time:** ~2-3 hours

### Step 1.1: Download Datasets

```bash
# Activate environment
conda activate bharatnyaya

# Create download script or run manually:
cd bharatnyaya-llm
```

**From HuggingFace (via Python):**
```python
from datasets import load_dataset

# 1. Aalap Instruction Dataset (22,272 legal instruction pairs)
aalap = load_dataset("opennyaiorg/aalap_instruction_dataset")
aalap["train"].to_json("data/raw/aalap_instructions.json")

# 2. Constitution Instructions (3,300 pairs)
constitution = load_dataset("nisaar/Articles_Constitution_3300_Instruction_Set")
constitution["train"].to_json("data/raw/constitution_instructions.json")

# 3. Indian Legal Summaries (Alpaca format — ready to use)
summaries = load_dataset("andrewmos/indian-legal-summaries-alpaca-format")
summaries["train"].to_json("data/raw/legal_summaries_alpaca.json")

# 4. Indian Penal Code
ipc_hf = load_dataset("harshitv804/Indian_Penal_Code")
ipc_hf["train"].to_json("data/raw/ipc_huggingface.json")
```

**From Kaggle (download manually or via kaggle CLI):**
```bash
# Install Kaggle CLI: pip install kaggle
# Set up ~/.kaggle/kaggle.json with your API key

# BNS All 358 Sections
kaggle datasets download -d nandr39/bharatiya-nyaya-sanhita-dataset-bns -p data/raw/ --unzip

# IPC Complete Dataset (511 sections)
kaggle datasets download -d omdabral/indian-penal-code-complete-dataset -p data/raw/ --unzip

# Indian Legal Texts Fine-Tuning
kaggle datasets download -d akshatgupta7/llm-fine-tuning-dataset-of-indian-legal-texts -p data/raw/ --unzip
```

**From GitHub:**
```bash
# IPC-to-BNS Mapping JSON
# Go to: https://github.com/Pranshu321/IndLegal
# Download: mapping/ipc_to_bns.json → save to data/raw/ipc_to_bns_mapping.json

# Or clone:
git clone https://github.com/Pranshu321/IndLegal.git /tmp/IndLegal
cp /tmp/IndLegal/mapping/ipc_to_bns.json data/raw/ipc_to_bns_mapping.json
```

### Step 1.2: Data Exploration Notebook

Create `notebooks/01_data_exploration.ipynb` — explore each dataset:

**What to look at for each dataset:**
1. How many rows/examples?
2. What columns/fields exist?
3. What does a sample entry look like?
4. What format is it in? (Alpaca, ShareGPT, raw text, CSV?)
5. What's the average length of inputs and outputs?
6. Are there duplicates?
7. How many are BNS-specific vs general legal?

**Key questions to answer:**
- How many of the 22K Aalap instructions mention BNS? (Likely very few — Aalap was trained on older data)
- Does the BNS CSV have enough detail per section for training?
- Is the IPC-BNS JSON mapping complete (all 511 IPC → BNS)?
- What's the quality of the Kaggle legal texts dataset?

### Checkpoint: How to Verify Stage 1

After running the exploration notebook, you should be able to answer:
- [ ] Total number of examples across all datasets
- [ ] How many are already in instruction format (ready for training)
- [ ] How many are raw text (need conversion)
- [ ] How many BNS-specific examples exist (likely <500 — this is the gap we fill in Stage 2)
- [ ] Sample row from each dataset printed and understood

---

## Stage 2 — Data Preparation

**Objective:** Create a high-quality, BNS-focused training dataset by merging existing data + generating new synthetic data.

**Time:** ~4-6 hours

### The Data Problem

Aalap has 22K legal instructions — but almost none mention BNS (it was built pre-July 2024). We need to **create BNS-specific training data** ourselves. This is the hardest and most valuable part of the project.

### Step 2.1: Create IPC↔BNS Mapping QA Pairs

Using the IPC-BNS mapping JSON + BNS CSV + IPC CSV, automatically generate QA pairs like:

```json
{
  "instruction": "What is the BNS equivalent of IPC Section 302?",
  "input": "",
  "output": "IPC Section 302 (Punishment for murder) corresponds to BNS Section 103(1) under the Bharatiya Nyaya Sanhita, 2023. The punishment remains death or imprisonment for life, and shall also be liable to fine. This section falls under Chapter VI of BNS (Of Offences Affecting the Human Body)."
}
```

**Variations to generate per IPC-BNS pair:**
1. "What is the BNS equivalent of IPC Section X?"
2. "What was IPC Section X? What is it now under BNS?"
3. "Under BNS, which section deals with [crime description]?"
4. "Compare IPC Section X and BNS Section Y"
5. "A person committed [crime]. Which BNS section applies?"

**Target: ~2,000-3,000 mapping QA pairs** (5-6 variations × ~400-500 unique mappings)

### Step 2.2: Generate Synthetic BNS QA Using Claude/GPT-4o

Use an LLM API to generate high-quality QA pairs from BNS section text:

```python
# Pseudocode for synthetic generation
for section in bns_sections:
    prompt = f"""
    Given this section of the Bharatiya Nyaya Sanhita (BNS):

    Section {section.number}: {section.title}
    {section.text}

    Generate 5 diverse question-answer pairs that a lawyer or law student
    might ask about this section. Include:
    1. A factual question about the section
    2. A scenario-based question (describe a crime, ask which section applies)
    3. A comparison question (how does this differ from the old IPC)
    4. A punishment-related question
    5. A procedural question

    Format as JSON with "instruction" and "output" fields.
    """
    response = call_llm(prompt)
    save_to_dataset(response)
```

**Target: ~2,000-3,000 synthetic QA pairs** (5 per section × 358 sections ≈ 1,790 + variations)

**Cost estimate:** ~$3-8 using Claude Haiku/GPT-4o-mini (cheap models for generation)

### Step 2.3: Combine All Datasets

Merge into a single dataset in **Alpaca format**:

```json
{
  "instruction": "The question or task",
  "input": "Optional additional context (often empty)",
  "output": "The expected answer"
}
```

**Final dataset composition (approximate):**

| Source | Count | Type |
|--------|-------|------|
| Aalap instructions | ~22,000 | General Indian legal QA |
| BNS mapping QA (generated in 2.1) | ~2,500 | IPC↔BNS mapping |
| Synthetic BNS QA (generated in 2.2) | ~2,500 | BNS section-specific |
| Constitution instructions | ~3,300 | Constitution articles |
| Indian Legal Summaries (Alpaca) | ~1,000-2,000 | Legal summaries |
| **Total** | **~28,000-32,000** | |

### Step 2.4: Quality Checks & Split

```python
# Deduplication
# Remove exact duplicates and near-duplicates (>90% token overlap)

# Length filtering
# Remove examples where output < 20 tokens (too short)
# Remove examples where input + output > 2048 tokens (won't fit in training context)

# Format validation
# Every entry must have non-empty "instruction" and "output"

# Train/val/test split
# 85% train, 10% validation, 5% test
# Stratify: ensure BNS-specific examples are proportionally represented in all splits
```

### Checkpoint: How to Verify Stage 2

- [ ] `data/processed/combined_dataset.json` exists with 25K-30K entries
- [ ] `data/splits/train.json`, `val.json`, `test.json` exist
- [ ] Random sample of 10 entries looks correct (well-formatted, factually plausible)
- [ ] BNS-specific examples are present in all splits
- [ ] No duplicate entries
- [ ] All entries have non-empty instruction and output fields

---

## Stage 3 — Fine-Tuning with Unsloth

**Objective:** Fine-tune Mistral 7B on your legal dataset using QLoRA on your RTX 3070 Ti.

**Time:** ~3-4 hours (includes ~1-2 hours of actual GPU training time)

### Step 3.1: Load Model with Unsloth

```python
from unsloth import FastLanguageModel
import torch

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="unsloth/mistral-7b-instruct-v0.3-bnb-4bit",  # Pre-quantized
    max_seq_length=1024,      # Reduced from 2048 to save VRAM on 8GB card
    dtype=None,               # Auto-detect
    load_in_4bit=True,        # 4-bit quantization (QLoRA)
)
```

### Step 3.2: Apply LoRA Adapters

```python
model = FastLanguageModel.get_peft_model(
    model,
    r=16,                     # LoRA rank — 16 is the sweet spot for 8GB
    target_modules=[          # Which layers to add adapters to
        "q_proj", "k_proj", "v_proj", "o_proj",    # Attention layers
        "gate_proj", "up_proj", "down_proj",        # MLP layers
    ],
    lora_alpha=32,            # Scaling factor (usually 2x rank)
    lora_dropout=0,           # Unsloth optimized — 0 is faster
    bias="none",
    use_gradient_checkpointing="unsloth",  # Saves ~30% VRAM
    random_state=42,
)
```

### Step 3.3: Format Dataset

```python
# Alpaca prompt template
alpaca_prompt = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
{instruction}

### Input:
{input}

### Response:
{output}"""

def formatting_prompts_func(examples):
    texts = []
    for instruction, input_text, output in zip(
        examples["instruction"], examples["input"], examples["output"]
    ):
        text = alpaca_prompt.format(
            instruction=instruction,
            input=input_text if input_text else "",
            output=output
        ) + tokenizer.eos_token
        texts.append(text)
    return {"text": texts}
```

### Step 3.4: Train

```python
from trl import SFTTrainer
from transformers import TrainingArguments

trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    dataset_text_field="text",
    max_seq_length=1024,
    dataset_num_proc=2,
    packing=False,
    args=TrainingArguments(
        # Batch settings (tuned for 8GB VRAM)
        per_device_train_batch_size=1,
        gradient_accumulation_steps=8,   # Effective batch size = 8

        # Training schedule
        num_train_epochs=2,              # 2 epochs for instruction data
        warmup_steps=50,
        learning_rate=2e-4,
        lr_scheduler_type="cosine",

        # Memory optimization
        fp16=not torch.cuda.is_bf16_supported(),
        bf16=torch.cuda.is_bf16_supported(),
        gradient_checkpointing=True,
        optim="adamw_8bit",             # 8-bit optimizer saves ~2GB VRAM

        # Logging
        logging_steps=10,
        eval_strategy="steps",
        eval_steps=100,
        save_strategy="steps",
        save_steps=200,
        output_dir="models/checkpoints",

        # Misc
        seed=42,
        report_to="none",               # We'll use MLflow manually
    ),
)

# Start training
trainer_stats = trainer.train()
```

**Expected on RTX 3070 Ti:**
- ~25K examples × 2 epochs = ~50K training steps ÷ 8 (grad accumulation) ≈ 6,250 optimizer steps
- Speed: ~2-3 iterations/second
- Total time: ~45-90 minutes per epoch, ~2-3 hours total
- Peak VRAM: ~6.5-7GB

### Step 3.5: Save the Trained Adapter

```python
# Save LoRA adapter (small, ~50-100MB)
model.save_pretrained("models/lora_adapter")
tokenizer.save_pretrained("models/lora_adapter")

# Quick test
FastLanguageModel.for_inference(model)
inputs = tokenizer(
    alpaca_prompt.format(
        instruction="What is the BNS equivalent of IPC Section 302?",
        input="",
        output=""
    ),
    return_tensors="pt"
).to("cuda")

outputs = model.generate(**inputs, max_new_tokens=256, temperature=0.7)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
```

### Checkpoint: How to Verify Stage 3

- [ ] Training completed without OOM (out of memory) errors
- [ ] Training loss decreased over time (check logs)
- [ ] `models/lora_adapter/` contains `adapter_model.safetensors` and `adapter_config.json`
- [ ] Quick test produces a coherent legal answer (doesn't need to be perfect yet)
- [ ] Peak VRAM usage stayed under 8GB

---

## Stage 4 — MLflow Experiment Tracking

**Objective:** Run multiple training experiments with different hyperparameters, track all of them in MLflow, and pick the best one.

**Time:** ~2 hours (mostly waiting for training runs)

### Step 4.1: Set Up MLflow

```bash
# Start MLflow tracking server (run in a separate terminal)
mlflow server --host 127.0.0.1 --port 5000

# Open in browser: http://127.0.0.1:5000
```

### Step 4.2: Wrap Training with MLflow Logging

```python
import mlflow

mlflow.set_tracking_uri("http://127.0.0.1:5000")
mlflow.set_experiment("bharatnyaya-llm")

with mlflow.start_run(run_name="run_1_baseline"):
    # Log hyperparameters
    mlflow.log_params({
        "base_model": "mistral-7b-instruct-v0.3",
        "lora_r": 16,
        "lora_alpha": 32,
        "learning_rate": 2e-4,
        "epochs": 2,
        "batch_size": 1,
        "grad_accumulation": 8,
        "max_seq_length": 1024,
        "dataset_size": len(train_dataset),
        "quantization": "4bit-nf4",
    })

    # Train (same code as Stage 3)
    trainer_stats = trainer.train()

    # Log final metrics
    mlflow.log_metrics({
        "train_loss": trainer_stats.training_loss,
        "train_runtime_seconds": trainer_stats.metrics["train_runtime"],
        "train_samples_per_second": trainer_stats.metrics["train_samples_per_second"],
    })

    # Log the adapter
    mlflow.log_artifacts("models/lora_adapter", artifact_path="lora_adapter")
```

### Step 4.3: Run 3-4 Experiments

| Run | LoRA Rank | Learning Rate | Epochs | What We're Testing |
|-----|-----------|---------------|--------|--------------------|
| **Run 1** (baseline) | r=16 | 2e-4 | 2 | Default settings |
| **Run 2** | r=32 | 2e-4 | 2 | Does higher rank improve legal accuracy? |
| **Run 3** | r=16 | 1e-4 | 3 | Does lower LR + more epochs help? |
| **Run 4** | r=16 | 2e-4 | 2 | BNS-only dataset (no Aalap) — does domain focus help? |

### Step 4.4: Compare & Register Best Model

```python
# In MLflow UI: compare runs side-by-side
# Look at: final train_loss, eval_loss, sample output quality

# Register the best run's model
from mlflow import MlflowClient

client = MlflowClient()
# Register model (use the run_id of best experiment)
model_version = mlflow.register_model(
    f"runs:/{best_run_id}/lora_adapter",
    "bharatnyaya-llm"
)

# Set alias
client.set_registered_model_alias("bharatnyaya-llm", "champion", model_version.version)
```

### Checkpoint: How to Verify Stage 4

- [ ] MLflow UI shows all 3-4 runs with logged parameters and metrics
- [ ] You can visually compare loss curves across runs
- [ ] Best model is registered in Model Registry with `@champion` alias
- [ ] You can articulate WHY the best run is the best (lower loss? better outputs?)

---

## Stage 5 — Evaluation

**Objective:** Measure how good the fine-tuned model actually is, and prove it's better than the base model.

**Time:** ~3-4 hours

### Step 5.1: Create Test Questions

Create `src/evaluation/test_questions.json` — 100 carefully crafted questions:

**Category 1: BNS Mapping (50 questions)**
- "What is the BNS equivalent of IPC Section 420?" (direct mapping)
- "A person committed robbery at gunpoint. Which BNS section applies?" (scenario → section)
- "Compare the punishment for murder under IPC 302 and BNS 103" (comparison)

**Category 2: General Legal QA (30 questions)**
- "What is the right to bail under BNSS?"
- "Explain the concept of mens rea in Indian criminal law"

**Category 3: Edge Cases (20 questions)**
- "What happens to IPC sections that have no BNS equivalent?" (tests knowledge of repealed sections)
- "Which BNS sections are entirely new with no IPC predecessor?" (tests deep knowledge)

### Step 5.2: Generate Answers from 3 Models

```python
# Generate answers from:
# 1. Base Mistral 7B (no fine-tuning) — baseline
# 2. Your fine-tuned BharatNyayaLLM — what we built
# 3. Aalap-Mistral-7B — existing best Indian legal model (benchmark)

models = {
    "base_mistral": load_base_mistral(),
    "bharatnyaya": load_finetuned_model(),
    "aalap": load_aalap_model(),  # For comparison
}

results = {}
for model_name, model in models.items():
    results[model_name] = []
    for question in test_questions:
        answer = generate(model, question)
        results[model_name].append(answer)
```

### Step 5.3: Automated Evaluation with DeepEval

```python
from deepeval.metrics import (
    AnswerRelevancyMetric,
    FaithfulnessMetric,
    HallucinationMetric,
)
from deepeval.test_case import LLMTestCase

# For each question-answer pair:
test_case = LLMTestCase(
    input="What is the BNS equivalent of IPC Section 302?",
    actual_output=model_answer,
    expected_output=ground_truth_answer,  # From our test set
    context=["BNS Section 103(1) corresponds to IPC Section 302..."]
)

# Run metrics
relevancy = AnswerRelevancyMetric(threshold=0.7)
relevancy.measure(test_case)
print(f"Relevancy score: {relevancy.score}")
```

### Step 5.4: LLM-as-Judge Evaluation

```python
# Use GPT-4o or Claude to rate each answer on a 1-5 scale
judge_prompt = """
Rate this answer on a scale of 1-5 for:
1. Factual Correctness: Are the BNS/IPC section numbers correct?
2. Completeness: Does it cover punishment, chapter, and key changes?
3. Clarity: Is the language clear and professional?

Question: {question}
Answer: {answer}
Ground Truth: {ground_truth}

Return JSON: {"correctness": X, "completeness": X, "clarity": X, "reasoning": "..."}
"""
```

### Step 5.5: Manual Spot-Check

For the 50 BNS mapping questions:
- Manually verify: Is the BNS section number the model outputs actually correct?
- Use [ipctobns.in](https://www.ipctobns.in/) or [devgan.in/bns/](https://devgan.in/all_sections_bns.php) as ground truth
- Calculate: **BNS mapping accuracy** = correct section numbers / total mapping questions

### Step 5.6: Results Table

| Metric | Base Mistral | BharatNyayaLLM | Aalap-Mistral |
|--------|-------------|----------------|---------------|
| BNS Mapping Accuracy | ?% | ?% | ?% |
| DeepEval Relevancy | ? | ? | ? |
| DeepEval Faithfulness | ? | ? | ? |
| Judge: Correctness (avg) | ?/5 | ?/5 | ?/5 |
| Judge: Completeness (avg) | ?/5 | ?/5 | ?/5 |

**Expected:** Base Mistral should score poorly on BNS questions (it doesn't know BNS). Aalap should score well on general legal but poorly on BNS (it was trained pre-BNS). BharatNyayaLLM should score best on BNS-specific questions.

### Checkpoint: How to Verify Stage 5

- [ ] Evaluation results table is populated for all 3 models
- [ ] BharatNyayaLLM outperforms base Mistral on BNS questions
- [ ] You can point to specific examples where fine-tuning helped
- [ ] Results are logged in MLflow
- [ ] You can explain WHY the model gets some answers wrong (data gaps, hallucination patterns)

---

## Stage 6 — Deployment

**Objective:** Convert the model to a compact format and deploy it as a web app anyone can try.

**Time:** ~2-3 hours

### Step 6.1: Convert to GGUF

```python
# Using Unsloth (one line!)
model.save_pretrained_gguf(
    "models/gguf",
    tokenizer,
    quantization_method="q4_k_m"  # 4-bit, ~4.1GB file
)

# Push to HuggingFace Hub
model.push_to_hub_gguf(
    "YOUR_USERNAME/BharatNyayaLLM-GGUF",
    tokenizer,
    quantization_method="q4_k_m",
    token="YOUR_HF_TOKEN"
)
```

### Step 6.2: Test Locally with Ollama

```bash
# Install Ollama: https://ollama.com/download

# Create a Modelfile
cat > Modelfile << 'EOF'
FROM ./models/gguf/unsloth.Q4_K_M.gguf
TEMPLATE """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
{{ .Prompt }}

### Input:


### Response:
"""
PARAMETER temperature 0.7
PARAMETER top_p 0.9
SYSTEM You are BharatNyayaLLM, an expert in Indian criminal law specializing in the Bharatiya Nyaya Sanhita (BNS) and its predecessor the Indian Penal Code (IPC). You provide accurate section mappings, punishments, and legal explanations.
EOF

# Create and run
ollama create bharatnyaya -f Modelfile
ollama run bharatnyaya "What is the BNS equivalent of IPC Section 420?"
```

### Step 6.3: Build Gradio Chat Interface

Create `src/deployment/app.py`:

```python
import gradio as gr
from llama_cpp import Llama

# Load GGUF model
llm = Llama.from_pretrained(
    repo_id="YOUR_USERNAME/BharatNyayaLLM-GGUF",
    filename="unsloth.Q4_K_M.gguf",
    n_ctx=2048,
    n_batch=512,
    verbose=False,
)

SYSTEM_PROMPT = """You are BharatNyayaLLM, an expert in Indian criminal law.
You specialize in the Bharatiya Nyaya Sanhita (BNS, 2023) which replaced the
Indian Penal Code (IPC, 1860) on 1 July 2024. You provide accurate section
mappings between IPC and BNS, explain punishments, and answer legal questions.
Always cite specific section numbers."""

def respond(message, history):
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    for user_msg, bot_msg in history:
        messages.append({"role": "user", "content": user_msg})
        messages.append({"role": "assistant", "content": bot_msg})
    messages.append({"role": "user", "content": message})

    response = llm.create_chat_completion(
        messages=messages,
        max_tokens=512,
        temperature=0.7,
        stream=True,
    )

    partial = ""
    for chunk in response:
        delta = chunk["choices"][0]["delta"].get("content", "")
        partial += delta
        yield partial

demo = gr.ChatInterface(
    fn=respond,
    title="BharatNyayaLLM",
    description="Indian Criminal Law AI — IPC to BNS mapping, legal QA, and statute explanation. Fine-tuned Mistral 7B on 28K+ Indian legal instructions.",
    examples=[
        "What is the BNS equivalent of IPC Section 302?",
        "A person stole a mobile phone from someone's pocket. Which BNS section applies?",
        "Compare the punishment for robbery under IPC and BNS",
        "What are the new offences introduced in BNS that didn't exist in IPC?",
        "Explain BNS Section 69 (sexual intercourse by deceitful means)",
    ],
    theme="soft",
)

demo.launch()
```

### Step 6.4: Deploy to HuggingFace Spaces

```bash
# Create a new Space on huggingface.co/new-space
# SDK: Gradio, Hardware: CPU Basic (free)

# Clone the space
git clone https://huggingface.co/spaces/YOUR_USERNAME/BharatNyayaLLM
cd BharatNyayaLLM

# Copy files
cp src/deployment/app.py .

# Create requirements.txt
cat > requirements.txt << 'EOF'
gradio
llama-cpp-python
huggingface_hub
EOF

# Push
git add . && git commit -m "Deploy BharatNyayaLLM" && git push
```

### Checkpoint: How to Verify Stage 6

- [ ] GGUF model uploaded to HuggingFace Hub
- [ ] Ollama runs the model locally and answers questions
- [ ] Gradio app runs locally (`python src/deployment/app.py`)
- [ ] HuggingFace Space is live and accessible via URL
- [ ] End-to-end test: type a crime description → get correct BNS section

---

## Resume Bullets This Project Gives You

Copy-paste these after completing the project (fill in your actual numbers):

> **BharatNyayaLLM — Fine-Tuned Legal LLM for Indian Criminal Law**
> - QLoRA fine-tuned Mistral 7B on 28K+ Indian legal instruction pairs for IPC-to-BNS statute mapping, achieving X% mapping accuracy vs Y% baseline, deployed as GGUF on HuggingFace Spaces.
> - Engineered data pipeline: merged 5 sources (Aalap, BNS sections, IPC-BNS mapping) + generated 5K synthetic QA pairs via Claude API, with automated deduplication and quality filtering.
> - Tracked 4 training experiments in MLflow (varying LoRA rank, learning rate, dataset composition), registered champion model in Model Registry, evaluated using DeepEval (faithfulness, relevancy) + LLM-as-judge scoring.
> - Converted to GGUF Q4_K_M (4.1GB) for CPU inference, built Gradio chat interface, deployed on HuggingFace Spaces with streaming responses.

**Keywords this adds to your resume:** LLM Fine-Tuning, QLoRA, LoRA, PEFT, PyTorch, Transformers, Hugging Face, MLflow, Model Registry, Experiment Tracking, DeepEval, LLM Evaluation, GGUF, Quantization, Gradio, HuggingFace Spaces, NLP, Machine Learning, Deep Learning, Synthetic Data Generation

---

## Interview Questions This Project Prepares You For

### Fine-Tuning (asked in 80% of GenAI interviews)

1. **"What is LoRA? How does it reduce training cost?"**
   → LoRA adds small rank-decomposition matrices to frozen model weights. Instead of training 7B parameters, you train ~10-50M (0.1-0.7%). Cuts VRAM from 100GB+ to 6-8GB.

2. **"What is QLoRA? Why not just use LoRA?"**
   → QLoRA combines 4-bit quantization of the base model with LoRA adapters. The base model is compressed to 4-bit (NF4 format), LoRA adapters are trained in 16-bit. This lets you fine-tune 7B models on 8GB VRAM.

3. **"How do you choose LoRA rank (r)?"**
   → Higher rank = more parameters = more capacity but more VRAM. For domain adaptation (not changing model behavior fundamentally), r=16-32 is standard. r=64+ for complex tasks.

4. **"What is the difference between SFT and RLHF/DPO?"**
   → SFT teaches the model to follow instructions. DPO/RLHF teaches it to prefer good answers over bad ones. SFT is always done first.

5. **"How do you evaluate a fine-tuned LLM?"**
   → Three levels: automated metrics (DeepEval faithfulness/relevancy), LLM-as-judge (GPT-4o rates outputs), human evaluation (domain experts verify factual correctness).

### MLOps (asked in 60% of AI Engineer interviews)

6. **"What is MLflow? What are its main components?"**
   → Experiment tracking (log params/metrics), Model Registry (version control for models), MLflow Models (packaging), MLflow Projects (reproducibility).

7. **"How would you do A/B testing between two models?"**
   → Register both in MLflow, assign aliases (@champion, @challenger), route traffic proportionally, compare metrics, promote winner.

8. **"How do you handle model versioning?"**
   → MLflow Model Registry: each training run produces a version, best version gets @champion alias. Deployment loads from alias, not version number.

### Data Engineering (asked in 50% of interviews)

9. **"How did you create your training dataset?"**
   → Combined 5 sources (HuggingFace, Kaggle, GitHub), generated synthetic QA pairs via Claude API, formatted as Alpaca JSON, deduplicated, split 85/10/5.

10. **"What's the risk of synthetic data? How do you mitigate it?"**
    → Risk: model learns teacher's biases and errors. Mitigation: cross-reference with ground truth (official BNS text), quality filtering, diverse prompt templates.

### Deployment (asked in 40% of interviews)

11. **"What is GGUF? Why not serve the model as-is?"**
    → GGUF is a quantized format for CPU/GPU inference via llama.cpp. Original model is 14GB and needs GPU. GGUF Q4_K_M is 4.1GB and runs on CPU.

12. **"How do you serve an LLM in production?"**
    → Options: GGUF + llama.cpp (single machine), vLLM (GPU server), TGI (HuggingFace), API wrapper (FastAPI + model). Choice depends on scale and budget.

### Domain-Specific (differentiators)

13. **"What changed from IPC to BNS?"**
    → IPC had 511 sections in 23 chapters. BNS has 358 sections in 20 chapters. Reorganized more logically, added new offences (mob lynching, organized crime, terrorism), removed outdated provisions.

14. **"Why fine-tune instead of using RAG for this task?"**
    → RAG is better for factual lookup. Fine-tuning is better for teaching the model a new "skill" (understanding legal language, mapping between law systems). Best approach is often hybrid: fine-tune for format/reasoning + RAG for factual grounding.

15. **"What would you do differently with unlimited budget?"**
    → Full fine-tune (not QLoRA) for maximum quality. DPO alignment phase with lawyer-annotated preference data. Continual pretraining on 702K NyayaAnumana cases before SFT. Multi-language support (Hindi, regional languages via IL-TUR benchmark).

---

## Existing Models to Study

Before building, study these existing Indian legal LLMs to understand the landscape:

| Model | What They Did | What We Do Differently |
|-------|--------------|----------------------|
| [Aalap-Mistral-7B](https://huggingface.co/opennyaiorg/Aalap-Mistral-7B-v0.1-bf16) | Mistral 7B SFT on 22K legal instructions | We add BNS-specific data (Aalap predates BNS) |
| [INLegalLLaMA](https://huggingface.co/sudipto-ducs/InLegalLLaMA) | LLaMA 2 continual pretraining + SFT | We use newer base model (Mistral v0.3), focus on BNS |
| [Indian-LawyerGPT](https://github.com/NisaarAgharia/Indian-LawyerGPT) | Falcon-7B/LLaMA 2 QLoRA | Outdated models, no BNS coverage |
| [Gemma-2-2B-Indian-Law](https://huggingface.co/Ananya8154/Gemma-2-2B-Indian-Law) | Small model fine-tuned for Indian law | Too small (2B) for complex legal reasoning |

**Our unique angle:** First fine-tuned LLM specifically trained on BNS (July 2024 law), with IPC→BNS mapping capability.

---

## Timeline Summary

| Stage | What | Time | Cumulative |
|-------|------|------|-----------|
| Stage 0 | Environment Setup | ~1 hour | 1 hour |
| Stage 1 | Data Collection & Exploration | ~2-3 hours | 4 hours |
| Stage 2 | Data Preparation | ~4-6 hours | 10 hours |
| Stage 3 | Fine-Tuning | ~3-4 hours | 14 hours |
| Stage 4 | MLflow Experiments | ~2 hours | 16 hours |
| Stage 5 | Evaluation | ~3-4 hours | 20 hours |
| Stage 6 | Deployment | ~2-3 hours | 23 hours |
| **Total** | | **~18-23 hours** | ~3-4 weekend days |

---

## How to Use This Roadmap

1. Open this README alongside Claude
2. Tell Claude: **"Let's start Stage 0"**
3. Follow the steps. When stuck, ask Claude to explain or debug
4. After completing each stage, use the **Checkpoint** section to verify
5. After each stage, ask Claude: **"Explain what we just built and what interview questions it prepares me for"**
6. Move to the next stage

Each stage builds on the previous one. Don't skip stages.

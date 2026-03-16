<div align="center">

# 🐛 Fine-Tuning CodeT5 for Python Bug Fixing

### with LoRA and Prompt Tuning

*Automated bug fixing using Parameter-Efficient Fine-Tuning on real Python bug–fix pairs*

![Python](https://img.shields.io/badge/Python-3.11+-blue?style=flat-square&logo=python)
![PyTorch](https://img.shields.io/badge/Framework-PyTorch-EE4C2C?style=flat-square&logo=pytorch)
![HuggingFace](https://img.shields.io/badge/Model-CodeT5%20Base-FFD21E?style=flat-square&logo=huggingface)
![LoRA](https://img.shields.io/badge/PEFT-LoRA-8A2BE2?style=flat-square)
![Colab](https://img.shields.io/badge/Environment-Google%20Colab-F9AB00?style=flat-square&logo=google-colab)

</div>

---

## 🧠 Project Overview

This project fine-tunes **Salesforce/CodeT5-base** — a code-aware encoder-decoder transformer — to automatically fix Python bugs. Using **LoRA (Low-Rank Adaptation)** from the PEFT library, the model is efficiently trained on 100 real bug–fix pairs from the PyTraceBugs dataset, significantly reducing the number of trainable parameters while preserving model quality.

The project explores **prompt tuning** by evaluating the model's sensitivity to different prompt templates, and benchmarks the fine-tuned model against the base CodeT5 model using BLEU score and exact match accuracy.

---

## ✨ Key Features

- 🔧 **Automated Python Bug Fixing** — Given buggy code, the model generates the corrected version
- ⚡ **Parameter-Efficient Fine-Tuning** — LoRA adapters train only ~1% of model parameters instead of the full model
- 💬 **Prompt Tuning** — Evaluates how different prompt templates affect output quality
- 📊 **Dual Evaluation** — Compares base vs. fine-tuned model on BLEU and Exact Match metrics
- 🗄️ **Real Bug Dataset** — Trained on curated PyTraceBugs data covering real-world Python errors (OverflowError, KeyError, AssertionError, FileNotFoundError, AttributeError, and more)

---

## 📂 Project Structure

```
Fine-Tuning-CodeT5/
├── codet5_bugfixing_100_samples_lora.py   # Main training & evaluation script
├── curated_pytracebugs_subset_100.jsonl   # 100 curated bug–fix pairs dataset
└── README.md
```

---

## 🗃️ Dataset

The dataset `curated_pytracebugs_subset_100.jsonl` contains **100 real Python bug–fix pairs** curated from the PyTraceBugs benchmark. Each entry includes:

| Field | Description |
|---|---|
| `id` | Unique sample identifier |
| `buggy_code` | The original buggy Python function |
| `fixed_code` | The correct fixed version |
| `traceback_type` | The type of error (e.g., `KeyError`, `AssertionError`, `OverflowError`) |

**Example bug types covered:** `OverflowError`, `KeyError`, `AssertionError`, `FileNotFoundError`, `AttributeError`

---

## ⚙️ Model & Training Configuration

### Base Model
```
Salesforce/codet5-base
```

### LoRA Configuration

| Parameter | Value |
|---|---|
| Rank (`r`) | 8 |
| Alpha (`lora_alpha`) | 16 |
| Target Modules | `q`, `v` (attention layers) |
| Dropout | 0.05 |
| Task Type | SEQ_2_SEQ_LM |

### Training Parameters

| Parameter | Value |
|---|---|
| Epochs | 5 |
| Batch Size | 4 |
| Learning Rate | 5e-5 |
| LR Scheduler | Linear |
| Max Token Length | 256 |
| Generation Max Length | 150 |
| Beam Search | 5 beams |
| Mixed Precision (fp16) | Yes (if GPU available) |

### Prompt Template
```
Fix the following Python code: {buggy_code}
```

---

## 📊 Evaluation Metrics

The model is evaluated by comparing predictions against reference fixed code using:

| Metric | Description |
|---|---|
| **Exact Match** | Percentage of predictions that exactly match the reference fix |
| **SacreBLEU** | N-gram overlap score between predicted and reference code |

Both the **base CodeT5** and the **fine-tuned LoRA model** are evaluated side-by-side for direct comparison.

---

## 🚀 Getting Started

### Run in Google Colab (Recommended)

1. Open the script in Google Colab
2. Upload `curated_pytracebugs_subset_100.jsonl` when prompted
3. Run all cells — training and evaluation will execute automatically

```python
# Install dependencies (first cell)
!pip install -q torch transformers==4.41.2 peft==0.10.0 accelerate bitsandbytes evaluate nltk sacrebleu datasets sentencepiece
```

### Run Locally

```bash
# 1. Clone the repository
git clone https://github.com/Aalezz/Fine-Tuning-CodeT5-for-Python-Bug-Fixing-with-LoRA-and-Prompt-Tuning.git
cd Fine-Tuning-CodeT5-for-Python-Bug-Fixing-with-LoRA-and-Prompt-Tuning

# 2. Install dependencies
pip install torch transformers==4.41.2 peft==0.10.0 accelerate bitsandbytes evaluate nltk sacrebleu datasets sentencepiece

# 3. Run the script
python codet5_bugfixing_100_samples_lora.py
```

> A GPU is strongly recommended. The script automatically detects and uses CUDA if available, otherwise falls back to CPU.

---

## 🔁 Workflow

```
Upload Dataset
     ↓
Load & Tokenize Bug–Fix Pairs
     ↓
Apply LoRA Adapters to CodeT5-base
     ↓
Fine-tune for 5 Epochs
     ↓
Save LoRA Adapters
     ↓
Evaluate Base Model  →  Compare  ←  Evaluate Fine-tuned Model
     ↓
Print BLEU & Exact Match Results
```

---

## 💾 Saving & Loading Adapters

After training, LoRA adapters are saved separately from the base model — meaning you only need to store a few MB of weights rather than the full model.

```python
# Save (automatic after training)
peft_model.save_pretrained("lora_adapters_100_sample")

# Load later
from peft import PeftModel, PeftConfig
config = PeftConfig.from_pretrained("lora_adapters_100_sample")
model = T5ForConditionalGeneration.from_pretrained(config.base_model_name_or_path)
model = PeftModel.from_pretrained(model, "lora_adapters_100_sample")
```

---

## 📄 License

This project is licensed under the MIT License.

---

<div align="center">
  Built with ❤️ using <a href="https://huggingface.co/Salesforce/codet5-base">CodeT5</a> · <a href="https://github.com/huggingface/peft">PEFT / LoRA</a> · <a href="https://colab.research.google.com">Google Colab</a>
  <br><br>
  <i>Fixing Python bugs, one fine-tune at a time.</i>
</div>

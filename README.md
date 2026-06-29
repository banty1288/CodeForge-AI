# 🚀 CodeForge AI

> Fine-tuning **Qwen2.5-3B-Instruct** using **QLoRA (PEFT + LoRA)** for Python software engineering tasks.

![Python](https://img.shields.io/badge/Python-3.12-blue)
![Transformers](https://img.shields.io/badge/🤗-Transformers-yellow)
![PEFT](https://img.shields.io/badge/PEFT-LoRA-green)
![QLoRA](https://img.shields.io/badge/QLoRA-4bit-red)
![License](https://img.shields.io/badge/License-MIT-blue)

---

# 📌 Project Overview

CodeForge AI is a learning-focused project demonstrating the complete workflow of **parameter-efficient fine-tuning (PEFT)** for Large Language Models.

The project fine-tunes **Qwen2.5-3B-Instruct** using **QLoRA** on a Python instruction-following dataset and publishes the resulting LoRA adapter to Hugging Face.

The objective of this project is **not to outperform the base model**, but to understand and implement a modern LLM fine-tuning pipeline from start to finish.

---

# 🎯 Objectives

* Learn how modern LLM fine-tuning works.
* Fine-tune a billion-parameter model on consumer hardware.
* Understand QLoRA, LoRA, PEFT, and Supervised Fine-Tuning.
* Publish the trained adapter to Hugging Face.
* Build a reusable workflow for future fine-tuning projects.

---

# 🧠 Base Model

* **Model:** Qwen2.5-3B-Instruct
* **Parameters:** ~3.1 Billion
* **Provider:** Alibaba Cloud

---

# ⚡ Fine-Tuning Method

This project uses:

* ✅ QLoRA
* ✅ LoRA
* ✅ PEFT
* ✅ Supervised Fine-Tuning (SFT)
* ✅ 4-bit NF4 Quantization
* ✅ Hugging Face Transformers
* ✅ TRL SFTTrainer

Only **0.96%** of the model parameters were trainable during fine-tuning.

---

# 📚 Dataset

Dataset:

* `iamtarun/python_code_instructions_18k_alpaca`

The dataset contains Python instruction-following examples for supervised fine-tuning.

For this proof-of-concept project, training was performed on a subset of the dataset to validate the complete fine-tuning pipeline.

---

# 🏗️ Project Architecture

```text
                Python Instruction Dataset
                           │
                           ▼
                     Tokenization
                           │
                           ▼
                Qwen2.5-3B-Instruct
                           │
                           ▼
                  4-bit Quantization
                           │
                           ▼
                      QLoRA + PEFT
                           │
                           ▼
                     LoRA Adapters
                           │
                           ▼
              Supervised Fine-Tuning
                           │
                           ▼
                  Fine-Tuned Adapter
                           │
                           ▼
                  Hugging Face Hub
```

---

# ⚙️ Training Configuration

| Parameter             | Value                   |
| --------------------- | ----------------------- |
| Base Model            | Qwen2.5-3B-Instruct     |
| Fine-Tuning           | QLoRA                   |
| LoRA Rank             | 16                      |
| Alpha                 | 32                      |
| Dropout               | 0.05                    |
| Quantization          | 4-bit NF4               |
| Learning Rate         | 2e-4                    |
| Batch Size            | 2                       |
| Gradient Accumulation | 4                       |
| Max Steps             | 100                     |
| Max Sequence Length   | 512                     |
| GPU                   | Tesla T4 (Google Colab) |

---

# 📊 Training Summary

* Training completed successfully.
* LoRA adapter generated and saved.
* Adapter uploaded to Hugging Face.
* Basic comparison performed between the base and fine-tuned models.

Training Loss:

```text
0.7047
```

Trainable Parameters:

```text
29,933,568
```

Total Parameters:

```text
3,115,872,256
```

Trainable Percentage:

```text
0.9607%
```

---

# 📁 Repository Structure

```text
CodeForge-AI/
│
├── train.ipynb
├── evaluate.ipynb
├── requirements.txt
├── README.md
├── LICENSE
│
├── images/
│
└── results/
```

---

# 🚀 Installation

```bash
git clone https://github.com/<your-username>/CodeForge-AI.git

cd CodeForge-AI

pip install -r requirements.txt
```

---

# 💻 Inference Example

```python
from transformers import AutoTokenizer
from peft import AutoPeftModelForCausalLM

model = AutoPeftModelForCausalLM.from_pretrained(
    "banty1614/codeforge-qwen-lora",
    device_map="auto"
)

tokenizer = AutoTokenizer.from_pretrained(
    "banty1614/codeforge-qwen-lora"
)
```

---

# 📈 Evaluation

The project includes a simple evaluation notebook comparing responses from:

* Base Qwen2.5-3B-Instruct
* Fine-Tuned LoRA Adapter

The comparison focuses on:

* Code correctness
* Explanation quality
* Response formatting
* Readability

This evaluation is qualitative and intended to verify that the end-to-end fine-tuning pipeline works. It should not be interpreted as a benchmark demonstrating overall superiority over the base model.

---

# ⚠️ Limitations

This project is a proof of concept.

Current limitations include:

* Training performed for only 100 optimization steps.
* Training used a subset of the available dataset.
* No standardized benchmark (e.g. HumanEval or MBPP) was used.
* The objective was to learn and implement the fine-tuning workflow rather than maximize model performance.

---

# 🔮 Future Improvements

* Train on the complete dataset.
* Increase training duration.
* Evaluate on HumanEval and MBPP.
* Fine-tune on software engineering datasets covering SQL, FastAPI, Docker, Git, and system design.
* Build a web interface for one-click fine-tuning.

---

# 🤗 Hugging Face Model

**Model Repository**

https://huggingface.co/banty1614/codeforge-qwen-lora

---

# 🛠️ Tech Stack

* Python
* PyTorch
* Hugging Face Transformers
* PEFT
* TRL
* BitsAndBytes
* Accelerate
* Google Colab
* QLoRA

---

# 🙏 Acknowledgements

* Alibaba Cloud for the Qwen2.5 models.
* Hugging Face for the Transformers ecosystem.
* The PEFT and TRL teams.
* The creators of the `iamtarun/python_code_instructions_18k_alpaca` dataset.

---

# 📜 License

This repository is released under the MIT License.

---

## ⭐ Learning Outcome

This project helped me gain practical experience with:

* Parameter-Efficient Fine-Tuning (PEFT)
* LoRA and QLoRA
* Supervised Fine-Tuning (SFT)
* Hugging Face model publishing
* Model evaluation
* End-to-end LLM engineering workflows

Contributions, suggestions, and feedback are welcome!


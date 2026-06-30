# PHEB: A European Portuguese High School-Level LLM Benchmark

[![Code](https://img.shields.io/badge/GitHub-PHEB%20Benchmark-blue?logo=github)](https://github.com/AMALIA-LLM/pheb)
[![Paper: LREC 2026](https://img.shields.io/badge/Paper-LREC%202026-red)](https://lrec.elra.info/lrec2026-main-367)

This repository provides the code and resources for running the **PHEB benchmark**, introduced in the paper  
[PHEB: An European Portuguese High School-Level LLM Benchmark](https://lrec.elra.info/lrec2026-main-367), accepted at **LREC 2026**.

PHEB (Portuguese High School Exams Benchmark) is designed to systematically evaluate large language models using authentic Portuguese national examination questions from high school curricula. The benchmark enables the assessment of LLMs across multiple subjects, testing both knowledge retention and language proficiency in European Portuguese.

## Overview

This benchmark comprises questions spanning **18 years (2006-2023)** across **six core subjects**. 
It evaluates LLMs on their performance in Portuguese high school curriculum, testing both knowledge retention and language proficiency.


## Dataset

The dataset is organized by subject, with each JSON file containing questions from national exams:

- `philosophy.json` - Philosophy questions
- `geography.json` - Geography questions  
- `bio_geo.json` - Biology and Geology questions
- `portuguese.json` - Portuguese Language and Literature questions
- `history_a.json` - History questions
- `mathematics_a.json` - Mathematics questions
- `all.json` - Combined dataset with all subjects

## Requirements

- Python 3.8+
- `transformers` - Hugging Face Transformers library
- `vllm` - Fast LLM inference
- `tqdm` - Progress bars

Install dependencies:
```bash
pip install transformers vllm tqdm
```

## Usage

### Loading the Dataset

Use the provided `loader.py` to load exam questions and corrections:

```python
from loader import load_all_files

all_exams, all_corrections = load_all_files()
```

### Evaluating Models

The `evaluate_mcq_instruct.py` script provides a comprehensive evaluation framework for instruction-tuned models on multiple-choice questions.

#### Basic Evaluation

```bash
python evaluate_mcq_instruct.py \
  --model_list "model_name_or_path" \
  --format_type boxed \
  --eval_method generate \
  --output_dir mcq-results
```

#### Evaluation Methods

**Format Types:**
- `boxed`: Chain-of-thought reasoning with answer in `\boxed{X}` format (recommended for capable models)
- `boxed_only`: Answer only in `\boxed{X}` format
- `letter_only`: Single letter answer (A/B/C/D)
- `number_only`: Single number answer (1/2/3/4)

**Evaluation Methods:**
- `generate`: Model generates a complete response
- `likelihood`: Compare next-token logits for each option (faster, more objective)

#### Other Options

```bash
python evaluate_mcq_instruct.py \
  --model_list "model1" "model2" \
  --format_type boxed \
  --eval_method generate \
  --custom_system_prompt "You are an expert in Portuguese exams." \
  --prepend_answer_prefix \
  --output_dir mcq-results
```

- `--custom_system_prompt`: Add a custom system prompt
- `--prepend_answer_prefix`: Append "A resposta é" before the answer token (recommended for likelihood evaluation)
- `--use_generation_prompt`: Use full generation instructions even in likelihood mode

### Output

The evaluation script provides:
- Overall accuracy statistics
- Per-subject performance breakdown
- Detailed results saved to JSON files with model responses and predictions


## How to Cite

If you use this repository or the PHEB benchmark in your work, please cite:

```bibtex
@inproceedings{tavares-etal-2026-pheb,
  title = {PHEB: An European Portuguese High School-Level LLM Benchmark},
  author = {Tavares, Diogo C. and Ferreira, Rafael and Simplício, Afonso and Vinagre, Gonçalo and Condez, Ana Carolina and Calvo, Inês and Vieira, Inês and Semedo, David and Magalhaes, Joao},
  booktitle = {Proceedings of the Fifteenth Language Resources and Evaluation Conference (LREC 2026)},
  month = {May},
  year = {2026},
  pages = {4673--4683},
  address = {Palma, Mallorca, Spain},
  publisher = {European Language Resources Association (ELRA)},
  doi = {10.63317/2o3fvueefvwj}
}
```

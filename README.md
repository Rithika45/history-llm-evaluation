# Exploring the Potential of LLMs as History Experts

## Overview

This project evaluates the ability of large language models (LLMs) to answer history-related questions, focusing on factual accuracy, reasoning, and hallucination behavior.

Designed a structured benchmarking pipeline to test both commercial and open-weight models across a diverse set of historical questions.

---

## Dataset

* Total Questions: 955
* Multiple Choice: 676
* True/False: 279
* Templates: 41
* Difficulty Levels: Easy and Hard

The dataset is built using structured templates to evaluate:

* Timeline reasoning
* Cause-and-effect understanding
* Fact-checking
* Hypothetical reasoning

---

## Models Evaluated

### Commercial Models

* GPT-4
* GPT-4 Turbo
* GPT-4o
* GPT-4o Mini
* GPT-3.5 Turbo

### Open-Weight Models

* LLaMA (8B, 70B)
* Qwen (32B, 72B)
* Mistral (7B, 24B, 123B)
* Gemma3 (27B)
* AYA / AYA Expanse
* GPT-OSS (20B, 120B)
* Phi-4

---

## Methods

* Zero-shot evaluation
* Few-shot (5-shot) prompting
* Template-based dataset construction
* Automatic dataset format detection
* Multi-model benchmarking

---

## Results

* Accuracy range: ~71% to 83%
* Larger models consistently outperform smaller models
* Few-shot prompting improves performance in most cases
* LLMs struggle with:

  * timeline consistency
  * hypothetical reasoning
  * hallucination control

---

## How to Run

```bash
pip install -r requirements.txt
python scripts/evaluate_openai.py
```

---

## Notes

Full dataset and additional results available upon request.

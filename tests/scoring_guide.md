# Evaluation Scoring Guide

This guide explains how chatbot answers are scored in the RAG evaluation module. Answers are scored on **three criteria**: Accuracy, Specificity, and Conciseness. Each criterion is rated 1–3; the final score is the sum of the three (maximum 9).

---

## Accuracy

Measures whether the answer is factually correct and aligns with expected key points.

| Score | Meaning |
|-------|---------|
| 1 | Incorrect or hallucinated |
| 2 | Partially correct |
| 3 | Fully correct |

---

## Specificity

Measures whether the answer uses course-specific tools and concepts (e.g. ChromaDB, Gradio, RAG) rather than generic language.

| Score | Meaning |
|-------|---------|
| 1 | Generic answer |
| 2 | Mentions some course tools |
| 3 | Clearly references course stack |

---

## Conciseness

Measures whether the answer is an appropriate length: clear and to the point, not overly long or vague.

| Score | Meaning |
|-------|---------|
| 1 | Overly long or vague |
| 2 | Moderate length |
| 3 | Clear and concise |

---

## Final Score

The **total score** is the sum of the three criteria:

**total = accuracy + specificity + conciseness**

- **Maximum score = 9** (3 + 3 + 3).
- Results are stored in `results.csv` with one row per question and model type (e.g. Base model, Fine tuned model, Fine tuned + RAG model).

---

## How to test and see accuracy results

**1. Run unit tests (pytest)**  
From the project root:

```bash
python -m pytest tests/scoring.py -v
```

This runs all scoring tests and confirms that `keyword_score`, `accuracy_score`, `conciseness_score`, `specificity_score`, and `total_score` behave as specified.

**2. Run the scoring demo**  
To see accuracy (and specificity, conciseness, total) printed for sample answers:

```bash
python tests/run_scoring_demo.py
```

The demo uses the first question from `tests/eval_test_set.json` and three sample answers (strong, partial, weak). For each answer it prints:

- `keyword_score` (0–1): fraction of expected keywords found  
- `accuracy` (1–3): derived from keyword_score  
- `specificity` (1–3): course terms (ChromaDB, Gradio, RAG, etc.)  
- `conciseness` (1–3): word count  
- `total` (3–9): accuracy + specificity + conciseness  

**3. Score your own answers**  
Use the functions in code after getting answers from your RAG pipeline or baseline:

```python
from rag.scoring import accuracy_score, specificity_score, conciseness_score, total_score

expected_points = ["vector store", "retrieval", "context documents"]
answer = "Your model's answer here."

acc = accuracy_score(answer, expected_points)
spec = specificity_score(answer)
conc = conciseness_score(answer)
total = total_score(acc, spec, conc)
print(f"Accuracy: {acc}, Total: {total}/9")
```

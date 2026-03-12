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

"""
Scoring utilities for RAG chatbot evaluation.
Scores answers on keyword coverage, accuracy, conciseness, and course-specificity.
"""

# Course-specific terms used for specificity scoring
COURSE_TERMS = frozenset({
    "chromadb", "gradio", "qwen", "phi", "embeddings", "rag"
})


def keyword_score(answer: str, expected_points: list[str]) -> float:
    """
    Compute a score between 0 and 1 based on how many expected keywords
    or phrases appear in the answer. Matching is case insensitive.

    Args:
        answer: The chatbot answer text.
        expected_points: List of keywords or phrases that should appear.

    Returns:
        Score from 0.0 to 1.0 (0 = none found, 1 = all found).
    """
    if not expected_points:
        return 1.0
    answer_lower = answer.lower().strip()
    found = sum(1 for point in expected_points if point.lower() in answer_lower)
    return found / len(expected_points)


def accuracy_score(answer: str, expected_points: list[str]) -> int:
    """
    Return an integer accuracy level based on keyword coverage.

    Args:
        answer: The chatbot answer text.
        expected_points: List of keywords or phrases that should appear.

    Returns:
        3 if most keywords appear (>= 2/3),
        2 if some keywords appear (1/3 to < 2/3),
        1 if few or none appear (< 1/3).
    """
    k = keyword_score(answer, expected_points)
    if k >= 2 / 3:
        return 3
    if k >= 1 / 3:
        return 2
    return 1


def conciseness_score(answer: str) -> int:
    """
    Score based on word count: prefer short, clear answers.

    Args:
        answer: The chatbot answer text.

    Returns:
        3 if < 40 words, 2 if 40–100 words, 1 if > 100 words.
    """
    word_count = len(answer.split()) if answer else 0
    if word_count < 40:
        return 3
    if word_count <= 100:
        return 2
    return 1


def specificity_score(answer: str) -> int:
    """
    Score based on use of course-specific tools and concepts.
    Checks for: ChromaDB, Gradio, Qwen, Phi, embeddings, RAG.

    Args:
        answer: The chatbot answer text.

    Returns:
        3 if multiple course-specific terms appear,
        2 if one term appears, 1 if none.
    """
    if not answer:
        return 1
    text_lower = answer.lower()
    count = sum(1 for term in COURSE_TERMS if term in text_lower)
    if count >= 2:
        return 3
    if count == 1:
        return 2
    return 1


def total_score(accuracy: int, specificity: int, conciseness: int) -> int:
    """
    Compute total evaluation score from the three component scores.

    Args:
        accuracy: Score 1–3.
        specificity: Score 1–3.
        conciseness: Score 1–3.

    Returns:
        Total score (max = 9).
    """
    return accuracy + specificity + conciseness

"""
LLM integration via OpenRouter.

OpenRouter is a unified gateway to 200+ models. We use the standard OpenAI SDK
client pointing base_url at OpenRouter — no extra dependencies needed.

Public API:
    generate_answer(query, context_chunks)  →  str   (RAG answer)
    baseline_answer(question)               →  str   (no-context answer for eval)
"""
from openai import OpenAI

from .config import OPENROUTER_API_KEY, LLM_MODEL

# ── Client singleton ──────────────────────────────────────────────────────────
_client: OpenAI | None = None


def _get_client() -> OpenAI:
    global _client
    if _client is None:
        _client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=OPENROUTER_API_KEY,
        )
    return _client


# ── Prompts ───────────────────────────────────────────────────────────────────
SYSTEM_PROMPT = """\
You are Ask Andela, an AI study assistant for the A3 AI Engineering Bootcamp cohort.

You answer student questions using only the course materials and channel discussions \
provided as context. Your answers should reflect the teaching style, tools, and \
terminology specific to this cohort — not generic internet knowledge.

Guidelines:
- Be concise: 3–6 sentences per answer.
- Ground every answer in the provided context. If the context does not contain enough \
information, say so clearly.
- Never start with "As an AI language model…" or similar preambles.
- Reference specific course tools (ChromaDB, Gradio, QLoRA, OpenRouter, etc.) by name \
when relevant.
- Every answer must be self-contained — do not reference "the slide above" or \
"as mentioned earlier".\
"""


def build_rag_prompt(query: str, context_chunks: list[dict]) -> str:
    """Assemble the user-turn message: retrieved context blocks + student question."""
    context_text = "\n\n---\n\n".join(
        f"[Source: {c['source']} | Category: {c['category']}]\n{c['content']}"
        for c in context_chunks
    )
    return (
        "Use the following course materials to answer the student's question.\n"
        "If the context does not cover the question, say so honestly.\n\n"
        f"CONTEXT:\n{context_text}\n\n"
        f"STUDENT QUESTION:\n{query}"
    )


# ── Generation ────────────────────────────────────────────────────────────────

def generate_answer(
    query: str,
    context_chunks: list[dict],
    model: str = LLM_MODEL,
    temperature: float = 0.1,
    max_tokens: int = 512,
) -> str:
    """Call the OpenRouter LLM with RAG context and return the answer string."""
    response = _get_client().chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": build_rag_prompt(query, context_chunks)},
        ],
        temperature=temperature,
        max_tokens=max_tokens,
        extra_headers={
            "HTTP-Referer": "https://github.com/ask-andela",
            "X-Title":      "Ask Andela",
        },
    )
    return response.choices[0].message.content.strip()


def baseline_answer(question: str, model: str = LLM_MODEL) -> str:
    """
    Call the LLM with NO retrieval context.
    Used as the baseline comparison column in evaluation.
    """
    response = _get_client().chat.completions.create(
        model=model,
        messages=[
            {
                "role":    "system",
                "content": "You are a helpful AI assistant. Answer the following question concisely.",
            },
            {"role": "user", "content": question},
        ],
        temperature=0.1,
        max_tokens=512,
    )
    return response.choices[0].message.content.strip()

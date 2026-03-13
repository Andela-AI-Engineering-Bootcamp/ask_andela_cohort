"""
LLM integration via OpenRouter.

OpenRouter is a unified gateway to 200+ models. We use the standard OpenAI SDK
client pointing base_url at OpenRouter - no extra dependencies needed.

Public API:
    generate_answer(query, context_chunks, history)  ->  str            (RAG answer, blocking)
    stream_answer(query, context_chunks, history)    ->  Iterator[str]  (RAG answer, streaming)
    baseline_answer(question)                        ->  str            (no-context answer for eval)
"""
from collections.abc import Iterator

from openai import OpenAI

from .config import OPENROUTER_API_KEY, LLM_MODEL, MAX_HISTORY_TURNS

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

You serve both students and course leads. Adapt accordingly - students need guidance on \
their own learning journey, while course leads may need programme-wide context and \
facilitation tips.

You answer questions using only the course materials and channel discussions provided as \
context. Your answers reflect the teaching style, tools, and terminology specific to this \
cohort - not generic internet knowledge.

Tone and style:
- Be warm and approachable, like a knowledgeable peer - not terse or robotic.
- Cover all the details the question calls for. A simple question deserves a concise \
answer; a detailed question deserves a thorough one. Never cut a response short \
just to keep it brief - completeness comes first, padding comes last.
- Use bullet points or short paragraphs when listing multiple items; it is easier \
to read than a wall of text.

Content rules:
- Ground every answer in the provided context. If the context is insufficient, say so \
clearly and suggest where the user might find the answer (e.g. "check the staff \
announcements channel" or "ask your squad lead").
- URLs matter: whenever a URL appears in the context that is relevant to your answer, \
include it verbatim and in full - never paraphrase, shorten, or silently drop a link.
- Use the full conversation history to resolve follow-up questions and ambiguous \
references (e.g. "Are you sure?", "Look at the staff announcements instead", \
"What about week 3?").
- Handle conversational turns naturally - respond warmly to "Thanks", "Got it", \
"Never mind" without forcing an answer from context or asking for a question.
- Never start with "As an AI language model..." or similar preambles.
- Reference specific course tools (ChromaDB, Gradio, QLoRA, OpenRouter, etc.) by name \
when relevant.
- Every answer must be self-contained - do not reference "the slide above" or \
"as mentioned earlier".
- Do not expose internal file names (like resource_program_expectations.txt) to the \
user; refer to content by its natural title or topic instead.\
"""


def build_rag_prompt(query: str, context_chunks: list[dict]) -> str:
    """Assemble the user-turn message: retrieved context blocks + question."""
    context_text = "\n\n---\n\n".join(
        f"[Source: {c['source']} | Category: {c['category']}]\n{c['content']}"
        for c in context_chunks
    )
    return (
        "Use the following course materials to answer the question.\n"
        "If the context does not cover the question, say so honestly.\n\n"
        f"CONTEXT:\n{context_text}\n\n"
        f"QUESTION:\n{query}"
    )


# ── Generation ────────────────────────────────────────────────────────────────

_EXTRA_HEADERS = {
    "HTTP-Referer": "https://github.com/ask-andela",
    "X-Title":      "Ask Andela",
}


def _build_messages(
    query: str,
    context_chunks: list[dict],
    history: list[dict] | None,
) -> list[dict]:
    """Assemble the full message list: system prompt + history window + RAG user turn."""
    messages: list[dict] = [{"role": "system", "content": SYSTEM_PROMPT}]
    if history:
        messages.extend(history[-(MAX_HISTORY_TURNS * 2):])
    messages.append({"role": "user", "content": build_rag_prompt(query, context_chunks)})
    return messages


def generate_answer(
    query: str,
    context_chunks: list[dict],
    model: str = LLM_MODEL,
    history: list[dict] | None = None,
    temperature: float = 0.3,
    max_tokens: int = 1024,
) -> str:
    """
    Call the OpenRouter LLM with RAG context and conversation history (blocking).
    Used by evaluate.py and the baseline comparison.
    """
    response = _get_client().chat.completions.create(
        model=model,
        messages=_build_messages(query, context_chunks, history),
        temperature=temperature,
        max_tokens=max_tokens,
        extra_headers=_EXTRA_HEADERS,
    )
    return response.choices[0].message.content.strip()


def stream_answer(
    query: str,
    context_chunks: list[dict],
    model: str = LLM_MODEL,
    history: list[dict] | None = None,
    temperature: float = 0.3,
    max_tokens: int = 1024,
) -> Iterator[str]:
    """
    Stream the LLM response token by token (non-blocking).
    Yields each text delta as it arrives from OpenRouter.
    Used by the Gradio UI for live token-by-token rendering.
    """
    with _get_client().chat.completions.create(
        model=model,
        messages=_build_messages(query, context_chunks, history),
        temperature=temperature,
        max_tokens=max_tokens,
        stream=True,
        extra_headers=_EXTRA_HEADERS,
    ) as stream:
        for chunk in stream:
            # Some chunks (role-only, [DONE] sentinel) arrive with empty choices
            if not chunk.choices:
                continue
            delta = chunk.choices[0].delta.content
            if delta:
                yield delta


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

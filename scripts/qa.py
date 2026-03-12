import requests
from langchain_chroma import Chroma
from scripts.embeddings import HFEmbeddings
from config import VECTOR_DB_DIR, OPENROUTER_API_KEY, OPENROUTER_API_BASE, OPENROUTER_MODEL

hf = HFEmbeddings()

# Load existing vector DB (do NOT rebuild)
vectorstore = Chroma(
    persist_directory=str(VECTOR_DB_DIR),
    embedding_function=hf
)

retriever = vectorstore.as_retriever(search_kwargs={"k": 4})


def query_llm(prompt):
    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json"
    }

    payload = {
        "model": OPENROUTER_MODEL,
        "messages": [
            {"role": "user", "content": prompt}
        ]
    }

    response = requests.post(
        f"{OPENROUTER_API_BASE}/chat/completions",
        headers=headers,
        json=payload
    )

    response.raise_for_status()
    return response.json()["choices"][0]["message"]["content"]


def ask(question: str):

    docs = retriever.invoke(question)

    context = "\n\n".join(doc.page_content for doc in docs)

    prompt = f"""
Answer the question using the context below.

Context:
{context}

Question:
{question}

Answer:
"""

    answer = query_llm(prompt)

    return {
        "result": answer,
        "source_documents": docs
    }

    
"""
Gradio UI definition for Ask Andela.

Public API:
    build(collection) -> gr.Blocks
        Constructs and returns the Gradio app, ready to launch.
"""
from pathlib import Path

import chromadb
import gradio as gr

from rag.pipeline import ask_andela

STYLES_CSS = (Path(__file__).parent / "styles.css").read_text()

EXAMPLES = [
    "What is RAG and why do we use it?",
    "What is QLoRA and why is it used for fine-tuning?",
    "What are the capstone project submission requirements?",
    "What is the format of submitting daily updates?",
    "What model should I use for fine-tuning on Google Colab?",
    "When are behavioral assignments due?",
]

ANDELA_LOGO = "https://cdn.worldvectorlogo.com/logos/andela.svg"


def build(collection: chromadb.Collection) -> gr.Blocks:
    """Return the fully wired Gradio Blocks app."""

    def chat(question: str, history: list[dict]) -> tuple[list[dict], str, str]:
        question = question.strip()
        if not question:
            return history, "", ""

        # Capture history *before* the current question so the pipeline can
        # use prior turns for retrieval contextualization and LLM context.
        prior_history = history

        try:
            result = ask_andela(question, collection, history=prior_history)
            reply = result["answer"]
            sources = "\n".join(f"- {s}" for s in result["sources"])
            sources_md = f"{sources}" if sources else ""
        except Exception as exc:
            reply = f"Something went wrong: {exc}"
            sources_md = ""

        history = history + [
            {"role": "user",      "content": question},
            {"role": "assistant", "content": reply},
        ]
        return history, "", sources_md

    def clear_chat() -> tuple[list, str, str]:
        return [], "", ""

    with gr.Blocks(title="Ask Andela", css=STYLES_CSS) as demo:

        gr.HTML(f"""
            <div id="andela-header">
                <img src="{ANDELA_LOGO}" alt="Andela" onerror="this.style.display='none'">
                <h1>Ask Andela</h1>
                <p>Your AI study assistant for the A3 AI Engineering Bootcamp</p>
            </div>
        """)

        with gr.Row():
            with gr.Column(scale=3):
                chatbot = gr.Chatbot(
                    elem_id="chatbot",
                    type="messages",
                    height=480,
                    show_label=False,
                    avatar_images=(None, ANDELA_LOGO),
                )
                with gr.Row():
                    question_box = gr.Textbox(
                        elem_id="question-box",
                        placeholder="Ask anything about the bootcamp materials…",
                        show_label=False,
                        lines=1,
                        scale=5,
                    )
                    ask_btn   = gr.Button("Ask",   variant="primary",   elem_id="ask-btn",   scale=1)
                    clear_btn = gr.Button("Clear", variant="secondary", elem_id="clear-btn", scale=1)

                gr.Examples(examples=EXAMPLES, inputs=question_box, label="Quick questions")

            with gr.Column(scale=1, min_width=220):
                gr.HTML("<p id='sources-label'>Sources</p>")
                sources_panel = gr.Markdown(
                    elem_id="sources-panel",
                    value="*Sources will appear here after you ask a question.*",
                )

        gr.HTML("""
            <div id="footer">
                Powered by <span>OpenRouter</span> &nbsp;·&nbsp;
                Embeddings: all-MiniLM-L6-v2 &nbsp;·&nbsp;
                Vector store: <span>ChromaDB</span>
            </div>
        """)

        shared = dict(
            fn=chat,
            inputs=[question_box, chatbot],
            outputs=[chatbot, question_box, sources_panel],
        )
        ask_btn.click(**shared)
        question_box.submit(**shared)
        clear_btn.click(fn=clear_chat, outputs=[chatbot, question_box, sources_panel])

    return demo

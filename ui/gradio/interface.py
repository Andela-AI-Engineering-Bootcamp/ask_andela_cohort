"""
Gradio UI definition for Ask Andela.

Public API:
    build(collection) -> gr.Blocks
        Constructs and returns the Gradio app, ready to launch.
"""
from pathlib import Path

import chromadb
import gradio as gr

from rag.pipeline import stream_andela

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

    def chat(question: str, history: list[dict]):
        """
        Streaming chat handler — yields partial UI state on every token.

        Phases:
            1. "sources" event: retrieval is done; populate the sources panel
               and open an empty assistant bubble immediately.
            2. "token" events: append each delta to the assistant bubble
               so the user sees the answer build word by word.
        """
        question = question.strip()
        if not question:
            yield history, "", ""
            return

        prior_history = list(history)
        sources_md = ""
        partial_reply = ""

        try:
            for event in stream_andela(question, collection, history=prior_history):
                if event["type"] == "sources":
                    sources_md = (
                        "\n".join(f"- {s}" for s in event["sources"])
                        if event["sources"] else ""
                    )
                    # Clear the input box and open the assistant bubble immediately
                    yield (
                        prior_history + [
                            {"role": "user",      "content": question},
                            {"role": "assistant", "content": ""},
                        ],
                        "",
                        sources_md,
                    )
                else:
                    partial_reply += event["token"]
                    yield (
                        prior_history + [
                            {"role": "user",      "content": question},
                            {"role": "assistant", "content": partial_reply},
                        ],
                        "",
                        sources_md,
                    )
        except Exception as exc:
            yield (
                prior_history + [
                    {"role": "user",      "content": question},
                    {"role": "assistant", "content": f"Something went wrong: {exc}"},
                ],
                "",
                "",
            )

    def clear_chat() -> tuple[list, str, str]:
        return [], "", ""

    with gr.Blocks(title="Ask Andela", css=STYLES_CSS) as demo:

        gr.HTML(f"""
            <div id="andela-header">
                <img src="{ANDELA_LOGO}" alt="Andela" onerror="this.style.display='none'">
                <h1>Ask Andela</h1>
                <p>Your Cohort's Single Source of Truth</p>
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
                        placeholder="Ask anything about the bootcamp materials...",
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
                Powered by <span>OpenRouter</span> &nbsp;|&nbsp;
                Embeddings: <span>BAAI/bge-small-en-v1.5</span> &nbsp;|&nbsp;
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

    demo.queue()
    return demo

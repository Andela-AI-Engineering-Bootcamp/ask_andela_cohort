import gradio as gr
from scripts.qa import ask


def chat(question):
    result = ask(question)

    answer = result["result"]
    sources = "\n".join(
        doc.metadata.get("source", "unknown")
        for doc in result["source_documents"]
    )

    return answer, sources


ui = gr.Interface(
    fn=chat,
    inputs=gr.Textbox(lines=2, placeholder="Ask about the Andela cohort"),
    outputs=[
        gr.Textbox(label="Answer"),
        gr.Textbox(label="Sources")
    ],
    title="Andela Cohort Knowledge Assistant"
)

ui.launch(inbrowser=True, share=True)
# ASK_ANDELA

## **Project Overview**
**One-liner:** An AI study buddy that knows your specific course.  

The **Cohort AI Study Buddy** is a local Q&A assistant designed to answer student questions using the cohort’s own course materials and discussion history. Unlike generic search engines, the assistant is **fine-tuned to respond in the cohort’s teaching style**, grounded entirely in course-specific knowledge such as Discourse posts, PDFs, assignments, and internal notes.  

It is demonstrated through a **Gradio interface**, allowing students to ask questions and receive accurate, style-consistent answers in real-time.

---

## **Key Features**
- **Course-Specific Knowledge:** Answers reflect your cohort’s curriculum, tools, terminology, and teaching style.  
- **Retrieval-Augmented Generation (RAG):** Retrieves relevant content chunks from course materials before generating answers.  
- **Live Gradio Interface:** Interactive, user-friendly UI for asking questions.  
- **Source Citation:** Optionally cites the course material used to answer.  
- **Feedback Loop:** Allows rating of responses to improve relevance over time.

---

## **Folder Structure**
```text
cohort_ai_study_buddy/
├── data/
│   ├── discourse_export.csv       # Exported posts/Q&A from Discourse
│   ├── course_materials/         # PDFs, DOCX, spreadsheets, etc.
│   │   ├── module1.pdf
│   │   ├── module2.pdf
│   │   └── ...
│   └── processed_chunks.pkl      # Preprocessed & chunked text for embeddings
│
├── embeddings/
│   └── cohort_vector_store/      # FAISS or Chroma vector DB files
│
├── scripts/
│   ├── preprocess.py             # Load, clean, and chunk course material
│   ├── build_embeddings.py       # Generate embeddings & save vector store
│   ├── query_ai.py               # Load vector store & run LLM query chain
│   └── fine_tune_llm.py          # Optional: fine-tune LLM on cohort Q&A style
│
├── gradio_ui/
│   └── app.py                    # Gradio interface script
│
├── requirements.txt              # Python dependencies
├── README.md                     # Project overview & instructions
└── .env                          # API keys (OpenAI, etc.)


---

## Installation

### 1. Clone the Repository
```bash
git clone <repository-url>
cd ask_andela_cohort

---

## Running App

### 1. Using UV
```bash
uv venv
uv sync
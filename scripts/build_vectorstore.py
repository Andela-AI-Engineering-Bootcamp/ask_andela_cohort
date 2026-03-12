import pickle
from langchain_community.vectorstores import Chroma
from scripts.embeddings import HFEmbeddings
from config import CHUNKS_FILE, VECTOR_DB_DIR

with open(CHUNKS_FILE, "rb") as f:
    chunks = pickle.load(f)


hf = HFEmbeddings()

vectorstore = Chroma.from_documents(
    documents=chunks,
    embedding=hf, 
    persist_directory=str(VECTOR_DB_DIR)
)

vectorstore.persist()
retriever = vectorstore.as_retriever(search_kwargs={"k":4})

print("Vectorstore ready!")
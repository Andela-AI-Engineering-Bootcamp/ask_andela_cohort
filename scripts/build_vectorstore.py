import pickle
from langchain_community.vectorstores import Chroma
from scripts.embeddings import HFEmbeddings
from config import CHUNKS_FILE, VECTOR_DB_DIR

# Load document chunks
with open(CHUNKS_FILE, "rb") as f:
    chunks = pickle.load(f)

# Initialize embeddings object
hf = HFEmbeddings()

# Create Chroma vectorstore (pass hf object, not method)
vectorstore = Chroma.from_documents(
    documents=chunks,
    embedding=hf,  # <-- object, must have embed_documents()
    persist_directory=str(VECTOR_DB_DIR)
)

vectorstore.persist()
retriever = vectorstore.as_retriever(search_kwargs={"k":4})

print("Vectorstore ready!")
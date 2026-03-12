import pickle
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter


from config import RAW_DIR, PROCESSED_DIR, CHUNKS_FILE

print("Loading cohort documents...")

docs = []
for file in RAW_DIR.glob("*.txt"):
    loader = TextLoader(file, encoding="utf-8")
    docs.extend(loader.load())

print(f"Loaded {len(docs)} documents")

splitter = RecursiveCharacterTextSplitter(
    chunk_size=800,
    chunk_overlap=150
)

chunks = splitter.split_documents(docs)

print(f"Created {len(chunks)} chunks")

with open(CHUNKS_FILE, "wb") as f:
    pickle.dump(chunks, f)

print("Chunks saved successfully.")
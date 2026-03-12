from sentence_transformers import SentenceTransformer

class HFEmbeddings:
    def __init__(self, model_name="all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)

    def embed_documents(self, texts):
        embeddings = self.model.encode(texts, show_progress_bar=True)
        # Convert numpy array to list of lists
        return embeddings.tolist() if hasattr(embeddings, "tolist") else embeddings

    def embed_query(self, text):
        embedding = self.model.encode([text])
        return embedding[0].tolist() if hasattr(embedding[0], "tolist") else embedding[0]
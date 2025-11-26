from sentence_transformers import SentenceTransformer

# -------------------------------
# Embedding Model
# -------------------------------

class EmbeddingModel:
    """Handles encoding of text descriptions using a Hugging Face transformer model."""

    def __init__(self, model_name="all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)

    def encode(self, sentences):
        embeddings = self.model.encode(sentences, convert_to_numpy=True)
        return embeddings


# -------------------------------
# Agent Class
# -------------------------------

class Agent:
    """Represents a household agent on the grid."""
    def __init__(self, desc_idx, embedding, pos, neighbourhood):
        self.desc_idx = desc_idx
        self.embedding = embedding
        self.pos = pos
        self.happy = False
        self.neighbourhood = neighbourhood

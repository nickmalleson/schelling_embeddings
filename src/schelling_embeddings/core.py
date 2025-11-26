from sentence_transformers import SentenceTransformer

# -------------------------------
# Embedding Model
# Used to calculate the embeddings from the sentences
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



# -------------------------------
# Neighbourhood Class
# Neighbourhoods contain cells and agents
# -------------------------------

class Neighbourhood:
    """Represents a rectangular neighbourhood block on the grid."""
    def __init__(self, nid):
        self.id = nid
        self.cells = []          # list of (i, j) tuples
        self.agents = set()      # set of Agent objects

    def add_cell(self, pos):
        self.cells.append(pos)

    def add_agent(self, agent):
        self.agents.add(agent)

    def remove_agent(self, agent):
        self.agents.discard(agent)

    def get_cells(self):
        return list(self.cells)

    def get_agents(self):
        return list(self.agents)

    def bounds(self):
        """Return (min_i, max_i, min_j, max_j) for cells (None if no cells)."""
        if not self.cells:
            return None
        rows = [c[0] for c in self.cells]
        cols = [c[1] for c in self.cells]
        return (min(rows), max(rows), min(cols), max(cols))

    def mean_embedding(self):
        """Return mean embedding of agents in the neighbourhood or None if empty."""
        if not self.agents:
            return None
        embs = np.array([a.embedding for a in self.agents])
        return np.mean(embs, axis=0)

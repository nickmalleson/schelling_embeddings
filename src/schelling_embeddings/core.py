from sentence_transformers import SentenceTransformer
import numpy as np

import random
import warnings

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

    def step(self, model):
        # Compute similarity
        sim = None
        if model.move_decision == "neighbours":
            # Move decision based on surrounding neighbours
            neighbours = model._get_neighbours(self.pos)
            if not neighbours:
                warnings.warn(f"Agent at {self.pos} has no neighbours")
                sim = 1.0
            else:
                sim = model._compute_similarity(self, neighbours)
        elif model.move_decision == "neighbourhood":
            # Move decision based on neighbourhood embedding
            sim = model._compute_similarity(self, self.neighbourhood)

        else:
            raise ValueError(f"Unknown move decision method: {model.move_decision}")

        if sim >= model.similarity_threshold:
            self.happy = True
        else:
            self.happy = False
            # Move unhappy agent to a random empty cell
            model.grid[self.pos] = None
            old_neigh = self.neighbourhood
            model.empty_cells.append(self.pos)
            new_pos = random.choice(model.empty_cells)
            model.empty_cells.remove(new_pos)
            self.pos = new_pos
            model.grid[new_pos] = self
            # update neighbourhood membership using objects
            new_nid = model.get_neighbourhood(new_pos)
            new_neigh = model.neighbourhoods.get(new_nid)
            if old_neigh is not None:
                old_neigh.remove_agent(self)
            self.neighbourhood = new_neigh
            if new_neigh is not None:
                new_neigh.add_agent(self)


# -------------------------------
# Neighbourhood Class
# Neighbourhoods contain cells and agents
# -------------------------------

class Neighbourhood:
    """Represents a rectangular neighbourhood block on the grid."""
    def __init__(self, nid, model):
        self.id = nid
        self.model = model
        self.cells = []          # list of (i, j) tuples
        self.agents = set()      # set of Agent objects
        self.neighbourhood_embedding = None

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

    def _mean_embedding(self):
        """Return mean embedding of agents in the neighbourhood or the average
        population embedding if empty."""
        if not self.agents:
            warnings.warn(f"No agents in neighbourhood found in {self.id}, "
                          f"returning average embedding.")
            return np.mean(np.array([a.embedding for a in self.agents]), axis=0)

        embs = np.array([a.embedding for a in self.agents])
        return np.mean(embs, axis=0)

    def step(self):
        self.neighbourhood_embedding = self._mean_embedding()



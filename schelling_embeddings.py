# ***********************************************************
# Schelling Model with Embeddings (v3)
# ***********************************************************
#
# Update from v1: households described using sentence embeddings, not bespoke variables.
# Update from v2: include wealth and neighbourhoods
#
# Created with (well, 'by' really!) ChatGPT.
# Agents are described with hypothetical text descriptions that describe three features of
# each household: structure, income and political beliefs. These are converted to embeddings.
# Neighbourhood 'character' is calcualted as the mean of the constituent agent embeddings
# Agent move decision based on the difference between their emedding and their neighbourhood
# embedding.

import numpy as np
import matplotlib.pyplot as plt
import math
import random
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.manifold import TSNE
import matplotlib.colors as mcolors
from sentence_transformers import SentenceTransformer
import matplotlib.patches as mpatches

# -------------------------------
# Household Descriptions.
# Some arbitrary descriptions of households, created by an LLM, that describe their household structure, income
# and political beliefs.
# The profiles themselves aren't important as it's just to test how well the embeddings work.
# Note that when the model runs in earnest (see schelling_embeddinggs.ipynb) it reads a large number of descriptions
# -------------------------------
household_descriptions = [
    "A retired couple living alone in a semi-detached house in a suburban area, relying on state pensions and modest savings, strongly supporting the Conservative party",
    "An elderly couple residing in a suburban, semi-detached house, drawing income from their savings and their state pensions, voting for the Conservative party consistently",
    "A young, single professional renting a studio flat in a city centre, earning a salary around £35,000 from a career in marketing, voting for the Liberal Democrats and actively campaigning for environmental causes",
    "A large, multi-generational family residing in a terraced house, with the patriarch working as a manual labourer on a zero-hours contract, the matriarch a part-time carer, and several children, identifying as Labour supporters and strongly union-backed",
    "A single parent with three children, living in a council flat, surviving on a tight budget that includes Universal Credit and Child Tax Credits, and staunchly supporting the Labour party, particularly its more left-wing elements",
]

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

# -------------------------------
# Schelling Model Class
# -------------------------------

class SchellingModel:
    """
    Implements an embedding-based version of the Schelling segregation model.
    Agents decide to move based on similarity of text-derived embeddings.
    Neighbourhoods are radial sectors around the grid centre; each cell maps to one sector id.
    """
    def __init__(self, descriptions, grid_size=20, num_agents=300,
                 similarity_threshold=0.85,
                 max_iters=20,
                 num_neighbourhoods=8,
                 color_method="tsne"):
        # Descriptions of the generic households
        self.descriptions = descriptions

        # Set up the model
        self.grid_size = grid_size
        self.grid = np.full((grid_size, grid_size), None)
        self.num_agents = num_agents
        self.similarity_threshold = similarity_threshold
        self.max_iters = max_iters
        self.empty_cells = [(i, j) for i in range(grid_size) for j in range(grid_size)]
        self.agents = []
        self.happy_counts = []

        # Neighbourhood configuration
        self.num_neighbourhoods = num_neighbourhoods
        # Links between cells and neighbourhoods _init_neighbourhoods
        self.cell_to_neighbourhood = {}
        self.neighbourhood_cells = {i: [] for i in range(self.num_neighbourhoods)}
        self.neighbourhood_agents = {i: set() for i in range(self.num_neighbourhoods)}
        self._init_neighbourhoods()

        # Calculate the embeddings for the household descriptions
        self.embedding_model = EmbeddingModel()
        self.description_embeddings = self.embedding_model.encode(self.descriptions)
        print(f"Embedding shape: {self.description_embeddings.shape}")
        self.desc_lookup = {i: desc for i, desc in enumerate(self.descriptions)}

        # ------------------------
        # COLOR MAPPING USING PCA or t-SNE (t-distributed Stochastic Neighbor Embedding)
        # ------------------------
        (self.rgb_map, self.desc_color_map) = SchellingModel._init_colours(
            color_method, self.description_embeddings)

        # Initialize the grid with agents
        self._init_agents()

    def _init_agents(self):
        """Randomly place agents on the grid with one of the household types."""
        for _ in range(self.num_agents):
            # Add the agent to the grid
            pos = random.choice(self.empty_cells)
            self.empty_cells.remove(pos)
            # Define the agent 'type' (from the descriptions)
            desc_idx = random.randint(0, len(self.descriptions) - 1)
            embedding = self.description_embeddings[desc_idx]
            nid = self.get_neighbourhood(pos)
            # Create the agent
            agent = Agent(desc_idx, embedding, pos, nid)
            # place on grid
            self.grid[pos] = agent
            # add to neighbourhood agent list
            self.neighbourhood_agents.setdefault(nid, set()).add(agent)

            self.agents.append(agent)

    def _init_neighbourhoods(self):
        """Partition the grid into `num_neighbourhoods` radial sectors around the grid centre."""
        n = self.num_neighbourhoods
        mid = (self.grid_size - 1) / 2.0
        two_pi = 2 * math.pi
        # assign each cell a sector id
        for i in range(self.grid_size):
            for j in range(self.grid_size):
                # angle where rows (i) are y and cols (j) are x
                angle = math.atan2(i - mid, j - mid)  # range [-pi, pi]
                if angle < 0:
                    angle += two_pi
                sector = int(angle / (two_pi / n))
                sector = min(sector, n - 1)
                self.cell_to_neighbourhood[(i, j)] = sector
                self.neighbourhood_cells.setdefault(sector, []).append((i, j))

    def get_neighbourhood(self, pos):
        """Return neighbourhood id for a given position tuple (i, j)."""
        return self.cell_to_neighbourhood.get(pos)

    def get_neighbourhood_cells(self, neighbourhood_id):
        """Return list of positions in the neighbourhood."""
        return list(self.neighbourhood_cells.get(neighbourhood_id, []))

    def get_neighbourhood_agents(self, neighbourhood_id):
        """Return list of Agent objects currently in the neighbourhood."""
        return list(self.neighbourhood_agents.get(neighbourhood_id, set()))

    def _get_neighbours(self, pos):
        """Return non-empty neighbouring agents in the Moore neighbourhood."""
        x, y = pos
        neighbours = []
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                if dx == 0 and dy == 0:
                    continue
                nx, ny = x + dx, y + dy
                if 0 <= nx < self.grid_size and 0 <= ny < self.grid_size:
                    neighbour = self.grid[nx, ny]
                    if neighbour is not None:
                        neighbours.append(neighbour)
        return neighbours

    def _compute_similarity(self, agent, neighbours):
        """Compute average cosine similarity between agent and neighbours."""
        if not neighbours:
            return 0
        # Put the neighbour's embeddings into a matrix
        emb_matrix = np.array([n.embedding for n in neighbours])
        # Calculate the cosine similarities between the agent and all neighbours
        sims = cosine_similarity([agent.embedding], emb_matrix)
        # Return the mean similarity
        return np.mean(sims)

    def _get_rgb(self, desc_idx):
        """Return RGB colour (0–1 range) for a description index via PCA projection."""
        rgb = self.rgb_map[desc_idx]
        scaled = (rgb - self.rgb_map.min()) / (self.rgb_map.max() - self.rgb_map.min())
        return scaled

    @classmethod
    def _init_colours(cls, color_method, description_embeddings):
        """Initialize colour mapping for descriptions using t-SNE or PCA."""
        if color_method == "tsne":
            perplexity = min(5, len(description_embeddings) - 1)
            tsne_map = TSNE(n_components=2, perplexity=perplexity, random_state=42).fit_transform(
                description_embeddings
            )
            t_min = tsne_map.min(axis=0)
            t_max = tsne_map.max(axis=0)
            tsne_norm = (tsne_map - t_min) / (t_max - t_min + 1e-8)
            rgb_map = []
            for i in range(len(tsne_norm)):
                hue = tsne_norm[i, 0]
                saturation = 0.8
                value = 0.9 - 0.4 * tsne_norm[i, 1]
                color_rgb = mcolors.hsv_to_rgb((hue, saturation, value))
                rgb_map.append(color_rgb)
        else:
            from sklearn.decomposition import PCA
            pca = PCA(n_components=3)
            pca_proj = pca.fit_transform(description_embeddings)
            pca_min = pca_proj.min(axis=0)
            pca_max = pca_proj.max(axis=0)
            norm = (pca_proj - pca_min) / (pca_max - pca_min + 1e-8)
            rgb_map = norm.tolist()

        rgb_map = np.array(rgb_map)
        desc_color_map = {i: rgb_map[i] for i in range(len(rgb_map))}
        return rgb_map, desc_color_map


    def plot_grid(self, iteration, show_neighbourhoods):
        """Plot the current state of the grid with agent types shown by colour.
        If show_neighbourhoods=True, overlay neighbourhood sectors for visualization.
        """
        img = np.ones((self.grid_size, self.grid_size, 3))
        for i in range(self.grid_size):
            for j in range(self.grid_size):
                agent = self.grid[i, j]
                if agent:
                    img[i, j] = self._get_rgb(agent.desc_idx)

        plt.figure(figsize=(6, 6))
        if show_neighbourhoods:
            # build neighbourhood map for background overlay
            neigh_map = np.zeros((self.grid_size, self.grid_size), dtype=int)
            for (i, j), nid in self.cell_to_neighbourhood.items():
                neigh_map[i, j] = nid
            cmap = plt.cm.get_cmap('tab20', self.num_neighbourhoods)
            plt.imshow(neigh_map, cmap=cmap, alpha=0.12, interpolation='nearest')
        plt.imshow(img, interpolation='nearest')
        plt.title(f"Iteration {iteration}")
        plt.axis('off')

        # Create legend patches for agent types
        legend_patches = []
        for idx, color in self.desc_color_map.items():
            label = f"{idx}: {self.desc_lookup[idx][:40]}..."
            patch = mpatches.Patch(color=color, label=label)
            legend_patches.append(patch)

        plt.legend(handles=legend_patches,
                   loc='upper center',
                   bbox_to_anchor=(0.5, -0.05),
                   fancybox=True,
                   shadow=True,
                   ncol=1)
        plt.subplots_adjust(bottom=0.3)
        plt.show()

    def plot_happiness(self, return_fig=False):
        """Plot number of happy agents per iteration. Return the plot if requested."""
        fig, ax = plt.subplots(figsize=(12, 5))
        ax.plot(self.happy_counts)
        ax.set_title("Number of Happy Agents per Iteration")
        ax.set_xlabel("Iteration")
        ax.set_ylabel("Happy Agents")
        fig.tight_layout()
        if return_fig:
            return fig
        else:
            plt.show()

    def run(self, do_plots=True):
        """Run the full simulation for the configured number of iterations."""
        for it in range(self.max_iters):
            happy = 0
            for agent in self.agents:
                neighbours = self._get_neighbours(agent.pos)
                sim = self._compute_similarity(agent, neighbours)
                if sim >= self.similarity_threshold:
                    agent.happy = True
                    happy += 1
                else:
                    agent.happy = False
                    # Move unhappy agent to a random empty cell
                    self.grid[agent.pos] = None
                    old_nid = agent.neighbourhood
                    self.empty_cells.append(agent.pos)
                    new_pos = random.choice(self.empty_cells)
                    self.empty_cells.remove(new_pos)
                    agent.pos = new_pos
                    self.grid[new_pos] = agent
                    # update neighbourhood membership
                    new_nid = self.get_neighbourhood(new_pos)
                    if old_nid is not None:
                        self.neighbourhood_agents.get(old_nid, set()).discard(agent)
                    agent.neighbourhood = new_nid
                    self.neighbourhood_agents.setdefault(new_nid, set()).add(agent)

            self.happy_counts.append(happy)
            print(f"Iteration {it}: {happy} happy agents")
            if do_plots:
                self.plot_grid(it, show_neighbourhoods=True)

# -------------------------------
# Main Execution
# -------------------------------

if __name__ == "__main__":

    model = SchellingModel(household_descriptions,
                           grid_size=20,
                           num_agents=300,
                           similarity_threshold=0.65,
                           max_iters=50,
                           num_neighbourhoods=8,
                           color_method="pca")
    model.run(do_plots=True)
    model.plot_happiness(return_fig=False)
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
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
from collections.abc import Iterable


# Import the sentence transformer model for embeddings.
# NOTE: the huggingface-cli command needs to be used to authenticate with huggingface locally.
# Try huggingface-cli logout, then huggingface-cli login, using a token from https://huggingface.co/settings/tokens
from src.schelling_embeddings.core import EmbeddingModel, Agent, Neighbourhood

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
                 color_method="tsne",
                 do_plots=True,
                 move_decision="neighbours"):

        # Descriptions of the generic households
        self.descriptions = descriptions

        # Set up the model
        self.grid_size = grid_size
        self.grid = np.full((grid_size, grid_size), None)
        self.num_agents = num_agents
        self.max_iters = max_iters
        self.empty_cells = [(i, j) for i in range(grid_size) for j in range(grid_size)]
        self.agents = []
        self.happy_counts = []

        # Agent's decision on whether to move
        self.similarity_threshold = similarity_threshold
        valid_move_decisions = ["neighbours", "neighbourhood"]
        if move_decision not in valid_move_decisions:
            raise ValueError(f"move_decision must be one of {valid_move_decisions}")
        self.move_decision = move_decision

        # Neighbourhood configuration
        self.num_neighbourhoods = num_neighbourhoods
        # Links between cells and neighbourhoods (populated in _init_neighbourhoods)
        # neighbourhood_cells stores _list_ of cells because these are static.
        # neighbourhood_agents stores _set_ of agents because these change dynamically.
        self.neighbourhoods = {i: Neighbourhood(i, self) for i in range(self.num_neighbourhoods)}
        self.cell_to_neighbourhood = {}
        self._init_neighbourhoods()

        # Calculate the embeddings for the household descriptions
        self.embedding_model = EmbeddingModel()
        self.description_embeddings = self.embedding_model.encode(self.descriptions)
        print(f"Embedding shape: {self.description_embeddings.shape}")
        self.desc_lookup = {i: desc for i, desc in enumerate(self.descriptions)}

        # ------------------------
        # Plots and COLOR MAPPING USING PCA or t-SNE (t-distributed Stochastic Neighbor Embedding)
        # ------------------------
        self.do_plots = do_plots
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
            neighbourhood_obj = self.neighbourhoods.get(nid)
            # Create the agent
            agent = Agent(desc_idx, embedding, pos, neighbourhood_obj)
            # place on grid
            self.grid[pos] = agent
            neighbourhood_obj.add_agent(agent)
            self.agents.append(agent)

    # TODO: remove get_neighbours and use neighbourhoods instead
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

    def _init_neighbourhoods(self):
        """Partition the grid into `num_neighbourhoods` rectangular blocks."""
        n = self.num_neighbourhoods
        g = self.grid_size

        # Choose rows/cols close to a square layout
        nrows = int(math.floor(math.sqrt(n))) or 1
        ncols = math.ceil(n / nrows)

        # Block size (ceiling so every cell is covered)
        block_h = math.ceil(g / nrows)
        block_w = math.ceil(g / ncols)

        # Create neighbourhoods and map cells to neighbourhood ids
        for i in range(g):
            for j in range(g):
                row_block = min(i // block_h, nrows - 1)
                col_block = min(j // block_w, ncols - 1)
                sector = row_block * ncols + col_block
                if sector >= n:
                    sector = n - 1
                self.cell_to_neighbourhood[(i, j)] = sector
                self.neighbourhoods[sector].add_cell((i, j))

    def get_neighbourhood(self, pos):
        """Return neighbourhood id for a given position tuple (i, j)."""
        return self.cell_to_neighbourhood.get(pos)

    def get_neighbourhood_cells(self, neighbourhood_id):
        """Return list of positions in the neighbourhood."""
        return self.neighbourhoods.get(neighbourhood_id).get_cells()

    def get_neighbourhood_agents(self, neighbourhood_id):
        """Return list of Agent objects currently in the neighbourhood."""
        return self.neighbourhoods.get(neighbourhood_id).get_agents()

    def _compute_similarity(self, agent, neighbours):
        """Compute average cosine similarity between agent and neighbours, or an agent and its neighbourhood.

        neighbours: list of Agent objects OR a single neighbourhood object (nothing else)
        """
        if isinstance(neighbours, Neighbourhood):
            # single neighbourhood
            return cosine_similarity([agent.embedding], [neighbours.neighbourhood_embedding])

        elif isinstance(neighbours, Iterable) and not isinstance(neighbours, (str, bytes)):
            # list (or iterable) of agents (but not strings, which are also iterable)
            if not neighbours:
                raise ValueError(f"Cannot compute similarity with empty neighbours list.")
            # Put the neighbour's embeddings into a matrix
            emb_matrix = np.array([n.embedding for n in neighbours])
            # Calculate the cosine similarities between the agent and all neighbours
            sims = cosine_similarity([agent.embedding], emb_matrix)
            # Return the mean similarity
            return np.mean(sims)

        raise ValueError(f"neighbours parameter must be either a Neighbourhood object or an "
                         f"iterable of Agent objects. Not {type(neighbours)}.")

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
            pca = PCA(n_components=3)
            pca_proj = pca.fit_transform(description_embeddings)
            pca_min = pca_proj.min(axis=0)
            pca_max = pca_proj.max(axis=0)
            norm = (pca_proj - pca_min) / (pca_max - pca_min + 1e-8)
            rgb_map = norm.tolist()

        rgb_map = np.array(rgb_map)
        desc_color_map = {i: rgb_map[i] for i in range(len(rgb_map))}
        return rgb_map, desc_color_map

    # python
    def plot_grid(self, iteration, show_neighbourhoods):
        """Plot grid side-by-side:
        - left: cell colours (agents) with neighbourhood boundaries
        - right: neighbourhood colours with neighbourhood boundaries
        """
        # build agent image
        img = np.ones((self.grid_size, self.grid_size, 3))
        for i in range(self.grid_size):
            for j in range(self.grid_size):
                agent = self.grid[i, j]
                if agent:
                    img[i, j] = self._get_rgb(agent.desc_idx)

        # prepare figure with two subplots
        fig, axs = plt.subplots(1, 2, figsize=(12, 6))
        ax_left, ax_right = axs

        # left: agents image
        ax_left.imshow(img, interpolation='nearest', origin='upper')
        ax_left.set_title(f"Iteration {iteration} — Cells")
        ax_left.axis('off')

        # right: neighbourhood colours

        # Build neighbourhood embeddings matrix (one row per neighbourhood).
        neigh_embeddings = np.vstack(
            [np.asarray(neigh.neighbourhood_embedding) for neigh in self.neighbourhoods.values()]
        )

        # Project neighbourhood embeddings to 3 components for RGB.
        n_samples, n_features = neigh_embeddings.shape
        n_comp = min(3, n_samples, n_features)
        if n_comp >= 1 and n_samples > 0:
            pca = PCA(n_components=n_comp)
            proj = pca.fit_transform(neigh_embeddings)  # (n_neigh, n_comp)
            # If fewer than 3 components, pad with zeros to get 3 channels
            if n_comp < 3:
                proj = np.hstack([proj, np.zeros((n_samples, 3 - n_comp))])
            # Normalize each channel to 0-1
            p_min = proj.min(axis=0)
            p_max = proj.max(axis=0)
            denom = (p_max - p_min) + 1e-8
            rgb_neigh = (proj - p_min) / denom
        else:
            # fallback single grey colour per neighbourhood
            rgb_neigh = np.tile(np.array([[0.6, 0.6, 0.6]]), (self.num_neighbourhoods, 1))

        # Build RGB image for neighbourhoods using the rgb_neigh mapping
        neigh_img = np.zeros((self.grid_size, self.grid_size, 3))
        for (i, j), nid in self.cell_to_neighbourhood.items():
            neigh_img[i, j] = rgb_neigh[nid]

        # right: neighbourhood colours (from neighbourhood embeddings)
        ax_right.imshow(neigh_img, interpolation='nearest', origin='upper')
        ax_right.set_title(f"Iteration {iteration} — Neighbourhoods")
        ax_right.axis('off')

        if show_neighbourhoods:
            for nid, neighbourhood in self.neighbourhoods.items():
                bounds = neighbourhood.bounds()
                if bounds is None:
                    continue
                min_i, max_i, min_j, max_j = bounds
                rect = mpatches.Rectangle(
                    (min_j - 0.5, min_i - 0.5),
                    width=(max_j - min_j + 1),
                    height=(max_i - min_i + 1),
                    fill=False,
                    edgecolor='k',
                    linewidth=1.5,
                    zorder=3,
                )
                ax_left.add_patch(mpatches.Rectangle(rect.get_xy(), rect.get_width(), rect.get_height(),
                                                     fill=False, edgecolor='k', linewidth=1.5, zorder=3))
                ax_right.add_patch(mpatches.Rectangle(rect.get_xy(), rect.get_width(), rect.get_height(),
                                                      fill=False, edgecolor='k', linewidth=1.5, zorder=3))


        # Create legend patches for agent types and place as figure legend
        legend_patches = []
        for idx, color in self.desc_color_map.items():
            label = f"{idx}: {self.desc_lookup[idx][:40]}..."
            patch = mpatches.Patch(color=color, label=label)
            legend_patches.append(patch)

        fig.legend(handles=legend_patches,
                   loc='upper center',
                   bbox_to_anchor=(0.5, -0.05),
                   fancybox=True,
                   shadow=True,
                   ncol=1)
        plt.subplots_adjust(bottom=0.25, wspace=0.12)
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

    def step(self, it):
        """Perform a single iteration of the simulation.

        it: iteration number
        do_plots: whether to plot the grid after the step
        """

        # Calculate the new neighbourhood embeddings
        for neighbourhood in self.neighbourhoods.values():
            neighbourhood.step()

        # Step the agents
        happy = 0
        for agent in self.agents:
            agent.step(self)
            if agent.happy:
                happy += 1

        self.happy_counts.append(happy)



    def run(self):
        """Run the full simulation for the configured number of iterations."""
        for it in range(self.max_iters):
            self.step(it)
            print(f"Iteration {it} complete. {self.happy_counts[-1]} happy agents")
            if self.do_plots:
                self.plot_grid(it, show_neighbourhoods=True)


# -------------------------------
# Main Execution
# -------------------------------

if __name__ == "__main__":

    model = SchellingModel(household_descriptions,
                           grid_size=20,
                           num_agents=300,
                           similarity_threshold=0.70,
                           max_iters=50,
                           num_neighbourhoods=8,
                           color_method="pca",
                           do_plots=True,
                           #move_decision="neighbours")
                           move_decision = "neighbourhood")
    model.run()
    model.plot_happiness(return_fig=False)
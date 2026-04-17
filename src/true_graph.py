import numpy as np
from tqdm import tqdm

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import networkx as nx


# Generate data from an underlying graph
class TrueGraph:
    def __init__(self, n, factors, alphabet_size=2):
        """
        Store a factor graph along with mechanisms to generate data.

        Parameters
        ----------
        n : int
            Number of variables.
        factors : list of (list[int], callable)
            Each element is a tuple:
              - scope : list[int]   -- variable indices involved in this factor
              - f     : callable    -- function f(*values) -> float (unnormalized
                                       factor value; need not be a probability).
                                       Values are passed in the order given by scope.
        alphabet_size : int or list[int]
            If int, every variable takes values in {0, ..., alphabet_size-1}.
            If list, alphabet_size[i] is the number of values variable i can take.
        """
        self.n = n
        self.factors = factors  # list of (scope, func)

        # Normalize alphabet_size to a per-variable list
        if isinstance(alphabet_size, int):
            self.alphabet_sizes = [alphabet_size] * n
        else:
            if len(alphabet_size) != n:
                raise ValueError("len(alphabet_size) must equal n")
            self.alphabet_sizes = list(alphabet_size)

        # Precompute: for each variable i, which factors involve it?
        self._var_to_factors = [[] for _ in range(n)]
        for k, (scope, _) in enumerate(factors):
            for i in scope:
                self._var_to_factors[i].append(k)

    def _log_factor_value(self, factor_idx, state):
        """
        Evaluate log of factor k given current state (length-n int array).
        Returns -inf if the factor value is <= 0.
        """
        scope, f = self.factors[factor_idx]
        val = f(*[state[i] for i in scope])
        if val <= 0:
            return -np.inf
        return np.log(val)
 
    def _conditional_log_probs(self, var_idx, state):
        """
        Compute the (unnormalized) log conditional distribution of variable
        var_idx given all other variables fixed to state.
 
        Only factors that involve var_idx need to be evaluated because all
        others are constant with respect to x_{var_idx}.
 
        Returns
        -------
        log_probs : np.ndarray, shape (alphabet_sizes[var_idx],)
            Unnormalized log probabilities for each possible value.
        """
        a = self.alphabet_sizes[var_idx]
        log_probs = np.zeros(a)
        relevant_factors = self._var_to_factors[var_idx]
 
        for v in range(a):
            state[var_idx] = v  # temporarily set
            for k in relevant_factors:
                log_probs[v] += self._log_factor_value(k, state)
 
        return log_probs
 
    def _log_probs_to_probs(self, log_probs):
        """Numerically stable softmax (shift by max before exp)."""
        lp = log_probs - np.max(log_probs)
        p = np.exp(lp)
        p /= p.sum()
        return p

    def factor_value(self, state):
        log_total = 0
        for idx in range(len(self.factors)):
            log_total += self._log_factor_value(idx, state)
        return np.exp(log_total)

    def sample(self, nsamples, n_burnin=1000, thinning=10, seed=None, progress=True):
        """
        Draw samples via Gibbs sampling.
 
        Parameters
        ----------
        nsamples : int
            Number of samples to return.
        n_burnin : int
            Number of Gibbs sweeps to discard before collecting samples.
        thinning : int
            Collect one sample every `thinning` sweeps to reduce
            autocorrelation.
        seed : int or None
 
        Returns
        -------
        np.ndarray, shape (nsamples, n), dtype int
            Each row is one sample; values are in {0, ..., alphabet_size_i - 1}.
        """
        rng = np.random.default_rng(seed)
 
        # Initialise state randomly
        state = np.array(
            [rng.integers(0, self.alphabet_sizes[i]) for i in range(self.n)],
            dtype=int,
        )
 
        def gibbs_sweep():
            for i in range(self.n):
                log_probs = self._conditional_log_probs(i, state)
                probs = self._log_probs_to_probs(log_probs)
                state[i] = rng.choice(self.alphabet_sizes[i], p=probs)
 
        # Burn-in
        for _ in range(n_burnin):
            gibbs_sweep()
 
        # Collect samples
        samples = np.zeros((nsamples, self.n), dtype=int)
        for s in tqdm(range(nsamples), disable = not progress):
            for _ in range(thinning):
                gibbs_sweep()
            samples[s] = state.copy()
 
        return samples

    def edge_set(self):
        edgeset = set()
        for scope, f in self.factors:
            if len(scope) < 2:
                continue
            for i in range(len(scope)):
                for j in range(i+1,len(scope)):
                    pair = (scope[i], scope[j])
                    pair_flip = (scope[j], scope[i])
                    if pair not in edgeset and pair_flip not in edgeset:
                        edgeset.add(pair)
        return edgeset

    def adjacency_list(self):
        adj_list = [[] for _ in range(self.n)]
        edgeset = self.edge_set()
        for i, j in edgeset:
            adj_list[i].append(j)
            adj_list[j].append(i)
        return adj_list
 
    def display_graph(self, var_names=None, factor_labels=None, ax=None,
                      figsize=(8, 6), seed=None):
        """
        Draw the factor graph using networkx and matplotlib.
 
        Variable nodes are drawn as circles; factor nodes as squares.
        Edges connect each factor to the variables in its scope.
 
        Parameters
        ----------
        var_names : list[str] or None
            Labels for variable nodes.  Defaults to ["x0", "x1", ...].
        factor_labels : list[str] or None
            Labels for factor nodes.  Defaults to ["f0", "f1", ...].
        ax : matplotlib Axes or None
            If None, a new figure is created.
        figsize : tuple
            Figure size (only used if ax is None).
        seed : int or None
            Layout seed for reproducibility.
 
        Returns
        -------
        fig, ax
        """
        if var_names is None:
            var_names = [f"x{i}" for i in range(self.n)]
        if factor_labels is None:
            factor_labels = [f"f{k}" for k in range(len(self.factors))]
 
        # Build bipartite graph
        G = nx.Graph()
 
        var_nodes = [f"var_{i}" for i in range(self.n)]
        fac_nodes = [f"fac_{k}" for k in range(len(self.factors))]
 
        G.add_nodes_from(var_nodes, bipartite=0)
        G.add_nodes_from(fac_nodes, bipartite=1)
 
        for k, (scope, _) in enumerate(self.factors):
            for i in scope:
                G.add_edge(f"fac_{k}", f"var_{i}")
 
        # Layout
        pos = nx.spring_layout(G, seed=seed)
 
        if ax is None:
            fig, ax = plt.subplots(figsize=figsize)
        else:
            fig = ax.get_figure()
 
        ax.set_aspect("equal")
        ax.axis("off")
 
        # Draw variable nodes (circles)
        nx.draw_networkx_nodes(
            G, pos,
            nodelist=var_nodes,
            node_shape="o",
            node_color="#4C9BE8",
            node_size=900,
            ax=ax,
        )
        # Draw factor nodes (squares)
        nx.draw_networkx_nodes(
            G, pos,
            nodelist=fac_nodes,
            node_shape="s",
            node_color="#F4A442",
            node_size=700,
            ax=ax,
        )
        # Edges
        nx.draw_networkx_edges(G, pos, width=1.8, alpha=0.7, edge_color="#555555", ax=ax)
 
        # Labels
        var_label_map = {f"var_{i}": var_names[i] for i in range(self.n)}
        fac_label_map = {f"fac_{k}": factor_labels[k] for k in range(len(self.factors))}
        label_map = {**var_label_map, **fac_label_map}
 
        nx.draw_networkx_labels(
            G, pos, labels=label_map,
            font_size=10, font_color="white", font_weight="bold",
            ax=ax,
        )
 
        # Legend
        legend_handles = [
            mpatches.Patch(color="#4C9BE8", label="Variable node"),
            mpatches.Patch(color="#F4A442", label="Factor node"),
        ]
        ax.legend(handles=legend_handles, loc="best", framealpha=0.9)
        ax.set_title("Factor Graph", fontsize=13, fontweight="bold")
 
        fig.tight_layout()
        return fig, ax
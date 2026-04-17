import numpy as np
import networkx as nx

from true_graph import TrueGraph


def random_potential_table(alphabet_size, rng, size=2, strength=1.0, symmetric=False):
    log_table = rng.normal(0.0, strength, size=tuple([alphabet_size]*size))
    # log_table = rng.normal(0.0, strength, size=(alphabet_size, alphabet_size))
    if symmetric and size == 2:
        log_table = (log_table + log_table.T) / 2.0
    table = np.exp(log_table)
    return table

def make_factor(tbl):
    def f(*args):
        return tbl[tuple(args)]
    return f

def generate_random_tree(n, alphabet_size, rng, strength=1.0, symmetric=False):
    sd = int(rng.integers(10000))
    rand_tree = nx.random_tree(n, seed=sd)
    edge_list = list(rand_tree.edges)
    potential_dict = {}
    for edge in edge_list:
        potential_dict[edge] = random_potential_table(alphabet_size, rng, size=2)
    factor_list = []
    for endpts, tbl in potential_dict.items():
        f = make_factor(tbl)
        factor_list.append((endpts, f))
    return TrueGraph(n, factor_list, alphabet_size)

def is_subset(scope, existing_scopes):
    s = set(scope)
    for e in existing_scopes:
        if s.issubset(e):
            return True
    return False

def generate_random_graph(n, alphabet_size, rng, n_factors=None, max_factor_size=3, strength=1.0):
    if n_factors is None:
        n_factors = rng.poisson(lam=n)
    factor_list = []
    scopes = []
    attempts = 0
    max_attempts = 10 * n_factors
    while len(scopes) < n_factors and attempts < max_attempts:
        attempts += 1
        # sample factor size (at least 2)
        size = int(rng.integers(2, max_factor_size + 1))
        # sample variables without replacement
        scope = tuple(sorted(rng.choice(n, size=size, replace=False)))
        # enforce: no subset condition
        if is_subset(scope, scopes):
            continue
        # also avoid exact duplicates
        if scope in scopes:
            continue
        scopes.append(set(scope))
        # generate potential table
        tbl = random_potential_table(alphabet_size, rng, size=size, strength=strength)
        f = make_factor(tbl)
        factor_list.append((scope, f))
    if len(scopes) < n_factors:
        print(f"Warning: only generated {len(scopes)} factors (target was {n_factors})")
    return TrueGraph(n, factor_list, alphabet_size)
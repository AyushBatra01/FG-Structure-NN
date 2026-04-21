import numpy as np
import itertools
import matplotlib.pyplot as plt

from true_graph import TrueGraph


def get_scopes(obj, mask_thresh=0.5, weight_thresh=0.5):
    """Extract factor scopes from either TrueGraph or learner."""
    # TrueGraph
    if hasattr(obj, "factors"):
        return [set(f[0]) for f in obj.factors]

    # Neural network learner
    elif hasattr(obj, "extract_graph"):
        recovered = obj.extract_graph(mask_thresh=mask_thresh,
                                      weight_thresh=weight_thresh)
        return [set(v) for v in recovered["factor_scopes"].values()]

    else:
        raise ValueError("Unknown object type")


def get_log_probs(obj, combinations):
    """Return log probabilities / scores."""
    # TrueGraph
    if hasattr(obj, "factor_value"):
        vals = np.array([obj.factor_value(x) for x in combinations])
        vals = vals / np.sum(vals)
        return np.log(vals + 1e-12)

    # Neural network learner
    elif hasattr(obj, "predict"):
        return obj.predict(combinations)

    else:
        raise ValueError("Unknown object type")


def get_probs(obj, combinations):
    logp = get_log_probs(obj, combinations)
    logp = logp - np.max(logp)
    p = np.exp(logp)
    return p / np.sum(p)


def get_combinations(obj, max_states=100000):
    if type(obj) == TrueGraph:
        alph_size = obj.alphabet_sizes[0]
        n_vars = obj.n
    elif type(obj) == FactorGraphLearner:
        alph_size = obj.network.alphabet_size
        n_vars = obj.network.n_vars
    else:
        raise ValueError("incorrect object type")
    total = alph_size ** n_vars
    if total > max_states:
        raise ValueError(f"Too many configurations: {total} > {max_states}")
    return np.array(list(itertools.product(range(alph_size), repeat=n_vars)))


def scopes_to_edges(scopes):
    edges = set()
    for s in scopes:
        for i, j in itertools.combinations(s, 2):
            edges.add(tuple(sorted((i, j))))
    return edges


def compare_graph(obj1, obj2):
    scopes1 = get_scopes(obj1)
    scopes2 = get_scopes(obj2)

    print("=== GRAPH 1 FACTORS ===")
    for s in scopes1:
        print(sorted(s))

    print("\n=== GRAPH 2 FACTORS ===")
    for s in scopes2:
        print(sorted(s))


def kl_divergence(obj1, obj2):
    eps = 1e-8
    combinations = get_combinations(obj1)

    p = get_probs(obj1, combinations)
    q = get_probs(obj2, combinations)

    p = np.clip(p, eps, 1.0)
    q = np.clip(q, eps, 1.0)

    return np.sum(p * np.log(p / q))


def factor_metrics(obj1, obj2):
    scopes1 = get_scopes(obj1)
    scopes2 = get_scopes(obj2)

    set1 = set(map(frozenset, scopes1))
    set2 = set(map(frozenset, scopes2))

    tp = len(set1 & set2)
    fp = len(set2 - set1)
    fn = len(set1 - set2)

    return {
        "precision": tp / (tp + fp + 1e-8),
        "recall": tp / (tp + fn + 1e-8),
        "tp": tp,
        "fp": fp,
        "fn": fn,
    }


def edge_metrics(obj1, obj2):
    edges1 = scopes_to_edges(get_scopes(obj1))
    edges2 = scopes_to_edges(get_scopes(obj2))

    tp = len(edges1 & edges2)
    fp = len(edges2 - edges1)
    fn = len(edges1 - edges2)

    return {
        "precision": tp / (tp + fp + 1e-8),
        "recall": tp / (tp + fn + 1e-8),
        "tp": tp,
        "fp": fp,
        "fn": fn,
    }


def structural_hamming_distance(obj1, obj2):
    edges1 = scopes_to_edges(get_scopes(obj1))
    edges2 = scopes_to_edges(get_scopes(obj2))
    return len(edges1.symmetric_difference(edges2))


def prob_diff_hist(obj1, obj2):
    combinations = get_combinations(obj1)
    p = get_probs(obj1, combinations)
    q = get_probs(obj2, combinations)

    plt.hist(p - q, bins=30)
    plt.title("Probability Difference Histogram")
    plt.show()


def kl_divergence_mle(graph, samples, alpha=0.1):
    eps = 1e-8
    alph_size = graph.alphabet_sizes[0]
    n_vars = graph.n
    total_configs = alph_size ** n_vars

    if total_configs > 100000:
        raise ValueError(f"Too many configurations: {total_configs} > 100000")

    combinations = get_combinations(graph)

    # true
    true = np.array([graph.factor_value(x) for x in combinations])
    true /= np.sum(true)
    true = np.clip(true, eps, 1.0)

    # empirical
    powers = alph_size ** np.arange(n_vars - 1, -1, -1)
    indices = (samples * powers).sum(axis=1)

    counts = np.bincount(indices, minlength=total_configs).astype(float)
    counts += alpha

    pred = counts / counts.sum()
    pred = np.clip(pred, eps, 1.0)

    return np.sum(true * np.log(true / pred))


def kl_mle_optimal(graph, samples):
    best = 999999
    for log_a in np.arange(-4,4.01,0.5):
        a = 10**log_a
        best = min(best, kl_divergence_mle(graph, samples, a))
    return best



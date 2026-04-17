import numpy as np
from true_graph import TrueGraph


# Copied from Chow-Liu project
def kruskal_max_spanning_tree(weights):
    """Compute a maximum spanning tree using Kruskal's algorithm.

    Parameters
    ----------
    weights : np.ndarray of shape (n, n)
        Symmetric edge-weight matrix (here: MI estimates).

    Returns
    -------
    list[tuple[int, int]]
        `n-1` edges of a maximum spanning tree.
    """
    n = weights.shape[0]
    edges = [(weights[i, j], i, j) for i in range(n) for j in range(i + 1, n)]
    edges.sort(key=lambda t: t[0], reverse=True)

    parent = list(range(n))
    rank = [0] * n

    def find(x):
        """Find set representative with path compression."""
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        """Union-by-rank; return True iff a merge happened."""
        ra, rb = find(a), find(b)
        if ra == rb:
            return False
        if rank[ra] < rank[rb]:
            parent[ra] = rb
        elif rank[ra] > rank[rb]:
            parent[rb] = ra
        else:
            parent[rb] = ra
            rank[ra] += 1
        return True

    tree = []
    for _, u, v in edges:
        if union(u, v):
            tree.append((u, v))
            if len(tree) == n - 1:
                break
    return tree



def empirical_pairwise_mi(samples, alphabet_size, eps=1e-12):
    """Estimate pairwise mutual information from samples.

    Parameters
    ----------
    samples : np.ndarray of shape (m, n)
        Binary data matrix with `m` samples and `n` variables.
    eps : float, default=1e-12
        Small constant for numerical stability in log-ratio terms.

    Returns
    -------
    np.ndarray of shape (n, n)
        Symmetric MI estimate matrix with zeros on the diagonal.

    Notes
    -----
    Implement using empirical marginals/joints:
    - `p_i(a)`, `p_j(b)`, `p_ij(a,b)`
    - `MI(i,j) = sum_{a,b} p_ij(a,b) log(p_ij(a,b)/(p_i(a)p_j(b)))`
    """
    m, n = samples.shape
    mi = np.zeros((n,n))
    # off diagonals (mutual information)
    for x in range(n):
        for y in range(x+1,n):
            mi_xy = 0
            for i in range(alphabet_size):
                for j in range(alphabet_size):
                    pxy = np.mean((samples[:,x] == i) & (samples[:,y] == j))
                    px = np.mean((samples[:,x] == i))
                    py = np.mean((samples[:,y] == j))
                    mi_xy += pxy * np.log((pxy + eps) / (px * py + eps))
            mi[x,y] = mi_xy
            mi[y,x] = mi_xy
    return mi

def chow_liu(samples, alphabet_size, alpha=1):
    nsamp, nvar = samples.shape
    # Step 1: MI + MST
    mi = empirical_pairwise_mi(samples, alphabet_size)
    mst = kruskal_max_spanning_tree(mi)
    # Step 2: build adjacency
    adj = {i: [] for i in range(nvar)}
    for i, j in mst:
        adj[i].append(j)
        adj[j].append(i)
    # Step 3: pick root and orient tree
    root = 0
    parent = {root: None}
    order = [root]
    # BFS to orient tree
    queue = [root]
    while queue:
        u = queue.pop(0)
        for v in adj[u]:
            if v not in parent:
                parent[v] = u
                queue.append(v)
                order.append(v)
    factors = []
    # Step 4: root marginal
    p_root = np.zeros(alphabet_size)
    for a in range(alphabet_size):
        count = np.sum(samples[:, root] == a)
        p_root[a] = (count + alpha) / (nsamp + alpha * alphabet_size)
    factors.append(((root,), make_factor(p_root)))
    # Step 5: conditional tables
    for child in order[1:]:
        par = parent[child]
        joint = np.zeros((alphabet_size, alphabet_size))
        for a in range(alphabet_size):
            for b in range(alphabet_size):
                count = np.sum((samples[:, par] == a) & (samples[:, child] == b))
                joint[a, b] = count + alpha
        # normalize rows
        cond = joint / joint.sum(axis=1, keepdims=True)
        factors.append(((par, child), make_factor(cond)))
    return TrueGraph(nvar, factors, alphabet_size)


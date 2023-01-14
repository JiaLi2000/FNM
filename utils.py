import networkx as nx
import numpy as np
import scipy.sparse as sp


def get_ncut(G: nx.graph, phi: np.array) -> float:
    n, W = len(phi), nx.adjacency_matrix(G, sorted(G.nodes))
    k = phi.max() + 1
    Y_0 = sp.coo_matrix(([1] * n, (range(n), phi)), dtype=int).tocsr()
    D = W.dot(Y_0).toarray()  # d_ij = sum_v of W_{iv}, where v is in cluster j
    degrees = D.sum(axis=1)
    volumes = np.array([degrees[phi == j].sum() for j in range(k)])
    cuts = volumes - (D * Y_0.toarray()).sum(axis=0)  # the right item just is association of clusters.
    f = (cuts / volumes).sum()
    return f


def get_fair_prop_vio(phi: np.array, C: np.array, alpha: np.array, beta: np.array) -> (float, float):
    n, m = C.shape
    k = phi.max() + 1
    Y = sp.coo_matrix(([1] * n, (range(n), phi)), dtype=int).tocsr()
    A = np.array([alpha] * k).T
    B = np.array([beta] * k).T
    F = C.T @ Y
    sizes = F.sum(axis=0)
    Ratio = F / sizes
    upper_diff = A - Ratio
    lower_diff = Ratio - B
    total = list(-upper_diff[upper_diff < 0]) + list(-lower_diff[lower_diff < 0])
    if len(total) == 0:
        avg, maxi = 0, 0
    else:
        avg, maxi = np.average(total), np.max(total)
    return avg, maxi


def get_balance_ICML(phi: np.array, C: np.array) -> (float, float):
    """
        returns balance of the dataset when phi == np.zeros(len(datasets))
    """
    n, m = C.shape
    k = phi.max() + 1
    Y = sp.coo_matrix(([1] * n, (range(n), phi)), dtype=int).tocsr()
    F = C.T @ Y
    balances = np.zeros(k)
    color_indexs = set(range(m))
    for j in range(k):
        if np.all(F[:, j]):
            ratios = [F[s, j] / F[s_prime, j] for s in color_indexs for s_prime in (color_indexs - {s})]
            balances[j] = min(ratios)
    return np.average(balances), np.min(balances)


def get_balance_NIPS(phi: np.array, C: np.array) -> (float, float):
    n, m = C.shape
    k = phi.max() + 1
    Y = sp.coo_matrix(([1] * n, (range(n), phi)), dtype=int).tocsr()
    F = C.T @ Y
    Ratio_clusters = F / F.sum(axis=0)
    Ratio_dataset = C.sum(axis=0) / n
    balances = np.zeros(k)
    for j in range(k):
        if np.all(Ratio_clusters[:, j]):
            r1 = [Ratio_clusters[i, j] / Ratio_dataset[i] for i in range(m)]
            r2 = [Ratio_dataset[i] / Ratio_clusters[i, j] for i in range(m)]
            balances[j] = min(r1 + r2)
    return np.average(balances), np.min(balances)

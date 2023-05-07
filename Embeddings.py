import copy
import time
import numpy as np
import pandas as pd
import scipy.sparse as sp
import scipy.linalg as sl
import networkx as nx
import karateclub as kc
from scipy.sparse.linalg import eigsh
from typing import Callable
from baseline.fairwalk import FairWalk


def spectral(G: nx.graph, k: int) -> (np.array, float, float):
    """
    Compute the normalized spectral embeddings H.
    :param G: nx.Graph, an undirected graph.
    :param k: int, the dimensions of embedding, also the number of clusters .
    :return:
            np.array, embeddings matrix H, sorted by node id ascending from 0 to n-1
            float, continuous ncut of H, conti_ncut(H) := Tr(H'LH)
            float, time
    """
    t1 = time.perf_counter()
    L = nx.laplacian_matrix(G, sorted(G.nodes)).astype(float)  # L = D - W
    Dsqinv = sp.diags(L.diagonal()).power(-0.5)
    L_sys = Dsqinv @ L @ Dsqinv
    eigenvalues, T = eigsh(L_sys, k=k, which='SM')
    embeddings = Dsqinv @ T
    t2 = time.perf_counter()
    return embeddings, eigenvalues.sum(), t2 - t1


def deepwalk(G: nx.graph, k: int, seed: int) -> (np.array, float, float):
    """
    Compute embeddings matrix with shape (n, k).
    :param G: nx.Graph, an undirected graph.
    :param k: int, the dimensions of embedding, also the number of clusters.
    :param seed: int, the seed for random walk
    :return:
            np.array, embeddings matrix, sorted by node id ascending from 0 to n-1
            float, continuous ncut of embeddings, conti_ncut(embeddings) := Tr(embeddings' L embeddings)
            float, time
    """
    G_copy = copy.deepcopy(G)  # the implementation in karateclub involves in-place computations
    t1 = time.perf_counter()
    model = kc.DeepWalk(walk_number=10, walk_length=80, dimensions=k, workers=1,  # set workers=1 for reproduction
                        window_size=5, epochs=1, learning_rate=0.05, min_count=1, seed=seed)
    model.fit(G_copy)
    embeddings = model.get_embedding()
    L = nx.laplacian_matrix(G, sorted(G.nodes))  # L = D - A
    conti_ncut = (embeddings * (L @ embeddings)).sum()
    t2 = time.perf_counter()
    return embeddings, conti_ncut, t2 - t1


def node2vec(G: nx.graph, k: int, seed: int) -> (np.array, float, float):
    """
    Compute embeddings matrix with shape (n, k).
    :param G: nx.Graph, an undirected graph.
    :param k: int, the dimensions of embedding, also the number of clusters.
    :param seed: int, the seed for random walk
    :return:
            np.array, embeddings matrix, sorted by node id ascending from 0 to n-1
            float, continuous ncut of embeddings, conti_ncut(embeddings) := Tr(embeddings' L embeddings)
            float, time
    """
    G_copy = copy.deepcopy(G)  # the implementation in karateclub involves in-place computations
    t1 = time.perf_counter()
    model = kc.Node2Vec(walk_number=10, walk_length=80, p=1.0, q=1.0, dimensions=k, workers=1,
                        # set workers=1 for reproduction
                        window_size=5, epochs=1, learning_rate=0.05, min_count=1, seed=seed)
    model.fit(G_copy)
    embeddings = model.get_embedding()
    L = nx.laplacian_matrix(G, sorted(G.nodes))  # L = D - A
    conti_ncut = (embeddings * (L @ embeddings)).sum()
    t2 = time.perf_counter()
    return embeddings, conti_ncut, t2 - t1


# code referred to https://github.com/urielsinger/fairwalk
def fair_walk(G: nx.graph, k: int, color: np.array, seed: int) -> (np.array, float, float):
    """
    Compute embeddings matrix with shape (n, k).
    :param G: nx.Graph, an undirected graph.
    :param k: int, the dimensions of embedding, also the number of clusters.
    :param color: 1xn np.array, mapping node id to its color.
    :param seed: int, the seed for random walk
    :return:
            np.array, embeddings matrix, sorted by node id ascending from 0 to n-1
            float, continuous ncut of embeddings, conti_ncut(embeddings) := Tr(embeddings' L embeddings)
            float, time
    """
    G_copy = copy.deepcopy(G)  # the implementation involves in-place computations
    t1 = time.perf_counter()
    node2group = {node: group for node, group in zip(sorted(G_copy.nodes()), color)}
    nx.set_node_attributes(G_copy, node2group, 'group')
    model = FairWalk(graph=G_copy, num_walks=10, walk_length=80, p=1.0, q=1.0, dimensions=k, workers=1, quiet=True,
                     seed=seed)  # set workers=1 for reproduction
    model = model.fit(window=5, min_count=1, seed=seed, epochs=1, alpha=0.05)
    embeddings = np.array([model.wv[str(n)] for n in range(G_copy.number_of_nodes())])
    L = nx.laplacian_matrix(G, sorted(G.nodes))  # L = D - A
    conti_ncut = (embeddings * (L @ embeddings)).sum()
    t2 = time.perf_counter()
    return embeddings, conti_ncut, t2 - t1


# code referred to https://github.com/matthklein/fair_spectral_clustering/blob/master/Fair_SC_normalized.m
def fair_equality(G: nx.graph, k: int, color: np.array) -> (np.array, float, float):
    """
    Read the clustering produced by matlab source code "Fair_SC_normalized.m" and compute the conti_ncut.
    :param G: nx.Graph, an undirected graph.
    :param k: int, the dimensions of embedding, also the number of clusters.
    :param color: 1xn np.array, mapping node id to its color.
    :return:
            np.array, embeddings matrix, sorted by node id ascending from 0 to n-1
            float, continuous ncut of embeddings, conti_ncut(embeddings) := Tr(embeddings' L embeddings)
            float, time
    """
    n2name = {1412: 'facebookNet', 21742: 'german', 2576: 'dblp', 27806: 'lastfm', 57156: 'SBM_1000'}
    n, m = G.number_of_edges(), color.max() + 1
    embeddings_file = f'./results/embeddings/fair_equality_embeddings_{n2name[n]}_k{str(k)}_m{str(m)}.csv'
    t1_file = f'./results/embeddings/fair_equality_t1_{n2name[n]}_k{str(k)}_m{str(m)}.csv'
    embeddings = pd.read_csv(embeddings_file, header=None).values
    t = pd.read_csv(t1_file, header=None)[0][0]
    t1 = time.perf_counter()
    L = nx.laplacian_matrix(G, sorted(G.nodes))  # L = D - A
    conti_ncut = (embeddings * (L @ embeddings)).sum()
    t2 = time.perf_counter()
    return embeddings, conti_ncut, t + t2 - t1


# code referred to https://github.com/jiiwang/scalable_fair_spectral_clustering
def fair_equality_s(G: nx.graph, k: int, color: np.array) -> (np.array, float, float):
    """
    Read the clustering produced by matlab source code "alg3.m" and compute the conti_ncut.
    :param G: nx.Graph, an undirected graph.
    :param k: int, the dimensions of embedding, also the number of clusters.
    :param color: 1xn np.array, mapping node id to its color.
    :return:
            np.array, embeddings matrix, sorted by node id ascending from 0 to n-1
            float, continuous ncut of embeddings, conti_ncut(embeddings) := Tr(embeddings' L embeddings)
            float, time
    """
    n2name = {1412: 'facebookNet', 21742: 'german', 2576: 'dblp', 27806: 'lastfm', 57156: 'SBM_1000',
              92752: 'deezer', 136196: 'credit_education', 10792894: 'pokec_age', 22301964: 'pokec_sex'}
    n, m = G.number_of_edges(), color.max() + 1
    embeddings_file = f'./results/embeddings/fair_equality_s_embeddings_{n2name[n]}_k{str(k)}_m{str(m)}.csv'
    t1_file = f'./results/embeddings/fair_equality_s_t1_{n2name[n]}_k{str(k)}_m{str(m)}.csv'
    embeddings = pd.read_csv(embeddings_file, header=None).values
    t = pd.read_csv(t1_file, header=None)[0][0]
    t1 = time.perf_counter()
    L = nx.laplacian_matrix(G, sorted(G.nodes))  # L = D - A
    conti_ncut = (embeddings * (L @ embeddings)).sum()
    t2 = time.perf_counter()
    return embeddings, conti_ncut, t + t2 - t1


def fair_proportion(G: nx.graph, k: int, C: np.array, alpha: np.array, beta: np.array,
                    xtol=1e-6, gtol=1e-6, ftol=1e-9, pho=2, mu=1e-3, omxitr=100, mxitr=2000) -> (
        np.array, float, float, float):
    """
    Compute the normalized spectral embeddings with alpha,beta fairness constraints.
    :param G: nx.Graph, a undirected graph.
    :param k: int, the dimension of embeddings, also the cluster number.
    :param C: np.array, n x m color indicator matrix, C_ij = 1 means node i belongs to color j
    :param alpha: np.array, fairness upper bound
    :param beta: np.array, fairness lower bound
    :param xtol: float, OptStiefelGBB stop control for ||X_k - X_{k-1}||
    :param gtol: float, OptStiefelGBB stop control for the projected gradient
    :param ftol: float, OptStiefelGBB stop control for |F_k - F_{k-1}|/(1+|F_{k-1}|)
    :param pho: float, updating rate of penalty
    :param mu: float, penalty for fairness violation
    :param omxitr: max iteration number of fair_proportion
    :param mxitr: max iteration number of OptStiefelGBB
    :return:
            np.array, embeddings matrix, sorted by node id ascending from 0 to n-1
            float, continuous ncut of embeddings, conti_ncut(embeddings) := Tr(embeddings' L embeddings)
            float, time
            float, objective value of Lagrange function
    """
    t1 = time.perf_counter()
    n, m = C.shape
    # initialize
    X = np.eye(n, k)
    A, B = np.array([alpha] * n), np.array([beta] * n)
    L = nx.laplacian_matrix(G, sorted(G.nodes)).astype(float)  # L = D - W
    Dsqinv = sp.diags(L.diagonal()).power(-0.5)
    L_sys = Dsqinv @ L @ Dsqinv
    Lower = (C - B).T @ Dsqinv
    Upper = (A - C).T @ Dsqinv
    P = np.hstack((Lower @ X, Upper @ X))
    Lmb = np.zeros((m, 2 * k))
    Out = {'itrsub': 0, 'nfe': 0}
    itr = 1
    # Solve i-th subproblem by OptStiefelGBB using the solution of (i-1)-th subproblem as the initial solution.
    for itr in range(1, omxitr):
        X, outs = OptStiefelGBB(X=X, fun=fairCut, L=L_sys, Lmb=Lmb, mu=mu, Lower=Lower, Upper=Upper, xtol=xtol,
                                gtol=gtol, ftol=ftol, mxitr=mxitr)
        Out['itrsub'] += outs['itr']
        Out['nfe'] += outs['nfe']
        Out['f_Larg'] = outs['fval']
        P = np.hstack((Lower @ X, Upper @ X))
        v_k = sl.norm(np.minimum(P, 0), 'fro')  # fairness constraints violation
        print(outs)
        print(f'{itr}-subproblem: , mu:{mu}, violation:{v_k}, fLarg:{outs["fval"]}')
        if v_k <= 0:
            Out['conti_ncut'] = (X * (L_sys @ X)).sum()  # conti_ncut = Tr(H'LH) = Tr(T'L_sysT)
            Out['msg'] = 'converge'
            embeddings = Dsqinv.dot(X)  # H = D^(-1/2) @ T
            break
        mu *= pho
        Lmb = np.maximum(Lmb - mu * P, 0)
    if itr >= omxitr:
        embeddings = Dsqinv.dot(X)
        Out['conti_ncut'] = (X * (L_sys @ X)).sum()
        Out['msg'] = 'exceed max iteration'
    t2 = time.perf_counter()
    Out['itr'] = itr
    Out['time'] = t2 - t1
    print(Out)
    return embeddings, Out['conti_ncut'], Out['time'], Out['f_Larg']


# auxiliary function
# Python implementation of OptStiefelGBB for k << n, referred to Matlab source code from https://github.com/optsuite/OptM
def OptStiefelGBB(X: np.array,
                  fun: Callable[[np.array, sp.csr_matrix, np.array, float, np.array, np.array], tuple[float, np.array]],
                  L: sp.csr_matrix, Lmb: np.array, Lower: np.array, Upper: np.array, mu: float, xtol=1e-6, gtol=1e-6,
                  ftol=1e-9, TAU=1e-3, mxitr=2000) -> (np.array, dict):
    """
    :param X: initial orthogonal solution
    :param fun: objective function AL which returns real f(X) and matrix f'(X)
    :param L: normalized laplacian matrix of G used in AL
    :param Lower, Upper: the constant coefficient matrix of two fairness constraints
    :param Lmb: multipliers matrix used in AL
    :param mu: penalty factor of AL
    :param gtol: stop control for the projected gradient
    :param TAU: initial step length for updating X_k
    :param mxitr: max iteration number
    :return: fair embeddings X, Iteration Log
    """
    t1 = time.perf_counter()
    # initial function value and gradient
    out = {'nfe': 1}  # num of calling fun()
    n, k = X.shape
    crit, nt = np.ones((mxitr + 1, 3)), 5
    F, G = fun(X, L, Lmb, mu, Lower, Upper)
    GX = G.T @ X
    # prepare variables for SMW formula to get inv( I + 0.5*tau*VU ) * V'*X
    U, V = np.hstack((G, X)), np.hstack((X, -G))
    VU, VX = V.T @ U, V.T @ X
    # compute gradient of lagrangian function
    dtX = G - X @ GX
    nrmG = sl.norm(dtX, 'fro')
    # step length
    Q, Cval, tau = 1, F, TAU
    eta, gamma, rhols = 0.1, 0.85, 1e-4
    # main iteration
    itr = 1
    for itr in range(1, mxitr + 1):
        # X, F(X), F'(X), L'(X) of previous iteration
        XP, FP, GP, dtXP = X, F, G, dtX
        # line search: scale step size
        nls = 1  # The number of line search attempts in each iteration
        deriv = rhols * (nrmG ** 2)
        while True:
            # calculate G, F
            aa = sl.solve(np.eye(2 * k) + (0.5 * tau) * VU, VX)
            X = XP - U @ (tau * aa)
            if sl.norm(X.T @ X - np.eye(k), 'fro') > 1e-13:
                X = myQR(X, k)[0]
            F, G = fun(X, L, Lmb, mu, Lower, Upper)
            out['nfe'] += 1
            if (F <= Cval - tau * deriv) or (nls >= 5):
                break
            tau *= eta
            nls += 1

        GX = G.T @ X
        U, V = np.hstack((G, X)), np.hstack((X, -G))
        VU, VX = V.T @ U, V.T @ X
        dtX = G - X @ GX
        nrmG = sl.norm(dtX, 'fro')
        # update tau by BB-steps
        S = X - XP
        tau = TAU
        Y = dtX - dtXP
        SY = np.abs(np.real((np.conj(S) * Y).sum()))
        if itr % 2 == 0:
            tau = (sl.norm(S, 'fro') ** 2) / SY
        else:
            tau = SY / (sl.norm(Y, 'fro') ** 2)
        tau = max(min(tau, 1e20), 1e-20)
        # convergence criterion
        XDiff = sl.norm(S, 'fro') / np.sqrt(n)
        FDiff = np.abs(FP - F) / (np.abs(FP) + 1)
        crit[itr, :] = np.array([nrmG, XDiff, FDiff])
        mcrit = np.mean(crit[itr - min(nt, itr) + 1: itr + 1, :], axis=0)
        if (XDiff < xtol and FDiff < ftol) or (nrmG < gtol) or (np.all(mcrit[1:3] < 10 * np.array([xtol, ftol]))):
            out['msg'] = 'converge'
            break
        Qp = Q
        Q = gamma * Qp + 1
        Cval = (gamma * Qp * Cval + F) / Q
    if itr >= mxitr:
        out['msg'] = 'exceed max iteration'
    # how similar does the current solution to identity matrix
    out['feasi'] = sl.norm(X.T @ X - np.eye(k), 'fro')
    if out['feasi'] > 1e-13:
        X = myQR(X, k)[0]
        F, G = fun(X, L, Lmb, mu, Lower, Upper)
        out['nfe'] += 1
        out['feasi'] = sl.norm(X.T @ X - np.eye(k), 'fro')
    t2 = time.perf_counter()
    out['nrmG'] = nrmG
    out['fval'] = F
    out['itr'] = itr
    out['time'] = t2 - t1
    return X, out


def myQR(XX: np.array, k: int) -> (np.array, np.array):
    Q, RR = sl.qr(XX, mode='economic')
    diagRR = np.sign(np.diag(RR))
    if np.count_nonzero(diagRR < 0) > 0:
        Q = Q @ sp.spdiags(diagRR, 0, k, k).toarray()
    return Q, RR


def fairCut(X: np.array, L: sp.csr_matrix, Lmb: np.array, mu: float, Lower: np.array, Upper: np.array) -> (
        float, np.array):
    '''
    Compute the objective value and gradient of Augmented Lagrange function at X
    :param X: np.array, the embeddings matrix at a certain iteration OptStiefelGBB
    :param L: sp.csr_matrix, L=D^{-0.5}(D-W)D^{-0.5} for ncut, L=(D-W) for ratiocut
    :param Lmb: np.array, multiplier matrix at a certain iteration of fair_proportion
    :param mu: float, penalty at a certain iteration of fair_proportion
    :param Lower: np.array, the constant coefficient matrix on the left of P
    :param Upper: np.array, the constant coefficient matrix on the right of P
    :return:
        float, objective value of Augmented Lagrange function at X
        np.array, gradient of Augmented Lagrange function at X
    '''
    # original F, G
    GCut = 2 * L.dot(X)
    FCut = 0.5 * (X * GCut).sum()
    # AL F
    P = np.hstack((Lower @ X, Upper @ X))
    Axlmb = P - Lmb / mu  # m * 2k
    Idx = (Axlmb <= 0)
    F = FCut + np.sum(-Lmb[Idx] * P[Idx] + 0.5 * mu * (P[Idx] ** 2)) - (0.5 / mu) * np.sum(
        Lmb[np.logical_not(Idx)] ** 2)
    # AL G
    Axlmb[np.logical_not(Idx)] = 0  # branch 2 of rho
    roundP = mu * Axlmb
    n, k = X.shape
    roundX = Lower.T @ roundP[:, range(0, k)] + Upper.T @ roundP[:, range(k, 2 * k)]
    G = GCut + roundX
    return F, G

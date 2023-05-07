import networkx as nx
import scipy.sparse as sp
import gurobipy as gp
from gurobipy import GRB
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import euclidean_distances as ecdist
import sys
import utils
import time
import numpy as np
import pandas as pd
from baseline.cplex_fair_assignment_lp_solver import fair_partial_assignment
from baseline.util.clusteringutil import vanilla_clustering


class Point:
    def __init__(self, idx: int, color: int, dists: list[float], membership: int):
        self.idx = idx
        self.color = color
        self.dists = dists
        self.membership = membership


def kmeans(X: np.array, G: nx.Graph, seed: int) -> (np.array, float, float):
    """
    Round embeddings H by kmeans.
    :param X: np.array, N x k embeddings matrix of G
    :param G: nx.Graph, a undirected graph.
    :param seed: int, random seed for select the first center
    :return:
            np.array,  1xN vector mapping node to its cluster membership.
            float, time
    """
    t1 = time.perf_counter()
    n, k = X.shape
    clustering = KMeans(n_clusters=k, init='k-means++', max_iter=200, random_state=seed, n_init=100)
    clustering.fit(X)
    phi = clustering.labels_
    ncut = utils.get_ncut(G, phi)
    t2 = time.perf_counter()
    return phi, ncut, t2 - t1


def reassigned_kmeans(X: np.array, G: nx.Graph, C: np.array, alpha: np.array, beta: np.array, seed: int) \
        -> (np.array, float, float, int):
    """
    Reassign the clustering of Kmeans on embeddings greedily towards F_fair to get a strictly fair one.
    :param X: np.array, N x k embeddings matrix of G
    :param G: nx.Graph, a undirected graph.
    :param C: np.array, n x m color indicator matrix, C_ij = 1 means node i belongs to color j
    :param alpha: np.array, fairness upper bound
    :param beta: np.array, fairness lower bound
    :param seed: int, random seed for select the first center
    :return:
            np.array,  1xN vector mapping node to its cluster membership.
            float, time
            int, #reassignment
    """
    t1 = time.perf_counter()
    n, k = X.shape
    clustering = KMeans(n_clusters=k, init='k-means++', max_iter=200, random_state=seed, n_init=100)
    clustering.fit(X)
    phi, S = clustering.labels_, clustering.cluster_centers_
    D_all = np.round(ecdist(X, S),
                     10)  # nxk, d_ij = euclidean_distance(point_i,center_j) , d_ij has the same clustering with d_ij^2
    phi_fair, num_of_reassignment = reassign_by_center(D_all, k, phi, C, alpha, beta)
    ncut = utils.get_ncut(G, phi_fair)
    t2 = time.perf_counter()
    return phi_fair, ncut, t2 - t1, num_of_reassignment


def reassigned_fair_kmeans(X: np.array, G: nx.Graph, C: np.array, alpha: np.array, beta: np.array) \
        -> (np.array, float, float, int):
    """
    Reassign the cluster
    ing of Kmeans on embeddings greedily towards F_fair to get a strictly fair one.
    :param X: np.array, N x k embeddings matrix of G
    :param G: nx.Graph, a undirected graph.
    :param C: np.array, n x m color indicator matrix, C_ij = 1 means node i belongs to color j
    :param alpha: np.array, fairness upper bound
    :param beta: np.array, fairness lower bound
    :return:
            np.array,  1xN vector mapping node to its cluster membership.
            float, time
            int, #reassignment
    """
    t1 = time.perf_counter()
    n, k = X.shape
    phi, S = fair_kmeans(pd.DataFrame(X), k, C, alpha, beta)
    D_all = np.round(ecdist(X, S), 10)
    phi_fair, num_of_reassignment = reassign_by_center(D_all, k, phi, C, alpha, beta)
    ncut = utils.get_ncut(G, phi_fair)
    t2 = time.perf_counter()
    return phi_fair, ncut, t2 - t1, num_of_reassignment


def fair_assign_exact(X: np.array, G: nx.Graph, C: np.array, alpha: np.array, beta: np.array, seed: int) \
        -> (np.array, float, float):
    """
    With fixed centers obtained from Kmeans on embeddings, run an IP to get a strictly fair clustering.
    :param X: np.array, N x k embeddings matrix of G
    :param G: nx.Graph, a undirected graph.
    :param C: np.array, n x m color indicator matrix, C_ij = 1 means node i belongs to color j
    :param alpha: np.array, fairness upper bound
    :param beta: np.array, fairness lower bound
    :param seed: int, random seed for select the first center
    :return:
            np.array,  1xN vector mapping node to its cluster membership.
            float, time
    """
    t1 = time.perf_counter()
    n, k = X.shape
    m = C.shape[1]
    ## 1.vanilla kmeans
    clustering = KMeans(n_clusters=k, init='k-means++', max_iter=200, random_state=seed, n_init=100)
    clustering.fit(X)
    S = clustering.cluster_centers_
    D = np.round(ecdist(X, S), 10) ** 2  # nxk, d_ij = euclidean_distance(point_i,center_j)^2
    ## 2. solve IP
    # Create a new model
    model = gp.Model("fair-assignment-exact")
    # Create variables
    Y = model.addMVar(shape=(n, k), vtype=GRB.BINARY, name="Y")
    # Set objective
    model.setObjective(gp.quicksum(gp.quicksum(D * Y)), GRB.MINIMIZE)
    # Build constraint matrix
    A, B = np.array([alpha] * n), np.array([beta] * n)
    # Add constraints
    model.addConstr((A - C).T @ Y >= np.zeros((m, k)), name="upper fairness constraints")
    model.addConstr((C - B).T @ Y >= np.zeros((m, k)), name="lower fairness constraints")
    model.addConstr(Y @ np.ones((k, 1)) == np.ones((n, 1)), name="node must belong to one cluster")
    model.addConstr(np.ones((1, n)) @ Y >= np.ones((1, k)), name="cluster not empty")
    # model.write('fair-assignment-exact.lp')
    # Optimize model
    model.setParam('MIPGap', 0.01)
    model.optimize()
    print(f'Obj value = {model.ObjVal}, solving time = {model.Runtime}')
    phi_fair = np.array(Y.X).argmax(axis=1)
    ncut = utils.get_ncut(G, phi_fair)
    t2 = time.perf_counter()
    return phi_fair, ncut, t2 - t1


def fair_assign_max(X: np.array, G: nx.Graph, C: np.array, alpha: np.array, beta: np.array, seed: int, rmaxiter=10) \
        -> (np.array, float, float, int):
    """
    With fixed centers obtained from Kmeans on embeddings, solve LP-relaxation of IP and select cluster j with
    max fractional value as the membership of point i. Then reassign the clustering to a strictly fair one.
    :param X: np.array, N x k embeddings matrix of G
    :param G: nx.Graph, a undirected graph.
    :param C: np.array, n x m color indicator matrix, C_ij = 1 means node i belongs to color j
    :param alpha: np.array, fairness upper bound
    :param beta: np.array, fairness lower bound
    :param seed: int, random seed for select the first center
    :param rmaxiter: max iteration number of fair_assign_max
    :return:
            np.array,  1xN vector mapping node to its cluster membership.
            float, ncut
            float, time
            int, number of reassignment
    """
    t1 = time.perf_counter()
    n, k = X.shape
    m = C.shape[1]
    A, B = np.array([alpha] * n), np.array([beta] * n)
    ncuts = []
    ## 1.vanilla kmeans
    clustering = KMeans(n_clusters=k, init='k-means++', max_iter=200, random_state=seed, n_init=100)
    clustering.fit(X)
    S = clustering.cluster_centers_
    D = np.round(ecdist(X, S), 10)  # nxk, d_ij = euclidean_distance(point_i,center_j)
    ## 2. solve LP
    # Create a new model
    model = gp.Model("fair-assignment-max")
    # Create variables
    Y = model.addMVar(shape=(n, k), lb=0, ub=1, name="Y")
    # Add constraints
    model.addConstr((A - C).T @ Y >= np.zeros((m, k)), name="upper fairness constraints")
    model.addConstr((C - B).T @ Y >= np.zeros((m, k)), name="lower fairness constraints")
    model.addConstr(Y @ np.ones((k, 1)) == np.ones((n, 1)), name="node must belong to one cluster")
    model.addConstr(np.ones((1, n)) @ Y >= np.ones((1, k)), name="cluster not empty")
    for iter in range(rmaxiter):
        # Set or update objective
        model.setObjective(gp.quicksum(gp.quicksum(D * Y)), GRB.MINIMIZE)
        # model.write('fair-assignment-max.lp')
        model.optimize()
        print(f'Obj value = {model.ObjVal}, solving time = {model.Runtime}')
        phi = np.array(Y.X).argmax(axis=1)  # assign point i to cluster j ,where j = max_j Y[i,j]
        model.reset(1)
        ## 3. rounding IP solution to an integral one
        phi_fair, num_of_reassignment = reassign_by_ncut(G, k, phi, C, alpha, beta)
        ncuts.append((iter, utils.get_ncut(G, phi_fair), phi_fair, num_of_reassignment))
        # update center
        last_S = S.copy()
        S = np.array([X[phi_fair == i].mean(axis=0) for i in range(k)])
        D = np.round(ecdist(X, S), 10)  # nxk, d_ij = euclidean_distance(point_i,center_j)
        if np.linalg.norm(S - last_S, 'fro') <= 1e-4:
            break
    id, ncut, phi_fair, num_of_reassignment = min(ncuts, key=lambda x: x[1])
    t2 = time.perf_counter()
    print([(item[0], item[1], item[3]) for item in ncuts])
    return phi_fair, ncut, t2 - t1, num_of_reassignment


# auxiliary function
def reassign_by_ncut(G: nx.Graph, k: int, phi: np.array, C: np.array, alpha: np.array, beta: np.array) \
        -> (np.array, int):
    """
    Transform an unfair clustering to the strictly alpha,beta-fair one.
    :param G: nx.Graph, a undirected graph.
    :param phi: 1xn np.array, initial clustering to be reassigned.
    :param C: np.array, n x m color indicator matrix, C_ij = 1 means node i belongs to color j
    :param alpha: np.array, fairness upper bound
    :param beta: np.array, fairness lower bound
    :return:
            np.array,  1xN vector mapping node to its cluster membership.
            int, #reassignmen
    """
    # transform indicator vector to 0-1 indicator matrix
    n, m = C.shape
    color = C.argmax(axis=1)
    Y = np.zeros((n, k), dtype=int)
    for i in range(n):
        Y[i, phi[i]] = 1
    F = C.T @ Y  # f_ij = number of color i in cluster j
    ## 2.construct IP to get F_fair
    # Create a new model
    model = gp.Model("fair-F")
    # Create variables
    row_sum = F.sum(axis=1)
    ub = np.array([[row_sum[i] - F[i, j] for j in range(k)] for i in range(m)])
    X = model.addVars(range(m), range(k), vtype=GRB.INTEGER, lb=-F, ub=ub, name="X")
    # we need add auxiliary variables abs_X explicitly to represent abs of X in gurobi, but the model remains linear.
    abs_X = model.addVars(range(m), range(k), vtype=GRB.INTEGER, name="abs_X")
    # Set objective
    model.setObjective(gp.quicksum(abs_X[i, j] for i in range(m) for j in range(k)), GRB.MINIMIZE)
    # Add constraints
    model.addConstrs((abs_X[i, j] == gp.abs_(X[i, j]) for i in range(m) for j in range(k)),
                     name='auxiliary variables for abs')
    model.addConstrs(
        ((alpha[i] * gp.quicksum((F[p, j] + X[p, j]) for p in range(m)) - X[i, j] >= F[i, j]) for i in range(m) for j in
         range(k)), name="alpha")
    model.addConstrs(
        ((beta[i] * gp.quicksum((F[p, j] + X[p, j]) for p in range(m)) - X[i, j] <= F[i, j]) for i in range(m) for j in
         range(k)), name="beta")
    model.addConstrs(((gp.quicksum((F[p, j] + X[p, j]) for p in range(m)) >= 1) for j in range(k)),
                     name='non-empty-cluster')
    model.addConstrs(((gp.quicksum(X[i, p] for p in range(k)) == 0) for i in range(m)), name='row-sum-stays-constants')
    # model.write('fair-F.lp')
    # Optimize model
    model.setParam('MIPGap', 0.01)
    model.optimize()
    if model.status != GRB.OPTIMAL:
        print('Error! IP F_fair has no solution.')
        sys.exit(-1)
    num_of_reassignment = 0.5 * np.round(model.ObjVal)
    print(f'#reassign = {num_of_reassignment}, solving time = {model.Runtime}')
    Delta_list = [[model.getVarByName(f'X[{i},{j}]').x for j in range(k)] for i in range(m)]
    Delta = np.round(Delta_list).astype(
        int)  # sometimes, gurobi returns int value such as -0.99999, we need round it to int
    ##3. Reassign the membership by Delta
    W = nx.adjacency_matrix(G, sorted(G.nodes))
    Y_sparse = sp.csr_matrix(Y)
    D = W.dot(Y_sparse).toarray()  # d_ij = sum of W_{iv}, where v is in cluster j
    degrees = D.sum(axis=1)
    volumes = np.array([degrees[phi == j].sum() for j in range(k)])
    cuts = volumes - (D * Y).sum(axis=0)  # the right item just is association of clusters.
    W = W.tocsc()
    points = [[i, color[i], phi[i]] for i in range(n)]
    get_delta = lambda node, new: (cuts[node[2]] - degrees[node[0]] + 2 * D[node[0], node[2]]) / (
            volumes[node[2]] - degrees[node[0]]) + (cuts[new] + degrees[node[0]] - 2 * D[node[0], new]) / (
                                          volumes[new] + degrees[node[0]]) - cuts[node[2]] / volumes[node[2]] - \
                                  cuts[new] / volumes[new]
    for i in range(m):
        under_filled_clusters = np.argwhere(Delta[i] > 0).T[0]
        over_filled_clusters = np.argwhere(Delta[i] < 0).T[0]
        out = [point for point in points if point[1] == i and point[
            2] in over_filled_clusters]  # transferable points, i.e. thoses whose membership is overfilled
        row_sum = Delta[i][Delta[i] > 0].sum()
        for _ in range(row_sum):  # reassign row by row
            min_relocation = [(node, new, get_delta(node, new)) for new in under_filled_clusters for node in out]
            p, new, _ = min(min_relocation, key=lambda tup: tup[2])
            old = p[2]
            Delta[i, new] -= 1
            Delta[i, old] += 1
            out.remove(p)
            if Delta[i, new] == 0:  # cluster new is fair now and no more nodes need to be added
                under_filled_clusters = np.argwhere(Delta[i] > 0).T[0]
            if Delta[i, old] == 0:  # cluster new is fair now and no more nodes need to be removed
                out = [point for point in out if point[2] != old]
            p[2] = new
            cuts[old] += -degrees[p[0]] + 2 * D[p[0], old]
            cuts[new] += degrees[p[0]] - 2 * D[p[0], new]
            col = W.getcol(p[0]).toarray().flatten()
            D[:, old] -= col
            D[:, new] += col
            volumes[old] -= degrees[p[0]]
            volumes[new] += degrees[p[0]]
    phi_fair = np.array([point[2] for point in points])
    return phi_fair, num_of_reassignment


def reassign_by_center(D_all: np.array, k: int, phi: np.array, C: np.array, alpha: np.array, beta: np.array) \
        -> (np.array, int):
    """
    Transform an unfair clustering to the strictly alpha,beta-fair one.
    :param D_all: np.array, n x k distance matrix
    :param k: int, the dimensions of embedding, also the number of clusters .
    :param C: np.array, n x m color indicator matrix, C_ij = 1 means node i belongs to color j
    :param alpha: np.array, fairness upper bound
    :param beta: np.array, fairness lower bound
    :return:
            np.array,  1xN vector mapping node to its cluster membership.
            int, #reassignment
    """
    # transform indicator vector to 0-1 indicator matrix
    n, m = C.shape
    color = C.argmax(axis=1)
    Y = np.zeros((n, k), dtype=int)
    for i in range(n):
        Y[i, phi[i]] = 1
    F = C.T @ Y  # f_ij = number of color i in cluster j
    ## 2.construct IP to get F_fair
    # Create a new model
    model = gp.Model("fair-F")
    # Create variables
    row_sum = F.sum(axis=1)
    ub = np.array([[row_sum[i] - F[i, j] for j in range(k)] for i in range(m)])
    X = model.addVars(range(m), range(k), vtype=GRB.INTEGER, lb=-F, ub=ub, name="X")
    # we need add auxiliary variables abs_X explicitly to represent abs of X in gurobi, but the model remains linear.
    abs_X = model.addVars(range(m), range(k), vtype=GRB.INTEGER, name="abs_X")
    # Set objective
    model.setObjective(gp.quicksum(abs_X[i, j] for i in range(m) for j in range(k)), GRB.MINIMIZE)
    # Add constraints
    model.addConstrs((abs_X[i, j] == gp.abs_(X[i, j]) for i in range(m) for j in range(k)),
                     name='auxiliary variables for abs')
    model.addConstrs(
        ((alpha[i] * gp.quicksum((F[p, j] + X[p, j]) for p in range(m)) - X[i, j] >= F[i, j]) for i in range(m) for j in
         range(k)), name="alpha")
    model.addConstrs(
        ((beta[i] * gp.quicksum((F[p, j] + X[p, j]) for p in range(m)) - X[i, j] <= F[i, j]) for i in range(m) for j in
         range(k)), name="beta")
    model.addConstrs(((gp.quicksum((F[p, j] + X[p, j]) for p in range(m)) >= 1) for j in range(k)),
                     name='non-empty-cluster')
    model.addConstrs(((gp.quicksum(X[i, p] for p in range(k)) == 0) for i in range(m)), name='row-sum-stays-constants')
    # model.write('fair-F.lp')
    # Optimize model
    model.setParam('MIPGap', 0.01)
    model.optimize()
    if model.status != GRB.OPTIMAL:
        print('Error! IP F_fair has no solution.')
        sys.exit(-1)
    num_of_reassignment = 0.5 * np.round(model.ObjVal)
    print(f'#reassign = {num_of_reassignment}, solving time = {model.Runtime}')
    Delta_list = [[model.getVarByName(f'X[{i},{j}]').x for j in range(k)] for i in range(m)]
    Delta = np.round(Delta_list).astype(
        int)  # sometimes, gurobi returns int value such as -0.99999, we need round it to int
    ##3. Reassign the membership greedily by Kmeans center
    points = [Point(i, color[i], D_all[i], phi[i]) for i in range(n)]
    for i in range(m):
        under_filled_clusters = np.argwhere(Delta[i] > 0).T[0]
        over_filled_clusters = np.argwhere(Delta[i] < 0).T[0]
        out = [point for point in points if
               point.color == i and point.membership in over_filled_clusters]
        D = np.array(
            [point.dists[under_filled_clusters] for point in out])  # 2D array ,len(out) * len(under_filled_clusters)
        row_sum = Delta[i][Delta[i] > 0].sum()
        for _ in range(row_sum):  # reassign row by row
            p, q = np.unravel_index(np.argmin(D, axis=None), D.shape)
            old, new = out[p].membership, under_filled_clusters[q]
            D[p] = sys.float_info.max  # node D[p] shouldn't be selected again
            Delta[i, new] -= 1
            Delta[i, old] += 1
            if Delta[i, new] == 0:  # cluster new is fair now and no more nodes need to be added
                D[:, q] = sys.float_info.max
            if Delta[i, old] == 0:  # cluster new is fair now and no more nodes need to be removed
                idxs = np.array([i for i in range(len(out)) if out[i].membership == old])
                if len(idxs) != 0:
                    D[idxs] = sys.float_info.max
            out[p].membership = new
    phi_fair = np.array([point.membership for point in points])
    return phi_fair, num_of_reassignment


# code referred to https://github.com/nicolasjulioflores/fair_algorithms_for_clustering
def fair_kmeans(df: pd.DataFrame, num_clusters: int, C: np.array, alpha: np.array, beta: np.array) -> (
        np.array, np.array):
    '''
    Algorithm in Fair Algorithms for Clustering with simplified interface. Others remain unchanged.
    :param df: pd.DataFrame, N x k embeddings matrix of G
    :param num_clusters: int, number of clusters
    :param C: np.array, n x m color indicator matrix, C_ij = 1 means node i belongs to color j
    :param alpha: np.array, fairness upper bound
    :param beta: np.array, fairness lower bound
    :return:
            np.array,  1xN vector mapping node to its cluster membership.
            np.array,  centers obtained from Kmeans on embeddings.
    '''
    n, k = df.shape
    m = C.shape[1]
    color = C.argmax(axis=1)
    _, _, cluster_centers = vanilla_clustering(df, num_clusters, 'kmeans')
    fp_alpha = {'sensitive_attribute': {i: alpha[i] for i in range(m)}}
    fp_beta = {'sensitive_attribute': {i: beta[i] for i in range(m)}}
    fp_color_flag = {'sensitive_attribute': list(color)}
    res = fair_partial_assignment(df, cluster_centers, fp_alpha, fp_beta, fp_color_flag, 'kmeans')
    assignment = np.array(res["assignment"])
    phi = np.array(np.split(assignment, n)).argmax(axis=1)
    return phi, np.array(cluster_centers)

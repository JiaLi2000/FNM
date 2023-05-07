from GraphClustering import graph_clustering
import utils
import networkx as nx
import numpy as np
import pandas as pd
import scipy.sparse as sp
import csv
import gc
import warnings

warnings.filterwarnings('ignore')

output = open("./results/results_ablation.csv", "w")
writer = csv.writer(output)
writer.writerow(["dataset", "#V", "#E", "group", "m", "k", "sigma", 'alpha', 'beta', "Embeddings", "Rounding",
                 'mu', 'pho', "ncut", 'conti_ncut', "f_Larg", 'balance_ICML_avg', 'balance_ICML_mini',
                 'balance_ICML_datasets', 'balance_NIPS_avg', 'balance_NIPS_mini',
                 "fair_prop_vio_avg", "fair_prop_vio_maxi", "#reassignment", "time1", "time2", "time"])

Embeddings_step = [('spectral', 'fair_assign_max'), ('deepwalk', 'fair_assign_max'), ('node2vec', 'fair_assign_max'),
                   ('fair_walk', 'fair_assign_max'), ('fair_equality', 'fair_assign_max'),
                   ('fair_equality_s', 'fair_assign_max'), ('fair_proportion', 'fair_assign_max')]
Rounding_step = [('fair_proportion', 'kmeans'), ('fair_proportion', 'reassigned_kmeans'),
                 ('fair_proportion', 'reassigned_fair_kmeans'),
                 ('fair_proportion', 'fair_assign_exact'), ('fair_proportion', 'fair_assign_max')]

# facebookNet
edgelist = pd.read_csv('./datasets/processed/facebookNet/facebookNet_edges.csv')
G = nx.from_pandas_edgelist(edgelist, source='source', target='target', edge_attr=None, create_using=nx.Graph)
colors = pd.read_csv('./datasets/processed/facebookNet/facebookNet_colors.csv', header=None)
colors.columns = ['id', 'sex']
color_name = 'sex'
color = colors[color_name].values
num_runs = 10

parameters_k_2 = {5: (1, 2), 20: (100, 8), }
parameters_k_8 = {5: (100, 2), 20: (1, 10), }
for k in [5, 20]:
    for sigma in [0.2, 0.8]:
        paras = eval(f'parameters_k_{int(sigma * 10)}')
        m = color.max() + 1
        n = G.number_of_nodes()
        C = sp.coo_matrix(([1] * n, (range(n), color)), dtype=int).toarray()
        ratio = C.sum(axis=0) / n
        beta = ratio * (1 - sigma)
        alpha = ratio / (1 - sigma)
        for ablation in [Embeddings_step, Rounding_step]:
            for (Embeddings, Rounding) in ablation:
                if Embeddings in ['deepwalk', 'node2vec', 'fair_walk']:
                    num_runs = 1
                for run in range(num_runs):
                    phi, result = graph_clustering(G=G, k=k, Embeddings=Embeddings, Rounding=Rounding,
                                                   C=C, alpha=alpha, beta=beta, color=color, seed=run,
                                                   mu=paras[k][0], pho=paras[k][1], xtol=1e-6, ftol=1e-9, gtol=1e-6,
                                                   omxitr=100, mxitr=2000, rmaxiter=10)
                    print(result)
                    fair_prop_vio_avg, fair_prop_vio_maxi = utils.get_fair_prop_vio(phi, C, alpha, beta)
                    balance_ICML_avg, balance_ICML_mini = utils.get_balance_ICML(phi, C)
                    balance_ICML_dataset0, balance_ICML_dataset = utils.get_balance_ICML(np.zeros(n, dtype=int), C)
                    writer.writerow(
                        ["facebookNet", G.number_of_nodes(), G.number_of_edges(), color_name, m, k, sigma, alpha, beta,
                         Embeddings, Rounding, paras[k][1], paras[k][0], "%.4f" % result['ncut'],
                         "%.8f" % result['conti_ncut'],
                         "%.8f" % result.get('f_Larg', -1),
                         "%.4f" % balance_ICML_avg, "%.4f" % balance_ICML_mini, "%.4f" % balance_ICML_dataset,
                         "%.4f" % result['balance_avg'], "%.4f" % result['balance_min'], "%.4f" % fair_prop_vio_avg,
                         "%.4f" % fair_prop_vio_maxi, result.get('num_of_reassignment', -1), "%.4f" % result['time1'],
                         "%.4f" % result['time2'], "%.4f" % result['time']])
                output.flush()
                num_runs = 10
del edgelist, G, colors
gc.collect()

# german
edgelist = pd.read_csv('./datasets/processed/german/german_edges.csv')
G = nx.from_pandas_edgelist(edgelist, source='source', target='target', edge_attr=None, create_using=nx.Graph)
colors = pd.read_csv('./datasets/processed/german/german_colors.csv', header=None)
colors.columns = ['id', 'sex']
color_name = 'sex'
color = colors[color_name].values
num_runs = 10

parameters_k_2 = {5: (0.0001, 4), 20: (1, 4), }
parameters_k_8 = {5: (1, 6), 20: (1, 6), }
for k in [5, 20]:
    for sigma in [0.2, 0.8]:
        paras = eval(f'parameters_k_{int(sigma * 10)}')
        m = color.max() + 1
        n = G.number_of_nodes()
        C = sp.coo_matrix(([1] * n, (range(n), color)), dtype=int).toarray()
        ratio = C.sum(axis=0) / n
        beta = ratio * (1 - sigma)
        alpha = ratio / (1 - sigma)
        for ablation in [Embeddings_step, Rounding_step]:
            for (Embeddings, Rounding) in ablation:
                if Embeddings in ['deepwalk', 'node2vec', 'fair_walk']:
                    num_runs = 1
                for run in range(num_runs):
                    phi, result = graph_clustering(G=G, k=k, Embeddings=Embeddings, Rounding=Rounding,
                                                   C=C, alpha=alpha, beta=beta, color=color, seed=run,
                                                   mu=paras[k][0], pho=paras[k][1], xtol=1e-6, ftol=1e-9, gtol=1e-6,
                                                   omxitr=100, mxitr=2000, rmaxiter=10)
                    print(result)
                    fair_prop_vio_avg, fair_prop_vio_maxi = utils.get_fair_prop_vio(phi, C, alpha, beta)
                    balance_ICML_avg, balance_ICML_mini = utils.get_balance_ICML(phi, C)
                    balance_ICML_dataset0, balance_ICML_dataset = utils.get_balance_ICML(np.zeros(n, dtype=int), C)
                    writer.writerow(
                        ["german", G.number_of_nodes(), G.number_of_edges(), color_name, m, k, sigma, alpha, beta,
                         Embeddings, Rounding, paras[k][1], paras[k][0], "%.4f" % result['ncut'],
                         "%.8f" % result['conti_ncut'],
                         "%.8f" % result.get('f_Larg', -1),
                         "%.4f" % balance_ICML_avg, "%.4f" % balance_ICML_mini, "%.4f" % balance_ICML_dataset,
                         "%.4f" % result['balance_avg'], "%.4f" % result['balance_min'], "%.4f" % fair_prop_vio_avg,
                         "%.4f" % fair_prop_vio_maxi, result.get('num_of_reassignment', -1), "%.4f" % result['time1'],
                         "%.4f" % result['time2'], "%.4f" % result['time']])
                output.flush()
                num_runs = 10
del edgelist, G, colors
gc.collect()

# SBM_1000
edgelist = pd.read_csv(f'datasets/processed/SBM_1000/SBM_1000_edges.csv')
G = nx.from_pandas_edgelist(edgelist, source='source', target='target', edge_attr=None, create_using=nx.Graph)
colors = pd.read_csv(f'datasets/processed/SBM_1000/SBM_1000_colors.csv', header=None)
colors.columns = ['id', '5']
color_name = '5'
color = colors[color_name].values
num_runs = 10

parameters_k_2 = {5: (1, 10), 20: (1, 4)}
parameters_k_8 = {5: (1, 8), 20: (1e-4, 6)}
for k in [5, 20]:
    for sigma in [0.2, 0.8]:
        paras = eval(f'parameters_k_{int(sigma * 10)}')
        m = color.max() + 1
        n = G.number_of_nodes()
        C = sp.coo_matrix(([1] * n, (range(n), color)), dtype=int).toarray()
        ratio = C.sum(axis=0) / n
        beta = ratio * (1 - sigma)
        alpha = ratio / (1 - sigma)
        for ablation in [Embeddings_step, Rounding_step]:
            for (Embeddings, Rounding) in ablation:
                if Embeddings in ['deepwalk', 'node2vec', 'fair_walk']:
                    num_runs = 1
                for run in range(num_runs):
                    phi, result = graph_clustering(G=G, k=k, Embeddings=Embeddings, Rounding=Rounding,
                                                   C=C, alpha=alpha, beta=beta, color=color, seed=run,
                                                   mu=paras[k][0], pho=paras[k][1], xtol=1e-6, ftol=1e-9, gtol=1e-6,
                                                   omxitr=100, mxitr=2000, rmaxiter=10)
                    print(result)
                    fair_prop_vio_avg, fair_prop_vio_maxi = utils.get_fair_prop_vio(phi, C, alpha, beta)
                    balance_ICML_avg, balance_ICML_mini = utils.get_balance_ICML(phi, C)
                    balance_ICML_dataset0, balance_ICML_dataset = utils.get_balance_ICML(np.zeros(n, dtype=int), C)
                    writer.writerow(
                        ["SBM_1000", G.number_of_nodes(), G.number_of_edges(), color_name, m, k, sigma, alpha, beta,
                         Embeddings, Rounding, paras[k][1], paras[k][0], "%.4f" % result['ncut'],
                         "%.8f" % result['conti_ncut'],
                         "%.8f" % result.get('f_Larg', -1),
                         "%.4f" % balance_ICML_avg, "%.4f" % balance_ICML_mini, "%.4f" % balance_ICML_dataset,
                         "%.4f" % result['balance_avg'], "%.4f" % result['balance_min'], "%.4f" % fair_prop_vio_avg,
                         "%.4f" % fair_prop_vio_maxi, result.get('num_of_reassignment', -1), "%.4f" % result['time1'],
                         "%.4f" % result['time2'], "%.4f" % result['time']])
                output.flush()
                num_runs = 10
del edgelist, G, colors
gc.collect()

# dblp
edgelist = pd.read_csv('./datasets/processed/dblp/dblp_edges.csv')
G = nx.from_pandas_edgelist(edgelist, source='source', target='target', edge_attr=None, create_using=nx.Graph)
colors = pd.read_csv('./datasets/processed/dblp/dblp_colors.csv', header=None)
colors.columns = ['id', 'continent']
color_name = 'continent'
color = colors[color_name].values
num_runs = 10

parameters_k_2 = {5: (0.01, 4), 20: (0.01, 6), }
parameters_k_8 = {5: (0.0001, 6), 20: (1, 10), }
for k in [5, 20]:
    for sigma in [0.2, 0.8]:
        paras = eval(f'parameters_k_{int(sigma * 10)}')
        m = color.max() + 1
        n = G.number_of_nodes()
        C = sp.coo_matrix(([1] * n, (range(n), color)), dtype=int).toarray()
        ratio = C.sum(axis=0) / n
        beta = ratio * (1 - sigma)
        alpha = ratio / (1 - sigma)
        for ablation in [Embeddings_step, Rounding_step]:
            for (Embeddings, Rounding) in ablation:
                if Embeddings in ['deepwalk', 'node2vec', 'fair_walk']:
                    num_runs = 1
                for run in range(num_runs):
                    phi, result = graph_clustering(G=G, k=k, Embeddings=Embeddings, Rounding=Rounding,
                                                   C=C, alpha=alpha, beta=beta, color=color, seed=run,
                                                   mu=paras[k][0], pho=paras[k][1], xtol=1e-6, ftol=1e-9, gtol=1e-6,
                                                   omxitr=100, mxitr=2000, rmaxiter=10)
                    print(result)
                    fair_prop_vio_avg, fair_prop_vio_maxi = utils.get_fair_prop_vio(phi, C, alpha, beta)
                    balance_ICML_avg, balance_ICML_mini = utils.get_balance_ICML(phi, C)
                    balance_ICML_dataset0, balance_ICML_dataset = utils.get_balance_ICML(np.zeros(n, dtype=int), C)
                    writer.writerow(
                        ["dblp", G.number_of_nodes(), G.number_of_edges(), color_name, m, k, sigma, alpha, beta,
                         Embeddings, Rounding, paras[k][1], paras[k][0], "%.4f" % result['ncut'],
                         "%.8f" % result['conti_ncut'],
                         "%.8f" % result.get('f_Larg', -1),
                         "%.4f" % balance_ICML_avg, "%.4f" % balance_ICML_mini, "%.4f" % balance_ICML_dataset,
                         "%.4f" % result['balance_avg'], "%.4f" % result['balance_min'], "%.4f" % fair_prop_vio_avg,
                         "%.4f" % fair_prop_vio_maxi, result.get('num_of_reassignment', -1), "%.4f" % result['time1'],
                         "%.4f" % result['time2'], "%.4f" % result['time']])
                output.flush()
                num_runs = 10
del edgelist, G, colors
gc.collect()

# lastfm
edgelist = pd.read_csv('./datasets/processed/lastfm/lastfm_edges.csv')
G = nx.from_pandas_edgelist(edgelist, source='source', target='target', edge_attr=None, create_using=nx.Graph)
colors = pd.read_csv('./datasets/processed/lastfm/lastfm_colors.csv', header=None)
colors.columns = ['id', 'country']
color_name = 'country'
color = colors[color_name].values
num_runs = 10

parameters_k_2 = {5: (0.01, 6), 20: (1, 8)}
parameters_k_8 = {5: (1, 10), 20: (1, 4)}
for k in [5, 20]:
    for sigma in [0.2, 0.8]:
        paras = eval(f'parameters_k_{int(sigma * 10)}')
        m = color.max() + 1
        n = G.number_of_nodes()
        C = sp.coo_matrix(([1] * n, (range(n), color)), dtype=int).toarray()
        ratio = C.sum(axis=0) / n
        beta = ratio * (1 - sigma)
        alpha = ratio / (1 - sigma)
        for ablation in [Embeddings_step, Rounding_step]:
            for (Embeddings, Rounding) in ablation:
                if Embeddings in ['deepwalk', 'node2vec', 'fair_walk']:
                    num_runs = 1
                for run in range(num_runs):
                    phi, result = graph_clustering(G=G, k=k, Embeddings=Embeddings, Rounding=Rounding,
                                                   C=C, alpha=alpha, beta=beta, color=color, seed=run,
                                                   mu=paras[k][0], pho=paras[k][1], xtol=1e-6, ftol=1e-9, gtol=1e-6,
                                                   omxitr=100, mxitr=2000, rmaxiter=10)
                    print(result)
                    fair_prop_vio_avg, fair_prop_vio_maxi = utils.get_fair_prop_vio(phi, C, alpha, beta)
                    balance_ICML_avg, balance_ICML_mini = utils.get_balance_ICML(phi, C)
                    balance_ICML_dataset0, balance_ICML_dataset = utils.get_balance_ICML(np.zeros(n, dtype=int), C)
                    writer.writerow(
                        ["lastfm", G.number_of_nodes(), G.number_of_edges(), color_name, m, k, sigma, alpha, beta,
                         Embeddings, Rounding, paras[k][1], paras[k][0], "%.4f" % result['ncut'],
                         "%.8f" % result['conti_ncut'],
                         "%.8f" % result.get('f_Larg', -1),
                         "%.4f" % balance_ICML_avg, "%.4f" % balance_ICML_mini, "%.4f" % balance_ICML_dataset,
                         "%.4f" % result['balance_avg'], "%.4f" % result['balance_min'], "%.4f" % fair_prop_vio_avg,
                         "%.4f" % fair_prop_vio_maxi, result.get('num_of_reassignment', -1), "%.4f" % result['time1'],
                         "%.4f" % result['time2'], "%.4f" % result['time']])
                output.flush()
                num_runs = 10
del edgelist, G, colors
gc.collect()

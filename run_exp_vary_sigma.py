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

output = open("./results/results_vary_sigma.csv", "w")
writer = csv.writer(output)
writer.writerow(["dataset", "#V", "#E", "group", "m", "k", "sigma", 'alpha', 'beta', "Embeddings", "Rounding",
                 'mu', 'pho', "ncut", 'conti_ncut', "f_Larg", 'balance_ICML_avg', 'balance_ICML_mini',
                 'balance_ICML_datasets', 'balance_NIPS_avg', 'balance_NIPS_mini',
                 "fair_prop_vio_avg", "fair_prop_vio_maxi", "#reassignment", "time1", "time2", "time"])

'''fair_prop'''

Embeddings = 'fair_proportion'
Rounding = 'fair_assign_max'

# facebookNet
edgelist = pd.read_csv('./datasets/processed/facebookNet/facebookNet_edges.csv')
G = nx.from_pandas_edgelist(edgelist, source='source', target='target', edge_attr=None, create_using=nx.Graph)
colors = pd.read_csv('./datasets/processed/facebookNet/facebookNet_colors.csv', header=None)
colors.columns = ['id', 'sex']
color_name = 'sex'
color = colors[color_name].values
num_runs = 10

paras = {0.1: (1, 4), 0.2: (1, 8), 0.3: (1, 6), 0.4: (1, 4), 0.5: (0.01, 2), 0.6: (1, 10), 0.7: (0.01, 2),
         0.8: (100, 8), 0.9: (0.0001, 10), 1: (0.01, 10), }
for k in [5]:
    for sigma in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:
        m = color.max() + 1
        n = G.number_of_nodes()
        C = sp.coo_matrix(([1] * n, (range(n), color)), dtype=int).toarray()
        ratio = C.sum(axis=0) / n
        if sigma == 1:
            beta = np.zeros(m)
            alpha = np.ones(m)
        else:
            beta = ratio * (1 - sigma)
            alpha = ratio / (1 - sigma)
        for run in range(num_runs):
            phi, result = graph_clustering(G=G, k=k, Embeddings=Embeddings, Rounding=Rounding,
                                           C=C, alpha=alpha, beta=beta, color=color, seed=run,
                                           mu=paras[sigma][0], pho=paras[sigma][1], xtol=1e-6, ftol=1e-9, gtol=1e-6,
                                           omxitr=100, mxitr=2000, rmaxiter=10)
            print(result)
            fair_prop_vio_avg, fair_prop_vio_maxi = utils.get_fair_prop_vio(phi, C, alpha, beta)
            balance_ICML_avg, balance_ICML_mini = utils.get_balance_ICML(phi, C)
            balance_ICML_dataset0, balance_ICML_dataset = utils.get_balance_ICML(np.zeros(n, dtype=int), C)
            writer.writerow(
                ["facebookNet", G.number_of_nodes(), G.number_of_edges(), color_name, m, k, sigma, alpha, beta,
                 Embeddings, Rounding, paras[sigma][1], paras[sigma][0], "%.4f" % result['ncut'],
                 "%.8f" % result['conti_ncut'],
                 "%.8f" % result.get('f_Larg', -1),
                 "%.4f" % balance_ICML_avg, "%.4f" % balance_ICML_mini, "%.4f" % balance_ICML_dataset,
                 "%.4f" % result['balance_avg'], "%.4f" % result['balance_min'], "%.4f" % fair_prop_vio_avg,
                 "%.4f" % fair_prop_vio_maxi, result.get('num_of_reassignment', -1), "%.4f" % result['time1'],
                 "%.4f" % result['time2'], "%.4f" % result['time']])
        output.flush()
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

paras = {0.1: (1, 4), 0.2: (0.0001, 4), 0.3: (0.01, 8), 0.4: (1, 10), 0.5: (0.01, 10), 0.6: (0.01, 6), 0.7: (0.01, 4),
         0.8: (1, 8), 0.9: (0.01, 10), 1: (0.0001, 8), }
for k in [5]:
    for sigma in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:
        m = color.max() + 1
        n = G.number_of_nodes()
        C = sp.coo_matrix(([1] * n, (range(n), color)), dtype=int).toarray()
        ratio = C.sum(axis=0) / n
        if sigma == 1:
            beta = np.zeros(m)
            alpha = np.ones(m)
        else:
            beta = ratio * (1 - sigma)
            alpha = ratio / (1 - sigma)
        for run in range(num_runs):
            phi, result = graph_clustering(G=G, k=k, Embeddings=Embeddings, Rounding=Rounding,
                                           C=C, alpha=alpha, beta=beta, color=color, seed=run,
                                           mu=paras[sigma][0], pho=paras[sigma][1], xtol=1e-6, ftol=1e-9, gtol=1e-6,
                                           omxitr=100, mxitr=2000, rmaxiter=10)
            print(result)
            fair_prop_vio_avg, fair_prop_vio_maxi = utils.get_fair_prop_vio(phi, C, alpha, beta)
            balance_ICML_avg, balance_ICML_mini = utils.get_balance_ICML(phi, C)
            balance_ICML_dataset0, balance_ICML_dataset = utils.get_balance_ICML(np.zeros(n, dtype=int), C)
            writer.writerow(
                ["german", G.number_of_nodes(), G.number_of_edges(), color_name, m, k, sigma, alpha, beta,
                 Embeddings, Rounding, paras[sigma][1], paras[sigma][0], "%.4f" % result['ncut'],
                 "%.8f" % result['conti_ncut'],
                 "%.8f" % result.get('f_Larg', -1),
                 "%.4f" % balance_ICML_avg, "%.4f" % balance_ICML_mini, "%.4f" % balance_ICML_dataset,
                 "%.4f" % result['balance_avg'], "%.4f" % result['balance_min'], "%.4f" % fair_prop_vio_avg,
                 "%.4f" % fair_prop_vio_maxi, result.get('num_of_reassignment', -1), "%.4f" % result['time1'],
                 "%.4f" % result['time2'], "%.4f" % result['time']])
        output.flush()
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

paras = {0.1: (0.0001, 8), 0.2: (1, 4), 0.3: (1, 4), 0.4: (0.0001, 10), 0.5: (0.0001, 10), 0.6: (1, 10), 0.7: (1, 4),
         0.8: (1, 10), 0.9: (1, 6), 1: (1, 8), }
for k in [5]:
    for sigma in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:
        m = color.max() + 1
        n = G.number_of_nodes()
        C = sp.coo_matrix(([1] * n, (range(n), color)), dtype=int).toarray()
        ratio = C.sum(axis=0) / n
        if sigma == 1:
            beta = np.zeros(m)
            alpha = np.ones(m)
        else:
            beta = ratio * (1 - sigma)
            alpha = ratio / (1 - sigma)
        for run in range(num_runs):
            phi, result = graph_clustering(G=G, k=k, Embeddings=Embeddings, Rounding=Rounding,
                                           C=C, alpha=alpha, beta=beta, color=color, seed=run,
                                           mu=paras[sigma][0], pho=paras[sigma][1], xtol=1e-6, ftol=1e-9, gtol=1e-6,
                                           omxitr=100, mxitr=2000, rmaxiter=10)
            print(result)
            fair_prop_vio_avg, fair_prop_vio_maxi = utils.get_fair_prop_vio(phi, C, alpha, beta)
            balance_ICML_avg, balance_ICML_mini = utils.get_balance_ICML(phi, C)
            balance_ICML_dataset0, balance_ICML_dataset = utils.get_balance_ICML(np.zeros(n, dtype=int), C)
            writer.writerow(
                ["SBM_1000", G.number_of_nodes(), G.number_of_edges(), color_name, m, k, sigma, alpha, beta,
                 Embeddings, Rounding, paras[sigma][1], paras[sigma][0], "%.4f" % result['ncut'],
                 "%.8f" % result['conti_ncut'],
                 "%.8f" % result.get('f_Larg', -1),
                 "%.4f" % balance_ICML_avg, "%.4f" % balance_ICML_mini, "%.4f" % balance_ICML_dataset,
                 "%.4f" % result['balance_avg'], "%.4f" % result['balance_min'], "%.4f" % fair_prop_vio_avg,
                 "%.4f" % fair_prop_vio_maxi, result.get('num_of_reassignment', -1), "%.4f" % result['time1'],
                 "%.4f" % result['time2'], "%.4f" % result['time']])
        output.flush()
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

paras = {0.1: (0.0001, 4), 0.2: (0.01, 4), 0.3: (0.0001, 8), 0.4: (0.01, 10), 0.5: (100, 8), 0.6: (1, 8), 0.7: (1, 2),
         0.8: (0.0001, 6), 0.9: (100, 6), 1: (0.01, 8), }
for k in [5]:
    for sigma in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:
        m = color.max() + 1
        n = G.number_of_nodes()
        C = sp.coo_matrix(([1] * n, (range(n), color)), dtype=int).toarray()
        ratio = C.sum(axis=0) / n
        if sigma == 1:
            beta = np.zeros(m)
            alpha = np.ones(m)
        else:
            beta = ratio * (1 - sigma)
            alpha = ratio / (1 - sigma)
        for run in range(num_runs):
            phi, result = graph_clustering(G=G, k=k, Embeddings=Embeddings, Rounding=Rounding,
                                           C=C, alpha=alpha, beta=beta, color=color, seed=run,
                                           mu=paras[sigma][0], pho=paras[sigma][1], xtol=1e-6, ftol=1e-9, gtol=1e-6,
                                           omxitr=100, mxitr=2000, rmaxiter=10)
            print(result)
            fair_prop_vio_avg, fair_prop_vio_maxi = utils.get_fair_prop_vio(phi, C, alpha, beta)
            balance_ICML_avg, balance_ICML_mini = utils.get_balance_ICML(phi, C)
            balance_ICML_dataset0, balance_ICML_dataset = utils.get_balance_ICML(np.zeros(n, dtype=int), C)
            writer.writerow(
                ["dblp", G.number_of_nodes(), G.number_of_edges(), color_name, m, k, sigma, alpha, beta,
                 Embeddings, Rounding, paras[sigma][1], paras[sigma][0], "%.4f" % result['ncut'],
                 "%.8f" % result['conti_ncut'],
                 "%.8f" % result.get('f_Larg', -1),
                 "%.4f" % balance_ICML_avg, "%.4f" % balance_ICML_mini, "%.4f" % balance_ICML_dataset,
                 "%.4f" % result['balance_avg'], "%.4f" % result['balance_min'], "%.4f" % fair_prop_vio_avg,
                 "%.4f" % fair_prop_vio_maxi, result.get('num_of_reassignment', -1), "%.4f" % result['time1'],
                 "%.4f" % result['time2'], "%.4f" % result['time']])
        output.flush()
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

paras = {0.1: (0.0001, 8), 0.2: (0.01, 6), 0.3: (1, 2), 0.4: (1, 2), 0.5: (1, 2), 0.6: (1, 8), 0.7: (1, 4),
         0.8: (1, 10), 0.9: (0.0001, 10), 1: (0.0001, 8), }
for k in [5]:
    for sigma in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:
        m = color.max() + 1
        n = G.number_of_nodes()
        C = sp.coo_matrix(([1] * n, (range(n), color)), dtype=int).toarray()
        ratio = C.sum(axis=0) / n
        if sigma == 1:
            beta = np.zeros(m)
            alpha = np.ones(m)
        else:
            beta = ratio * (1 - sigma)
            alpha = ratio / (1 - sigma)
        for run in range(num_runs):
            phi, result = graph_clustering(G=G, k=k, Embeddings=Embeddings, Rounding=Rounding,
                                           C=C, alpha=alpha, beta=beta, color=color, seed=run,
                                           mu=paras[sigma][0], pho=paras[sigma][1], xtol=1e-6, ftol=1e-9, gtol=1e-6,
                                           omxitr=100, mxitr=2000, rmaxiter=10)
            print(result)
            fair_prop_vio_avg, fair_prop_vio_maxi = utils.get_fair_prop_vio(phi, C, alpha, beta)
            balance_ICML_avg, balance_ICML_mini = utils.get_balance_ICML(phi, C)
            balance_ICML_dataset0, balance_ICML_dataset = utils.get_balance_ICML(np.zeros(n, dtype=int), C)
            writer.writerow(
                ["lastfm", G.number_of_nodes(), G.number_of_edges(), color_name, m, k, sigma, alpha, beta,
                 Embeddings, Rounding, paras[sigma][1], paras[sigma][0], "%.4f" % result['ncut'],
                 "%.8f" % result['conti_ncut'],
                 "%.8f" % result.get('f_Larg', -1),
                 "%.4f" % balance_ICML_avg, "%.4f" % balance_ICML_mini, "%.4f" % balance_ICML_dataset,
                 "%.4f" % result['balance_avg'], "%.4f" % result['balance_min'], "%.4f" % fair_prop_vio_avg,
                 "%.4f" % fair_prop_vio_maxi, result.get('num_of_reassignment', -1), "%.4f" % result['time1'],
                 "%.4f" % result['time2'], "%.4f" % result['time']])
        output.flush()
del edgelist, G, colors
gc.collect()

# credit_education
edgelist = pd.read_csv('./datasets/processed/credit_education/credit_education_edges.csv')
G = nx.from_pandas_edgelist(edgelist, source='source', target='target', edge_attr=None, create_using=nx.Graph)
colors = pd.read_csv('./datasets/processed/credit_education/credit_education_colors.csv', header=None)
colors.columns = ['id', 'education']
color_name = 'education'
color = colors[color_name].values
num_runs = 10

paras = {0.1: (0.0001, 10), 0.2: (0.0001, 10), 0.3: (0.0001, 6), 0.4: (0.0001, 2), 0.5: (0.0001, 2), 0.6: (0.0001, 10),
         0.7: (0.0001, 2), 0.8: (1, 2), 0.9: (0.0001, 2), 1: (100, 6), }
for k in [5]:
    for sigma in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:
        m = color.max() + 1
        n = G.number_of_nodes()
        C = sp.coo_matrix(([1] * n, (range(n), color)), dtype=int).toarray()
        ratio = C.sum(axis=0) / n
        if sigma == 1:
            beta = np.zeros(m)
            alpha = np.ones(m)
        else:
            beta = ratio * (1 - sigma)
            alpha = ratio / (1 - sigma)
        for run in range(num_runs):
            phi, result = graph_clustering(G=G, k=k, Embeddings=Embeddings, Rounding=Rounding,
                                           C=C, alpha=alpha, beta=beta, color=color, seed=run,
                                           mu=paras[sigma][0], pho=paras[sigma][1], xtol=1e-6, ftol=1e-9, gtol=1e-6,
                                           omxitr=100, mxitr=2000, rmaxiter=10)
            print(result)
            fair_prop_vio_avg, fair_prop_vio_maxi = utils.get_fair_prop_vio(phi, C, alpha, beta)
            balance_ICML_avg, balance_ICML_mini = utils.get_balance_ICML(phi, C)
            balance_ICML_dataset0, balance_ICML_dataset = utils.get_balance_ICML(np.zeros(n, dtype=int), C)
            writer.writerow(
                ["credit_education", G.number_of_nodes(), G.number_of_edges(), color_name, m, k, sigma, alpha, beta,
                 Embeddings, Rounding, paras[sigma][1], paras[sigma][0], "%.4f" % result['ncut'],
                 "%.8f" % result['conti_ncut'],
                 "%.8f" % result.get('f_Larg', -1),
                 "%.4f" % balance_ICML_avg, "%.4f" % balance_ICML_mini, "%.4f" % balance_ICML_dataset,
                 "%.4f" % result['balance_avg'], "%.4f" % result['balance_min'], "%.4f" % fair_prop_vio_avg,
                 "%.4f" % fair_prop_vio_maxi, result.get('num_of_reassignment', -1), "%.4f" % result['time1'],
                 "%.4f" % result['time2'], "%.4f" % result['time']])
        output.flush()
del edgelist, G, colors
gc.collect()

# deezer
edgelist = pd.read_csv('./datasets/processed/deezer/deezer_edges.csv')
G = nx.from_pandas_edgelist(edgelist, source='source', target='target', edge_attr=None, create_using=nx.Graph)
colors = pd.read_csv('./datasets/processed/deezer/deezer_colors.csv', header=None)
colors.columns = ['id', 'sex']
color_name = 'sex'
color = colors[color_name].values
num_runs = 10

paras = {0.1: (1, 6), 0.2: (100, 2), 0.3: (100, 6), 0.4: (1, 2), 0.5: (100, 2), 0.6: (100, 6), 0.7: (100, 10),
         0.8: (1, 10), 0.9: (100, 2), 1: (0.0001, 6), }
for k in [5]:
    for sigma in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:
        m = color.max() + 1
        n = G.number_of_nodes()
        C = sp.coo_matrix(([1] * n, (range(n), color)), dtype=int).toarray()
        ratio = C.sum(axis=0) / n
        if sigma == 1:
            beta = np.zeros(m)
            alpha = np.ones(m)
        else:
            beta = ratio * (1 - sigma)
            alpha = ratio / (1 - sigma)
        for run in range(num_runs):
            phi, result = graph_clustering(G=G, k=k, Embeddings=Embeddings, Rounding=Rounding,
                                           C=C, alpha=alpha, beta=beta, color=color, seed=run,
                                           mu=paras[sigma][0], pho=paras[sigma][1], xtol=1e-6, ftol=1e-9, gtol=1e-6,
                                           omxitr=100, mxitr=2000, rmaxiter=10)
            print(result)
            fair_prop_vio_avg, fair_prop_vio_maxi = utils.get_fair_prop_vio(phi, C, alpha, beta)
            balance_ICML_avg, balance_ICML_mini = utils.get_balance_ICML(phi, C)
            balance_ICML_dataset0, balance_ICML_dataset = utils.get_balance_ICML(np.zeros(n, dtype=int), C)
            writer.writerow(
                ["deezer", G.number_of_nodes(), G.number_of_edges(), color_name, m, k, sigma, alpha, beta,
                 Embeddings, Rounding, paras[sigma][1], paras[sigma][0], "%.4f" % result['ncut'],
                 "%.8f" % result['conti_ncut'],
                 "%.8f" % result.get('f_Larg', -1),
                 "%.4f" % balance_ICML_avg, "%.4f" % balance_ICML_mini, "%.4f" % balance_ICML_dataset,
                 "%.4f" % result['balance_avg'], "%.4f" % result['balance_min'], "%.4f" % fair_prop_vio_avg,
                 "%.4f" % fair_prop_vio_maxi, result.get('num_of_reassignment', -1), "%.4f" % result['time1'],
                 "%.4f" % result['time2'], "%.4f" % result['time']])
        output.flush()
del edgelist, G, colors
gc.collect()

Rounding = 'reassigned_kmeans'
# pokec_age
edgelist = pd.read_csv('./datasets/processed/pokec_age/pokec_age_edges.csv')
G = nx.from_pandas_edgelist(edgelist, source='source', target='target', edge_attr=None, create_using=nx.Graph)
colors = pd.read_csv('./datasets/processed/pokec_age/pokec_age_colors.csv', header=None)
colors.columns = ['id', 'age']
color_name = 'age'
color = colors[color_name].values
num_runs = 3

for k in [5]:
    for sigma in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:
        m = color.max() + 1
        n = G.number_of_nodes()
        C = sp.coo_matrix(([1] * n, (range(n), color)), dtype=int).toarray()
        ratio = C.sum(axis=0) / n
        if sigma == 1:
            beta = np.zeros(m)
            alpha = np.ones(m)
        else:
            beta = ratio * (1 - sigma)
            alpha = ratio / (1 - sigma)
        for run in range(num_runs):
            phi, result = graph_clustering(G=G, k=k, Embeddings=Embeddings, Rounding=Rounding,
                                           C=C, alpha=alpha, beta=beta, color=color, seed=run,
                                           mu=1e-4, pho=2, xtol=1e-3, ftol=1e-6, gtol=1e-3,
                                           omxitr=100, mxitr=2000, rmaxiter=10)
            print(result)
            fair_prop_vio_avg, fair_prop_vio_maxi = utils.get_fair_prop_vio(phi, C, alpha, beta)
            balance_ICML_avg, balance_ICML_mini = utils.get_balance_ICML(phi, C)
            balance_ICML_dataset0, balance_ICML_dataset = utils.get_balance_ICML(np.zeros(n, dtype=int), C)
            writer.writerow(
                ["pokec_age", G.number_of_nodes(), G.number_of_edges(), color_name, m, k, sigma, alpha, beta,
                 Embeddings, Rounding, '1e-4', '2', "%.4f" % result['ncut'],
                 "%.8f" % result['conti_ncut'],
                 "%.8f" % result.get('f_Larg', -1),
                 "%.4f" % balance_ICML_avg, "%.4f" % balance_ICML_mini, "%.4f" % balance_ICML_dataset,
                 "%.4f" % result['balance_avg'], "%.4f" % result['balance_min'], "%.4f" % fair_prop_vio_avg,
                 "%.4f" % fair_prop_vio_maxi, result.get('num_of_reassignment', -1), "%.4f" % result['time1'],
                 "%.4f" % result['time2'], "%.4f" % result['time']])
        output.flush()
del edgelist, G, colors
gc.collect()

# pokec_sex
edgelist = pd.read_csv('./datasets/processed/pokec_sex/pokec_sex_edges.csv')
G = nx.from_pandas_edgelist(edgelist, source='source', target='target', edge_attr=None, create_using=nx.Graph)
colors = pd.read_csv('./datasets/processed/pokec_sex/pokec_sex_colors.csv', header=None)
colors.columns = ['id', 'sex']
color_name = 'sex'
color = colors[color_name].values
num_runs = 3

for k in [5]:
    for sigma in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:
        m = color.max() + 1
        n = G.number_of_nodes()
        C = sp.coo_matrix(([1] * n, (range(n), color)), dtype=int).toarray()
        ratio = C.sum(axis=0) / n
        if sigma == 1:
            beta = np.zeros(m)
            alpha = np.ones(m)
        else:
            beta = ratio * (1 - sigma)
            alpha = ratio / (1 - sigma)
        for run in range(num_runs):
            phi, result = graph_clustering(G=G, k=k, Embeddings=Embeddings, Rounding=Rounding,
                                           C=C, alpha=alpha, beta=beta, color=color, seed=run,
                                           mu=1e-4, pho=2, xtol=1e-3, ftol=1e-6, gtol=1e-3,
                                           omxitr=100, mxitr=2000, rmaxiter=10)
            print(result)
            fair_prop_vio_avg, fair_prop_vio_maxi = utils.get_fair_prop_vio(phi, C, alpha, beta)
            balance_ICML_avg, balance_ICML_mini = utils.get_balance_ICML(phi, C)
            balance_ICML_dataset0, balance_ICML_dataset = utils.get_balance_ICML(np.zeros(n, dtype=int), C)
            writer.writerow(
                ["pokec_sex", G.number_of_nodes(), G.number_of_edges(), color_name, m, k, sigma, alpha, beta,
                 Embeddings, Rounding, '1e-4', '2', "%.4f" % result['ncut'],
                 "%.8f" % result['conti_ncut'],
                 "%.8f" % result.get('f_Larg', -1),
                 "%.4f" % balance_ICML_avg, "%.4f" % balance_ICML_mini, "%.4f" % balance_ICML_dataset,
                 "%.4f" % result['balance_avg'], "%.4f" % result['balance_min'], "%.4f" % fair_prop_vio_avg,
                 "%.4f" % fair_prop_vio_maxi, result.get('num_of_reassignment', -1), "%.4f" % result['time1'],
                 "%.4f" % result['time2'], "%.4f" % result['time']])
        output.flush()
del edgelist, G, colors
gc.collect()

'''
others, vary sigma to compute alpha/beta-proportion violation.
'''
Embeddings_list = ['spectral', 'fair_equality']
Rounding = 'kmeans'
# facebookNet
edgelist = pd.read_csv('./datasets/processed/facebookNet/facebookNet_edges.csv')
G = nx.from_pandas_edgelist(edgelist, source='source', target='target', edge_attr=None, create_using=nx.Graph)
colors = pd.read_csv('./datasets/processed/facebookNet/facebookNet_colors.csv', header=None)
colors.columns = ['id', 'sex']
color_name = 'sex'
color = colors[color_name].values
num_runs = 10
m = color.max() + 1
n = G.number_of_nodes()
C = sp.coo_matrix(([1] * n, (range(n), color)), dtype=int).toarray()
ratio = C.sum(axis=0) / n

for k in [5]:
    for Embeddings in Embeddings_list:
        for run in range(num_runs):
            phi, result = graph_clustering(G=G, k=k, Embeddings=Embeddings, Rounding=Rounding, C=C, color=color,
                                           seed=run)
            print(result)
            for sigma in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:
                if sigma == 1:
                    beta = np.zeros(m)
                    alpha = np.ones(m)
                else:
                    beta = ratio * (1 - sigma)
                    alpha = ratio / (1 - sigma)
                fair_prop_vio_avg, fair_prop_vio_maxi = utils.get_fair_prop_vio(phi, C, alpha, beta)
                balance_ICML_avg, balance_ICML_mini = utils.get_balance_ICML(phi, C)
                balance_ICML_dataset0, balance_ICML_dataset = utils.get_balance_ICML(np.zeros(n, dtype=int), C)
                writer.writerow(
                    ["facebookNet", G.number_of_nodes(), G.number_of_edges(), color_name, m, k, sigma, alpha, beta,
                     Embeddings, Rounding, '-', '-', "%.4f" % result['ncut'], "%.8f" % result['conti_ncut'],
                     "%.8f" % result.get('f_Larg', -1), "%.4f" % balance_ICML_avg, "%.4f" % balance_ICML_mini,
                     "%.4f" % balance_ICML_dataset,
                     "%.4f" % result['balance_avg'], "%.4f" % result['balance_min'], "%.4f" % fair_prop_vio_avg,
                     "%.4f" % fair_prop_vio_maxi, result.get('num_of_reassignment', -1), "%.4f" % result['time1'],
                     "%.4f" % result['time2'], "%.4f" % result['time']])
            output.flush()
del edgelist, G, colors, C
gc.collect()

# german
edgelist = pd.read_csv('./datasets/processed/german/german_edges.csv')
G = nx.from_pandas_edgelist(edgelist, source='source', target='target', edge_attr=None, create_using=nx.Graph)
colors = pd.read_csv('./datasets/processed/german/german_colors.csv', header=None)
colors.columns = ['id', 'sex']
color_name = 'sex'
color = colors[color_name].values
num_runs = 10
m = color.max() + 1
n = G.number_of_nodes()
C = sp.coo_matrix(([1] * n, (range(n), color)), dtype=int).toarray()
ratio = C.sum(axis=0) / n

for k in [5]:
    for Embeddings in Embeddings_list:
        for run in range(num_runs):
            phi, result = graph_clustering(G=G, k=k, Embeddings=Embeddings, Rounding=Rounding, C=C, color=color,
                                           seed=run)
            print(result)
            for sigma in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:
                if sigma == 1:
                    beta = np.zeros(m)
                    alpha = np.ones(m)
                else:
                    beta = ratio * (1 - sigma)
                    alpha = ratio / (1 - sigma)
                fair_prop_vio_avg, fair_prop_vio_maxi = utils.get_fair_prop_vio(phi, C, alpha, beta)
                balance_ICML_avg, balance_ICML_mini = utils.get_balance_ICML(phi, C)
                balance_ICML_dataset0, balance_ICML_dataset = utils.get_balance_ICML(np.zeros(n, dtype=int), C)
                writer.writerow(
                    ["german", G.number_of_nodes(), G.number_of_edges(), color_name, m, k, sigma, alpha, beta,
                     Embeddings, Rounding, '-', '-', "%.4f" % result['ncut'], "%.8f" % result['conti_ncut'],
                     "%.8f" % result.get('f_Larg', -1), "%.4f" % balance_ICML_avg, "%.4f" % balance_ICML_mini,
                     "%.4f" % balance_ICML_dataset,
                     "%.4f" % result['balance_avg'], "%.4f" % result['balance_min'], "%.4f" % fair_prop_vio_avg,
                     "%.4f" % fair_prop_vio_maxi, result.get('num_of_reassignment', -1), "%.4f" % result['time1'],
                     "%.4f" % result['time2'], "%.4f" % result['time']])
            output.flush()
del edgelist, G, colors, C
gc.collect()

# SBM_1000
edgelist = pd.read_csv(f'datasets/processed/SBM_1000/SBM_1000_edges.csv')
G = nx.from_pandas_edgelist(edgelist, source='source', target='target', edge_attr=None, create_using=nx.Graph)
colors = pd.read_csv(f'datasets/processed/SBM_1000/SBM_1000_colors.csv', header=None)
colors.columns = ['id', '5']
color_name = '5'
color = colors[color_name].values
num_runs = 10
m = color.max() + 1
n = G.number_of_nodes()
C = sp.coo_matrix(([1] * n, (range(n), color)), dtype=int).toarray()
ratio = C.sum(axis=0) / n

for k in [5]:
    for Embeddings in Embeddings_list:
        for run in range(num_runs):
            phi, result = graph_clustering(G=G, k=k, Embeddings=Embeddings, Rounding=Rounding, C=C, color=color,
                                           seed=run)
            print(result)
            for sigma in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:
                if sigma == 1:
                    beta = np.zeros(m)
                    alpha = np.ones(m)
                else:
                    beta = ratio * (1 - sigma)
                    alpha = ratio / (1 - sigma)
                fair_prop_vio_avg, fair_prop_vio_maxi = utils.get_fair_prop_vio(phi, C, alpha, beta)
                balance_ICML_avg, balance_ICML_mini = utils.get_balance_ICML(phi, C)
                balance_ICML_dataset0, balance_ICML_dataset = utils.get_balance_ICML(np.zeros(n, dtype=int), C)
                writer.writerow(
                    ["SBM_1000", G.number_of_nodes(), G.number_of_edges(), color_name, m, k, sigma, alpha, beta,
                     Embeddings, Rounding, '-', '-', "%.4f" % result['ncut'], "%.8f" % result['conti_ncut'],
                     "%.8f" % result.get('f_Larg', -1), "%.4f" % balance_ICML_avg, "%.4f" % balance_ICML_mini,
                     "%.4f" % balance_ICML_dataset,
                     "%.4f" % result['balance_avg'], "%.4f" % result['balance_min'], "%.4f" % fair_prop_vio_avg,
                     "%.4f" % fair_prop_vio_maxi, result.get('num_of_reassignment', -1), "%.4f" % result['time1'],
                     "%.4f" % result['time2'], "%.4f" % result['time']])
            output.flush()
del edgelist, G, colors, C
gc.collect()

# dblp
edgelist = pd.read_csv('./datasets/processed/dblp/dblp_edges.csv')
G = nx.from_pandas_edgelist(edgelist, source='source', target='target', edge_attr=None, create_using=nx.Graph)
colors = pd.read_csv('./datasets/processed/dblp/dblp_colors.csv', header=None)
colors.columns = ['id', 'continent']
color_name = 'continent'
color = colors[color_name].values
num_runs = 10
m = color.max() + 1
n = G.number_of_nodes()
C = sp.coo_matrix(([1] * n, (range(n), color)), dtype=int).toarray()
ratio = C.sum(axis=0) / n

for k in [5]:
    for Embeddings in Embeddings_list:
        for run in range(num_runs):
            phi, result = graph_clustering(G=G, k=k, Embeddings=Embeddings, Rounding=Rounding, C=C, color=color,
                                           seed=run)
            print(result)
            for sigma in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:
                if sigma == 1:
                    beta = np.zeros(m)
                    alpha = np.ones(m)
                else:
                    beta = ratio * (1 - sigma)
                    alpha = ratio / (1 - sigma)
                fair_prop_vio_avg, fair_prop_vio_maxi = utils.get_fair_prop_vio(phi, C, alpha, beta)
                balance_ICML_avg, balance_ICML_mini = utils.get_balance_ICML(phi, C)
                balance_ICML_dataset0, balance_ICML_dataset = utils.get_balance_ICML(np.zeros(n, dtype=int), C)
                writer.writerow(
                    ["dblp", G.number_of_nodes(), G.number_of_edges(), color_name, m, k, sigma, alpha, beta,
                     Embeddings, Rounding, '-', '-', "%.4f" % result['ncut'], "%.8f" % result['conti_ncut'],
                     "%.8f" % result.get('f_Larg', -1), "%.4f" % balance_ICML_avg, "%.4f" % balance_ICML_mini,
                     "%.4f" % balance_ICML_dataset,
                     "%.4f" % result['balance_avg'], "%.4f" % result['balance_min'], "%.4f" % fair_prop_vio_avg,
                     "%.4f" % fair_prop_vio_maxi, result.get('num_of_reassignment', -1), "%.4f" % result['time1'],
                     "%.4f" % result['time2'], "%.4f" % result['time']])
            output.flush()
del edgelist, G, colors, C
gc.collect()

# lastfm
edgelist = pd.read_csv('./datasets/processed/lastfm/lastfm_edges.csv')
G = nx.from_pandas_edgelist(edgelist, source='source', target='target', edge_attr=None, create_using=nx.Graph)
colors = pd.read_csv('./datasets/processed/lastfm/lastfm_colors.csv', header=None)
colors.columns = ['id', 'country']
color_name = 'country'
color = colors[color_name].values
num_runs = 10
m = color.max() + 1
n = G.number_of_nodes()
C = sp.coo_matrix(([1] * n, (range(n), color)), dtype=int).toarray()
ratio = C.sum(axis=0) / n

for k in [5]:
    for Embeddings in Embeddings_list:
        for run in range(num_runs):
            phi, result = graph_clustering(G=G, k=k, Embeddings=Embeddings, Rounding=Rounding, C=C, color=color,
                                           seed=run)
            print(result)
            for sigma in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:
                if sigma == 1:
                    beta = np.zeros(m)
                    alpha = np.ones(m)
                else:
                    beta = ratio * (1 - sigma)
                    alpha = ratio / (1 - sigma)
                fair_prop_vio_avg, fair_prop_vio_maxi = utils.get_fair_prop_vio(phi, C, alpha, beta)
                balance_ICML_avg, balance_ICML_mini = utils.get_balance_ICML(phi, C)
                balance_ICML_dataset0, balance_ICML_dataset = utils.get_balance_ICML(np.zeros(n, dtype=int), C)
                writer.writerow(
                    ["lastfm", G.number_of_nodes(), G.number_of_edges(), color_name, m, k, sigma, alpha, beta,
                     Embeddings, Rounding, '-', '-', "%.4f" % result['ncut'], "%.8f" % result['conti_ncut'],
                     "%.8f" % result.get('f_Larg', -1), "%.4f" % balance_ICML_avg, "%.4f" % balance_ICML_mini,
                     "%.4f" % balance_ICML_dataset,
                     "%.4f" % result['balance_avg'], "%.4f" % result['balance_min'], "%.4f" % fair_prop_vio_avg,
                     "%.4f" % fair_prop_vio_maxi, result.get('num_of_reassignment', -1), "%.4f" % result['time1'],
                     "%.4f" % result['time2'], "%.4f" % result['time']])
            output.flush()
del edgelist, G, colors, C
gc.collect()

Embeddings_list = ['spectral']
Rounding = 'kmeans'

# credit_education
edgelist = pd.read_csv('./datasets/processed/credit_education/credit_education_edges.csv')
G = nx.from_pandas_edgelist(edgelist, source='source', target='target', edge_attr=None, create_using=nx.Graph)
colors = pd.read_csv('./datasets/processed/credit_education/credit_education_colors.csv', header=None)
colors.columns = ['id', 'education']
color_name = 'education'
color = colors[color_name].values
num_runs = 10
m = color.max() + 1
n = G.number_of_nodes()
C = sp.coo_matrix(([1] * n, (range(n), color)), dtype=int).toarray()
ratio = C.sum(axis=0) / n

for k in [5]:
    for Embeddings in Embeddings_list:
        for run in range(num_runs):
            phi, result = graph_clustering(G=G, k=k, Embeddings=Embeddings, Rounding=Rounding, C=C, color=color,
                                           seed=run)
            print(result)
            for sigma in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:
                if sigma == 1:
                    beta = np.zeros(m)
                    alpha = np.ones(m)
                else:
                    beta = ratio * (1 - sigma)
                    alpha = ratio / (1 - sigma)
                fair_prop_vio_avg, fair_prop_vio_maxi = utils.get_fair_prop_vio(phi, C, alpha, beta)
                balance_ICML_avg, balance_ICML_mini = utils.get_balance_ICML(phi, C)
                balance_ICML_dataset0, balance_ICML_dataset = utils.get_balance_ICML(np.zeros(n, dtype=int), C)
                writer.writerow(
                    ["credit_education", G.number_of_nodes(), G.number_of_edges(), color_name, m, k, sigma, alpha, beta,
                     Embeddings, Rounding, '-', '-', "%.4f" % result['ncut'], "%.8f" % result['conti_ncut'],
                     "%.8f" % result.get('f_Larg', -1), "%.4f" % balance_ICML_avg, "%.4f" % balance_ICML_mini,
                     "%.4f" % balance_ICML_dataset,
                     "%.4f" % result['balance_avg'], "%.4f" % result['balance_min'], "%.4f" % fair_prop_vio_avg,
                     "%.4f" % fair_prop_vio_maxi, result.get('num_of_reassignment', -1), "%.4f" % result['time1'],
                     "%.4f" % result['time2'], "%.4f" % result['time']])
            output.flush()
del edgelist, G, colors, C
gc.collect()

# deezer
edgelist = pd.read_csv('./datasets/processed/deezer/deezer_edges.csv')
G = nx.from_pandas_edgelist(edgelist, source='source', target='target', edge_attr=None, create_using=nx.Graph)
colors = pd.read_csv('./datasets/processed/deezer/deezer_colors.csv', header=None)
colors.columns = ['id', 'sex']
color_name = 'sex'
color = colors[color_name].values
num_runs = 10
m = color.max() + 1
n = G.number_of_nodes()
C = sp.coo_matrix(([1] * n, (range(n), color)), dtype=int).toarray()
ratio = C.sum(axis=0) / n

for k in [5]:
    for Embeddings in Embeddings_list:
        for run in range(num_runs):
            phi, result = graph_clustering(G=G, k=k, Embeddings=Embeddings, Rounding=Rounding, C=C, color=color,
                                           seed=run)
            print(result)
            for sigma in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:
                if sigma == 1:
                    beta = np.zeros(m)
                    alpha = np.ones(m)
                else:
                    beta = ratio * (1 - sigma)
                    alpha = ratio / (1 - sigma)
                fair_prop_vio_avg, fair_prop_vio_maxi = utils.get_fair_prop_vio(phi, C, alpha, beta)
                balance_ICML_avg, balance_ICML_mini = utils.get_balance_ICML(phi, C)
                balance_ICML_dataset0, balance_ICML_dataset = utils.get_balance_ICML(np.zeros(n, dtype=int), C)
                writer.writerow(
                    ["deezer", G.number_of_nodes(), G.number_of_edges(), color_name, m, k, sigma, alpha, beta,
                     Embeddings, Rounding, '-', '-', "%.4f" % result['ncut'], "%.8f" % result['conti_ncut'],
                     "%.8f" % result.get('f_Larg', -1), "%.4f" % balance_ICML_avg, "%.4f" % balance_ICML_mini,
                     "%.4f" % balance_ICML_dataset,
                     "%.4f" % result['balance_avg'], "%.4f" % result['balance_min'], "%.4f" % fair_prop_vio_avg,
                     "%.4f" % fair_prop_vio_maxi, result.get('num_of_reassignment', -1), "%.4f" % result['time1'],
                     "%.4f" % result['time2'], "%.4f" % result['time']])
            output.flush()
del edgelist, G, colors, C
gc.collect()

# pokec_age
edgelist = pd.read_csv('./datasets/processed/pokec_age/pokec_age_edges.csv')
G = nx.from_pandas_edgelist(edgelist, source='source', target='target', edge_attr=None, create_using=nx.Graph)
colors = pd.read_csv('./datasets/processed/pokec_age/pokec_age_colors.csv', header=None)
colors.columns = ['id', 'age']
color_name = 'age'
color = colors[color_name].values
num_runs = 3
m = color.max() + 1
n = G.number_of_nodes()
C = sp.coo_matrix(([1] * n, (range(n), color)), dtype=int).toarray()
ratio = C.sum(axis=0) / n

for k in [5]:
    for Embeddings in Embeddings_list:
        for run in range(num_runs):
            phi, result = graph_clustering(G=G, k=k, Embeddings=Embeddings, Rounding=Rounding, C=C, color=color,
                                           seed=run)
            print(result)
            for sigma in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:
                if sigma == 1:
                    beta = np.zeros(m)
                    alpha = np.ones(m)
                else:
                    beta = ratio * (1 - sigma)
                    alpha = ratio / (1 - sigma)
                fair_prop_vio_avg, fair_prop_vio_maxi = utils.get_fair_prop_vio(phi, C, alpha, beta)
                balance_ICML_avg, balance_ICML_mini = utils.get_balance_ICML(phi, C)
                balance_ICML_dataset0, balance_ICML_dataset = utils.get_balance_ICML(np.zeros(n, dtype=int), C)
                writer.writerow(
                    ["pokec_age", G.number_of_nodes(), G.number_of_edges(), color_name, m, k, sigma, alpha, beta,
                     Embeddings, Rounding, '-', '-', "%.4f" % result['ncut'], "%.8f" % result['conti_ncut'],
                     "%.8f" % result.get('f_Larg', -1), "%.4f" % balance_ICML_avg, "%.4f" % balance_ICML_mini,
                     "%.4f" % balance_ICML_dataset,
                     "%.4f" % result['balance_avg'], "%.4f" % result['balance_min'], "%.4f" % fair_prop_vio_avg,
                     "%.4f" % fair_prop_vio_maxi, result.get('num_of_reassignment', -1), "%.4f" % result['time1'],
                     "%.4f" % result['time2'], "%.4f" % result['time']])
            output.flush()
del edgelist, G, colors, C
gc.collect()

# pokec_sex
edgelist = pd.read_csv('./datasets/processed/pokec_sex/pokec_sex_edges.csv')
G = nx.from_pandas_edgelist(edgelist, source='source', target='target', edge_attr=None, create_using=nx.Graph)
colors = pd.read_csv('./datasets/processed/pokec_sex/pokec_sex_colors.csv', header=None)
colors.columns = ['id', 'sex']
color_name = 'sex'
color = colors[color_name].values
num_runs = 3
m = color.max() + 1
n = G.number_of_nodes()
C = sp.coo_matrix(([1] * n, (range(n), color)), dtype=int).toarray()
ratio = C.sum(axis=0) / n

for k in [5]:
    for Embeddings in Embeddings_list:
        for run in range(num_runs):
            phi, result = graph_clustering(G=G, k=k, Embeddings=Embeddings, Rounding=Rounding, C=C, color=color,
                                           seed=run)
            print(result)
            for sigma in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:
                if sigma == 1:
                    beta = np.zeros(m)
                    alpha = np.ones(m)
                else:
                    beta = ratio * (1 - sigma)
                    alpha = ratio / (1 - sigma)
                fair_prop_vio_avg, fair_prop_vio_maxi = utils.get_fair_prop_vio(phi, C, alpha, beta)
                balance_ICML_avg, balance_ICML_mini = utils.get_balance_ICML(phi, C)
                balance_ICML_dataset0, balance_ICML_dataset = utils.get_balance_ICML(np.zeros(n, dtype=int), C)
                writer.writerow(
                    ["pokec_sex", G.number_of_nodes(), G.number_of_edges(), color_name, m, k, sigma, alpha, beta,
                     Embeddings, Rounding, '-', '-', "%.4f" % result['ncut'], "%.8f" % result['conti_ncut'],
                     "%.8f" % result.get('f_Larg', -1), "%.4f" % balance_ICML_avg, "%.4f" % balance_ICML_mini,
                     "%.4f" % balance_ICML_dataset,
                     "%.4f" % result['balance_avg'], "%.4f" % result['balance_min'], "%.4f" % fair_prop_vio_avg,
                     "%.4f" % fair_prop_vio_maxi, result.get('num_of_reassignment', -1), "%.4f" % result['time1'],
                     "%.4f" % result['time2'], "%.4f" % result['time']])
            output.flush()
del edgelist, G, colors, C
gc.collect()

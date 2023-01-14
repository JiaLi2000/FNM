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

'''
grid k
'''
output = open("./results/grid_k.csv", "w")
writer = csv.writer(output)
writer.writerow(["dataset", "#V", "#E", "group", "m", "k", "sigma", 'alpha', 'beta', "Embeddings", "Rounding",
                 'mu', 'pho', "ncut", 'conti_ncut', "f_Larg", 'balance_ICML_avg', 'balance_ICML_mini',
                 'balance_ICML_datasets', 'balance_NIPS_avg', 'balance_NIPS_mini',
                 "fair_prop_vio_avg", "fair_prop_vio_maxi", "#reassignment", "time1", "time2", "time"])

# facebookNet
edgelist = pd.read_csv('./datasets/processed/facebookNet/facebookNet_edges.csv')
G = nx.from_pandas_edgelist(edgelist, source='source', target='target', edge_attr=None, create_using=nx.Graph)
colors = pd.read_csv('./datasets/processed/facebookNet/facebookNet_colors.csv', header=None)
colors.columns = ['id', 'sex']
color_name = 'sex'
for k in range(2, 11, 1):
    for sigma in [0.2, 0.5, 0.8]:
        color = colors[color_name].values
        m = color.max() + 1
        n = G.number_of_nodes()
        C = sp.coo_matrix(([1] * n, (range(n), color)), dtype=int).toarray()
        ratio = C.sum(axis=0) / n
        beta = ratio * (1 - sigma)
        alpha = ratio / (1 - sigma)
        if sigma == 1:
            beta = np.zeros_like(beta)
            alpha = np.ones_like(alpha)
        for pho in [2, 4, 6, 8, 10]:
            for mu in [1e-4, 1e-2, 1, 100]:
                phi, result = graph_clustering(G=G, k=k, Embeddings='fair_proportion', Rounding='fair_assign_max',
                                               C=C, alpha=alpha, beta=beta, color=color, seed=1,
                                               mu=mu, pho=pho, xtol=1e-6, ftol=1e-9, gtol=1e-6,
                                               omxitr=100, mxitr=2000, rmaxiter=10)
                print(result)
                fair_prop_vio_avg, fair_prop_vio_maxi = utils.get_fair_prop_vio(phi, C, alpha, beta)
                balance_ICML_avg, balance_ICML_mini = utils.get_balance_ICML(phi, C)
                balance_ICML_dataset0, balance_ICML_dataset = utils.get_balance_ICML(np.zeros(n, dtype=int), C)
                writer.writerow(
                    ["facebookNet", G.number_of_nodes(), G.number_of_edges(), color_name, m, k, sigma, alpha, beta,
                     'fair_proportion', 'fair_assign_max', mu, pho, "%.4f" % result['ncut'],
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
for k in range(2, 11, 1):
    for sigma in [0.2, 0.5, 0.8]:
        color = colors[color_name].values
        m = color.max() + 1
        n = G.number_of_nodes()
        C = sp.coo_matrix(([1] * n, (range(n), color)), dtype=int).toarray()
        ratio = C.sum(axis=0) / n
        beta = ratio * (1 - sigma)
        alpha = ratio / (1 - sigma)
        if sigma == 1:
            beta = np.zeros_like(beta)
            alpha = np.ones_like(alpha)
        for pho in [2, 4, 6, 8, 10]:
            for mu in [1e-4, 1e-2, 1, 100]:
                phi, result = graph_clustering(G=G, k=k, Embeddings='fair_proportion', Rounding='fair_assign_max',
                                               C=C, alpha=alpha, beta=beta, color=color, seed=1,
                                               mu=mu, pho=pho, xtol=1e-6, ftol=1e-9, gtol=1e-6,
                                               omxitr=100, mxitr=2000, rmaxiter=10)
                print(result)
                fair_prop_vio_avg, fair_prop_vio_maxi = utils.get_fair_prop_vio(phi, C, alpha, beta)
                balance_ICML_avg, balance_ICML_mini = utils.get_balance_ICML(phi, C)
                balance_ICML_dataset0, balance_ICML_dataset = utils.get_balance_ICML(np.zeros(n, dtype=int), C)
                writer.writerow(
                    ["german", G.number_of_nodes(), G.number_of_edges(), color_name, m, k, sigma, alpha, beta,
                     'fair_proportion', 'fair_assign_max', mu, pho, "%.4f" % result['ncut'],
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
m = color.max() + 1
n = G.number_of_nodes()
C = sp.coo_matrix(([1] * n, (range(n), color)), dtype=int).toarray()
ratio = C.sum(axis=0) / n
for k in range(2, 21, 2):
    for sigma in [0.2, 0.5, 0.8]:
        beta = ratio * (1 - sigma)
        alpha = ratio / (1 - sigma)
        if sigma == 1:
            beta = np.zeros_like(beta)
            alpha = np.ones_like(alpha)
        for pho in [2, 4, 6, 8, 10]:
            for mu in [1e-4, 1e-2, 1, 100]:
                phi, result = graph_clustering(G=G, k=k, Embeddings='fair_proportion', Rounding='fair_assign_max',
                                               C=C, alpha=alpha, beta=beta, color=color, seed=1,
                                               mu=mu, pho=pho, xtol=1e-6, ftol=1e-9, gtol=1e-6,
                                               omxitr=100, mxitr=2000, rmaxiter=10)
                print(result)
                fair_prop_vio_avg, fair_prop_vio_maxi = utils.get_fair_prop_vio(phi, C, alpha, beta)
                balance_ICML_avg, balance_ICML_mini = utils.get_balance_ICML(phi, C)
                balance_ICML_dataset0, balance_ICML_dataset = utils.get_balance_ICML(np.zeros(n, dtype=int), C)
                writer.writerow(
                    ["SBM_1000", G.number_of_nodes(), G.number_of_edges(), color_name, m, k, sigma, alpha, beta,
                     'fair_proportion', 'fair_assign_max', mu, pho, "%.4f" % result['ncut'],
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
for k in range(2, 11, 1):
    for sigma in [0.2, 0.5, 0.8]:
        color = colors[color_name].values
        m = color.max() + 1
        n = G.number_of_nodes()
        C = sp.coo_matrix(([1] * n, (range(n), color)), dtype=int).toarray()
        ratio = C.sum(axis=0) / n
        beta = ratio * (1 - sigma)
        alpha = ratio / (1 - sigma)
        if sigma == 1:
            beta = np.zeros_like(beta)
            alpha = np.ones_like(alpha)
        for pho in [2, 4, 6, 8, 10]:
            for mu in [1e-4, 1e-2, 1, 100]:
                phi, result = graph_clustering(G=G, k=k, Embeddings='fair_proportion', Rounding='fair_assign_max',
                                               C=C, alpha=alpha, beta=beta, color=color, seed=1,
                                               mu=mu, pho=pho, xtol=1e-6, ftol=1e-9, gtol=1e-6,
                                               omxitr=100, mxitr=2000, rmaxiter=10)
                print(result)
                fair_prop_vio_avg, fair_prop_vio_maxi = utils.get_fair_prop_vio(phi, C, alpha, beta)
                balance_ICML_avg, balance_ICML_mini = utils.get_balance_ICML(phi, C)
                balance_ICML_dataset0, balance_ICML_dataset = utils.get_balance_ICML(np.zeros(n, dtype=int), C)
                writer.writerow(
                    ["dblp", G.number_of_nodes(), G.number_of_edges(), color_name, m, k, sigma, alpha, beta,
                     'fair_proportion', 'fair_assign_max', mu, pho, "%.4f" % result['ncut'],
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
for k in range(5, 51, 5):
    for sigma in [0.2, 0.5, 0.8]:
        color = colors[color_name].values
        m = color.max() + 1
        n = G.number_of_nodes()
        C = sp.coo_matrix(([1] * n, (range(n), color)), dtype=int).toarray()
        ratio = C.sum(axis=0) / n
        beta = ratio * (1 - sigma)
        alpha = ratio / (1 - sigma)
        if sigma == 1:
            beta = np.zeros_like(beta)
            alpha = np.ones_like(alpha)
        for pho in [2, 4, 6, 8, 10]:
            for mu in [1e-4, 1e-2, 1]:
                phi, result = graph_clustering(G=G, k=k, Embeddings='fair_proportion', Rounding='fair_assign_max',
                                               C=C, alpha=alpha, beta=beta, color=color, seed=1,
                                               mu=mu, pho=pho, xtol=1e-6, ftol=1e-9, gtol=1e-6,
                                               omxitr=100, mxitr=2000, rmaxiter=10)
                print(result)
                fair_prop_vio_avg, fair_prop_vio_maxi = utils.get_fair_prop_vio(phi, C, alpha, beta)
                balance_ICML_avg, balance_ICML_mini = utils.get_balance_ICML(phi, C)
                balance_ICML_dataset0, balance_ICML_dataset = utils.get_balance_ICML(np.zeros(n, dtype=int), C)
                writer.writerow(
                    ["lastfm", G.number_of_nodes(), G.number_of_edges(), color_name, m, k, sigma, alpha, beta,
                     'fair_proportion', 'fair_assign_max', mu, pho, "%.4f" % result['ncut'],
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
for k in range(5, 51, 5):
    for sigma in [0.2, 0.5, 0.8]:
        color = colors[color_name].values
        m = color.max() + 1
        n = G.number_of_nodes()
        C = sp.coo_matrix(([1] * n, (range(n), color)), dtype=int).toarray()
        ratio = C.sum(axis=0) / n
        beta = ratio * (1 - sigma)
        alpha = ratio / (1 - sigma)
        if sigma == 1:
            beta = np.zeros_like(beta)
            alpha = np.ones_like(alpha)
        for pho in [2, 6, 10]:
            for mu in [1e-4, 1e-2, 1, 100]:
                phi, result = graph_clustering(G=G, k=k, Embeddings='fair_proportion', Rounding='fair_assign_max',
                                               C=C, alpha=alpha, beta=beta, color=color, seed=1,
                                               mu=mu, pho=pho, xtol=1e-6, ftol=1e-9, gtol=1e-6,
                                               omxitr=100, mxitr=2000, rmaxiter=10)
                print(result)
                fair_prop_vio_avg, fair_prop_vio_maxi = utils.get_fair_prop_vio(phi, C, alpha, beta)
                balance_ICML_avg, balance_ICML_mini = utils.get_balance_ICML(phi, C)
                balance_ICML_dataset0, balance_ICML_dataset = utils.get_balance_ICML(np.zeros(n, dtype=int), C)
                writer.writerow(
                    ["credit_education", G.number_of_nodes(), G.number_of_edges(), color_name, m, k, sigma, alpha, beta,
                     'fair_proportion', 'fair_assign_max', mu, pho, "%.4f" % result['ncut'],
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
for k in range(5, 51, 5):
    for sigma in [0.2, 0.5, 0.8]:
        color = colors[color_name].values
        m = color.max() + 1
        n = G.number_of_nodes()
        C = sp.coo_matrix(([1] * n, (range(n), color)), dtype=int).toarray()
        ratio = C.sum(axis=0) / n
        beta = ratio * (1 - sigma)
        alpha = ratio / (1 - sigma)
        if sigma == 1:
            beta = np.zeros_like(beta)
            alpha = np.ones_like(alpha)
        for pho in [2, 6, 10]:
            for mu in [1e-4, 1e-2, 1, 100]:
                phi, result = graph_clustering(G=G, k=k, Embeddings='fair_proportion', Rounding='fair_assign_max',
                                               C=C, alpha=alpha, beta=beta, color=color, seed=1,
                                               mu=mu, pho=pho, xtol=1e-6, ftol=1e-9, gtol=1e-6,
                                               omxitr=100, mxitr=2000, rmaxiter=10)
                print(result)
                fair_prop_vio_avg, fair_prop_vio_maxi = utils.get_fair_prop_vio(phi, C, alpha, beta)
                balance_ICML_avg, balance_ICML_mini = utils.get_balance_ICML(phi, C)
                balance_ICML_dataset0, balance_ICML_dataset = utils.get_balance_ICML(np.zeros(n, dtype=int), C)
                writer.writerow(
                    ["deezer", G.number_of_nodes(), G.number_of_edges(), color_name, m, k, sigma, alpha, beta,
                     'fair_proportion', 'fair_assign_max', mu, pho, "%.4f" % result['ncut'],
                     "%.8f" % result['conti_ncut'],
                     "%.8f" % result.get('f_Larg', -1),
                     "%.4f" % balance_ICML_avg, "%.4f" % balance_ICML_mini, "%.4f" % balance_ICML_dataset,
                     "%.4f" % result['balance_avg'], "%.4f" % result['balance_min'], "%.4f" % fair_prop_vio_avg,
                     "%.4f" % fair_prop_vio_maxi, result.get('num_of_reassignment', -1), "%.4f" % result['time1'],
                     "%.4f" % result['time2'], "%.4f" % result['time']])
            output.flush()
del edgelist, G, colors
gc.collect()
output.close()

'''
grid sigma
'''

output = open("./results/grid_sigma.csv", "w")
writer = csv.writer(output)
writer.writerow(["dataset", "#V", "#E", "group", "m", "k", "sigma", 'alpha', 'beta', "Embeddings", "Rounding",
                 'mu', 'pho', "ncut", 'conti_ncut', "f_Larg", 'balance_ICML_avg', 'balance_ICML_mini',
                 'balance_ICML_datasets', 'balance_NIPS_avg', 'balance_NIPS_mini',
                 "fair_prop_vio_avg", "fair_prop_vio_maxi", "#reassignment", "time1", "time2", "time"])
# facebookNet
edgelist = pd.read_csv('./datasets/processed/facebookNet/facebookNet_edges.csv')
G = nx.from_pandas_edgelist(edgelist, source='source', target='target', edge_attr=None, create_using=nx.Graph)
colors = pd.read_csv('./datasets/processed/facebookNet/facebookNet_colors.csv', header=None)
colors.columns = ['id', 'sex']
color_name = 'sex'
for k in [5]:
    for sigma in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:
        color = colors[color_name].values
        m = color.max() + 1
        n = G.number_of_nodes()
        C = sp.coo_matrix(([1] * n, (range(n), color)), dtype=int).toarray()
        ratio = C.sum(axis=0) / n
        beta = ratio * (1 - sigma)
        alpha = ratio / (1 - sigma)
        if sigma == 1:
            beta = np.zeros_like(beta)
            alpha = np.ones_like(alpha)
        for pho in [2, 4, 6, 8, 10]:
            for mu in [1e-4, 1e-2, 1, 100]:
                phi, result = graph_clustering(G=G, k=k, Embeddings='fair_proportion', Rounding='fair_assign_max',
                                               C=C, alpha=alpha, beta=beta, color=color, seed=1,
                                               mu=mu, pho=pho, xtol=1e-6, ftol=1e-9, gtol=1e-6,
                                               omxitr=100, mxitr=2000, rmaxiter=10)
                print(result)
                fair_prop_vio_avg, fair_prop_vio_maxi = utils.get_fair_prop_vio(phi, C, alpha, beta)
                balance_ICML_avg, balance_ICML_mini = utils.get_balance_ICML(phi, C)
                balance_ICML_dataset0, balance_ICML_dataset = utils.get_balance_ICML(np.zeros(n, dtype=int), C)
                writer.writerow(
                    ["facebookNet", G.number_of_nodes(), G.number_of_edges(), color_name, m, k, sigma, alpha, beta,
                     'fair_proportion', 'fair_assign_max', mu, pho, "%.4f" % result['ncut'],
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
for k in [5]:
    for sigma in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:
        color = colors[color_name].values
        m = color.max() + 1
        n = G.number_of_nodes()
        C = sp.coo_matrix(([1] * n, (range(n), color)), dtype=int).toarray()
        ratio = C.sum(axis=0) / n
        beta = ratio * (1 - sigma)
        alpha = ratio / (1 - sigma)
        if sigma == 1:
            beta = np.zeros_like(beta)
            alpha = np.ones_like(alpha)
        for pho in [2, 4, 6, 8, 10]:
            for mu in [1e-4, 1e-2, 1, 100]:
                phi, result = graph_clustering(G=G, k=k, Embeddings='fair_proportion', Rounding='fair_assign_max',
                                               C=C, alpha=alpha, beta=beta, color=color, seed=1,
                                               mu=mu, pho=pho, xtol=1e-6, ftol=1e-9, gtol=1e-6,
                                               omxitr=100, mxitr=2000, rmaxiter=10)
                print(result)
                fair_prop_vio_avg, fair_prop_vio_maxi = utils.get_fair_prop_vio(phi, C, alpha, beta)
                balance_ICML_avg, balance_ICML_mini = utils.get_balance_ICML(phi, C)
                balance_ICML_dataset0, balance_ICML_dataset = utils.get_balance_ICML(np.zeros(n, dtype=int), C)
                writer.writerow(
                    ["german", G.number_of_nodes(), G.number_of_edges(), color_name, m, k, sigma, alpha, beta,
                     'fair_proportion', 'fair_assign_max', mu, pho, "%.4f" % result['ncut'],
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
for k in [5]:
    for sigma in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:
        color = colors[color_name].values
        m = color.max() + 1
        n = G.number_of_nodes()
        C = sp.coo_matrix(([1] * n, (range(n), color)), dtype=int).toarray()
        ratio = C.sum(axis=0) / n
        beta = ratio * (1 - sigma)
        alpha = ratio / (1 - sigma)
        if sigma == 1:
            beta = np.zeros_like(beta)
            alpha = np.ones_like(alpha)
        for pho in [2, 4, 6, 8, 10]:
            for mu in [1e-4, 1e-2, 1, 100]:
                phi, result = graph_clustering(G=G, k=k, Embeddings='fair_proportion', Rounding='fair_assign_max',
                                               C=C, alpha=alpha, beta=beta, color=color, seed=1,
                                               mu=mu, pho=pho, xtol=1e-6, ftol=1e-9, gtol=1e-6,
                                               omxitr=100, mxitr=2000, rmaxiter=10)
                print(result)
                fair_prop_vio_avg, fair_prop_vio_maxi = utils.get_fair_prop_vio(phi, C, alpha, beta)
                balance_ICML_avg, balance_ICML_mini = utils.get_balance_ICML(phi, C)
                balance_ICML_dataset0, balance_ICML_dataset = utils.get_balance_ICML(np.zeros(n, dtype=int), C)
                writer.writerow(
                    ["SBM_1000", G.number_of_nodes(), G.number_of_edges(), color_name, m, k, sigma, alpha, beta,
                     'fair_proportion', 'fair_assign_max', mu, pho, "%.4f" % result['ncut'],
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
for k in [5]:
    for sigma in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:
        color = colors[color_name].values
        m = color.max() + 1
        n = G.number_of_nodes()
        C = sp.coo_matrix(([1] * n, (range(n), color)), dtype=int).toarray()
        ratio = C.sum(axis=0) / n
        beta = ratio * (1 - sigma)
        alpha = ratio / (1 - sigma)
        if sigma == 1:
            beta = np.zeros_like(beta)
            alpha = np.ones_like(alpha)
        for pho in [2, 4, 6, 8, 10]:
            for mu in [1e-4, 1e-2, 1, 100]:
                phi, result = graph_clustering(G=G, k=k, Embeddings='fair_proportion', Rounding='fair_assign_max',
                                               C=C, alpha=alpha, beta=beta, color=color, seed=1,
                                               mu=mu, pho=pho, xtol=1e-6, ftol=1e-9, gtol=1e-6,
                                               omxitr=100, mxitr=2000, rmaxiter=10)
                print(result)
                fair_prop_vio_avg, fair_prop_vio_maxi = utils.get_fair_prop_vio(phi, C, alpha, beta)
                balance_ICML_avg, balance_ICML_mini = utils.get_balance_ICML(phi, C)
                balance_ICML_dataset0, balance_ICML_dataset = utils.get_balance_ICML(np.zeros(n, dtype=int), C)
                writer.writerow(
                    ["dblp", G.number_of_nodes(), G.number_of_edges(), color_name, m, k, sigma, alpha, beta,
                     'fair_proportion', 'fair_assign_max', mu, pho, "%.4f" % result['ncut'],
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
for k in [5]:
    for sigma in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:
        color = colors[color_name].values
        m = color.max() + 1
        n = G.number_of_nodes()
        C = sp.coo_matrix(([1] * n, (range(n), color)), dtype=int).toarray()
        ratio = C.sum(axis=0) / n
        beta = ratio * (1 - sigma)
        alpha = ratio / (1 - sigma)
        if sigma == 1:
            beta = np.zeros_like(beta)
            alpha = np.ones_like(alpha)
        for pho in [2, 4, 6, 8, 10]:
            for mu in [1e-4, 1e-2, 1]:
                phi, result = graph_clustering(G=G, k=k, Embeddings='fair_proportion', Rounding='fair_assign_max',
                                               C=C, alpha=alpha, beta=beta, color=color, seed=1,
                                               mu=mu, pho=pho, xtol=1e-6, ftol=1e-9, gtol=1e-6,
                                               omxitr=100, mxitr=2000, rmaxiter=10)
                print(result)
                fair_prop_vio_avg, fair_prop_vio_maxi = utils.get_fair_prop_vio(phi, C, alpha, beta)
                balance_ICML_avg, balance_ICML_mini = utils.get_balance_ICML(phi, C)
                balance_ICML_dataset0, balance_ICML_dataset = utils.get_balance_ICML(np.zeros(n, dtype=int), C)
                writer.writerow(
                    ["lastfm", G.number_of_nodes(), G.number_of_edges(), color_name, m, k, sigma, alpha, beta,
                     'fair_proportion', 'fair_assign_max', mu, pho, "%.4f" % result['ncut'],
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
for k in [5]:
    for sigma in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:
        color = colors[color_name].values
        m = color.max() + 1
        n = G.number_of_nodes()
        C = sp.coo_matrix(([1] * n, (range(n), color)), dtype=int).toarray()
        ratio = C.sum(axis=0) / n
        beta = ratio * (1 - sigma)
        alpha = ratio / (1 - sigma)
        if sigma == 1:
            beta = np.zeros_like(beta)
            alpha = np.ones_like(alpha)
        for pho in [2, 6, 10]:
            for mu in [1e-4, 1e-2, 1, 100]:
                phi, result = graph_clustering(G=G, k=k, Embeddings='fair_proportion', Rounding='fair_assign_max',
                                               C=C, alpha=alpha, beta=beta, color=color, seed=1,
                                               mu=mu, pho=pho, xtol=1e-6, ftol=1e-9, gtol=1e-6,
                                               omxitr=100, mxitr=2000, rmaxiter=10)
                print(result)
                fair_prop_vio_avg, fair_prop_vio_maxi = utils.get_fair_prop_vio(phi, C, alpha, beta)
                balance_ICML_avg, balance_ICML_mini = utils.get_balance_ICML(phi, C)
                balance_ICML_dataset0, balance_ICML_dataset = utils.get_balance_ICML(np.zeros(n, dtype=int), C)
                writer.writerow(
                    ["credit_education", G.number_of_nodes(), G.number_of_edges(), color_name, m, k, sigma, alpha, beta,
                     'fair_proportion', 'fair_assign_max', mu, pho, "%.4f" % result['ncut'],
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
for k in [5]:
    for sigma in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:
        color = colors[color_name].values
        m = color.max() + 1
        n = G.number_of_nodes()
        C = sp.coo_matrix(([1] * n, (range(n), color)), dtype=int).toarray()
        ratio = C.sum(axis=0) / n
        beta = ratio * (1 - sigma)
        alpha = ratio / (1 - sigma)
        if sigma == 1:
            beta = np.zeros_like(beta)
            alpha = np.ones_like(alpha)
        for pho in [2, 6, 10]:
            for mu in [1e-4, 1e-2, 1, 100]:
                phi, result = graph_clustering(G=G, k=k, Embeddings='fair_proportion', Rounding='fair_assign_max',
                                               C=C, alpha=alpha, beta=beta, color=color, seed=1,
                                               mu=mu, pho=pho, xtol=1e-6, ftol=1e-9, gtol=1e-6,
                                               omxitr=100, mxitr=2000, rmaxiter=10)
                print(result)
                fair_prop_vio_avg, fair_prop_vio_maxi = utils.get_fair_prop_vio(phi, C, alpha, beta)
                balance_ICML_avg, balance_ICML_mini = utils.get_balance_ICML(phi, C)
                balance_ICML_dataset0, balance_ICML_dataset = utils.get_balance_ICML(np.zeros(n, dtype=int), C)
                writer.writerow(
                    ["deezer", G.number_of_nodes(), G.number_of_edges(), color_name, m, k, sigma, alpha, beta,
                     'fair_proportion', 'fair_assign_max', mu, pho, "%.4f" % result['ncut'],
                     "%.8f" % result['conti_ncut'],
                     "%.8f" % result.get('f_Larg', -1),
                     "%.4f" % balance_ICML_avg, "%.4f" % balance_ICML_mini, "%.4f" % balance_ICML_dataset,
                     "%.4f" % result['balance_avg'], "%.4f" % result['balance_min'], "%.4f" % fair_prop_vio_avg,
                     "%.4f" % fair_prop_vio_maxi, result.get('num_of_reassignment', -1), "%.4f" % result['time1'],
                     "%.4f" % result['time2'], "%.4f" % result['time']])
            output.flush()
del edgelist, G, colors
gc.collect()

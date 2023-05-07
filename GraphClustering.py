from Embeddings import *
from Roundings import *
import networkx as nx
import numpy as np


def graph_clustering(G: nx.graph, k: int, Embeddings: str, Rounding: str, **kwargs) -> (np.ndarray, dict):
    if Embeddings == 'spectral':
        embeddings, conti_ncut, t1 = spectral(G, k)
    elif Embeddings == 'deepwalk':
        embeddings, conti_ncut, t1 = deepwalk(G, k, kwargs['seed'])
    elif Embeddings == 'node2vec':
        embeddings, conti_ncut, t1 = node2vec(G, k, kwargs['seed'])
    elif Embeddings == 'fair_walk':
        embeddings, conti_ncut, t1 = fair_walk(G, k, kwargs['color'], kwargs['seed'])
    elif Embeddings == 'fair_equality':
        embeddings, conti_ncut, t1 = fair_equality(G, k, kwargs['color'])
    elif Embeddings == 'fair_equality_s':
        embeddings, conti_ncut, t1 = fair_equality_s(G, k, kwargs['color'])
    elif Embeddings == 'fair_proportion':
        embeddings, conti_ncut, t1, f_Larg = fair_proportion(G, C=kwargs['C'], k=k, alpha=kwargs['alpha'],
                                                             beta=kwargs['beta'], xtol=kwargs['xtol'],
                                                             gtol=kwargs['gtol'], ftol=kwargs['ftol'],
                                                             pho=kwargs['pho'], mu=kwargs['mu'],
                                                             omxitr=kwargs['omxitr'], mxitr=kwargs['mxitr'])
    else:
        print('No such Embeddings method!')
        sys.exit(-1)

    if Rounding == 'kmeans':
        phi, ncut, t2 = kmeans(embeddings, G, seed=kwargs['seed'])
    elif Rounding == 'fair_assign_exact':
        phi, ncut, t2 = fair_assign_exact(embeddings, G, C=kwargs['C'], alpha=kwargs['alpha'],
                                          beta=kwargs['beta'], seed=kwargs['seed'])
    elif Rounding == 'reassigned_kmeans':
        phi, ncut, t2, num_of_reassignment = reassigned_kmeans(embeddings, G, C=kwargs['C'],
                                                               alpha=kwargs['alpha'], beta=kwargs['beta'],
                                                               seed=kwargs['seed'])
    elif Rounding == 'reassigned_fair_kmeans':
        phi, ncut, t2, num_of_reassignment = reassigned_fair_kmeans(embeddings, G, C=kwargs['C'],
                                                                    alpha=kwargs['alpha'], beta=kwargs['beta'])
    elif Rounding == 'fair_assign_max':
        phi, ncut, t2, num_of_reassignment = fair_assign_max(embeddings, G, C=kwargs['C'],
                                                             alpha=kwargs['alpha'], beta=kwargs['beta'],
                                                             seed=kwargs['seed'],
                                                             rmaxiter=kwargs['rmaxiter'])
    else:
        print('No such Rounding method!')
        sys.exit(-1)
    balance_avg, balance_min = utils.get_balance_NIPS(phi, C=kwargs['C'])
    result = {'ncut': ncut, 'conti_ncut': conti_ncut, 'balance_avg': balance_avg,
              'balance_min': balance_min, 'time1': t1, 'time2': t2, 'time': t1 + t2}
    if Embeddings == 'fair_proportion':
        result['f_Larg'] = f_Larg
    if Rounding in ['reassigned_kmeans', 'reassigned_fair_kmeans', 'fair_assign_max']:
        result['num_of_reassignment'] = num_of_reassignment
    return phi, result

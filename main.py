import config
import numpy as np
import scipy.sparse as sp
from sklearn.preprocessing import normalize
from data.datasets import load_data
import argparse
from sklearn.preprocessing import normalize
import random
from cluster import cluster


def parse_args():
    p = argparse.ArgumentParser(description='Set parameter')
    p.add_argument('--dataset', type=str, default='cora', help='dataset name (e.g.: acm, dblp, imdb)')
    p.add_argument('--datatype', type=str, default='hg', help='type of dataset format (e.g.: hg, han)')
    p.add_argument('--tmax', type=int, default=200, help='t_max parameter')
    p.add_argument('--seeds', type=int, default=0, help='seed for randomness')
    p.add_argument('--alpha', type=float, default=0.2, help='mhc parameter')
    p.add_argument('--beta', type=float, default=0.5, help='weight of knn random walk')
    p.add_argument('--metric', type=bool, default=False, help='calculate additional metrics: modularity')
    p.add_argument('--verbose', action='store_true', help='print verbose logs')
    p.add_argument('--scale', action='store_true', help='use configurations for large-scale data')
    p.add_argument('--interval', type=int, default=5, help='interval between cluster predictions during orthogonal iterations')
    args = p.parse_args()
    config.dataset = args.dataset
    config.datatype = args.datatype
    config.tmax = args.tmax
    config.seeds = args.seeds
    config.alpha = args.alpha
    # config.beta = args.beta
    config.metric = args.metric
    config.verbose = args.verbose
    config.cluster_interval = args.interval
    return args

def run_ahcka(args):
    dataset = load_data(config.dataset)
    features = dataset['features']
    labels = dataset['labels']

    labels = np.asarray(np.argmax(labels, axis=1)).flatten() if labels.ndim == 2 else labels
    config.labels = labels
    k  = len(np.unique(labels))
    config.labels_nk = np.eye(k)[labels]


    seed = config.seeds
    np.random.seed(seed)
    random.seed(seed)

    hg_adjs = dataset['hypergraphs']
    g_adjs = dataset['graphs'] if 'graphs' in dataset else []
    # config.hg_adj = hg_adj
    # config.features = features.copy()
    d_vec = sum([np.asarray(hg_adj.sum(0)).flatten() for hg_adj in hg_adjs] + [np.asarray(g_adj.sum(0)).flatten() for g_adj in g_adjs])
    deg_dict = {i: d_vec[i] for i in range(len(d_vec))}
    # p_mat = (normalize(hg_adj.T, norm='l1', axis=1),  normalize(hg_adj, norm='l1', axis=1))

    results = cluster(hg_adjs, g_adjs, features, k, deg_dict, alpha=config.alpha, view_weights=config.beta[config.dataset], tmax=config.tmax, ri=False)

    return results

if __name__ == '__main__':
    args = parse_args()
    config.dataset = args.dataset
    config.metric = args.metric
    config.tmax = args.tmax
    # config.beta = args.beta
    config.alpha = args.alpha
    config.seeds = args.seeds
    # config.verbose = args.verbose
    config.cluster_interval = args.interval
    if args.scale:
        config.approx_knn = True
        config.init_iter = 1

    run_ahcka(args)

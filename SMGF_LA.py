import resource
import config_simple as config
import numpy as np
import time
import scipy.sparse as sp
import scipy.sparse.linalg as sla
from spectral import discretize
from evaluate import clustering_metrics
from dataset_simple import load_data
from scipy.optimize import minimize
import argparse
from evaluate import ovr_evaluate
from sparse_dot_mkl import dot_product_mkl

def parse_args():
    p = argparse.ArgumentParser(description='Set parameter')
    p.add_argument('--dataset', type=str, default='acm', help='dataset name (e.g.: acm, dblp, imdb)')
    p.add_argument('--embedding', action='store_true', help='run embedding task')
    p.add_argument('--verbose', action='store_true', help='print verbose logs')
    p.add_argument('--knn_k', type=int, default=10, help='k neighbors except imdb=500, yelp=200' )
    args = p.parse_args()
    config.dataset = args.dataset
    config.verbose = args.verbose
    config.embedding = args.embedding
    config.knn_k = args.knn_k
    return args

def SMGF_LA(dataset):
    num_clusters = dataset['k']
    n = dataset['n']
    nv = dataset['nv']
    g_adjs = dataset['graphs']
    features = dataset['features']
    view_weights = np.full(nv, 1.0/nv)
    start_time = time.time()
    knn_adjs = []

    for X in features:    
        import faiss
        if sp.issparse(X):
            X=X.astype(np.float32).tocoo()
            ftd = np.zeros(X.shape, X.dtype)
            ftd[X.row, X.col] = X.data
        else :
            ftd = X.astype(np.float32)
            ftd = np.ascontiguousarray(ftd)
        faiss.normalize_L2(ftd)
        if config.approx_knn:
            index = faiss.index_factory(ftd.shape[1], "IVF1000,PQ40", faiss.METRIC_INNER_PRODUCT)
            index.train(ftd)
        else:
            index = faiss.index_factory(ftd.shape[1], "Flat", faiss.METRIC_INNER_PRODUCT)
        index.add(ftd)
        distances, neighbors = index.search(ftd, config.knn_k+1)
        knn = sp.csr_matrix(((distances.ravel()), neighbors.ravel(), np.arange(0, neighbors.size+1, neighbors.shape[1])), shape=(n, n))
        knn.setdiag(0)
        knn = knn + knn.T
        knn_adjs.append(knn + knn.T)

    if config.verbose:
        print(f'KNN graph construction time: {time.time()-start_time:.2f}s')
    g_dvs = [sp.diags(np.asarray(g_adjs[i].sum(1)).flatten()).tocsr() for i in range(len(g_adjs))]
    knn_dvs = [sp.diags(np.asarray(knn_adjs[i].sum(1)).flatten()).tocsr() for i in range(len(knn_adjs))]
    for dv in [*g_dvs, *knn_dvs]:
        dv.data[dv.data==0] = 1
        dv.data = dv.data**-0.5
    
    # Sparse linear operator for the aggregated Laplacian matrix
    def mv_lap(mat):
        if config.approx_knn:
            product = np.zeros_like(mat)
        else:
            product = np.zeros(mat.shape)
        iv = 0
        for i in range(len(g_adjs)):
            product += view_weights[iv] * g_dvs[i]@dot_product_mkl(g_adjs[i], (g_dvs[i]@mat), cast=True)
            iv += 1
        for i in range(len(knn_adjs)):
            product += view_weights[iv] * knn_dvs[i]@dot_product_mkl(knn_adjs[i], (knn_dvs[i]@mat), cast=True)
            iv += 1
        return mat-product
    lapLO = sla.LinearOperator((n, n), matvec=mv_lap)
    if config.verbose:
        print('Time for constructing linear operator: {:.4f}s'.format(time.time()-start_time))
    
    opt_time = time.time()

    if config.optimize_weights:
        obj_num_eigs = { 'con': 2, 'gap': num_clusters+1, 'combine': num_clusters+1, 'reg':num_clusters+1,'ori': num_clusters+1}
        eig_vec = None
        def eig_obj(w):
            nonlocal eig_vec
            view_weights[:-1] = w
            view_weights[-1] = 1.0 - np.sum(w)
            eig_val, eig_vec = sla.eigsh(lapLO, obj_num_eigs[config.opt_objective], which='SM', tol=config.opt_eig_tol, maxiter=1000)
            eig_val = eig_val.real
            eig_val.sort()
            return eig_val[num_clusters-1] / eig_val[num_clusters] - config.obj_alpha*eig_val[1] + config.obj_regular*np.power(np.asarray(view_weights),2).sum()
        
        w_constraint = [{'type': 'ineq', 'fun': lambda w: 1.0 - np.sum(w) - config.weight_lb}, {'type': 'ineq', 'fun': lambda w: min(w)-config.weight_lb}, {'type': 'ineq', 'fun': lambda w: 1.0-(nv-1)*config.weight_lb-max(w)}]
        opt_w = minimize(eig_obj, np.full((nv-1), 1.0/nv), method='COBYLA', tol=config.opt_w_tol, constraints=w_constraint, options={'maxiter': 1000, 'rhobeg': config.opt_cobyla_rhobeg, 'disp': config.verbose})
    
    elif len(config.fixed_weights) == nv:
        view_weights = config.fixed_weights
        eig_val, eig_vec = sla.eigsh(lapLO, num_clusters+1, which='SM', tol=config.opt_eig_tol, maxiter=1000)

    if config.verbose:
        print(f"opt_time: {time.time()-opt_time}")

    if config.embedding:
        delta=sp.eye(dataset['n'])-mv_lap(sp.eye(dataset['n']))
        if config.approx_knn:
            from sketchne_graph import sketchne_graph
            emb = sketchne_graph(delta, dim = config.embed_dim, spec_propagation=False, window_size=30, eta1=32, eta2=32, eig_rank=256, power_iteration=20)
        else:
            from embedding import netmf
            emb = netmf(delta, dim = config.embed_dim)
        embed_time = time.time() - start_time
        peak_memory_MBs = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0
        emb_results = ovr_evaluate(emb, dataset['labels'])
        print(f"Time: {embed_time:.3f}s RAM: {int(peak_memory_MBs)}MB Weights: {', '.join([f'{w:.2f}' for w in view_weights])}")
    else:
        predict_clusters, _ = discretize(eig_vec[:, :num_clusters])
        cluster_time = time.time() - start_time
        peak_memory_MBs = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0
        cm = clustering_metrics(dataset['labels'], predict_clusters)
        acc, nmi, f1, _, ari, _ = cm.evaluationClusterModelFromLabel()
        print(f"Acc: {acc:.3f} F1: {f1:.3f} NMI: {nmi:.3f} ARI: {ari:.3f} Time: {cluster_time:.3f}s RAM: {int(peak_memory_MBs)}MB Weights: {', '.join([f'{w:.2f}' for w in view_weights])}")

    if config.embedding:
        return list(emb_results) + [embed_time]
    else:
        return [acc, f1, nmi, ari, cluster_time]

if __name__ == '__main__':
    args = parse_args()
    dataset = load_data(config.dataset)
    if config.dataset.startswith("mag"):
        config.approx_knn = True
    SMGF_LA(dataset)



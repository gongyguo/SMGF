import resource
import config_simple as config
import numpy as np
import time
import scipy.sparse as sp
import scipy.sparse.linalg as sla
from sklearn.neighbors import kneighbors_graph
from spectral import discretize
from evaluate import clustering_metrics
from dataset_simple import load_data
from scipy.optimize import minimize
from embedding import netmf
from sketchne_graph import sketchne_graph
import argparse
from sklearn.cluster import KMeans
from evaluate import ovr_evaluate
from sparse_dot_mkl import dot_product_mkl
import random

def parse_args():
    p = argparse.ArgumentParser(description='Set parameter')
    p.add_argument('--dataset', type=str, default='acm', help='dataset name (e.g.: acm, dblp, imdb)')
    p.add_argument('--embedding', action='store_true', help='run embedding task')
    p.add_argument('--verbose', action='store_true', help='print verbose logs')
    p.add_argument('--knn_k', type=int, default=10, help='k neighbors')
    args = p.parse_args()
    config.dataset = args.dataset
    config.verbose = args.verbose
    config.embedding = args.embedding
    config.knn_k = args.knn_k
    return args

def SMGF(dataset):
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
        knn.setdiag(0.0) 
        knn = knn + knn.T
        knn_adjs.append(knn + knn.T)


    print(f'KNN graph construction time: {time.time()-start_time:.2f}s')
    g_dvs = [sp.diags(np.asarray(g_adjs[i].sum(1)).flatten()).tocsr() for i in range(len(g_adjs))]
    knn_dvs = [sp.diags(np.asarray(knn_adjs[i].sum(1)).flatten()).tocsr() for i in range(len(knn_adjs))]
    for dv in [*g_dvs, *knn_dvs]:
        dv.data[dv.data==0] = 1
        dv.data = dv.data**-0.5
    
    # Sparse linear operator for the aggregated Laplacian matrix
    lapnum = optnum = 0
    laps = [sp.identity(n)-g_dvs[i]@dot_product_mkl(g_adjs[i], (g_dvs[i]), cast=True) for i in range(len(g_adjs))] + [sp.identity(n)-knn_dvs[i]@dot_product_mkl(knn_adjs[i], (knn_dvs[i]), cast=True) for i in range(len(knn_adjs))]
    def mv_lap(mat):
        nonlocal lapnum
        lapnum+=1
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
            grads = [-eig_vec[:,1].T@dot_product_mkl((laps[i]-laps[-1]), eig_vec[:,1])+(eig_val[num_clusters]*eig_vec[:,num_clusters-1].T@dot_product_mkl((laps[i]-laps[-1]), eig_vec[:,num_clusters-1])-eig_val[num_clusters-1]*eig_vec[:,num_clusters].T@dot_product_mkl((laps[i]-laps[-1]), eig_vec[:,num_clusters]))/eig_val[num_clusters]**2+config.obj_regular*2*w[i] for i in range(nv-1)]
            return eig_val[num_clusters-1] / eig_val[num_clusters] - config.obj_alpha*eig_val[1] + config.obj_regular*(np.sum(np.power(np.asarray(view_weights),config.po_lambda))), np.array(grads)

        Ir = np.eye(nv-1)
        w_constraint = [{'type': 'ineq', 'fun': lambda w: 1.0 - np.sum(w) - config.weight_lb, 'jac': lambda w: np.full_like(w, -1)}] + [{'type': 'ineq', 'fun': lambda w: min(w)-config.weight_lb, 'jac': lambda w: Ir[i,:]} for i in range(nv-1)]
        opt_w = minimize(eig_obj, np.full((nv-1), 1.0/nv), method='SLSQP', tol=1e-5, constraints=w_constraint, jac=True, options={'maxiter': 1000, 'disp': config.verbose}, bounds=[(0, 1) for i in range(nv-1)])
        view_weights[:-1] = opt_w.x
        view_weights[-1] = 1.0 - np.sum(opt_w.x)

    elif len(config.fixed_weights) == nv:
        view_weights = config.fixed_weights
        eig_val, eig_vec = sla.eigsh(lapLO, num_clusters+1, which='SM', tol=config.opt_eig_tol, maxiter=1000)

    if config.verbose:
        print(f"opt_time: {time.time()-opt_time}")

    if config.embedding:
        delta=sp.eye(dataset['n'])-mv_lap(sp.eye(dataset['n']))
        if config.approx_knn:
            emb = sketchne_graph(delta, dim = config.embed_dim, spec_propagation=False, window_size=30, eta1=32, eta2=32, eig_rank=256, power_iteration=20)
        else:
            emb = netmf(delta, dim = config.embed_dim)
        visual(emb,num_clusters,dataset['labels'],config.dataset)
        embed_time = time.time() - start_time
        peak_memory_MBs = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0
        emb_results = ovr_evaluate(emb, dataset['labels'])
        print(f"Time: {embed_time:.3f}s RAM: {int(peak_memory_MBs)}MB Weights: {', '.join([f'{w:.2f}' for w in view_weights])}")
    else:
        if config.kmeans:
            predict_clusters = KMeans(n_clusters=num_clusters).fit_predict(eig_vec[:, :num_clusters])
        else:
            print(eig_vec[:, :num_clusters].sum())
            predict_clusters, _ = discretize(eig_vec[:, :num_clusters])
        cluster_time = time.time() - start_time
        peak_memory_MBs = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0
        cm = clustering_metrics(dataset['labels'], predict_clusters)
        acc, nmi, f1, _, ari, _ = cm.evaluationClusterModelFromLabel()
        print(f"Acc: {acc:.3f} F1: {f1:.3f} NMI: {nmi:.3f} ARI: {ari:.3f} Time: {cluster_time:.3f}s RAM: {int(peak_memory_MBs)}MB Weights: {', '.join([f'{w:.6f}' for w in view_weights])}")
    print(f"lap_time: {lapnum}, opt_num: {optnum}")
    if config.verbose:
        print(f"Eigenvalues: {' '.join([str(e) for e in eig_val])}")
        print(f"f_con={eig_val[1]} f_gap={eig_val[num_clusters]/eig_val[num_clusters-1]}")
    if config.embedding:
        return list(emb_results) + [embed_time]
    else:
        return [acc, f1, nmi, ari, cluster_time]

if __name__ == '__main__':
    args = parse_args()
    dataset = load_data(config.dataset)
    if config.dataset in ['magphy', 'magsoc', 'mageng']:
        config.approx_knn = True
    config.knn_k = args.knn_k
    config.seed = args.seed
    config.obj_regular = args.obj_regular
    config.weight_lb = args.weight_lb
    config.opt_cobyla_rhobeg = args.opt_cobyla_rhobeg
    config.opt_objective = 'combine'
    config.alpha = args.alpha
    SMGF(dataset)


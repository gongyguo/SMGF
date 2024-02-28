
dataset = None
embedding = False

knn_k = 10
sc_eig_tol = 1e-2
kmeans = False # use K-means instead of Discretize
optimize_weights = True # disable for UNIFORM variant
fixed_weights = [1./3] * 3
opt_cobyla_rhobeg = 0.05
opt_max_iters = 1000
opt_eig_tol = 0.01 # convergence threshold for eigensolver used for weight optimization
weight_lb = 0.00 # lower bound of weights
opt_w_tol = 1e-2 # convergence threshold for COBYLA optimizer
opt_objective = 'combine' # Optimization objective. 'con': connectivity, 'gap': eigengap, 'reg': relative eigen-gap
obj_alpha = 1.0 #connectivity obj weight
obj_regular = 0.5 # regularizer weight gamma
poly_degree = 2  #polynomial feature degree
seed = 0 #random seed
ridge_alpha = 0.05 #ridge linear regression
embed_dim = 64 # embedding demension
sample_weight=0.1 #sample weight for OPT funciton
approx_knn = False # large-scale approx knn config
verbose = False
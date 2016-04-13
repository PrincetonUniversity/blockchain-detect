from preamble import *

matrix = sys.argv[1]
ncomp = [int(x) for x in sys.argv[2].split(' ')]
alpha = [float(x) for x in sys.argv[3].split(' ')]
l1_ratio = [float(x) for x in sys.argv[4].split(' ')]
init = sys.argv[5]

def eval_fold(estimator_class, cvfold, param):
    param_key = str(sorted(param.items()))    
    est = estimator_class(**param)
    train_ix, test_ix = cvfold
    print(' ' + param_key + '\n', end='')
    sys.stdout.flush()
    est.fit(iX[train_ix], Ytrain[train_ix])
    return param_key, (est.score(iX[test_ix], Ytrain[test_ix]), len(test_ix), param)

def conglomerate_cvs(cvs):
    full_dict = {}
    for key, (auc, n, param) in cvs:
        totauc, tot, param = full_dict.get(key, (0, 0, param))
        full_dict[key] = totauc + auc * n, tot + n, param
    return max((a / n, param) for k, (a, n, param) in full_dict.items())

def cvAUC_broken(estimator_class, grid, name):
    print('Running grid search for {}'.format(name))
    cv_params = product(cv, ParameterGrid(grid))
    t = time.time()
    all_cv = Parallel(n_jobs=nproc)(
        delayed(eval_fold)(estimator_class, cv, p) for cv, p in cv_params)
    t = int(time.time() - t)
    best_auc, best_p = conglomerate_cvs(all_cv)
    print('\tran cv grid with best AUC {:.4f} in {} s '.format(best_auc, t))
    print('\tmodel:', best_p)
    est = estimator_class(**best_p)
    est.fit(iX, Ytrain)
    return est

def make_NMF(M, n_components, alpha, l1_ratio, init):
    from sklearn.decomposition import NMF
    nmf = NMF(n_components=n_components, init=init, alpha=alpha, l1_ratio=l1_ratio)
    W = nmf.fit_transform(M)
    return W, nmf.components_

class NMFPredictor(CachedTrainingMatrixEstimator):
    def __init__(self, A, n_components, alpha, l1_ratio, init):
        super(NMFPredictor, self).__init__()
        self.A = A
        self.args = (n_components, alpha, l1_ratio, init)
        
    def fit(self, iX, y):
        X = masked_matrix(self.A, iX)
        self.W, self.H = make_NMF(X, *self.args)
        
    def predict(self, X):
        senders, receivers = X[:, 0], X[:, 1]
        return (self.W[senders, :] * self.H.transpose()[receivers, :]).sum(axis=1)
           
grid = {
    'A': [matrix],
    'n_components': ncomp, 
    'alpha': alpha,
    'l1_ratio': l1_ratio, 
    'init': [init]}

import operator
from functools import reduce

nitems = reduce(operator.mul, (len(x) for x in grid.values())) 
print('running grid search on',nitems, 'configs')

cvAUC_broken(NMFPredictor, grid, 'NMF')

from preamble import *

def make_NMF(M, n_components, alpha, l1_ratio, init):
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
    'A': ['counts', 'binary', 'log1p'],
    'n_components': [4],
    'alpha': [0], #[0.001, 0.01, 0.1, 1, 10, 100, 1000],
    'l1_ratio': [0], #np.arange(0, 1.01, .1),
    'init': ['nnsvd']} #['random', 'nndsvd','nndsvda']}

nmf = cvAUC(NMFPredictor(None, None, None, None, None), grid, 'NMF', verbosity=2) 
print('NMF')

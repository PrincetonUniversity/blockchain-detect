from preamble import *

class SVDFactorization(CachedTrainingMatrixEstimator):
    def __init__(self, A, k):
        super(SVDFactorization, self).__init__()
        self.A = A
        self.k = k
        
    def fit(self, iX, y):
        X = masked_matrix(self.A, iX)
        self.u, self.s, self.vt = svds(X, k=self.k, tol=1e-10, which='LM')
        
    def predict(self, X):
        senders, receivers = X[:, 0], X[:, 1]
        sv = self.s[:, np.newaxis] * self.vt[:, receivers]
        u = self.u[senders, :]
        return (u * sv.transpose()).sum(axis=1)

print('defined cv procedure')
        
grid = {
    'A': ['counts', 'binary', 'log1p'],
    'k': list(range(1, 21))}

svd = cvAUC(SVDFactorization(None, None), grid, 'SVD', verbosity=2)
print('svd')

from preamble import *

import lda

matrix = sys.argv[1]
ks = [int(x) for x in sys.argv[2].split(' ')]

class LDAPredict(CachedTrainingMatrixEstimator):
    def __init__(self, A, k):
        super(LDAPredict, self).__init__()
        self.A = A
        self.k = k
        
    def fit(self, iX, y):
        X = masked_matrix(self.A, iX).astype('int')
        ldaf = lda.LDA(n_topics=self.k)
        self.left = ldaf.fit_transform(X)
        self.right = ldaf.components_
        
    def predict(self, X):
        senders, receivers = X[:, 0], X[:, 1]
        u = self.left[senders, :]
        v = self.right[:, receivers].transpose()
        return (u * v).sum(axis=1)

grid = {
    'A': [matrix],
    'k': ks}

lda = cvAUC(LDAPredict(None, None), grid, 'LDA', verbosity=2)

print('lda')

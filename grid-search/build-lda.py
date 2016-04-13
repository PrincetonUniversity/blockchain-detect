from preamble import *

import lda

matrices = sys.argv[1].split(' ')
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

import itertools
import pickle, time

def make_lda(m, k):
    ldam = LDAPredict(m, k)
    print('Building {} {}\n'.format(m, k), end='')
    sys.stdout.flush() 
    t = time.time()
    ldam.fit(iX, Ytrain)
    with open('ldamodels/{}-{}.p'.format(m, k), 'wb') as w:
        pickle.dump(ldam, w)
    t = time.time() - t
    print('  built and dumpted {} {} in {} s\n'.format(
        m, k, t), end='')
    sys.stdout.flush()
    
print('making', len(matrices) * len(ks), 'ldas')
Parallel(n_jobs=nproc)(
    delayed(make_lda)(matrix, k) for matrix, k in itertools.product(matrices, ks))
print('finished everything')


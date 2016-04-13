from preamble import *

import lda

matrices = sys.argv[1].split(' ')
ks = [int(x) for x in sys.argv[2].split(' ')]
alphas  = [float(x) for x in sys.argv[3].split(' ')]
l1_ratio  = [float(x) for x in sys.argv[4].split(' ')]
init  = sys.argv[5].split(' ')

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

import itertools
import pickle, time

def printnmf(t):
    return '{}-{}-{}-{}-{}'.format(*t)

def make_lda(m, k, alpha, l1, init):
    tup = m, k, alpha, l1, init
    nmf = NMFPredictor(m, k, alpha, l1, init)
    print('Building {}\n'.format(printnmf(tup)), end='')
    sys.stdout.flush() 
    t = time.time()
    nmf.fit(iX, Ytrain)
    with open('ldamodels/nmf-{}.p'.format(tup), 'wb') as w:
        pickle.dump(nmf, w)
    t = time.time() - t
    print('  built and dumpted {} in {} s\n'.format(tup, t), end='')
    sys.stdout.flush()
    
print('making', len(matrices) * len(ks) * len(alphas) * len(l1_ratio) * len(init), 'ldas')
Parallel(n_jobs=nproc)(
    delayed(make_lda)(*t) for t in itertools.product(matrices, ks, alphas, l1_ratio, init))
print('finished everything')


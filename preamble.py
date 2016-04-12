

import numpy as np
import pickle
import pandas as pd
import os, os.path
from itertools import *
import math
import random
import scipy.stats
import sys
import random
from joblib import Parallel, delayed
import multiprocessing
nproc = max(1, multiprocessing.cpu_count())
from scipy.sparse import csr_matrix
import scipy.sparse
from sklearn import cross_validation, base
from sklearn.metrics import roc_curve, auc
from sklearn.grid_search import ParameterGrid, GridSearchCV
from scipy.sparse.linalg import svds
from sklearn.decomposition import NMF
import networkx as nx
import scipy
import time
    
# Warnings

import warnings
warnings.filterwarnings('ignore')
        
# Extract data in df form and as a sparse matrix

dftrain = pd.read_csv('data/txTripletsCounts.txt',
                      header=None,
                      index_col=None,
                      sep=' ',
                      names=['sender','receiver','transaction'])

dftest = pd.read_csv('data/testTriplets.txt',
                     header=None,
                     index_col=None,
                     sep=' ',
                     names=['sender','receiver','transaction'])

dim = max(df[c].max() for df in (dftrain, dftest) for c in ['sender', 'receiver'])
dim += 1

# both the matrices here have m[i, j] return the number of transactions from i to j
# in training.
train_csr = csr_matrix((dftrain['transaction'],(dftrain['sender'],dftrain['receiver'])),
                       shape=(dim,dim),
                       dtype=float)

train_csc = train_csr.tocsc()

def maxlgbin(series): return math.ceil(np.log2(series.max()))

bincsr = train_csr.sign()
bincsc = train_csc.sign()
logcsr = train_csr.log1p()
logcsc = train_csc.log1p()

print('loaded data')

# Create a simple CV scheme (relying on our time-stationarity assumption)
# for evaluation.

Xtrain = dftrain[['sender', 'receiver']].values.astype('int')
Ytrain = dftrain['transaction'].values.astype('bool')
Xtest = dftest[['sender', 'receiver']].values.astype('int')
Ytest = dftest['transaction'].values.astype('bool')
iX = np.arange(len(Xtrain))
n_ones = Ytest.sum()
n_zeros = len(Ytest) - n_ones

n_folds = 10

cv = list(islice(cross_validation.KFold(
    len(Ytrain), n_folds=(len(Xtrain) // n_ones), shuffle=True, random_state=0), 0, n_folds))

def named_matrix(A):
    if A == 'counts': return train_csr
    elif A == 'binary': return bincsr
    elif A == 'log1p': return logcsr
    return None


def masked_matrix(A, iX):
    M = named_matrix(A)
    mask = np.ones(len(Xtrain), dtype='bool')
    mask[iX] = 0
    X = Xtrain[mask]
    vals = np.ones(len(X))
    X = X.transpose()
    disable = csr_matrix((vals, (X[0], X[1])), shape=M.shape)    
    return M - M.multiply(disable)

def AUC(exact, pred):
    fpr, tpr, thresholds = roc_curve(exact, pred)
    return auc(fpr, tpr)

def sample_one():
    while True:
        i, j = (random.randint(0, dim - 1) for i in range(2))
        if i == j: continue
        if not bincsr[i, j]: return [i, j]

class CachedTrainingMatrixEstimator(base.BaseEstimator):
    def __init__(self):
        super(CachedTrainingMatrixEstimator, self).__init__()
    
    def score(self, iX, Y):
        assert abs(len(Y) - n_ones) <= 1
        assert np.all(Y > 0)
        assert len(iX) == len(Y)
        yes_train = Xtrain[iX]
        no_train = np.array([sample_one() for i in range(n_zeros)])
        pred_yes = self.predict(yes_train)
        pred_no = self.predict(no_train)
        exact_yes = np.ones(len(pred_yes), dtype='int')
        exact_no = np.zeros(len(pred_no), dtype='int')
        exact = np.concatenate((exact_yes, exact_no))
        pred = np.concatenate((pred_yes, pred_no))
        return AUC(exact, pred)

def cvAUC(estimator, grid, name, verbosity=0):
    print('Running grid search for {}'.format(name))
    gse = GridSearchCV(estimator, grid, n_jobs=nproc, verbose=verbosity, cv=cv)
    t = time.time()
    gse.fit(iX, Ytrain)
    t = int(time.time() - t)
    print('\tran cv grid with best AUC {:.4f} in {} s '.format(gse.best_score_, t))
    print('\tmodel:', gse.best_params_)
    return gse.best_estimator_

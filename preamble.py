
import numpy as np
import pandas as pd
import os, os.path
from itertools import *
import math
import random
import scipy.stats
import sys
from joblib import Parallel, delayed
import multiprocessing
nproc = max(1, multiprocessing.cpu_count())
from scipy.sparse import csr_matrix
from scipy import sparse

if nproc > 4:
    # timesharing env, only for part of them
    nproc = nproc * 4 // 5
    
# Warnings

import warnings
warnings.filterwarnings('ignore')


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
print('loaded csr')

assert 'cuda' in os.environ.get('PATH', '')

assert 'cuda' in os.environ.get('LD_LIBRARY_PATH', '')

os.environ['THEANO_FLAGS'] = 'mode=FAST_RUN,device=gpu,floatX=float32,cuda.root=/usr/local/cuda-7.5'#,lib.cnmem=0.5'

import theano, theano.sparse
print('Loaded theano', theano.config.device, 'cuda', theano.config.cuda.root)

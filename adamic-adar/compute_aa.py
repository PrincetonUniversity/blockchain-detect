#/bin/echo run this as python3

import pickle
import networkx as nx
import sys, itertools
from joblib import Parallel, delayed
import multiprocessing
nproc = max(1, multiprocessing.cpu_count() - 1)

if nproc > 4:
    # timesharing env, only for part of them
    nproc = nproc * 4 // 5

if __name__ != '__main__':
    print('Should only be run as a script', file=sys.stderr)
    sys.exit(1)

if len(sys.argv) != 3:
    print('Usage: python compute_aa.py start end\n\n' + \
          'Computes AA score for edges [start, end) according to the' + \
          ' ordering induced by g.edges(), where g is the graph from ' + \
          'the file loaded in the cwd called simple_graph.p. Pickles output' + \
          'as start-end.p', file=sys.stderr)
    sys.exit(1)

start = int(sys.argv[1])
end = int(sys.argv[2])

def chunks(l, n):
    return [l[x:x+n] for x in range(0, len(l), n)]

import time

t = time.time()
with open('../data/simple_graph.p', 'rb') as r:
    g = pickle.load(r)
print('Loaded graph in {} s'.format(time.time() - t))

E = g.edges()
tot_edges = end - start

assert start >= 0
assert end <= len(E)
assert start <= end

step = max(tot_edges // nproc, 1)

def getaa(i):
    t = time.time()
    l = list(nx.adamic_adar_index(g, E[i:i + step]))
    t = time.time() - t
    print('  finished edges [{: 7d}, {: 7d}) in {: 7.2f} s\n'
          .format(i, i + step, t), end='')
    sys.stdout.flush()
    return l

t = time.time()
preds = Parallel(n_jobs=nproc)(
    delayed(getaa)(i) for i in range(start, end, step))
t = time.time() - t
print('Completed all {} edges [{}, {}) in {} s'
      .format(end - start, start, end, t))

print('Dumping...', end='')
sys.stdout.flush()
t = time.time()
with open('{}-{}.p'.format(start, end), 'wb') as w:
    pickle.dump(list(itertools.chain.from_iterable(preds)), w)
print('  finished in {} s'.format(time.time() - t))


#!/usr/bin/env python

import numpy as np
import os


structype = os.path.basename(os.getcwd())

if structype == 'equil_tempdot5':
    lbls = np.arange(199)
elif structype == 'equil_tempdot6':
    lbls = np.arange(200)
elif structype == 'tempdot5':
    lbls = np.load('ifiltered_MRO_tempdot5.npy')
elif structype == 'tempdot6':
    lbls = np.load('ifiltered_MRO_tempdot6.npy')
else:
    print(f'bad structype {structype}')



ring_stats = np.load(f'sample-{lbls[0]}/ring_counts.npy')
N = 1

for n in lbls[1:]:
    print(n, end = ' --> ')
    try:
        ring_stats += np.load(f'sample-{n}/ring_counts.npy')
        print('ye')
        N += 1
    except FileNotFoundError:
        print('missing NPY!')

np.save(f'avg_ring_counts_{structype}_MROfiltered.npy', ring_stats/N)
#!/usr/bin/env python

import numpy as np
import os


structype = os.path.basename(os.getcwd())

if structype == 'equil_tempdot5':
    lbls = np.arange(199)
elif structype == 'equil_tempdot6':
    lbls = np.arange(200)
elif structype == 'tempdot5':
    lbls = np.arange(217)
elif structype == 'tempdot6':
    lbls = np.arange(218)
else:
    print(f'bad structype {structype}')



ring_stats = np.load(f'sample-{lbls[0]}/ring_counts.npy')

for n in lbls[1:]:
    print(n, end = ' --> ')
    try:
        ring_stats += np.load(f'sample-{n}/ring_counts.npy')
        print('ye')
    except FileNotFoundError:
        print('missing NPY!')

np.save(f'ring_counts_{structype}.npy', ring_stats)
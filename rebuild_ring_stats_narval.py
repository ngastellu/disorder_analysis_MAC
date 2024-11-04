#!/usr/bin/env python

import numpy as np
from ring_analysis_MPI import ring_stats_rebuild
import os

structype = os.path.basename(os.getcwd())

if structype == 'equil_tempdot5':
    lbls = range(199)
elif structype == 'equil_tempdot6':
    lbls = range(200)
elif structype == 'tempdot6':
    lbls = range(218)
elif structype == 'tempdot5':
    lbls = range(217)
else:
    print(f'Invalid structype {structype}.')


n = lbls[0]
ring_stats = ring_stats_rebuild(f'sample-{n}', n)

for n in lbls[1:]:
    ring_stats += ring_stats_rebuild(f'sample-{n}', n)

np.save(f'ring_stats_{structype}.npy', ring_stats)



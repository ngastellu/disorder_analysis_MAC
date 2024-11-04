#!/usr/bin/env python

import numpy as np
from ring_analysis_MPI import ring_stats_rebuild
import sys

n = sys.argv[1]
ring_stats = ring_stats_rebuild(f'sample-{n}', n)

np.save(f'sample-{n}/ring_counts.npy', ring_stats)
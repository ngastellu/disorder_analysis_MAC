#!/usr/bin/env python

import numpy as np
from plot_ring_stats_ndatasets import compare_ring_stats


datadir = '/Users/nico/Desktop/simulation_outputs/ring_stats_40x40_pCNN_MAC/'
t1_files  = ['avg_ring_counts_t1_relaxed.npy','avg_ring_counts_t1.npy']
tdot25_files  = ['avg_ring_counts_tdot25_relaxed.npy','avg_ring_counts_tdot25.npy']


compare_ring_stats(datadir,t1_files,normalised=[False,False],labels=['relaxed', 'unrelaxed'])
# compare_ring_stats(datadir,tdot25_files,normalised=[False,False],labels=['relaxed', 'raw'])

#!/usr/bin/env python

import numpy as np
from plot_ring_stats_ndatasets import plot_ring_stats



ddir = '/Users/nico/Desktop/simulation_outputs/ring_stats_40x40_pCNN_MAC/'
dfile = 'avg_ring_counts_t1_new_model_relaxed.npy'
title = 'Ring stats for conditional model ($N = 101$, $T = 1$)'

plot_ring_stats(ddir, dfile, title=title, normalise=True, usetex=True)
#!/usr/bin/python
import numpy as np


datadir = '/Users/nico/Desktop/simulation_outputs/ring_stats_40x40_pCNN_MAC/'


N_tdot5 = 217
N_eqtdot5 = 196

N_tdot6 = 218
N_eqtdot6 = 200


ring_stats_tdot5 = np.load(datadir + 'avg_ring_counts_tempdot5.npy')
ring_stats_eqtdot5 = np.load(datadir + 'avg_ring_counts_equil_tempdot5.npy')

ring_stats_tdot6 = np.load(datadir + 'avg_ring_counts_tempdot6.npy')
ring_stats_eqtdot6 = np.load(datadir + 'avg_ring_counts_equil_tempdot6.npy')


ring_stats_tdot5_all = (N_tdot5 * ring_stats_tdot5 + N_eqtdot5 * ring_stats_eqtdot5) / (N_tdot5 + N_eqtdot5)
ring_stats_tdot6_all = (N_tdot6 * ring_stats_tdot6 + N_eqtdot6 * ring_stats_eqtdot6) / (N_tdot6 + N_eqtdot6)

np.save(datadir + 'avg_ring_counts_tempdot5_w_equil.npy', ring_stats_tdot5_all)
np.save(datadir + 'avg_ring_counts_tempdot6_w_equil.npy', ring_stats_tdot6_all)
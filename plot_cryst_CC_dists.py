#!/usr/bin/env python

import numpy as np
import matplotlib.pyplot as plt
from qcnico.plt_utils import setup_tex


stypes = ['40x40', 'tempdot5', 'tempdot6']
lbls = ['PixelCNN', '$\\tilde{T} = 0.5$', '$\\tilde{T} = 0.6$']


fig, ax = plt.subplots()
setup_tex()

bin_edges = np.load(f'/Users/nico/Desktop/simulation_outputs/structural_characteristics_MAC/crystalline_CC_dists/dist_hist_edges.npy')
centers = 0.5 * (bin_edges[1:] + bin_edges[:-1])
dx = bin_edges[1] - bin_edges[0]

rmin = 2.0
rmax = 3.0

for st, lbl in zip(stypes,lbls):
    counts = np.load(f'/Users/nico/Desktop/simulation_outputs/structural_characteristics_MAC/crystalline_CC_dists/dist_hists_{st}.npy')
    counts = counts / counts.sum()
    ax.bar(centers, counts, width=dx, alpha=0.5,label=lbl)

    relevant_bin_inds = ((centers > rmin) * (centers < rmax)).nonzero()[0]
    relevant_bins = centers[relevant_bin_inds]
    relevant_counts = counts[relevant_bin_inds]

    ipeaks = np.argsort(relevant_counts)[-5:]
    xpeaks = relevant_bins[ipeaks]
    print('Peak positions = ', xpeaks)

plt.legend()
plt.show()
#!/usr/bin/env python

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams
from qcnico.graph_tools import adjacency_matrix_sparse, count_rings, label_atoms, cycle_centers, classify_hexagons
from qcnico.qcplots import plot_rings_MAC
from qcnico.coords_io import read_xyz


pos_path = 'conditionedp6dot9.xyz'
pos = read_xyz(pos_path)
rCC = 1.8
M = adjacency_matrix_sparse(pos,rCC)
ring_data, cycles = count_rings(pos,rCC,max_size=10,return_cycles=True,distinguish_hexagons=True)
ring_cntrs = cycle_centers(cycles, pos)
print(ring_data)

atom_lbls = label_atoms(pos,cycles,ring_data,distinguish_hexagons=True)
print(atom_lbls)

ring_sizes = np.array([len(c) for c in cycles])
hex_inds = (ring_sizes == 6).nonzero()[0]
hexs = np.array([c for c in cycles if len(c) == 6])
i6, c6 = classify_hexagons(hexs)
i6 = list(i6)
iso_inds = hex_inds[i6]
ring_sizes[iso_inds] *= -1


rcParams['figure.figsize'] = [12.8,9.6]
plot_rings_MAC(pos,M,ring_sizes,ring_cntrs,atom_labels=atom_lbls,dotsize_atoms=10.0,dotsize_centers=50.0,show=False)
plt.savefig('labelled_conditionedp6dot9.png',bbox_inches='tight')
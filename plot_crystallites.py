#!/usr/bin/env python


import pickle
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams
from qcnico.coords_io import read_xyz
from qcnico.graph_tools import adjacency_matrix_sparse, count_rings, classify_hexagons, cycle_centers
from qcnico.qcplots import plot_atoms, plot_rings_MAC

rCC = 1.8

ddir = '/Users/nico/Desktop/simulation_outputs/structural_characteristics_MAC/crystallite_sizes/tempdot5/other_structure_data/'
pos = read_xyz(ddir + 'tempdot5n108_relaxed_no-dangle.xyz')
pos = pos[:,:2]
M = adjacency_matrix_sparse(pos,rCC)

with open(ddir + 'clusters-108.pkl', 'rb') as fo: 
    clusters = pickle.load(fo)


with open(ddir + 'centres_hashmap-108.pkl', 'rb') as fo: 
    centres_dict = pickle.load(fo)


hex_centres = np.zeros((len(centres_dict),2))

for r, k in centres_dict.items():
    hex_centres[k] = r

cluster_centres = [np.array([hex_centres[i] for i in c]) for c in clusters]


# cluster_clrs = get_cm(np.arange(len(clusters)),'rainbow',min_val=0.0,max_val=1.0)
cluster_clrs = ['limegreen'] * len(clusters)



rcParams['figure.figsize'] = [12.8,9.6]
fig, ax  = plot_atoms(pos,dotsize=1.0,show=False)

for cc, clr in zip(cluster_centres, cluster_clrs):
    ax.scatter(*cc.T,c=clr,s=1.0,zorder=4)

plt.savefig('/Users/nico/Desktop/GM_presentations/gm_presentation_2024-06-25/tempdot5_example.png',bbox_inches='tight')
plt.show()

x_bounds = [100,190]
y_bounds = [30,120]

pos = pos[(pos[:,0] >= x_bounds[0]) * (pos[:,0] <= x_bounds[1]) * (pos[:,1] >= y_bounds[0]) * (pos[:,1] <= y_bounds[1])]



rCC = 1.8
M = adjacency_matrix_sparse(pos,rCC)
ring_data, cycles = count_rings(pos,rCC,max_size=10,return_cycles=True,distinguish_hexagons=True)
ring_cntrs = cycle_centers(cycles, pos)
print(ring_data)

# atom_lbls = label_atoms(pos,cycles,ring_data,distinguish_hexagons=True)
# print(atom_lbls)

ring_sizes = np.array([len(c) for c in cycles])
hex_inds = (ring_sizes == 6).nonzero()[0]
hexs = np.array([c for c in cycles if len(c) == 6])
i6, c6 = classify_hexagons(hexs)
i6 = list(i6)
iso_inds = hex_inds[i6]
ring_sizes[iso_inds] *= -1


rcParams['figure.figsize'] = [12.8,9.6]
plot_rings_MAC(pos,M,ring_sizes,ring_cntrs,atom_labels=None,dotsize_atoms=5.0,dotsize_centers=50.0,show=True)
# plt.savefig('labelled_conditionedp6dot9.png',bbox_inches='tight')
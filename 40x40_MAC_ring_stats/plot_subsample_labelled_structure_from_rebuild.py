#!/usr/bin/env python


import pickle
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams
from qcnico.coords_io import read_xyz, read_xsf
from qcnico.graph_tools import adjacency_matrix_sparse, count_rings, classify_hexagons, cycle_centers
from qcnico.qcplots import plot_atoms, plot_atoms_w_bonds, plot_rings_MAC, size_to_clr
from qcnico.pixel2xyz import pxl2xyz




rCC = 1.8

structype = 'amc300'

if structype == 'amc300':
    xyz_prefix = 'sAMC300-'
    unofficial_structype = 'tempdot5'

nn = 13
pos = read_xyz(f'/Users/nico/Desktop/scripts/disorder_analysis_MAC/structures/sAMC-300/{xyz_prefix}{nn}.xyz')
pos = pos[:,:2]


ddir = f'/Users/nico/Desktop/simulation_outputs/structural_characteristics_MAC/labelled_rings/{structype}/sample-{nn}/'

print(f'Got M.')

with open(ddir + f'clusters-{nn}.pkl', 'rb') as fo: 
    cryst_clusters = pickle.load(fo)

cluster_sizes = [len(c) for c in cryst_clusters]



with open(ddir + f'centres_hashmap-{nn}.pkl', 'rb') as fo: 
    hex_centres_dict = pickle.load(fo)


hex_centres = np.zeros((len(hex_centres_dict),2))

for r, k in hex_centres_dict.items():
    hex_centres[k] = r

cluster_centres = [np.array([hex_centres[i] for i in c]) for c in cryst_clusters]

print('Got cluster centres.')


cryst_mask = np.load(f'/Users/nico/Desktop/simulation_outputs/structural_characteristics_MAC/crystalline_masks/{unofficial_structype}/crystalline_atoms_mask-{nn}.npy')

# rcParams['figure.figsize'] = [19.2,14.4]
# rcParams['figure.figsize'] = [12.8,9.6]
# fig, ax  = plot_atoms_w_bonds(pos,M,dotsize=4.0,show=False)
full_clrs = ['black']*pos.shape[0]
for i in range(len(full_clrs)):
    if cryst_mask[i]:
        full_clrs[i] = 'darkorchid'
print('Plotted C skeletton.')


# First plot all ring centers assuming all hexagons are isolated
ring_lengths = np.load(ddir + f'all_ring_lengths-{nn}.npy')
ring_centers = np.load(ddir + f'all_ring_centers-{nn}.npy')

ring_lengths[ring_lengths == 6] = -6 # label all hecagons as isolated; will fix this over-assignment later

x_bounds = np.array([100,150])
y_bounds = np.array([200,250])

pos_filter = (pos[:,0] >= x_bounds[0]) * (pos[:,0] <= x_bounds[1]) * (pos[:,1] >= y_bounds[0]) * (pos[:,1] <= y_bounds[1])
pos = pos[pos_filter]
clrs = [full_clrs[k] for k in pos_filter.nonzero()[0]]


M = adjacency_matrix_sparse(pos,rCC)
fig, ax  = plot_atoms_w_bonds(pos,M,dotsize=40.0,show=False,colour=clrs, bond_lw=2.0)

centers_filter = (ring_centers[:,0] >= x_bounds[0]) * (ring_centers[:,0] <= x_bounds[1]) * (ring_centers[:,1] >= y_bounds[0]) * (ring_centers[:,1] <= y_bounds[1])
ring_centers = ring_centers[centers_filter]
ring_lengths = ring_lengths[centers_filter]


center_clrs = list(map(size_to_clr,ring_lengths)) 



ax.scatter(*ring_centers.T, c=center_clrs, s=300, zorder=3)
# ax.scatter(*ring_centers.T, c=center_clrs, s=16.0, zorder=3)
print('Plotted ring centers, except 6c.')

# Now plot crystalline clusters over the mislabeled isolated hexs
for cc in cluster_centres:
    ax.scatter(*cc.T,c='limegreen',s=300,zorder=4)

x_bounds_plot = np.array([110,135])
y_bounds_plot = np.array([210,235])
ax.set_xlim(x_bounds_plot)
ax.set_ylim(y_bounds_plot)

print('Plotted 6c.')
print('Crystalline cluster sizes: ', cluster_sizes)

plt.show()

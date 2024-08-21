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

# structype = 'amc400'

# if structype == 'amc400':
#     xyz_prefix = 'amc400-'

nn = 181

# ddir = f'/Users/nico/Desktop/simulation_outputs/structural_characteristics_MAC/plot_labelled_tdot5_structure/sample-{nn}/'
# pos = read_xyz(f'/Users/nico/Desktop/simulation_outputs/MAC_structures/relaxed_no_dangle/{structype}/{xyz_prefix}{nn}_relaxed_no-dangle.xyz')
# pos = pos[:,:2]

# nn = 3
# mm = 2

# ddir = f'/Users/nico/Desktop/simulation_outputs/structural_characteristics_MAC/labelled_rings/amc400/sample-{nn}/'
ddir = f'/Users/nico/Desktop/simulation_outputs/structural_characteristics_MAC/labelled_rings/small_kMC_test/sample-{nn}/'

# readme = ddir + 'README'
# with open(readme) as fo:
#     lines = fo.readlines()

# ddir += f'sample-{nn}/'

# strucfile_names = [l.strip().split()[1] for l in lines]
# strucfile = f'/Users/nico/Desktop/simulation_outputs/MAC_structures/Ata_structures/testing/{strucfile_names[nn]}'
# struc_filename = 'check_0_plainconditiondot99.npy'
# strucfile = f'/Users/nico/Desktop/simulation_outputs/MAC_structures/Ata_structures/testing/{struc_filename}'
strucfile = ddir + f'sample-{nn}.xsf'

# pos = pxl2xyz(np.load(strucfile)[0,0,:,:],0.2)
pos, _ = read_xsf(strucfile,read_forces=False)


M = adjacency_matrix_sparse(pos,rCC)

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


# rcParams['figure.figsize'] = [19.2,14.4]
# rcParams['figure.figsize'] = [12.8,9.6]
# fig, ax  = plot_atoms_w_bonds(pos,M,dotsize=4.0,show=False)
fig, ax  = plot_atoms_w_bonds(pos,M,dotsize=10,show=False)
print('Plotted C skeletton.')


# First plot all ring centers assuming all hexagons are isolated
ring_lengths = np.load(ddir + f'all_ring_lengths-{nn}.npy')
ring_centers = np.load(ddir + f'all_ring_centers-{nn}.npy')

ring_lengths[ring_lengths == 6] = -6 # label all hecagons as isolated; will fix this over-assignment later

center_clrs = list(map(size_to_clr,ring_lengths)) 

ax.scatter(*ring_centers.T, c=center_clrs, s=40, zorder=3)
# ax.scatter(*ring_centers.T, c=center_clrs, s=16.0, zorder=3)
print('Plotted ring centers, except 6c.')

# Now plot crystalline clusters over the mislabeled isolated hexs
for cc in cluster_centres:
    ax.scatter(*cc.T,c='limegreen',s=40,zorder=4)

print('Plotted 6c.')
print('Crystalline cluster sizes: ', cluster_sizes)

# plt.savefig('/Users/nico/Desktop/figures_worth_saving/charge_hopping_paper_intermediate_figs/tdot5_full_structure.eps',bbox_inches='tight')
# plt.suptitle(strucfile_names[nn] + f' \# {mm} ')
# plt.suptitle(struc_filename)
plt.xlim([0,20])
plt.ylim([0,20])
plt.show()

# x_bounds = [100,190]
# y_bounds = [30,120]

# pos = pos[(pos[:,0] >= x_bounds[0]) * (pos[:,0] <= x_bounds[1]) * (pos[:,1] >= y_bounds[0]) * (pos[:,1] <= y_bounds[1])]



# rCC = 1.8
# M = adjacency_matrix_sparse(pos,rCC)
# ring_data, cycles = count_rings(pos,rCC,max_size=10,return_cycles=True,distinguish_hexagons=True)
# ring_cntrs = cycle_centers(cycles, pos)
# print(ring_data)

# # atom_lbls = label_atoms(pos,cycles,ring_data,distinguish_hexagons=True)
# # print(atom_lbls)

# ring_sizes = np.array([len(c) for c in cycles])
# hex_inds = (ring_sizes == 6).nonzero()[0]
# hexs = np.array([c for c in cycles if len(c) == 6])
# i6, c6 = classify_hexagons(hexs)
# i6 = list(i6)
# iso_inds = hex_inds[i6]
# ring_sizes[iso_inds] *= -1


# rcParams['figure.figsize'] = [12.8,9.6]
# plot_rings_MAC(pos,M,ring_sizes,ring_cntrs,atom_labels=None,dotsize_atoms=5.0,dotsize_centers=50.0,show=True)
# # plt.savefig('labelled_conditionedp6dot9.png',bbox_inches='tight')
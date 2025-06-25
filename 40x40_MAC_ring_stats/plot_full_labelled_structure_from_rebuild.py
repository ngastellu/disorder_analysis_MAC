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
    strucdir = 'sAMC-300'
elif structype == 'amc500':
    xyz_prefix = 'sAMC500-'
    unofficial_structype = '40x40'
    strucdir='sAMC-500'

nn = 0
run_name = 'newparamamorphous_graphenetdot0.9'


pxls = np.load(f'/Users/nico/Desktop/simulation_outputs/equivariant_MAC/ata_new_batch_2025-06-23/{run_name}.npy')[nn,0,:,:]
pos = pxl2xyz(pxls,pixel2angstroms=0.2)
pos = pos[:,:2]

# nn = 3
# mm = 2

ddir = f'/Users/nico/Desktop/simulation_outputs/equivariant_MAC/labelled_rings/ata_new_batch_2025-06-23/{run_name}/'

M = adjacency_matrix_sparse(pos,rCC)

print(f'Got M.')


try:
    with open(ddir + f'clusters-{nn}.pkl', 'rb') as fo: 
        cryst_clusters = pickle.load(fo)

    cluster_sizes = [len(c) for c in cryst_clusters]

except FileNotFoundError as e:
    print(e)
    cryst_clusters = []
    cluster_sizes = []



with open(ddir + f'centres_hashmap-{nn}.pkl', 'rb') as fo: 
    hex_centres_dict = pickle.load(fo)


hex_centres = np.zeros((len(hex_centres_dict),2))

for r, k in hex_centres_dict.items():
    hex_centres[k] = r

cluster_centres = [np.array([hex_centres[i] for i in c]) for c in cryst_clusters]

print('Got cluster centres.')


# cryst_mask = np.load(f'/Users/nico/Desktop/simulation_outputs/structural_characteristics_MAC/crystalline_masks/{unofficial_structype}/crystalline_atoms_mask-{nn}.npy')

# rcParams['figure.figsize'] = [19.2,14.4]
# rcParams['figure.figsize'] = [12.8,9.6]
# rcParams['figure.dpi'] = 300
# fig, ax  = plot_atoms_w_bonds(pos,M,dotsize=4.0,show=False)
clrs = ['black']*pos.shape[0]
# for i in range(len(clrs)):
#     if cryst_mask[i]:
#         clrs[i] = 'darkorchid'
fig, ax  = plot_atoms_w_bonds(pos,M,dotsize=0.3,show=False,colour=clrs,bond_lw=0.3,zorder_atoms=10,zorder_bonds=10)
print('Plotted C skeletton.')


# First plot all ring centers assuming all hexagons are isolated
ring_lengths = np.load(ddir + f'all_ring_lengths-{nn}.npy')
ring_centers = np.load(ddir + f'all_ring_centers-{nn}.npy')

ring_lengths[ring_lengths == 6] = -6 # label all hecagons as isolated; will fix this over-assignment later

center_clrs = list(map(size_to_clr,ring_lengths)) 

ax.scatter(*ring_centers.T, c=center_clrs, s=8, zorder=3)
# ax.scatter(*ring_centers.T, c=center_clrs, s=16.0, zorder=3)
print('Plotted ring centers, except 6c.')

# Now plot crystalline clusters over the mislabeled isolated hexs
for cc in cluster_centres:
    ax.scatter(*cc.T,c='limegreen',s=8,zorder=4)

print('Plotted 6c.')
print('Crystalline cluster sizes: ', cluster_sizes)
plt.suptitle(run_name)

# plt.savefig('/Users/nico/Desktop/figures_worth_saving/charge_hopping_paper_intermediate_figs/tdot5_full_structure.eps',bbox_inches='tight')
# plt.suptitle(strucfile_names[nn] + f' \# {mm} ')
# plt.suptitle(struc_filename)
# plt.xlim([0,20])
# plt.ylim([0,20])
plt.show()

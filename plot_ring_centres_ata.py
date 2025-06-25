#!/usr/bin/env python


import pickle
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from ring_analysis_MPI import rebuild_rings
from plot_utils import plot_atoms_w_bonds, size_to_clr
from graph_tools import adjacency_matrix_sparse

def restitch_atomic_stucture(nn, datadir):
    return np.unique(np.vstack([np.load(npy) for npy in datadir.expanduser().glob(f"pos_sample-{nn}*npy")]),axis=0)


rCC = 1.8 # max C-C bond length considered when building adjacency matrix
nn = 0 # index of 


path_string = '/Users/nico/Desktop/simulation_outputs/plot_rings/newMACmodeldot5/'
ddir = Path(path_string)
cryst_clusters_file = ddir / f"clusters-{nn}.pkl"

if not cryst_clusters_file.exists():
    rebuild_rings(nn, ddir)


pos = restitch_atomic_stucture(nn, ddir)


M = adjacency_matrix_sparse(pos,rCC)

print(f'Got M.')

with open(ddir / f'clusters-{nn}.pkl', 'rb') as fo: 
    cryst_clusters = pickle.load(fo)

cluster_sizes = [len(c) for c in cryst_clusters]



with open(ddir / f'centres_hashmap-{nn}.pkl', 'rb') as fo: 
    hex_centres_dict = pickle.load(fo)


hex_centres = np.zeros((len(hex_centres_dict),2))

for r, k in hex_centres_dict.items():
    hex_centres[k] = r

cluster_centres = [np.array([hex_centres[i] for i in c]) for c in cryst_clusters]

print('Got cluster centres.')


clrs = ['black']*pos.shape[0]
fig, ax  = plot_atoms_w_bonds(pos,M,dotsize=0.5,show=False,colour=clrs,bond_lw=0.9,zorder_atoms=10,zorder_bonds=10)
print('Plotted C skeletton.')


# First plot all ring centers assuming all hexagons are isolated
ring_lengths = np.load(ddir / f'all_ring_lengths-{nn}.npy')
ring_centers = np.load(ddir / f'all_ring_centers-{nn}.npy')

ring_lengths[ring_lengths == 6] = -6 # label all hecagons as isolated; will fix this over-assignment later

center_clrs = list(map(size_to_clr,ring_lengths)) 

ax.scatter(*ring_centers.T, c=center_clrs, s=8, zorder=3)
print('Plotted ring centers, except 6c.')

# Now plot crystalline clusters over the mislabeled isolated hexs
for cc in cluster_centres:
    ax.scatter(*cc.T,c='limegreen',s=8,zorder=4)

print('Plotted 6c.')
print('Crystalline cluster sizes: ', cluster_sizes)
plt.savefig('/Users/nico/Desktop/figures_worth_saving/newMACmodeldot5_rings.png', dpi=300)

plt.show()

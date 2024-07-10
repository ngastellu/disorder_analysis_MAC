#!/usr/bin/env python


import pickle
import numpy as np
import matplotlib.pyplot as plt
from qcnico.coords_io import read_xyz
from qcnico.graph_tools import adjacency_matrix_sparse
from qcnico.plt_utils import get_cm
from qcnico.qcplots import plot_atoms 

rCC = 1.8

ddir = '/Users/nico/Desktop/simulation_outputs/structural_characteristics_MAC/crystallite_sizes/tempdot5/most_crystalline_structure_data/'
pos = read_xyz(ddir + 'tempdot5n15_relaxed_no-dangle.xyz')
pos = pos[:,:2]
M = adjacency_matrix_sparse(pos,rCC)

with open(ddir + 'clusters-15.pkl', 'rb') as fo: 
    clusters = pickle.load(fo)


with open(ddir + 'centres_hashmap-15.pkl', 'rb') as fo: 
    centres_dict = pickle.load(fo)


hex_centres = np.zeros((len(centres_dict),2))

for r, k in centres_dict.items():
    hex_centres[k] = r

cluster_centres = [np.array([hex_centres[i] for i in c]) for c in clusters]


cluster_clrs = get_cm(np.arange(len(clusters)),'rainbow',min_val=0.0,max_val=1.0)



fig, ax  = plot_atoms(pos,dotsize=1.0,show=False)

for cc, clr in zip(cluster_centres, cluster_clrs):
    ax.scatter(*cc.T,c=clr,s=5.0,zorder=4)

plt.show()

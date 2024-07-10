#!/usr/bin/env python

import numpy as np
import matplotlib.pyplot as plt
from qcnico.coords_io import read_xyz
from qcnico.qcplots import plot_atoms
from qcnico.plt_utils import get_cm
import pickle




structype = '40x40'
nn = 64

ddir = '/Users/nico/Desktop/simulation_outputs/structural_characteristics_MAC/rho_sites/testing/'

# all_pos = read_xyz(ddir + f'{structype}n{nn}_relaxed_no-dangle.xyz')
all_pos = read_xyz(ddir + f'bigMAC-{nn}_relaxed_no-dangle.xyz')
undisC = np.load(ddir + f'undistorted_atoms_{structype}-{nn}.npy')
with open(ddir + f'undistorted_clusters_{structype}-{nn}.pkl', 'rb') as fo:
    clusters = pickle.load(fo)

nclrs = 30
clrs = get_cm(np.arange(nclrs),'turbo',max_val=1.0,min_val=0.0)


N = all_pos.shape[0]
distorted = np.ones(N,dtype='bool')
distorted[undisC] = False

disC = all_pos[distorted,:]

fig, ax = plot_atoms(disC, dotsize=1.0,show=False)
max_cluster_size = 0
for k, c in enumerate(clusters):
    size = len(c)
    if size == 1: continue
    print(size)
    if size > max_cluster_size: max_cluster_size = size
    clr = clrs[k % nclrs]
    pos = all_pos[c]
    fig, ax = plot_atoms(pos,colour=clr,dotsize=1.0,show=False,plt_objs=(fig,ax))

plt.show()



print('Total number of atoms = ', N)
print('Max cluster size = ', max_cluster_size)
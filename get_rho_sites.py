#!/usr/bin/env python


import sys
from os import path
import numpy as np
from qcnico.coords_io import read_xyz



def get_structure_area(pos):
    xmin = np.min(pos[:,0])
    ymin = np.min(pos[:,1])
    xmax = np.max(pos[:,0])
    ymax = np.max(pos[:,1])

    return (xmax - xmin) * (ymax - ymin)


structype = sys.argv[1]


if structype == '40x40':
    xyz_prefix = 'bigMAC-'
    lbls = range(1,300)
    cluster_dir = path.expanduser('~/scratch/structural_characteristics_MAC/crystallite_sizes/40x40/')
else:
    xyz_prefix = structype + 'n'
    if structype == 'tempdot5':
        lbls = range(117)
        cluster_dir = path.expanduser('~/scratch/structural_characteristics_MAC/crystallite_sizes/tempdot5/sparse_matrices/') 
    else: # tempdot6
        lbls = range(132)
        cluster_dir = path.expanduser('~/scratch/structural_characteristics_MAC/crystallite_sizes/tempdot6/subsampled/') 


rho_sites = np.ones(len(lbls)) * -1
for k, nn in enumerate(lbls):
    print(nn)
    try:
        ncrystallites = np.load(cluster_dir + f'sample-{nn}/cryst_cluster_sizes-{nn}.npy').shape[0]
    except FileNotFoundError as e:
        print(e)
        continue

    pos = read_xyz(path.expanduser(f'~/scratch/clean_bigMAC/{structype}/relaxed_structures_no_dangle/{xyz_prefix}{nn}_relaxed_no-dangle.xyz'))
    pos = pos[:,:2]

    area = get_structure_area(pos)

    rho_sites[k] = ncrystallites / area


rho_sites = rho_sites[rho_sites > 0]
np.save(f'rho_sites_{structype}.npy', rho_sites)

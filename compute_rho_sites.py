#!/usr/bin/env python

import sys
from os import path
import numpy as np
import pickle
from qcnico.coords_io import read_xyz


structype = sys.argv[1]


if structype == '40x40':
    xyz_prefix = 'bigMAC-'
    lbls = np.arange(1,301)
else:
    xyz_prefix = structype + 'n'
    if structype == 'tempdot5':
        lbls = np.arange(117)
    else: #tempdot6
        lbls = np.arange(132)

rho = np.zeros(lbls.shape[0])
succ = np.ones(lbls.shape[0],dtype='bool')
for k, nn in enumerate(lbls):
    print(f'\n{k}', end = ' ')
    try:
        full_pos = read_xyz(path.expanduser(f'~/scratch/clean_bigMAC/{structype}/relaxed_structures_no_dangle/{xyz_prefix}{nn}_relaxed_no-dangle.xyz'))
        full_pos = full_pos[:,:2]
        with open(f'sample-{nn}/undistorted_clusters_{structype}-{nn}.pkl', 'rb') as fo:
            clusters = pickle.load(fo)
    except FileNotFoundError as e:
        print('File not found!')
        succ[k] = False
        continue

    xmin = np.min(full_pos[:,0])
    ymin = np.min(full_pos[:,1])

    xmax = np.max(full_pos[:,0])
    ymax = np.max(full_pos[:,1])

    area = (xmax - xmin) * (ymax - ymin)
     
    nclusters = 0
    for c in clusters:
        if len(c) > 1: nclusters +=1
    
    rho[k] = nclusters/area

lbls = lbls[succ]
rho = rho[succ]

np.save(f'rho_sites_{structype}.npy', np.vstack((lbls,rho)).T)
#!/usr/bin/env python 

import numpy as np
from scipy.integrate import trapezoid
from os import path
import sys


def A_MRO(radii,pair_corr_func,rstart=4.0,rend=12.0):
    # Define integration_range
    integ_mask = (radii >= rstart) * (radii <= rend)

    # Integrate |g(r) - 1| over the desired range
    area = trapezoid(np.abs(pair_corr_func[integ_mask]-1) ,radii[integ_mask]) 
    
    return area


structype = sys.argv[1]


if structype == '40x40':
    lbls = np.arange(1,301)
elif structype == 'tempdot5':
    lbls = np.arange(117)
else: #tempdot6
    lbls = np.arange(132)

graphene_dir = path.expanduser('~/scratch/structural_characteristics_MAC/pair_func/graphene/')
structype_dir = path.expanduser(f'~/scratch/structural_characteristics_MAC/pair_func/{structype}/')
radii_graphene = np.load(graphene_dir + 'r_nvt300_93000.npy')
pair_func_graphene = np.load(graphene_dir + 'pair_func_nvt300_93000.npy') * 0.5

A_graphene = A_MRO(radii_graphene,pair_func_graphene)
print('A_graphene = ', A_graphene)

eta_MRO = np.zeros(lbls.shape[0])
succ = np.ones(lbls.shape[0],dtype='bool')
for k, n in enumerate(lbls):
    print(f'\n{k}', end = ' ')
    try:
        g = np.load(structype_dir + f'pair_func-{n}.npy') * 0.5
        r = np.load(structype_dir + f'r-{n}.npy')
    except FileNotFoundError as e:
        print('NPY not found!')
        succ[k] = False
        continue
    eta_MRO[k] = A_MRO(r,g)

inds = lbls[succ]
eta_MRO = np.log(eta_MRO[succ] / A_graphene)

np.save(f'eta_MRO_{structype}.npy',np.vstack((inds,eta_MRO)).T)

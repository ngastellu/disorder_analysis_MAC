#!/usr/bin/env python 

import numpy as np
from scipy.integrate import trapezoid


def A_MRO(radii,pair_corr_func,rstart=4.0,rend=12.0):
    # Define integration_range
    integ_mask = (radii >= rstart) * (radii <= rend)

    # Integrate |g(r) - 1| over the desired range
    area = trapezoid(np.abs(pair_corr_func[integ_mask]-1) ,radii[integ_mask]) 
    
    return area

ddir = '/Users/nico/Desktop/simulation_outputs/structural_characteristics_MAC/pair_corr_func/'
radii = np.load(ddir + 'radii.npy')
pair_func_pCNN = np.load(ddir + 'avg_pair_func_pCNN.npy')[:,1] * 0.5
pair_func_tempdot6 = np.load(ddir + 'avg_pair_func_tempdot6.npy')[:,1] * 0.5
pair_func_tempdot5 = np.load(ddir + 'avg_pair_func_tempdot5.npy')[:,1] * 0.5
radii_graphene = np.load(ddir + 'r_nvt300_93000.npy')
pair_func_graphene = np.load(ddir + 'pair_func_nvt300_93000.npy') * 0.5


pair_funcs = [pair_func_pCNN,pair_func_tempdot6,pair_func_tempdot5,pair_func_graphene]
rr = [radii,radii,radii,radii_graphene]
areas = np.zeros(4)

for k, g, r in zip(range(4), pair_funcs,rr):
    areas[k] = A_MRO(r,g)

areas /= areas[-1] # divide by graphene area to obtain eta_MRO parameter


labels = ('pCNN','tdot6', 'tdot5','graphene')
for k in range(4):
    print(f'{labels[k]} --> {np.log(areas[k])}')
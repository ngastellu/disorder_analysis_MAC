#!/usr/bin/env python 

import numpy as np
import matplotlib.pyplot as plt
from qcnico.plt_utils import setup_tex, MAC_ensemble_colours



ddir = '/Users/nico/Desktop/simulation_outputs/structural_characteristics_MAC/pair_corr_func/'
r1 = np.load(ddir + 'radii_graphene.npy')
r2 = np.load(ddir + 'r_nvt300_avg.npy')
pair_func_graphene = np.load(ddir + 'pair_func_graphene.npy') * 0.5
pair_func_graphene2 = np.load(ddir + 'pair_func_nvt300_avg.npy') * 0.5

print(pair_func_graphene.shape)


clrs = MAC_ensemble_colours()

setup_tex()

fig,ax = plt.subplots()
ax.plot(r1, pair_func_graphene,c='b',label='pristine',lw=0.8)
ax.plot(r2, pair_func_graphene2,c='r',label='md',lw=0.8)
# ax.set_xlim([0,12])
ax.set_xlabel('Pair distance [\AA]')
ax.set_ylabel('Pair correlation function')

plt.legend()
plt.show()
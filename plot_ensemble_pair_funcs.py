#!/usr/bin/env python 

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams
from qcnico.plt_utils import setup_tex, MAC_ensemble_colours



ddir = '/Users/nico/Desktop/simulation_outputs/structural_characteristics_MAC/pair_corr_func/'
radii = np.load(ddir + 'radii.npy')
pair_func_pCNN = np.load(ddir + 'avg_pair_func_pCNN.npy')[:,1] * 0.5
pair_func_tempdot6 = np.load(ddir + 'avg_pair_func_tempdot6.npy')[:,1] * 0.5
pair_func_tempdot5 = np.load(ddir + 'avg_pair_func_tempdot5.npy')[:,1] * 0.5
radii_graphene = np.load(ddir + 'r_nvt300_avg.npy')
pair_func_graphene = np.load(ddir + 'pair_func_nvt300_avg.npy') * 0.5
print(pair_func_graphene.shape)


# clrs = MAC_ensemble_colours(clr_type='two_ensembles')
clrs = MAC_ensemble_colours()

setup_tex(fontsize=90)

fig,ax = plt.subplots()
# rcParams['font.size'] = 40
# rcParams['figure.figsize'] =   [12.8,9.6]
# ax.plot(radii, pair_func_pCNN, c=clrs[0],label='$\delta$-aG',lw=2.5 ,zorder=2)
# ax.plot(radii, pair_func_pCNN, c='#5ca904',label='sAMC-500',lw=2.5 ,zorder=4)
ax.plot(radii, pair_func_pCNN, c='limegreen',label='sAMC-500',lw=2.5 ,zorder=4)
ax.plot(radii, pair_func_tempdot6, c=clrs[1],label='sAMC-q400',lw=2.5,zorder=3)
# ax.plot(radii, pair_func_tempdot5, c=clrs[1],label='$\chi$-aG',lw=2.5   ,zorder=2)
ax.plot(radii, pair_func_tempdot5, c=clrs[2],label='sAMC-300',lw=2.5   ,zorder=2)
#ax.plot(radii_graphene, pair_func_graphene,c='r',label='graphene',lw=0.8)
# ax.plot(radii_graphene, pair_func_graphene,c='#8018ff',label='graphene',lw=2.5  ,zorder=1)
ax.plot(radii_graphene, pair_func_graphene,c='#1897ff',label='graphene',lw=2.0  ,zorder=1)
ax.set_xlim([0,12])
ax.set_xlabel('Pair distance $r$ [\AA]')
ax.set_ylabel('Pair correlation function $g(r)$')

plt.legend()
plt.show()
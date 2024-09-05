#!/usr/bin/env python 

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams
from qcnico.plt_utils import setup_tex, MAC_ensemble_colours



ddir = '/Users/nico/Desktop/simulation_outputs/structural_characteristics_MAC/pair_corr_func/'

rfiles = ['amc400_subsample/r-8.npy','amc400/r-8.npy']
gfiles = ['amc400_subsample/pair_func1-8.npy','amc400/pair_func-8.npy']
labels = ['amc400-8 subsample 1','amc400-8']

nsysts = len(rfiles)

setup_tex(fontsize=20)

fig,ax = plt.subplots()

for k in range(nsysts):
    radii = np.load(ddir + rfiles[k])
    pair_func = np.load(ddir + gfiles[k])
    ax.plot(radii, pair_func,label=labels[k],lw=0.8)


# rcParams['font.size'] = 40
# rcParams['figure.figsize'] = [12.8,9.6]
# ax.plot(radii, pair_func_pCNN, c=clrs[0],label='$\delta$-aG',lw=2.5 ,zorder=2)
# ax.plot(radii, pair_func_pCNN, c='#5ca904',label='sAMC-500',lw=2.5 ,zorder=4)
ax.set_xlim([4,12])
ax.set_xlabel('Pair distance $r$ [\AA]')
ax.set_ylabel('Pair correlation function $g(r)$')

plt.legend()
plt.show()
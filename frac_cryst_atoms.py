#!/usr/bin/env python 

import numpy as np
import matplotlib.pyplot as plt
from qcnico.plt_utils import setup_tex, MAC_ensemble_colours
import os


structypes = ['40x40', 'tempdot6', 'tempdot5']
#lbls= ['PixelCNN', '$\\tilde{T} = 0.6$', '$\\tilde{T} = 0.5$']
lbls= ['sAMC-500', 'sAMC-q400', 'sAMC-300']
clrs = MAC_ensemble_colours()

simdir = '/Users/nico/Desktop/simulation_outputs/'
percdir = simdir + 'percolation/'

setup_tex()
fig, axs = plt.subplots(3,1,sharex=True)
bins = np.linspace(0,1,100)

for ax, st, lbl, c in zip(axs,structypes,lbls,clrs):

    datadir = '/Users/nico/Desktop/simulation_outputs/structural_characteristics_MAC/fraction_cryst_atoms/'
    data = np.load(datadir + f'frac_cryst_atoms_{st}.npy')
    data = data[data > 0]

    # datadir_lo = percdir + f'{st}/electronic_crystallinity/loMO_crystallinities_crystalline_{st}_renormd/'
    # npys_lo = [datadir_lo + f for f in os.listdir(datadir_lo)]
    # data_lo = np.hstack([np.load(f) for f in npys_lo])

    # datadir_hi = percdir + f'{st}/electronic_crystallinity/hiMO_crystallinities_crystalline_{st}_renormd/'
    # npys_hi = [datadir_hi + f for f in os.listdir(datadir_hi)]
    # data_hi = np.hstack([np.load(f) for f in npys_hi])

    # print(np.unique(data_hi))
    # print(np.min(data_hi))

    # data = np.hstack((data_lo,data_hi))
    hist, bins = np.histogram(data,bins=bins)
    centers = (bins[1:] + bins[:-1]) * 0.5
    dx = centers[1] - centers[0]

    ax.bar(centers, hist,align='center',width=dx,color=c,label=lbl)
    ax.legend()
    ax.set_ylabel('Counts')
    # ax.set_yscale('log')
    # print(f'{st} ensemble has {ntiny} radii <= 1')

ax.set_xlabel('Fraction crystalline atoms')# / \# of crystalline atoms in structure')

# plt.suptitle('100 lowest- and highest-lying MOs')

plt.show()



#!/usr/bin/env python

import numpy as np
import matplotlib.pyplot as plt
from qcnico.plt_utils import setup_tex, MAC_ensemble_colours, histogram




structypes = ['40x40', 'tempdot6', 'tempdot5']
lbls = ['sAMC-500', 'sAMC-q400', 'sAMC-300']
clrs = MAC_ensemble_colours()
sdir = '/Users/nico/Desktop/simulation_outputs/structural_characteristics_MAC/'

ndir = sdir + 'nb_cryst_atoms/'
fdir = sdir + 'fraction_cryst_atoms/'

setup_tex()

fig, axs = plt.subplots(2,1)

nbins = 200
nb_bins = np.linspace(8700, 53900,nbins)
frac_bins = np.linspace(0.15, 0.88,nbins) * 100

for st, c, lbl in zip(structypes,clrs, lbls):
    nb_cryst = np.load(ndir + f'nb_cryst_atoms_{st}.npy')
    frac_cryst = np.load(fdir + f'frac_cryst_atoms_{st}.npy')

    filter = nb_cryst != 0

    nb_cryst = nb_cryst[filter]
    frac_cryst = frac_cryst[filter] * 100

    fig, axs[0] = histogram(nb_cryst,nb_bins,plt_kwargs={'color': c, 'alpha': 0.6, 'label': lbl},show=False, density=True, plt_objs=(fig,axs[0]))
    fig, axs[1] = histogram(frac_cryst,frac_bins,plt_kwargs={'color': c, 'alpha': 0.6},show=False, density=True, plt_objs=(fig,axs[1]))



axs[0].legend()

axs[0].set_xlabel('\# of crystalline atoms')
axs[1].set_xlabel('\% of crystalline atoms')

axs[0].set_ylabel('Density')
axs[1].set_ylabel('Density')
plt.show()







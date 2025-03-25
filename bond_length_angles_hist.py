#!/usr/bin/env python

import numpy as np
import matplotlib.pyplot as plt
from qcnico.plt_utils import setup_tex, MAC_ensemble_colours
import os
from time import perf_counter
from matplotlib import rcParams




datadir = '/Users/nico/Desktop/simulation_outputs/structural_characteristics_MAC/bond_hists/'
softmax_temps = ['500', 'q400', '300']
clrs = MAC_ensemble_colours()

# Find neighbors within reasonable bond range
bond_cutoff = 1.8

nbins_bonds = 100
min_bond_length = 1.25
max_bond_length = 1.75
bond_bins = np.linspace(min_bond_length,max_bond_length,nbins_bonds)
bond_bin_centres = (bond_bins[1:] + bond_bins[:-1]) * 0.5
dx_bl = bond_bin_centres[1] - bond_bin_centres[0]

nbins_angles = 100
min_angle = 60
max_angle = 180
angle_bins = np.linspace(min_angle,max_angle,nbins_angles)
angle_bin_centres = (angle_bins[1:] + angle_bins[:-1]) * 0.5
dx_angles = angle_bin_centres[1] - angle_bin_centres[0]

rCC_graphene = 1.42
theta_graphene = 120
theta_sp3 = 109.5
alphas = [0.8,0.7,0.6]


setup_tex()
rcParams['figure.dpi'] = 180
fig, axs = plt.subplots(1,2)
fig.subplots_adjust(bottom=0.142,top=0.96,left=0.067,right=0.98)

for sT, c, a in zip(softmax_temps, clrs, alphas):
    print(f'\n********** sAMC-{sT} **********')

    if nbins_bonds != 100:
        bond_length_hist = np.load(os.path.join(datadir, f'bond_length_hist-sAMC{sT}_{nbins_bonds}bins.npy'))
    else:
        bond_length_hist = np.load(os.path.join(datadir, f'bond_length_hist-sAMC{sT}.npy'))

    if nbins_angles != 100:
        angles_hist = np.load(os.path.join(datadir, f'bond_angle_hist-sAMC{sT}_{nbins_angles}bins.npy'))
    else:
        angles_hist = np.load(os.path.join(datadir, f'bond_angle_hist-sAMC{sT}.npy'))


    print(angles_hist.shape)
    print(angles_hist.shape)

    axs[0].bar(bond_bin_centres, bond_length_hist, width=dx_bl, color=c, alpha=a,label=f'sAMC-{sT}')
    axs[1].bar(angle_bin_centres, angles_hist, width=dx_angles, color=c, alpha=a,label=f'sAMC-{sT}')

    axs[0].axvline(x=rCC_graphene,ymin=0,ymax=1,ls='--',c='k',lw=1.0)
    axs[1].axvline(x=theta_graphene,ymin=0,ymax=1,ls='--',c='k',lw=1.0)
    # axs[1].axvline(x=theta_sp3,ymin=0,ymax=1,ls='--',c='k',lw=1.0)
for ax in axs:
    ax.set_yticks([])
# for ax in axs:
#     ax.legend()

axs[0].legend() # don't need both
axs[0].set_ylabel('Frequency')
axs[1].set_ylabel('Frequency')

axs[0].set_xlabel('Bond length [\AA]')
axs[1].set_xlabel('Bond angle [$^\circ$]')


axs[0].set_xticks(np.linspace(1.3,1.7,5))
axs[1].set_xticks(np.linspace(70,170,6))

# axs[1].set_xlim([75,180])

plt.show()

#!/usr/bin/env python

import numpy as np
import matplotlib.pyplot as plt
from qcnico.plt_utils import setup_tex


rho_sites = np.load('/Users/nico/Desktop/simulation_outputs/structural_characteristics_MAC/rho_sites/rho_sites_volker_training_set.npy') * 100
eta_MRO_12 = np.load('/Users/nico/Desktop/simulation_outputs/structural_characteristics_MAC/MRO_param/eta_MRO_volker_full_data_rmax12.npy')[:,1]
eta_MRO_10 = np.load('/Users/nico/Desktop/simulation_outputs/structural_characteristics_MAC/MRO_param/eta_MRO_volker_full_data_rmax10.npy')[:,1 ]
ii = np.load('AMC400_selected_indices.npy')

s_ii = set(ii)
s_all = set(range(rho_sites.shape[0]))
s_notsel = s_all - s_ii
i_not_selected = np.array(list(s_notsel))

avg = [np.mean(eta_MRO_10[ii]),np.mean(rho_sites[ii])]
target = np.array([-1.2005550812548542, 0.3246434726705032])


fig, ax = plt.subplots()
ax.scatter(eta_MRO_10[i_not_selected],rho_sites[i_not_selected],s=4.0,alpha=0.6)
ax.scatter(eta_MRO_10[ii],rho_sites[ii],s=4.0,alpha=0.6,c='r')
ax.scatter(*avg,marker='*',s=30,c='r')
ax.scatter(*target,marker='*',s=30,c='limegreen')

plt.show()
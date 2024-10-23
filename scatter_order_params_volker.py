#!/usr/bin/env python

import numpy as np
from qcnico.plt_utils import setup_tex
import matplotlib.pyplot as plt



def match_order_params(etas_dat, rhos_dat, selected_inds = None):
    """The purpose of this function is to ensure that rho[i] and eta[i] of a given ensemble correspond to the same structure.
    This function is necessary because not all of the order param calculations were successful, so some structures are not 
    respresented in eta and/or rho arrays. The missing structures are not always consistent between both arrays."""
    
    if selected_inds is None:
        inds_rhos = set(rhos_dat[:,0])
        inds_etas = set(etas_dat[:,0])

        ii = np.array(list(inds_rhos & inds_etas))
    
    else:
        ii = selected_inds
    
    for i in ii:
        # print(etas_dat[:,0] == i)
        eta = etas_dat[etas_dat[:,0]==i,1]
        rho = rhos_dat[rhos_dat[:,0]==i,1] * 100 #converts from angstrom^-2 to nm^-2
    
    return eta, rho



selected_inds = np.load('/Users/nico/Desktop/scripts/disorder_analysis_MAC/new_AMC400_selected_indices.npy')
print(selected_inds.shape)
eta_volker = np.load('/Users/nico/Desktop/simulation_outputs/structural_characteristics_MAC/MRO_param/eta_MRO_volker_full_data_rmax10.npy')[:,1]
rho_volker = np.load('/Users/nico/Desktop/simulation_outputs/structural_characteristics_MAC/rho_sites/rho_sites_volker_training_set.npy') * 100

# eta_volker, rho_volker = match_order_params(eta_volker,rho_volker,selected_inds=selected_inds)

eta_volker_sel = eta_volker[selected_inds]
rho_volker_sel = rho_volker[selected_inds]

eta_avg_training = np.mean(eta_volker_sel)
rho_avg_training = np.mean(rho_volker_sel)

eta_sAMC400 = np.load('/Users/nico/Desktop/simulation_outputs/structural_characteristics_MAC/MRO_param/eta_MRO_tempdot6_rmax10.npy')
rho_sites_sAMC400 = np.load('/Users/nico/Desktop/simulation_outputs/structural_characteristics_MAC/rho_sites/rho_sites_tempdot6.npy')

eta_sAMC400, rho_sites_sAMC400 = match_order_params(eta_sAMC400,rho_sites_sAMC400)



eta_target = np.mean(eta_sAMC400)
rho_target = np.mean(rho_sites_sAMC400)


setup_tex(fontsize=85)

fig, ax = plt.subplots()

ax.scatter(eta_volker, rho_volker, s=1.0, alpha=0.7,zorder=1,label='Volker AMC (not in training set)')
ax.scatter(eta_sAMC400, rho_sites_sAMC400, s=1.0, c='limegreen', alpha=0.7,zorder=1,label='sAMC-q400')
ax.scatter(eta_volker_sel, rho_volker_sel,s=1.0,c='r',alpha=0.7,zorder=2,label='Volker AMC (training set)')
ax.scatter(eta_avg_training, rho_avg_training, marker='*', s=100.0, facecolor='r',edgecolor='k',lw=1.0,zorder=3,label='training set avg')
ax.scatter(eta_target, rho_target, marker='*', s=100.0, facecolor='limegreen', edgecolor='k',lw=1.0,zorder=3,label='sAMC-q400 avg')
ax.set_xlabel("$\eta'_{\\text{MRO}}$")
ax.set_ylabel("$\\rho_{\\text{sites}}$ [nm$^{-2}$]")
ax.legend()
plt.show()

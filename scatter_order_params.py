#!/usr/bin/env python

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams
from qcnico.plt_utils import setup_tex, MAC_ensemble_colours



ddir_rho = '/Users/nico/Desktop/simulation_outputs/structural_characteristics_MAC/rho_sites/'
ddir_eta = '/Users/nico/Desktop/simulation_outputs/structural_characteristics_MAC/MRO_param/old_data/'

structypes = ['40x40', 'tempdot6', 'tempdot5']
labels = ['sAMC-500', 'sAMC-400', 'sAMC-300']
clrs = MAC_ensemble_colours()

# structypes = ['40x40', 'tempdot5']
# labels = ['$\delta$-aG','$\chi$-aG']
# clrs = MAC_ensemble_colours('two_ensembles')

setup_tex(fontsize=40)

fig, ax = plt.subplots()
rcParams['font.size'] = 40
rcParams['figure.figsize'] = [12.8,9.6]

for st, lbl, c in zip(structypes,labels,clrs):
    rhos_dat = np.load(ddir_rho + f'rho_sites_{st}.npy')
    etas_dat = np.load(ddir_eta + f'eta_MRO_{st}.npy')

    inds_rhos = set(rhos_dat[:,0])
    inds_etas = set(etas_dat[:,0])

    ii = np.array(list(inds_rhos & inds_etas))
    scat_dat = np.zeros((ii.shape[0],2))
    
    for k,i in enumerate(ii):
        print(etas_dat[:,0] == i)
        eta = etas_dat[etas_dat[:,0]==i,1]
        rho = rhos_dat[rhos_dat[:,0]==i,1] * 100
        scat_dat[k,0] = eta
        scat_dat[k,1] = rho
    
    ax.scatter(*scat_dat[:-1].T,s=70.0,c=c,alpha=0.7,zorder=1)
    ax.scatter(*scat_dat[-1].T,s=70.0,c=c,alpha=0.5,zorder=1)
    ax.scatter(*np.mean(scat_dat,axis=0).T,s=320.0,marker='*',c=c,zorder=2,edgecolor='k',lw=1.5,label=lbl)

ax.scatter(1/1600,0,s=320.0,marker='*',c='#8018ff',edgecolor='k',lw=1.5,label='graphene')
ax.set_xlabel('$\log\eta_{\\text{MRO}}$',fontsize=40)
ax.set_ylabel('$\\rho_{\\text{sites}}$ [nm$^{-2}$]',fontsize=40)
ax.legend()
plt.show()



#!/usr/bin/env python

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams
from qcnico.plt_utils import setup_tex, MAC_ensemble_colours



ddir_rho = '/Users/nico/Desktop/simulation_outputs/structural_characteristics_MAC/rho_sites/'
ddir_eta = '/Users/nico/Desktop/simulation_outputs/structural_characteristics_MAC/MRO_param/'

structypes = ['40x40', 'tempdot6', 'amc400', 'tempdot5', 'amc400_subsample', 'T1']
labels = ['sAMC-500', 'sAMC-q400', 'sAMC-400b', 'sAMC-300', 'sAMC-400b (50x50 subsample)', 'T1']
# clrs = MAC_ensemble_colours()
clrs = ['#008744', '#0057e7', '#d62d20', '#ff5e00' , '#ffa700', '#20c9d6']

# structypes = ['40x40', 'tempdot5']
# labels = ['$\delta$-aG','$\chi$-aG']
# clrs = MAC_ensemble_colours('two_ensembles')

setup_tex(fontsize=20)

fig, ax = plt.subplots()
# rcParams['font.size'] = 40
# rcParams['figure.figsize'] = [12.8,9.6]
# rcParams['figure.figsize'] = [12,10.6]

for st, lbl, c in zip(structypes,labels,clrs):
    print(st)
    rhos_dat = np.load(ddir_rho + f'rho_sites_{st}.npy')
    etas_dat = np.load(ddir_eta + f'eta_MRO_{st}.npy')

    inds_rhos = set(rhos_dat[:,0])
    inds_etas = set(etas_dat[:,0])

    ii = np.array(list(inds_rhos & inds_etas))
    scat_dat = np.zeros((ii.shape[0],2))
    
    for k,i in enumerate(ii):
        # print(etas_dat[:,0] == i)
        eta = etas_dat[etas_dat[:,0]==i,1]
        rho = rhos_dat[rhos_dat[:,0]==i,1] * 100
        scat_dat[k,0] = eta
        scat_dat[k,1] = rho
    
    ax.scatter(*scat_dat[:-1].T,s=70.0,c=c,alpha=0.7,zorder=1)
    ax.scatter(*scat_dat[-1].T,s=70.0,c=c,alpha=0.5,zorder=1)
    ax.scatter(*np.mean(scat_dat,axis=0).T,s=320.0,marker='*',c=c,zorder=2,edgecolor='k',lw=1.5,label=lbl)
    print(*np.mean(scat_dat,axis=0))

# ax.scatter(1/1600,0,s=320.0,marker='*',c='#8018ff',edgecolor='k',lw=1.5,label='graphene')
ax.scatter(1/1600,0,s=320.0,marker='*',c='#1897ff',edgecolor='k',lw=1.5,label='graphene')
ax.set_xlabel('$\log\eta_{\\text{MRO}}$',fontsize=20)
ax.set_ylabel('$\\rho_{\\text{sites}}$ [nm$^{-2}$]',fontsize=20)
ax.legend()
plt.show()



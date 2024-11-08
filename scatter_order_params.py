#!/usr/bin/env python

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams
from qcnico.plt_utils import setup_tex, MAC_ensemble_colours

def filterMRO(scat_dat,eta_min,eta_max):
    etas = scat_dat[:,0]
    efilter = (etas >= eta_min) * (etas <= eta_max)
    print(f'Keeping {efilter.sum()}/{etas.shape[0]} structures')
    return scat_dat[efilter,:]

ddir_rho = '/Users/nico/Desktop/simulation_outputs/structural_characteristics_MAC/rho_sites/'
ddir_eta = '/Users/nico/Desktop/simulation_outputs/structural_characteristics_MAC/MRO_param/'

# structypes = ['40x40', 'tempdot6', 'amc400', 'tempdot5', 'amc400_subsample', 'T1']
structypes = ['40x40', 'tempdot6', 'tempdot5']
labels = ['sAMC-500', 'sAMC-q400', 'sAMC-300']
clrs = MAC_ensemble_colours() 
# clrs = ['#008744', '#0057e7', '#d62d20', '#ff5e00' , '#ffa700', '#20c9d6']

# structypes = ['40x40', 'tempdot5']
# labels = ['$\delta$-aG','$\chi$-aG']
# clrs = MAC_ensemble_colours('two_ensembles')

eta_max_tdot6 = -0.99
eta_min_tdot6 = -2

eta_max_tdot5 = -0.05
eta_min_tdot5 = -0.90

eta_bounds = [[-np.inf,np.inf], [eta_min_tdot6, eta_max_tdot6], [eta_min_tdot5, eta_max_tdot5]]

fontsize=45
fontsize_axes = 65
# setup_tex(fontsize=fontsize)

rcParams['font.size'] = fontsize # define font size BEFORE instantiating figure
rcParams['figure.figsize'] = [6,5.7]
rcParams['mathtext.fontset'] = 'cm'
rcParams['font.family'] = 'sans-serif'

fig, ax = plt.subplots()
fig.subplots_adjust(bottom=0.185,top=0.99,left=0.17,right=0.885)
# rcParams['figure.figsize'] = [12,10.6]

for st, eb, lbl, c in zip(structypes, eta_bounds,labels,clrs):
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
    
    emin, emax = eb
    scat_dat = filterMRO(scat_dat, emin, emax)
    
    ax.scatter(*scat_dat[:-1].T,s=80.0,c=c,alpha=0.7,zorder=1)
    ax.scatter(*scat_dat[-1].T,s=80.0,c=c,alpha=0.5,zorder=1)
    ax.scatter(*np.mean(scat_dat,axis=0).T,s=1000.0,marker='*',c=c,zorder=2,edgecolor='k',lw=2.0,label=lbl)
    print(*np.mean(scat_dat,axis=0))

ax.scatter(1/1600,0,s=1000.0,marker='*',c='#1897ff',edgecolor='k',lw=2.0,label='graphene')
ax.set_xlabel(r'$\log\eta_{\mathsf{MRO}}$',fontsize=fontsize_axes)
ax.set_ylabel(r'$\rho_{\mathsf{sites}}$ [nm$^{-2}$]',fontsize=fontsize_axes)
ax.legend()
plt.show()



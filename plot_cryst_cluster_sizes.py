#!/usr/bin/env python

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams
from glob import glob
import os
from qcnico.plt_utils import multiple_histograms, histogram, MAC_ensemble_colours


def load_filtered_data(datadir, prefix, ifiltered):
    dat = np.hstack([np.load(os.path.join(datadir,f'{prefix}{n}.npy')) for n in ifiltered])
    return dat


a = 1.42 # edge length of crystalline hexagon
area = 3 * np.sqrt(3) * a * a / 2



datadir = '/Users/nico/Desktop/simulation_outputs/structural_characteristics_MAC/crystallite_sizes/tempdot5/'
ifiltered_tempdot5 = np.load('/Users/nico/Desktop/simulation_outputs/structural_characteristics_MAC/ifiltered_MRO_tempdot5.npy')
sizes_tempdot5 = load_filtered_data(datadir, 'cryst_cluster_sizes-', ifiltered_tempdot5)

datadir = '/Users/nico/Desktop/simulation_outputs/structural_characteristics_MAC/crystallite_sizes/tempdot6/'
ifiltered_tempdot6 = np.load('/Users/nico/Desktop/simulation_outputs/structural_characteristics_MAC/ifiltered_MRO_tempdot6.npy')
sizes_tempdot6 = load_filtered_data(datadir, 'cryst_cluster_sizes-', ifiltered_tempdot6)

npys = glob('/Users/nico/Desktop/simulation_outputs/structural_characteristics_MAC/crystallite_sizes/40x40/*npy')
sizes_pCNN = np.hstack([np.load(f) for f in npys]) # * area


# multiple_histograms((sizes_pCNN,sizes_tempdot6,sizes_tempdot5),('PixelCNN','$\\tilde{T} = 0.6$','$\\tilde{T} = 0.5$'),nbins=200,xlabel='Area [\AA$^2$]',log_counts=True)

sizes = (sizes_pCNN,sizes_tempdot6,sizes_tempdot5)
# # labels = ('PixelCNN','$\\tilde{T} = 0.6$','$\\tilde{T} = 0.5$') 
labels = ('sAMC-500', 'sAMC-q400', 'sAMC-300')
clrs = MAC_ensemble_colours()

# sizes = (sizes_pcnn,sizes_tempdot5)
# labels = ('$\delta$-aG','$\chi$-aG')


# clrs = MAC_ensemble_colours('two_ensembles')



# rcParams['font.size'] = 50
# rcParams['figure.figsize'] = [12.8,9.6]

fontsize = 45
fontsize_axes = 60
rcParams['font.size'] = fontsize # define font size BEFORE instantiating figure
# rcParams['figure.figsize'] = [6,5.7]
rcParams['mathtext.fontset'] = 'cm'
rcParams['font.family'] = 'sans-serif'
fig, axs = plt.subplots(len(sizes),1,sharey=True)
fig.subplots_adjust(bottom=0.185,top=0.99,left=0.17,right=0.885,hspace=0.39)

for k, s, l, c, ax in zip(range(len(sizes)),sizes, labels, clrs, axs):
    if k == 1:
        ylabel= 'Counts (log)'
    else:
        ylabel = ' '
    fig, ax = histogram(s,bins=80,xlabel='',ylabel=ylabel,log_counts=True,show=False,plt_objs=(fig,ax),plt_kwargs={'color':c, 'label':l},usetex=False,y_axis_fontsize=fontsize_axes)
    # ax.legend()
    ax.tick_params('both',length=7.0,width=1.5)#,labelsize=37)
    # ax.set_box_aspect(1)
    # ax.tick_params('x',length=5.0,width=1.0,labelsize=37)

# axs[-1].set_xlabel('Area [\AA$^2$]')
axs[-1].set_xlabel('Crystallite size [# hexagons]',fontsize=fontsize_axes)
# fig.set_box_aspect(1)

# plt.suptitle('Distribution of crystallite sizes')
plt.tight_layout()
plt.show()


print('Max radius for tempdot5 = ', np.sqrt(np.max(sizes_tempdot5)*area/np.pi))
print('Max radius for tempdot6 = ', np.sqrt(np.max(sizes_tempdot6)*area/np.pi))
print('Max radius for PixelCNN = ', np.sqrt(np.max(sizes_pCNN)*area/np.pi))

# print('Mean radius for tempdot5 = ', np.sqrt(np.mean(sizes_tempdot5)/np.pi))
# # print('Mean radius for tempdot6 = ', np.sqrt(np.mean(sizes_tempdot6)/np.pi))
# print('Mean radius for PixelCNN = ', np.sqrt(np.mean(sizes_pCNN)/np.pi))
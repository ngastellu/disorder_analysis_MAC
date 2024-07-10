#!/usr/bin/env python

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams
from glob import glob
from qcnico.plt_utils import multiple_histograms, histogram, setup_tex, get_cm, MAC_ensemble_colours


a = 1.42 # edge length of crystalline hexagon
area = 3 * np.sqrt(3) * a * a / 2



npys = glob('/Users/nico/Desktop/simulation_outputs/structural_characteristics_MAC/crystallite_sizes/tempdot5/*npy')
sizes_tempdot5 = np.hstack([np.load(f) for f in npys]) * area

npys = glob('/Users/nico/Desktop/simulation_outputs/structural_characteristics_MAC/crystallite_sizes/tempdot6/*npy')
sizes_tempdot6 = np.hstack([np.load(f) for f in npys]) * area

npys = glob('/Users/nico/Desktop/simulation_outputs/structural_characteristics_MAC/crystallite_sizes/40x40/*npy')
sizes_pCNN = np.hstack([np.load(f) for f in npys]) * area

setup_tex(fontsize=40)

# multiple_histograms((sizes_pCNN,sizes_tempdot6,sizes_tempdot5),('PixelCNN','$\\tilde{T} = 0.6$','$\\tilde{T} = 0.5$'),nbins=200,xlabel='Area [\AA$^2$]',log_counts=True)

sizes = (sizes_pCNN,sizes_tempdot6,sizes_tempdot5)
# # labels = ('PixelCNN','$\\tilde{T} = 0.6$','$\\tilde{T} = 0.5$') 
labels = ('sAMC-500', 'sAMC-q400', 'sAMC-300')
clrs = MAC_ensemble_colours()

# sizes = (sizes_pcnn,sizes_tempdot5)
# labels = ('$\delta$-aG','$\chi$-aG')


# clrs = MAC_ensemble_colours('two_ensembles')



rcParams['font.size'] = 30
rcParams['figure.figsize'] = [12.8,9.6]
fig, axs = plt.subplots(len(sizes),1)

for s, l, c, ax in zip(sizes, labels, clrs, axs):
    fig, ax = histogram(s,nbins=80,xlabel='',log_counts=True,show=False,plt_objs=(fig,ax),plt_kwargs={'color':c, 'label':l},usetex=False)
    ax.legend()

axs[-1].set_xlabel('Area [\AA$^2$]')
plt.suptitle('Distribution of crystallite sizes')
plt.show()


print('Max radius for tempdot5 = ', np.sqrt(np.max(sizes_tempdot5)/np.pi))
# print('Max radius for tempdot6 = ', np.sqrt(np.max(sizes_tempdot6)/np.pi))
print('Max radius for PixelCNN = ', np.sqrt(np.max(sizes_pCNN)/np.pi))

print('Mean radius for tempdot5 = ', np.sqrt(np.mean(sizes_tempdot5)/np.pi))
# print('Mean radius for tempdot6 = ', np.sqrt(np.mean(sizes_tempdot6)/np.pi))
print('Mean radius for PixelCNN = ', np.sqrt(np.mean(sizes_pCNN)/np.pi))
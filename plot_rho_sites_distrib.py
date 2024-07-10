#!/usr/bin/env python

import numpy as np
from qcnico.plt_utils import histogram, multiple_histograms, MAC_ensemble_colours, setup_tex
import matplotlib.pyplot as plt


ddir = '/Users/nico/Desktop/simulation_outputs/structural_characteristics_MAC/rho_crystallites/'
labels = ['40x40', 'tempdot6', 'tempdot5']
rhos_list = []
for l in labels:
    rhos_list.append(np.load(ddir + f'rho_sites_{l}.npy')*100)

clrs = MAC_ensemble_colours()

setup_tex()

multiple_histograms(rhos_list, labels, colors=clrs, nbins=30, normalised=True,alpha=0.8)
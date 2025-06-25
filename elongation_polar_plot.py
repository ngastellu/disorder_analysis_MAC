#!/usr/bin/env python

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.stats import gaussian_kde



outerdir = Path("~/Desktop/simulation_outputs/structural_characteristics_MAC/shapes_dat").expanduser()
structypes = ['40x40', 'tempdot5']

fig, axs = plt.subplots(1,2, subplot_kw={'projection': 'polar'})

for st, ax in zip(structypes,axs):

    # fig, axs = plt.subplots(1,2, subplot_kw={'projection': 'polar'})


    datadir = outerdir / st
    npys = datadir.glob("*.npy")
    data = np.vstack([np.load(npy) for npy in npys])
    shape_ratio = data[:,0]
    theta_m = data[:,1]
    theta_M = data[:,2] - (5*np.pi/4)

    plot_data_M = np.vstack((theta_M, shape_ratio))
    kernel_M = gaussian_kde(plot_data_M)
    clrs = kernel_M(plot_data_M)

    ax.scatter(theta_M, np.abs(shape_ratio), c=clrs, alpha=0.5, s=1.0)
    ax.set_title('$\\theta_M$')
    ax.set_rlabel_position(-22.5) 

    # plot_data_m = np.vstack((theta_m, shape_ratio))
    # kernel_m = gaussian_kde(plot_data_m)
    # clrs = kernel_m(plot_data_m)

    # axs[1].scatter(theta_m, np.abs(shape_ratio), c=clrs, alpha=0.5)
    # axs[1].set_title('$\\theta_m$')



    # plt.suptitle(st)
plt.show()
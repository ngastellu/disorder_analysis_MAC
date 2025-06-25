#!/usr/bin/env python

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from qcnico.plt_utils import histogram





outerdir = Path("~/Desktop/simulation_outputs/structural_characteristics_MAC/shapes_dat").expanduser()
structypes = ['40x40', 'tempdot5']

for st in structypes:

    fig, axs = plt.subplots(2,1,sharey=True)

    datadir = outerdir / st
    npys = datadir.glob("*.npy")
    data = np.vstack([np.load(npy) for npy in npys])
    # shape_ratio = data[:,0]
    # theta_m = data[:,1]
    # theta_M = data[:,2]

    data = data[:,[0,2]]

    labels = ['Shape ratio', '$\\theta_M$']
    for d, label, ax in zip(data.T, labels, axs):
        _ = histogram(d,bins=200,xlabel=label, plt_objs=(fig, ax), show = False)

    plt.suptitle(st)
    plt.tight_layout()
    plt.show()
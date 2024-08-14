#!/usr/bin/env python

import matplotlib.pyplot as plt
from qcnico.plt_utils import setup_tex
from qcnico.pair_corr_func import pair_correlation_hist
from qcnico.coords_io import read_xyz
import numpy as np



np.random.seed(0)

posdir = '/Users/nico/Desktop/simulation_outputs/MAC_structures/relaxed_no_dangle/amc400/'

# nn = np.random.randint(113,size=2)
nn = [1,99]
rmin = 0.0
rmax = 13.0
nbins = 2000

setup_tex()

for n in nn:
    pos = read_xyz(posdir + f'amc400-{n}_relaxed_no-dangle.xyz')[:,:2]
    r, g = pair_correlation_hist(pos, rmin, rmax, nbins)
    plt.plot(r,g,label=f'\# {n}')

plt.legend()
plt.show()

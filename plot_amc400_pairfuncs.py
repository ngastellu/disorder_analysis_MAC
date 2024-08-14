#!/usr/bin/env python

import numpy as np
import matplotlib.pyplot as plt
from qcnico.plt_utils import setup_tex


setup_tex()


ddir = '/Users/nico/Desktop/simulation_outputs/structural_characteristics_MAC/pair_corr_func/amc400_unrelaxed/'
# nn = [1, 12, 34, 69, 103]
nn = [1,99,42,64]

r0 = np.load(ddir + f'r-{nn[0]}.npy')
g = np.load(ddir + f'pair_func-{nn[0]}.npy')
plt.plot(r0,g,lw=0.8,label= f'\# {nn[0]}')

for n in nn[1:]:
    r = np.load(ddir + f'r-{n}.npy')
    print(np.all(r == r0))
    g = np.load(ddir + f'pair_func-{n}.npy')
    plt.plot(r,g,lw=0.8,label= f'\# {n}')

plt.legend()
plt.xlabel('$r$ [\AA]')
plt.ylabel('$g(r)$')
plt.suptitle('unrelaxed amc400')
plt.show() 
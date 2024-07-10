#!/usr/bin/env python

import numpy as np
from qcnico.coords_io import read_dump
from qcnico.pair_corr_func import pair_correlation_hist

supercell = []
with open('dump-93000.xsf') as fo:
    for n in range(5):
        l = fo.readline()
    for n in range(2):
        l = float(fo.readline().strip().split()[1])
        supercell.append(l)


pos, *_ = read_dump('dump-93000.xsf')
pos = pos[:,:2]

rmin = 0.0
rmax = 15.0
nbins = 2500


r, pair_func = pair_correlation_hist(pos,rmin,rmax,nbins,L=supercell,eps=0.0)
np.save(f'r_nvt300_93000.npy',r)
np.save(f'pair_func_nvt300_93000.npy',pair_func)

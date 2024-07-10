#!/usr/bin/env python

import numpy as np
from glob import glob



npys = glob('/Users/nico/Desktop/simulation_outputs/structural_characteristics_MAC/crystallite_sizes/tempdot5/*npy')
max_size = 0
structure_ind = -1

for npy in npys:
    istruc = npy.split('/')[-1].split('-')[1].split('.')[0]
    sizes = np.load(npy)
    smax = np.max(sizes)
    if smax > max_size:
        structure_ind = istruc
        max_size = smax
        print(f'New max found! istruc = {istruc} ; smax = {smax}')
    
print(f'\n**** Final results: istruc = {structure_ind} ; max_size = {max_size} ****')
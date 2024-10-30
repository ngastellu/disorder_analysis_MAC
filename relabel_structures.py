#!/usr/bin/env python

from glob import glob
import os

# This script renames structures such that all of the indices form an interrupted sequence of integers

lbls = [int(f.split('-')[1].split('.')[0]) for f in glob('*.xyz')]

lbls_sorted = sorted(lbls)

fo = open('old2new_lbls.txt', 'r')
fo.write('This file keeps track of which new file index corresponds to which old file index.\n\n')
fo.write('old ---> new\n')

for k, n in enumerate(lbls_sorted):
    fo.write(f'{n} ---> {k}\n')
    os.rename(f'tempdot6n{n}_relaxed_no-dangle.xyz',f'tempdot6n{k}_relaxed_no-dangle.xyz')
fo.close()
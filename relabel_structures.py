#!/usr/bin/env python

from glob import glob
import os

# This script renames structures such that all of the indices form an interrupted sequence of integers
strucdir = 'relaxed_structures_no_dangle'
structype = os.path.basename(os.getcwd())

if structype == '40x40':
    split_str = '-'
else:
    split_str = 'n'

lbls = [int(f.split(split_str)[1].split('_')[0]) for f in glob(f'{strucdir}/*.xyz')]

lbls_sorted = sorted(lbls)

fo = open('old2new_lbls.txt', 'r')
fo.write('This file keeps track of which new file index corresponds to which old file index.\n\n')
fo.write('old ---> new\n')

for k, n in enumerate(lbls_sorted):
    fo.write(f'{n} ---> {k}\n')
    os.rename(f'{strucdir}/{structype}{split_str}{n}_relaxed_no-dangle.xyz',f'{strucdir}/{structype}{split_str}{k}_relaxed_no-dangle.xyz')
fo.close()
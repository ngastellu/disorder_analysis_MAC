#!/usr/bin/env python

import numpy as np
import sys


structype = sys.argv[1]

if structype == '40x40':
    lbls = np.arange(1,301)
elif structype == 'tempdot6':
    lbls = np.arange(132)
elif structype == 'tempdot5':
    lbls = np.arange(117)
else:
    print(f'{structype} is an invalid structure type.')
    sys.exit()


fraction_cryst_atoms = np.zeros(lbls.shape[0])
for k, n in enumerate(lbls):
    try: 
        cryst_mask = np.load(f'sample-{n}/crystalline_atoms_mask-{n}.npy')
    except FileNotFoundError as e:
        print(e)
        continue
    
    ncryst = cryst_mask.sum()
    N = cryst_mask.shape[0]
    fraction_cryst_atoms[k] = ncryst / N
    print(f'{n} ---> {fraction_cryst_atoms[k]*100}% cryst. atoms')

np.save(f'frac_cryst_atoms_{structype}.npy', fraction_cryst_atoms)


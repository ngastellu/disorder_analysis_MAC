#!/usr/bin/env python

import numpy as np
from ring_analysis_MPI import crystalline_atoms
from qcnico.coords_io import read_xyz
import sys
from os import path




structype = sys.argv[1]
nn = int(sys.argv[2])

if structype == '40x40':
    xyz_prefix = 'bigMAC-'
elif structype == 'amc400':
    xyz_prefix = 'amc400-'
else:
    xyz_prefix = structype + 'n'

full_pos = read_xyz(path.expanduser(f'~/scratch/clean_bigMAC/{structype}/relaxed_structures_no_dangle/{xyz_prefix}{nn}_relaxed_no-dangle.xyz'))
full_pos = full_pos[:,:2]

crystalline_mask = crystalline_atoms(full_pos, nn)
np.load(f'crystalline_atoms_mask-{nn}.npy', crystalline_mask)
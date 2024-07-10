#!/usr/bin/env python

import sys
import os
import numpy as np
from qcnico.graph_tools import count_rings, label_atoms
from qcnico.coords_io import read_xyz


nepochs = int(sys.argv[1])
nn = int(sys.argv[2])

pos = read_xyz(os.path.expanduser(f'~/scratch/clean_bigMAC/labelled_MAP/relaxed_structures_no_dangle/labelled{nepochs}-{nn}_relaxed_no-dangle.xyz'))
pos = pos[:,:2]


rCC = 1.8
ring_data, cycles = count_rings(pos,rCC,max_size=7,return_cycles=True,distinguish_hexagons=True)
atom_lbls = label_atoms(pos,cycles,ring_data,distinguish_hexagons=True)

labelled_pos = np.vstack((pos,atom_lbls)).T

np.save(f'labelled_MAP_labelled_atoms/labelled_coords_{nepochs}-{nn}.npy', labelled_pos)

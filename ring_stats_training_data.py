#!/usr/bin/env python


from os import path
import sys
import numpy as np
from qcnico.graph_tools import count_rings

nn = int(sys.argv[1])
njobs = int(sys.argv[2])


all_coords = np.load(path.expanduser('~/scratch/MAP_MAC_training/data/coords_13944p6.npy'))

N = all_coords.shape[0]
rCC = 1.8

nstrucs_per_job = N // njobs

if nn < njobs - 1:
    istrucs = np.arange(nn*nstrucs_per_job,(nn+1)*nstrucs_per_job)
else:
    istrucs = np.arange(nn*nstrucs_per_job,N)

ring_stats = np.zeros((istrucs.shape[0],7))

for k, n in enumerate(istrucs):
    pos = all_coords[n][:,[0,2]]
    ring_stats[n,:] = count_rings(pos, rCC, distinguish_hexagons=True,max_size=8)[0]

np.save(f'ring_stats_all_strucs-{nn}.npy', ring_stats)

#!/usr/bin/env python

from ring_analysis_MPI import get_rings_from_subsamp
import numpy as np
from qcnico.coords_io import read_xyz
from mpi4py import MPI
from os import path
import sys


comm = MPI.COMM_WORLD
rank = comm.Get_rank()
nprocs = comm.Get_size()

print(f'[{rank+1}] Hello from process {rank+1} of {nprocs}!', flush=True)


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

get_rings_from_subsamp(full_pos,nprocs,rank, nn, save_explicit_rings=True, max_ring_size=7,outdir=f'sample-{nn}')

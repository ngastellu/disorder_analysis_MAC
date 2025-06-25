#!/usr/bin/env python

from ring_analysis_MPI import get_rings_from_subsamp
from qcnico.coords_io import read_xyz
import numpy as np
from mpi4py import MPI
from pathlib import Path
import sys


def pxl2xyz(image,pixel2angstroms=0.2):
    occupied_pixels = np.vstack(image.nonzero()).astype(float).T #get atomic coordinates in pixel space
    occupied_pixels *= pixel2angstroms #convert coordinates to angstroms

    return occupied_pixels


comm = MPI.COMM_WORLD
rank = comm.Get_rank()
nprocs = comm.Get_size()


model_type = sys.argv[1]
run_name = sys.argv[2]
Tsoftmax = sys.argv[3]

pos_file = Path(f"~/scratch/equivariant_MAC/{model_type}/{run_name}/samples/{run_name}_Tsoftmax_{Tsoftmax}.npy")



if pos_file.suffix == '.npy':
    full_pos_list = []
    for k in range(3):
        full_pos_pixel = np.load(pos_file.expanduser())[k,0] #expect this array to be of shape (1,1,L,L)
        full_pos_list.append(pxl2xyz(full_pos_pixel))
elif pos_file.suffix == '.xyz':
    full_pos = read_xyz(pos_file.expanduser())

for nn, full_pos in enumerate(full_pos_list):
    full_pos = full_pos[:,:2]

    get_rings_from_subsamp(full_pos,nprocs,rank, nn, save_explicit_rings=True,outdir=f"{model_type}/{run_name}/sample-{nn}")
    print('Exiting normally!', flush=True)

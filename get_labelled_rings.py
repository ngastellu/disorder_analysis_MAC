
#!/usr/bin/env python

"""This script divides the structure into overlapping squares, and obtains the lengths and positions of the carbon
rings in each subsample, in parallel (one MPI process ---> one subsample)."""

import os
import sys
import numpy as np
from qcnico.coords_io import read_xyz
from qcnico.graph_tools import adjacency_matrix_sparse, count_rings, cycle_centers, hexagon_adjmat
from qcnico.lattice import cartesian_product
from mpi4py import MPI

def subsample_MAC_half_step(pos,l,m,n,m_max,n_max):
    """Returns the l*l square sample of a L*L MAC structure corresponding to coordinates m*l ≤ x ≤ (m+1)*l and 
    n*l ≤ y ≤ (n+1)*l."""
    
    N = pos.shape[0]

    if m == 0:
        x_mask1 = np.ones(N,dtype=bool)
    else:
        x_mask1 = pos[:,0] >= m * l 
    if m == m_max: 
        x_mask2 = np.ones(N,dtype=bool)
    else:
        x_mask2 = pos[:,0] < (m+2) * l

    if n == 0:
        y_mask1 = np.ones(N,dtype=bool)
    else:
        y_mask1 = pos[:,1] >= n * l 
    if n == n_max: 
        y_mask2 = np.ones(N,dtype=bool)
    else:
        y_mask2 = pos[:,1] < (n+2) * l

    mask = x_mask1 * x_mask2 * y_mask1 * y_mask2
    return pos[mask,:]




comm = MPI.COMM_WORLD
rank = comm.Get_rank()
nprocs = comm.Get_size()

print(f'[{rank+1}] Hello from process {rank+1} of {nprocs}!', flush=True)


structype = sys.argv[1]
nn = int(sys.argv[2])

if structype == '40x40':
    xyz_prefix = 'bigMAC-'
else:
    xyz_prefix = structype + 'n'

full_pos = read_xyz(os.path.expanduser(f'~/scratch/clean_bigMAC/{structype}/relaxed_structures_no_dangle/{xyz_prefix}{nn}_relaxed_no-dangle.xyz'))
full_pos = full_pos[:,:2]

L = 400
l = 80
a = (L // l) - 1

ii_sample = cartesian_product(np.arange(a),np.arange(a))
m,n = ii_sample[rank]

print(f'[{rank+1}] Sample indices: ({m,n})')
pos = subsample_MAC_half_step(full_pos,l,m,n,a-1,a-1)

np.save(f'pos_sample-{nn}_{m}_{n}.npy', pos)

rCC = 1.8

_, rings, M = count_rings(pos,rCC,max_size=7,return_cycles=True,return_M=True)

hexs = [c for c in rings if len(c)==6]
ring_lengths = np.array([len(c) for c in rings])
ring_centers = cycle_centers(rings, pos)
hex_centers = cycle_centers(hexs, pos)
Mhex = hexagon_adjmat(hexs)

np.save(f'M_hex-{nn}_{m}_{n}.npy', Mhex)
np.save(f'M_atoms-{nn}_{m}_{n}.npy', M)
np.save(f'hex_centers-{nn}_{m}_{n}.npy', hex_centers)
np.save(f'ring_centers-{nn}_{m}_{n}.npy', ring_centers)
np.save(f'ring_lengths-{nn}_{m}_{n}.npy', ring_lengths)

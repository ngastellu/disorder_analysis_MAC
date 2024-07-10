#!/usr/bin/env python

import numpy as np
from qcnico.coords_io import read_dump
from qcnico.pair_corr_func import pair_correlation_hist
# from qcnico.md_utils import parse_LAMMPS_log


def partition_function(T,energies):
    kB = 8.617e-5
    beta = 1.0/(kB*T)
    return np.sum(np.exp(energies*beta))

# thermal_avg_bool = int(sys.argv[1])

# if thermal_avg_bool == 1:
#     logfile = 'log.lammps'
#     nframes = 10000000
#     temps, energy = parse_LAMMPS_log(logfile, nframes, [1,2])
#     Z = partition_function(T, energy)


frames_dir = '/home/ngaste/scratch/graphene_MD/MD_frames/'

init_equil_frame = 20000
nframes = 100000
nsamples = 100
sample_frames = np.random.randint(init_equil_frame,nframes,size=nsamples)

supercell = []
with open(frames_dir + f'dump-{sample_frames[0]}.xsf') as fo:
    for n in range(5):
        l = fo.readline()
    for n in range(2):
        l = float(fo.readline().strip().split()[1])
        supercell.append(l)

pos, *_ = read_dump(frames_dir + f'dump-{sample_frames[0]}.xsf')
pos = pos[:,:2]

rmin = 0.0
rmax = 15.0
nbins = 2500

r0, avg_pair_func = pair_correlation_hist(pos,rmin,rmax,nbins,L=supercell,eps=0.0)

for n in sample_frames[1:]:
    supercell = []
    with open(frames_dir + f'dump-{n}.xsf') as fo:
        for n in range(5):
            l = fo.readline()
        for n in range(2):
            l = float(fo.readline().strip().split()[1])
            supercell.append(l)
    pos, *_ = read_dump(frames_dir + f'dump-{n}.xsf')
    pos = pos[:,:2]
    r, pair_func = pair_correlation_hist(pos,rmin,rmax,nbins,L=supercell,eps=0.0)
    assert np.all(r == r0), 'radius arrays are mismatched!'
    avg_pair_func += pair_func

avg_pair_func /= (2*nsamples)

np.save(f'r_nvt300_avg.npy',r)
np.save(f'pair_func_nvt300_avg.npy',avg_pair_func)
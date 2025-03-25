#!/usr/bin/env python

import numpy as np
import matplotlib.pyplot as plt
from qcnico.graph_tools import adjacency_matrix_sparse, get_triplets
from qcnico.coords_io import read_xyz
import os
from time import perf_counter



def bond_length_hist_stucture(xyz, pairs, bins):
    # Compute bond lengths
    bond_lengths = np.linalg.norm(xyz[list(pairs)][:, 0] - xyz[list(pairs)][:, 1], axis=1)
    hist, _ = np.histogram(bond_lengths,bins,density=True)
    return hist


def angles_hist_structure(xyz, M, bins):
    T,C = get_triplets(M)
    print(f'# of triangles = {len(C)}', end = ' ')
    all_triplets = T | C
    angles = np.zeros(len(all_triplets))
    for n,t in enumerate(all_triplets):
        i, j, k = t # i is the central atom 
        v1 = xyz[j] - xyz[i]
        v2 = xyz[k] - xyz[i]
        # Normalize vectors
        v1 /= np.linalg.norm(v1)
        v2 /= np.linalg.norm(v2)
        
        # Compute angle (dot product)
        angles[n] = np.arccos(np.clip(np.dot(v1, v2), -1.0, 1.0)) #clip forces the dot product to be within [-1, 1]
    angles *= 180.0/np.pi # convert to degrees
    hist, _ = np.histogram(angles, bins, density=True)
    return hist


def analyse_structure(xyz,rcut,bond_bins, angle_bins):
    start = perf_counter()
    M, pairs = adjacency_matrix_sparse(xyz, rcut, return_pairs=True)
    bond_length_hist = bond_length_hist_stucture(xyz, pairs, bond_bins)
    angle_hist = angles_hist_structure(xyz, M, angle_bins)
    end = perf_counter()
    print(f' [{end - start} seconds]')
    return bond_length_hist, angle_hist

def structure_index(xyzfile,narval=False):
    if narval:
        if os.path.basename(xyzfile[0]) == 't': # e.g. tempdot5n1_relaxed_no-dangle.xyz
            return int(os.path.basename(xyzfile).split('n')[1].split('_')[0])
        elif os.path.basename(xyzfile[0]) == 'b': # e.g. bigMAC-10_relaxed_no-dangle.xyz
            return int(os.path.basename(xyzfile).split('-')[1].split('_')[0])
    else:
        return int(os.path.basename(xyzfile).split('.')[0].split('-')[1])

def analyse_ensemble(structype, rcut, bond_bins, angle_bins, narval=False):
    posdir = os.path.expanduser(f'~/scratch/clean_bigMAC/{structype}/relaxed_structures_no_dangle')
    xyz_files = os.listdir(posdir)
    xyz = read_xyz(os.path.join(posdir,xyz_files[0]))
    n = structure_index(xyz_files[0],narval=narval)
    print(f'\nistruc = {n}', end = ' ')
    bond_length_hist_tot, angle_hist_tot = analyse_structure(xyz,rcut, bond_bins, angle_bins)

    for fxyz in xyz_files[1:]:
        n = structure_index(fxyz,narval=narval)
        print(f'\nistruc = {n}', end = ' ')
        xyz = read_xyz(os.path.join(posdir,fxyz))
        bl_hist, a_hist = analyse_structure(xyz, rcut, bond_bins, angle_bins)
        bond_length_hist_tot += bl_hist
        angle_hist_tot += a_hist
    
    Nstruc = len(xyz_files)
    bond_length_hist_tot /= Nstruc
    angle_hist_tot /= Nstruc

    return bond_length_hist_tot, angle_hist_tot





# Load XYZ coordinates (assuming an Nx3 numpy array)

posdir = '/Users/nico/Desktop/simulation_outputs/MAC_structures/'
structypes = ['40x40', 'tempdot6', 'tempdot5']
softmax_temps = ['500', 'q400', '300']


# Find neighbors within reasonable bond range
bond_cutoff = 1.8

nbins_bonds = 100
min_bond_length = 1.25
max_bond_length = 1.75
bond_bins = np.linspace(min_bond_length,max_bond_length,nbins_bonds)
# bond_bin_centres = (bond_bins[1:] + bond_bins[:-1]) * 0.5
# dx_bl = bond_bin_centres[1] - bond_bin_centres[0]

nbins_angles = 100
min_angle = 60
max_angle = 180
angle_bins = np.linspace(min_angle,max_angle,nbins_angles)
# angle_bin_centres = (angle_bins[1:] + angle_bins[:-1]) * 0.5
# dx_angles = angle_bin_centres[1] - angle_bin_centres[0]



for st, sT in zip(structypes, softmax_temps):
    print(f'\n********** {st} **********')
    bond_length_hist, angles_hist = analyse_ensemble(st,bond_cutoff,bond_bins, angle_bins, narval=True)

    np.save(f'bond_length_hist-sAMC{sT}.npy',bond_length_hist)
    np.save(f'bond_angle_hist-sAMC{sT}.npy',angles_hist)

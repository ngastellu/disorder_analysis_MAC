#!/usr/bin/env python

import sys
import numpy as np
from qcnico.pixel2xyz import pxl2xyz
from qcnico.data_utils import resize_arr
from ring_analysis_MPI import rebuild_rings, crystalline_centres
from time import perf_counter

"""Script to measure degree of elongation in MAP-generated amorphous graphene structures"""


def point_cloud_cov(X):
    """Computes the covariance matrix of point cloud X, assuming X is an array of row vectors.
    Tested timing, this function is marginally faster than `np.cov` and returns identical results with 1e-11."""
    N = X.shape[0]
    X -= np.mean(X,axis=0)
    return (X.T @ X) / (N-1)

def extract_shape(X):
    """From the point cloud's covariance matrix, determine its shape ratio and major/minor axes."""
    C = point_cloud_cov(X)
    eigvals, eigvecs = np.linalg.eig(C)
    # eigvecs /= np.linalg.norm(eigvecs,axis=0)
    sorted_inds = np.argsort(eigvals)
    um, uM = eigvals[sorted_inds]
    vm, vM = eigvecs[:,sorted_inds]

    shape_ratio = um / uM
    thetam = np.arctan2(vm[1],vm[0]) 
    thetaM = np.arctan2(vM[1],vM[0]) 

    return shape_ratio, thetam, thetaM

def crystallite_shapes(nn,datadir):
    """Obtains shape ratios and preferred orientation axes of crystallites in a structure whose rings data (as obtained by 
    `get_rings_from_subsamp` and `rebuild_rings` functions in `ring_analysis_MPI`) is contained in `datadir`."""

    cryst_cluster_centres = crystalline_centres(datadir, nn)
    nmax = 1000 # assume structure has under `nmax` cryst clusters
    out = np.zeros((nmax, 3))

    for k, X in enumerate(cryst_cluster_centres):
        if k > nmax:
            out = resize_arr(out)
        out[k,:] = extract_shape(X)
    
    return out[:k+1,:]


structype = sys.argv[1]
rundir = structype
rebuild_rings(0,rundir)
cryst_shape_out = crystallite_shapes(0,rundir)
print(f'Found {cryst_shape_out.shape[0]} clusters.')

np.save(f'{rundir}/shape-0.npy', cryst_shape_out)

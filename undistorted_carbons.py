#!/usr/bin/env python

import sys 
from os import path
import numpy as np
from qcnico.coords_io import read_xyz
from time import perf_counter
from qcnico.jitted_cluster_utils import jitted_components
from scipy.spatial import KDTree
from scipy.sparse import find
from itertools import combinations



def undistorted_carbons(pos, rCC_graphene=1.42, theta_graphene=2*np.pi/3, eps_rel=0.1):
    """This function finds all of the 'undistorted carbons' of a MAC structure. A carbon is deemed undistorted if it has three neighbours (i.e. other carbons within `rCC_max` of it) and
    * the three bond lengths between it and its neighbours deviate by less than `eps_rel` (default = 10%) from rCC_graphene = 1.4 angstroms.
    * the three bond angles surrounding it deviate by less than `eps_rel` theta_graphene=120°.
    """

    rmax = rCC_graphene * (1+eps_rel)
    rmin = rCC_graphene * (1-eps_rel)
    print('Building tree... ', end='')
    start = perf_counter()
    tree = KDTree(pos)
    end = perf_counter()
    print(f'Done! [{end-start}]')
    
    # DOK --> COO is fast and COO --> CSR is fast, and CSR is useful for summing/slicing along rows
    print('Building adjmat...', end = ' ')
    start = perf_counter()
    A = tree.sparse_distance_matrix(tree, rmax).tocoo().tocsr()
    Abool = A.astype('bool') 
    end = perf_counter()
    print(f'Done! [{end-start}]')

    # apply neighbour filter first
    print('Neighbour filter...', end =' ')
    start = perf_counter()
    undistorted = (Abool.sum(axis=1) == 3).nonzero()[0]
    print(undistorted)
    A = A[undistorted,:]
    end = perf_counter()
    print(f'Done! [{end-start}]')
    print('Nb. undistorted = ', undistorted.shape)
    print(undistorted)

    # distance filter
    print('\nDistance filter...', end=' ')
    start = perf_counter()
    dfilter = np.zeros(undistorted.shape[0],dtype='bool')
    for k in range(len(undistorted)):
        _, _, dists = find(A.getrow(k))
        dfilter[k] = np.all((dists >= rmin) * (dists <= rmax))
    undistorted = undistorted[dfilter.nonzero()[0]]
    end = perf_counter()
    print(f'Done! [{end-start}]')
    print('Nb. undistorted = ', undistorted.shape)
    print(undistorted)
     
    undistorted_neighbours = Abool[undistorted,:].nonzero()

    # angle filter
    print('\nAngle filter...', end=' ')
    start = perf_counter()
    undistorted = angle_filter(pos, undistorted, undistorted_neighbours,theta_graphene,eps_rel)
    end = perf_counter()
    print(f'Done! [{end-start}]')
    print('Nb. undistorted = ', undistorted.shape)
    print(undistorted)

    return undistorted
        

def angle_filter(pos,undistorted,neighbours,theta_graphene=2*np.pi/3, eps_rel=0.1):
    """Checks angle criterion (see docstring of `undistorted_carbons` above) for undistorted atoms."""
    theta_max = theta_graphene * (1 + eps_rel)
    theta_min = theta_graphene * (1 - eps_rel)
    good_theta = np.zeros(undistorted.shape[0],dtype='bool')
    print('yeboii: ', undistorted)
    print(undistorted.shape)
    for k, n in enumerate(undistorted):
        r0 = pos[n,:]
        nn = neighbours[1][(neighbours[0] == k).nonzero()[0]] # k instead of n here bc Abool has already been filtered (see def of undistorted_neihgbours in undistorted_carbons)
        print(f'{n} -->  {nn}')
        neighbour_pairs = combinations(nn,2)
        for m1, m2 in neighbour_pairs:
            r1 = pos[m1,:]
            r2 = pos[m2,:]
            v1 = r0 - r1
            v2 = r0 - r2
            theta = np.arccos( np.dot(v1,v2) / (np.linalg.norm(v1)*np.linalg.norm(v2)) )
            good_theta[k] = (theta <= theta_max) and (theta >= theta_min)
    return undistorted[good_theta]


def undistorted_clusters(pos, undistorted, rmax):
    """Obtains the clusters of undistorted carbons in a MAC structure."""
    undistorted_pos = pos[undistorted,:]
    tree = KDTree(undistorted_pos)
    M = tree.sparse_distance_matrix(tree,rmax).todense()
    clusters = jitted_components(M)
    global_clusters = [[undistorted[k] for k in c] for c in clusters] # re-index atoms in `clusters` using their indices in `pos`
    return global_clusters
    



# -------- MAIN ---------

structype = sys.argv[1]
nn = int(sys.argv[2])

if structype == '40x40':
    xyz_prefix = 'bigMAC-'
else:
    xyz_prefix = structype + 'n'

full_pos = read_xyz(path.expanduser(f'~/scratch/clean_bigMAC/{structype}/relaxed_structures_no_dangle/{xyz_prefix}{nn}_relaxed_no-dangle.xyz'))
full_pos = full_pos[:,:2]

eps_rel = 0.1
rCC_graphene = 1.42
rmax = rCC_graphene * (1+eps_rel)
undisC = undistorted_carbons(full_pos)
print('Doing cluster search...', end = ' ')
start = perf_counter()
clusters = undistorted_clusters(full_pos, undisC,rmax)
end = perf_counter()
print(f'Done! [{end-start} seconds]')
print(f'Found {len(clusters)} clusters!')
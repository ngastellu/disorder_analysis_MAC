#!/usr/bin/env python

import numpy as np
import pickle
from os import path
from glob import glob
from qcnico.lattice import cartesian_product
from qcnico.graph_tools import adjacency_matrix_sparse, count_rings, cycle_centers, hexagon_adjmat
from scipy import sparse
from time import perf_counter



def subsample_MAC_half_step(pos,l,m,n,m_max,n_max,return_global_indices=False):
    """Returns the l*l square sample of a L*L MAC structure corresponding to coordinates m*l ≤ x ≤ (m+1)*l and 
    n*l ≤ y ≤ (n+1)*l.
    This half-step sampling ensures that no region of the sample will lie solely on the edge of a subsample.
    
    If `return_global_indices` is set to True, this function will also return the indices of the subsampled atoms in
    full `pos` array."""
    
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
    if return_global_indices:
        return pos[mask,:], mask.nonzero()[0]
    else:
        return pos[mask,:]

def get_rings_from_subsamp(full_pos,nprocs,rank, nn, save_explicit_rings=True, max_ring_size=8,outdir='.'):

    """This is a driver function which runs the MPI-parallelised ring analysis code for a given structure.
    Given a number of MPI jobs `nprocs` (which HAS to be a perfect square), it determines the size of the subsamples,
    divides the structure up appropriately, and does the ring analysis (i.e. identifies the rings, their centers, and 
    builds the local atom and hexagon adjacency matrices).
    
    If `save_explicit_rings` is set to `True`, then this code will save the rings (each represented by the list of
    indices of its component atoms) as pickled list.
    
    The variable `nn` is basically just a label, useful to differentiate outputs from different structures if multiple 
    structures are stored in the same file (i.e. mupltiple Ata test structures in the same NPY)."""
    
    full_pos = full_pos[:,:2]

    a = np.sqrt(nprocs) #nprocs should always be a perfect square
    if a % 1 != 0:
        print('ERROR: Number of MPI processes must be a perfect square! It is currently set to: ', nprocs)
        print('Returning None.')
        return None
    
    a = int(a)


    Lx = np.max(full_pos[:,0]) - np.min(full_pos[:,0])
    Ly = np.max(full_pos[:,1]) - np.min(full_pos[:,1])
    L = np.max(np.ceil([Lx,Ly]))
    l = L // (a+1) # I did the math, this ensures a proper partitioning of the structure into nprocs subsamples

    ii_sample = cartesian_product(np.arange(a),np.arange(a))
    m,n = ii_sample[rank]

    print(f'[{rank+1}] Sample indices: ({m,n})')
    if save_explicit_rings:
        pos, iatoms = subsample_MAC_half_step(full_pos,l,m,n,a-1,a-1,return_global_indices=True)
    else:
        pos = subsample_MAC_half_step(full_pos,l,m,n,a-1,a-1,return_global_indices=False)

    np.save(path.join(outdir,f'pos_sample-{nn}_{m}_{n}.npy'),pos)

    rCC = 1.8

    _, rings, M = count_rings(pos,rCC,max_size=max_ring_size,return_cycles=True,return_M=True)

    hexs = [c for c in rings if len(c)==6]
    ring_lengths = np.array([len(c) for c in rings])
    ring_centers = cycle_centers(rings, pos)
    hex_centers = cycle_centers(hexs, pos)
    Mhex = hexagon_adjmat(hexs)

    np.save(path.join(outdir,f'M_hex-{nn}_{m}_{n}.npy'), Mhex)
    np.save(path.join(outdir,f'M_atoms-{nn}_{m}_{n}.npy'), M)
    np.save(path.join(outdir,f'hex_centers-{nn}_{m}_{n}.npy'), hex_centers)
    np.save(path.join(outdir,f'ring_centers-{nn}_{m}_{n}.npy'), ring_centers)
    np.save(path.join(outdir,f'ring_lengths-{nn}_{m}_{n}.npy'), ring_lengths)

    if save_explicit_rings:
        rings_global = [[iatoms[i] for i in c] for c in rings] #list of all rings in subsample with globally indexed atoms
        hexs_global = np.array([[iatoms[i] for i in h] for h in hexs]) #list of hexagons in subsamp with globally indexed atoms

        with open(path.join(outdir, f'cycles-{nn}_{m}_{n}.pkl'), 'wb') as fo:
            pickle.dump(rings_global, fo)

       # save hexs separately because this will make my life easier to determine 
       # which atoms are in crystalline clusters 
        np.save(path.join(outdir, f'hexs-{nn}_{m}_{n}.npy'), hexs_global)
    

def get_a(datadir, nn):
    """Estimates the number of paritions along one direction from the output of an MPI-parallelised ring
    analysis job (i.e. variable `a` from `get_rings_from_subsamp`), using the number of 'M_hex' files.
    Assumes that all of the output files are formated as 'M_hex-nn_m_n.npy'."""

    Mhex_files = glob(path.join(datadir, f'M_hex-{nn}_*.npy'))
    nvals = [int(f.split('_')[-1].split('.')[0]) for f in Mhex_files]
    return max(nvals) + 1



def rebuild_rings(nn,datadir=None):
    """Reconstructs rings from subsampled MPI runs:
        * Places all of the rings (identified by their center of mass) into a single array

        * Creates a single array of all of the ring lengths, ordered in the same as the ring
           centers array.

        * Constructs the adjacency matrix of all hexagons in the structure and uses it to 
           determine which hexagons are crystalline
        
    This whole procedure basically removes redundant rings from overlapping subsamples and stitches the
    hexagon network of the full structure back together using the local hexagon adjacency matrices.
    
    If `explicit_rings` is set to `True`, this function also creates a list of all of the rings in the structure
    where the ring is represented by the list of the global indices of it component atoms"""

    from qcnico.jitted_cluster_utils import get_clusters

    if datadir is None:
        datadir = f'sample-{nn}'

    a = get_a(datadir,nn)

    slice_inds = cartesian_product(np.arange(a),np.arange(a))

    start = perf_counter()

    m,n = slice_inds[0,:]

    print(f'Initialising hash maps (m,n) = ({m,n})',flush=True)
    hex_pos_global = {tuple(r):k for k,r in enumerate(np.load(path.join(datadir,f'hex_centers-{nn}_{m}_{n}.npy')))} # global hashtable mapping hexagon centers to integer indices
    all_pos_global = {tuple(r):k for k,r in enumerate(np.load(path.join(datadir,f'ring_centers-{nn}_{m}_{n}.npy')))} # global hashtable mapping ring centers to integer indices
    all_lengths = np.load(path.join(datadir, f'ring_lengths-{nn}_{m}_{n}.npy'))


    M = np.load(path.join(datadir, f'M_hex-{nn}_{m}_{n}.npy'))
    neighb_list = {k:tuple(M[k,:].nonzero()[0]) for k in range(M.shape[0])}

    ncentres_tot = M.shape[0]
    ncentres_all_tot = all_lengths.shape[0]

    assert len(hex_pos_global) == ncentres_tot, f'Mismatch between number of centers ({hex_pos_global.shape[0]}) and shape of hexagon adjacency matrix {M.shape}!'
    print('Done! Commencing loop over other subsamples...',flush=True)


    for mn in slice_inds[1:]:
        m,n = mn
        print(f'\n------ {(m,n)} ------',flush=True)
        hex_pos = np.load(path.join(datadir, f'hex_centers-{nn}_{m}_{n}.npy'))
        print(f'{hex_pos.shape[0]} distinct crystalline centers.', flush=True)
        local_map_hex = {k:-1 for k in range(hex_pos.shape[0])} # hashtable that maps centre indices local to the NPY being processed to their global index (i.e. in `hex_pos_global`)  

        print('Loop 1: ', end='')
        # first, update the global hashtable to properly index centers in subsample (m,n)
        for k, r in enumerate(hex_pos):
            r = tuple(r)
            if r in hex_pos_global:
                #print(f'* {r} in hex_pos_global *', flush=True)
                local_map_hex[k] = hex_pos_global[r] # if this centre has been seen, add its global index to the local hashmap
            else:
                #print(f'~ Adding {r} to hex_pos_global ~', flush=True)
                hex_pos_global[r] = ncentres_tot # if this centre hasn't yet been seen; assign a new index to it in global hashmap
                local_map_hex[k] = ncentres_tot # idem for local hashmap
                ncentres_tot += 1 # prepare index for next unseen centre
        print('Done!',flush=True)
        vals = np.array(local_map_hex.values())

        print('Loop 2: ', end='',flush=True)
        # next, update neighbour list using global hashmap
        M = np.load(path.join(datadir, f'M_hex-{nn}_{m}_{n}.npy'))
        for k in range(hex_pos.shape[0]):
            k_global = local_map_hex[k]
            ineighbs_local = tuple(M[:,k].nonzero()[0])
            # handle case 
            if k_global in neighb_list:
                neighb_list[k_global] = neighb_list[k_global] + tuple(local_map_hex[p] for p in ineighbs_local)
            else:
                neighb_list[k_global] = tuple(local_map_hex[p] for p in ineighbs_local)
        print('Done!',flush=True)

        print('Loop 3 (all rings): ', end='', flush=True)
        all_pos = np.load(path.join(datadir, f'ring_centers-{nn}_{m}_{n}.npy'))
        lengths = np.load(path.join(datadir, f'ring_lengths-{nn}_{m}_{n}.npy'))
        new_lengths_local = []
        for k, r in enumerate(all_pos):
            r = tuple(r)
            if r in all_pos_global:
                continue
            else:
                all_pos_global[r] = ncentres_all_tot
                new_lengths_local.append(lengths[k]) # ring lengths are sorted in same order as ring centres
                ncentres_all_tot += 1
        all_lengths = np.hstack((all_lengths,new_lengths_local))
        print(f'Done! Added {len(new_lengths_local)} rings to global hashmap.', flush=True)

    end = perf_counter()
    print(f'\n**** Building hashtables took {end-start} seconds. ****\n',flush=True)


    print('Constructing global hexagon adjacency matrix...', flush=True)
    start = perf_counter()

    Mglobal = np.zeros((ncentres_tot,ncentres_tot),dtype=bool)

    isnucleus = np.zeros(ncentres_tot,dtype=bool)
    isweird = np.zeros(ncentres_tot,dtype=bool)

    for k in range(ncentres_tot):
        ineighbs = neighb_list[k]
        Mglobal[k,ineighbs] = True
        Mglobal[ineighbs,k] = True

        nb_neighbs = np.unique(ineighbs).shape[0]
        if nb_neighbs == 6:
            isnucleus[k] = True
        elif nb_neighbs > 6:
            isweird[k] = True
    end = perf_counter()
    print(f'**** Done! [{end - start} seconds] ****\nSaving stuff.', flush=True)

    np.save(path.join(datadir, f'hex_global-{nn}.npy'),Mglobal)

    with open(path.join(datadir, f'centres_hashmap-{nn}.pkl'), 'wb') as fo:
        pickle.dump(hex_pos_global,fo)

    with open(path.join(datadir, f'neighbs_dict-{nn}.pkl'), 'wb') as fo:
        pickle.dump(neighb_list,fo)

    nuclei = isnucleus.nonzero()[0]
    print(f'*** Found {nuclei.shape[0]} crystalline nuclei ***', flush=True)

    if isweird.sum() > 0:
        weird = isweird.nonzero()[0]
        print(f'!!!! Foundi {weird.shape[0]} weird nuclei !!!! Printing their number of neighbours now: ', flush=True)
        for w in weird:
            print(f'{w} --> {Mglobal[w,:].sum()}', flush=True)


    print('Searching for clusters...',flush=True)
    start = perf_counter()
    Mglobal = sparse.csr_array(Mglobal.astype(np.int8)) #use sparse CSR matrix: DRAMATICALLY speeds up matrix product
    nuclei_neighbs = np.unique(Mglobal[nuclei,:].nonzero()[1])
    Mglobal2 = Mglobal @ Mglobal
    nuclei_next_neighbs = np.unique(Mglobal2[nuclei,:].nonzero()[1])
    strict_6c = set(np.concatenate((nuclei,nuclei_neighbs,nuclei_next_neighbs)))
    cluster_start = perf_counter()
    print(f'[{cluster_start - start} seconds later] Starting `get_clusters`...',flush=True)
    Mglobal = Mglobal.tolil() #convert to LIL format: fast row-slicing and efficient updates to sparsity structure
    crystalline_clusters = get_clusters(nuclei, Mglobal.toarray(), strict_6c)
    end = perf_counter()
    print(f'**** Done! Total time = {end - start} seconds. Time spent in `get_cluster` = {end - cluster_start} seconds ****',flush=True)

    cluster_sizes = np.array([len(c) for c in crystalline_clusters])
    np.save(path.join(datadir, f'cryst_cluster_sizes-{nn}.npy'),cluster_sizes)

    print('Building all_centres to match order in `all_lengths`...')
    start = perf_counter()
    all_centres = np.zeros((all_lengths.shape[0], 2))
    for r, k in all_pos_global.items():
        all_centres[k] = r
    end = perf_counter()
    np.save(path.join(datadir, f'all_ring_centers-{nn}.npy'), all_centres)
    np.save(path.join(datadir, f'all_ring_lengths-{nn}.npy'), all_lengths)
    print(f'Done! Total time = {end - start} seconds.',flush=True)


    with open(path.join(datadir, f'clusters-{nn}.pkl'), 'wb')    as fo:
        pickle.dump(crystalline_clusters,fo)

def crystalline_atoms(full_pos, nn,datadir=None):
    """Generates a mask `m` filtering which atoms in a given structure which belong to a crystalline cluster from the 
    REBUILT output of MPI parallel ring analysis: m[n] = True iff nth atom is in a crystalline hexagon."""


    if datadir is None:
        datadir = f'sample-{nn}'

    N = full_pos.shape[0]

    # build list of all crystalline hexagons
    with open(path.join(datadir, f'clusters-{nn}.pkl'), 'rb') as fo:
        cryst_clusters = pickle.load(fo)

    cryst_hexs = cryst_clusters[0].union(*cryst_clusters[1:])
    crystalline_mask = np.zeros(N,dtype=bool)

    # obtain global hashmap hex center ---> hex index
    with open(path.join(datadir, f'centres_hashmap-{nn}.pkl'), 'rb') as fo:
        hex_centres_hashmap = pickle.load(fo)
    
    hex_centres = np.zeros((len(hex_centres_hashmap),2))
    for r, k in hex_centres_hashmap.items():
        hex_centres[k] = r

    cryst_centres = hex_centres[list(cryst_hexs)]
    cryst_centres = set(tuple(r) for r in cryst_centres)
    
    # loop over all local outputs to determine which atoms are in the cryst clusters
    a = get_a(datadir,nn)

    slice_inds = cartesian_product(np.arange(a),np.arange(a))

    for mn in slice_inds:
        m,n = mn
        local_hex_centres = np.load(path.join(datadir, f'hex_centers-{nn}_{m}_{n}.npy'))
        local_hex_atoms = np.load(path.join(datadir, f'hexs-{nn}_{m}_{n}.npy'))
        for k,r in local_hex_centres:
            r = tuple(r)
            if r in cryst_centres:
                icryst = local_hex_atoms[k]
                crystalline_mask[icryst] = True
    
    return crystalline_mask

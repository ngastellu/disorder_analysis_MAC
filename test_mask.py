#!/usr/bin/env python

import numpy as np



def get_site_atoms(atom_array,mask,n):
    """
    This function provides a way to identify which carbon atoms belong to a given hopping site using a
    mask.
    
    A mask M is (Ns * Na) array of bools, where Ns = number of hopping sites, and Na = number of atoms.
    Its jth row M[j,:] is associated with the jth hopping site of its associated AMC structure.
    If M[j,n] = True, then the structure's nth carbon atom (as indexed in its XYZ file) contributes
    to the AMC structure's jth hopping site.

    Masks can also be used to directly extract any per-atom property of interest from each site. 
    Supposing you had an array E of atomic energies for a given AMC structure, where E[n] corresponds
    to the energy of the nth carbon atom (again, as indexed in its XYZ file), E[M[j]] would yield the
    energies of all carbons belonging to the jth hopping site.

    Parameters
    ----------
    atom_array: `np.ndarray`, shape=(Na,d) or (Na,), dtype = any
        An array of atomic properties, could be positions (as in this example), energies, indices, or any other
        property in a AMC structure. 
        ** N.B.: Make sure the atoms in `atoms_array` are sorted in the same order as they are in the 
                position files in the GitHub repo. **

    mask: `np.ndarray`, shape=(Ns, Na), dtype = bool
        Mask associated with the structure whose atomic properties are described by `atom_array`.

    n: `int` between 0 and Ns - 1
        Index of the site whose atoms you want to identify.

    Output
    ------
    out: `np.ndarray`, shape=(Nn,)
        Subset of `atom_array` corresponding to the Nn atoms which belong to the site `n`
    """

    return atom_array[mask[n]]


def read_xyz(filepath,return_symbols=False):
    """Returns the coordinates (and symbols, optionally) of all atoms stored in a .xyz file.

    Parameter
    ----------
    filepath: `str`
        Path to the .xyz file whose coordinates we wish to obtain.

    Output
    ------
    coords: `ndarray`, shape=(N,3), dtype=`float`
        Array of coordinates stored in the input file.
    symbols: `ndarray`, shape=(N,), dtype=`str`
        Symbols corresponding to the atoms whose coordinates are stored in the input file.
    """
    with open(filepath) as fo:
        natoms = int(fo.readline().rstrip().lstrip().lstrip('#'))
        fo.readline() #skip 2nd line
        lines = fo.readlines()

    if return_symbols:
        N = len(lines)
        symbols = [None] * N
        coords = np.zeros((N,3),dtype=float)

        for k, line in enumerate(lines):
            split_line = line.rstrip().lstrip().split()
            symbols[k] = split_line[0]
            coords[k,:] = list(map(float, split_line[1:4]))

        return coords, symbols

    else:
        coords = np.array([list(map(float,line.lstrip().rstrip().split()[1:4])) for line in lines])

        return coords


if __name__ == '__main__':
    """
    This script illustrates how to use a mask array to determine which atoms belong to a particular
    hopping site.

    It reads the atomic coordinates of a given AMC structure from a user-specified XYZ file, and a mask
    array (defined below) associated with that structure.
    It then prints out the atomic coordinates of the atoms belong to the 10th hopping site of that 
    fragment, for the sake of example. It also runs a little consistency check at the end.

    In this example, I'm using the 'naive' site masks (see explanatory note).
    """

    # Specify desired AMC structure
    structure_type = 'q400'
    structure_index = 42

    # Read its atomic coordinates
    posfile = f'structures/sAMC-{structure_type}/sAMC{structure_type}-{structure_index}.xyz' #path to the file containing the atomic positions of the structure at hand
    coords = read_xyz(posfile)

    # Load its hopping site mask
    mask_file = f'hopping_site_masks/sAMC-{structure_type}/hopping_site_masks-{structure_index}.npy' #path to the file containing the masks for each site
    M = np.load(mask_file)

    # Apply mask to obtain the coords of carbon atoms belonging to one of its hopping sites (e.g. the 10th site)
    site_index = 9
    site_atomic_coords = get_site_atoms(coords,M,site_index)
 
    print(f"Atomic coords of hopping site {site_index}:")
    for r in site_atomic_coords:
        print(r)
    
    # Apply mask to obtain indices of the carbon atoms belonging to the same site as above
    atom_indices = np.arange(coords.shape[0])
    site_atom_indices = get_site_atoms(atom_indices, M, site_index)

    print("Results are consistent (should be True): ", np.all(site_atomic_coords == coords[site_atom_indices]))
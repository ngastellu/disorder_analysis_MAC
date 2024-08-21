#!/usr/bin/env python 

import numpy as np
import pickle



def ring_stats_rebuild(datadir, xlim=np.array([-np.inf,np.inf]), ylim=np.array([-np.inf, np.inf]),nn=0):
    rl_filename = f'all_ring_lengths-{nn}.npy'
    cluster_filename = f'clusters-{nn}.pkl'

    all_ring_lengths = np.load(datadir + rl_filename)

    ring_lengths = np.arange(3,9) 
    ring_stats = np.zeros(ring_lengths.shape[0] + 1) # add one slot to accomodate for 6i/6c distinction

    with open(datadir + cluster_filename, 'rb') as fo:
        cryst_clusters = pickle.load(fo) 
    
    cryst_hexs = cryst_clusters[0].union(*cryst_clusters[1:])

    if np.any(~np.isinf(xlim)) or np.any(~np.isinf(ylim)):
        all_centres = np.load(datadir + f'all_ring_centers-{nn}.npy')

        # find inds of all n-rings that lie in the desired region
        x_mask = (all_centres[:,0] >= xlim[0]) * (all_centres[:,0] <= xlim[1])
        y_mask = (all_centres[:,1] >= ylim[0]) * (all_centres[:,1] <= ylim[1])

        mask = x_mask * y_mask


        all_ring_lengths = all_ring_lengths[mask] #keep only the rings in region of interest


        # find inds of hexagons that lie in the desired region
        with open(datadir + f'centres_hashmap-{nn}.pkl', 'rb') as fo:
            hex_centres_dict = pickle.load(fo)
        
        # this loads the dictionary into an array which preserves the hex centre --> index mapping
        hex_centres = np.zeros((len(hex_centres_dict),2))
        for r, k in hex_centres_dict.items():
            hex_centres[k] = r
            
        x_mask = (hex_centres[:,0] >= xlim[0]) * (hex_centres[:,0] <= xlim[1])
        y_mask = (hex_centres[:,1] >= ylim[0]) * (hex_centres[:,1] <= ylim[1])

        mask = x_mask * y_mask

        hexs_filtered = set(mask.nonzero()[0])
        nhexs = (all_ring_lengths == 6).sum()

        print('Number of hexagons match: ', nhexs == len(hexs_filtered)) #sanity check

        # finally keep only crsytalline heaxgons that lie in the desired region
        cryst_hexs = cryst_hexs & hexs_filtered

    nb_6c = len(cryst_hexs)
    nb_6i = (all_ring_lengths == 6).sum() - nb_6c

    for k in range(3):
        ring_stats[k] = (all_ring_lengths == 3+k).sum()

    # store 6c and 6i in 'wrong' order (wrt fig in Tian paper) bc plotting function swaps them anyways
    ring_stats[3] = nb_6i
    ring_stats[4] = nb_6c

    for k, n in enumerate(ring_lengths[4:]): #ring lengths starts at 3; only consider ring lenghts > 6 here
        ring_stats[5+k] = (all_ring_lengths == n).sum()
    
    return ring_stats



nn = 181
datadir = f'/Users/nico/Desktop/simulation_outputs/structural_characteristics_MAC/labelled_rings/small_kMC_test/sample-{nn}/'

# system_name = 'conditiondot99'
# ringstats_dir = '/Users/nico/Desktop/simulation_outputs/ring_stats_40x40_pCNN_MAC/'
ring_stats = ring_stats_rebuild(datadir,nn=nn,xlim=(-np.inf,20),ylim=(-np.inf,20))
print(ring_stats)
# ring_stats /= ring_stats.sum()

# np.save(ringstats_dir + f'ring_stats_{system_name}_righthalf.npy', ring_stats)
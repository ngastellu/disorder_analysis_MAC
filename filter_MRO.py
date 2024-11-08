#!/usr/bin/env python

import numpy as np


def filterMRO(eta_MRO_dat,eta_min,eta_max):
    "Returns indices of structures whose eta_MRO param lies between `eta_min` and `eta_max`."
    inds = eta_MRO_dat[:,0]
    etas = eta_MRO_dat[:,1]
    efilter = (etas >= eta_min) * (etas <= eta_max)
    print(f'Keeping {efilter.sum()}/{etas.shape[0]} structures')
    return inds[efilter]


eta_max_tdot6 = -0.99
eta_min_tdot6 = -2

eta_max_tdot5 = -0.05
eta_min_tdot5 = -0.90


eta_bounds = [[eta_min_tdot6, eta_max_tdot6], [eta_min_tdot5, eta_max_tdot5]]
structypes = ['tempdot6', 'tempdot5']
for st, eb in zip(structypes, eta_bounds):
    eta_MRO_dat = np.load(f'eta_MRO_{st}_rmax12.npy')
    eta_min, eta_max = eb
    good_inds = filterMRO(eta_MRO_dat, eta_min, eta_max)
    np.save(f'ifiltered_MRO_{st}.npy', good_inds.astype('int'))
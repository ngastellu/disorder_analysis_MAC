#!/usr/bin/env python

from time import perf_counter
import os
import sys
import numpy as np
from qcnico.coords_io import read_xyz
from qcnico.graph_tools import count_rings, classify_hexagons, cycle_centers



structype = sys.argv[1]
nn = int(sys.argv[2])

if structype == '40x40':
    xyz_prefix = 'bigMAC-'
else:
    xyz_prefix = structype + 'n'

pos = read_xyz(os.path.expanduser(f'~/scratch/clean_bigMAC/{structype}/relaxed_structures_no_dangle/{xyz_prefix}{nn}_relaxed_no-dangle.xyz'))
pos = pos[:,:2]

rCC = 1.8


print('Getting rings...',end=' ',flush=True)
start = perf_counter()
_, rings = count_rings(pos,rCC,max_size=7,return_cycles=True)
end = perf_counter()
print(f'Done! [{end-start} seconds]', flush=True)

print('Classifying hexagons and getting clusters...', end = ' ', flush=True)
start = perf_counter()
hexs = [c for c in rings if len(c)==6]
_, _, clusters = classify_hexagons(hexs,return_cryst_clusters=True)
end = perf_counter()
print(f'Done! [{end-start} seconds]', flush=True)


print(f'Calculating cluster sizes and saving errythang...', flush=True)
clusters_hexs = [[hexs[n] for n in c] for c in clusters]
cluster_sizes = [len(C) for C in clusters_hexs]

np.save(f'cryst_cluster_sizes-{nn}.npy',cluster_sizes)

cluster_centres = [cycle_centers(c, pos) for c in clusters]

np.save(f'cluster_centres-{nn}.npy', cluster_centres)

end = perf_counter()
print(f'Done! [{end-start} seconds]', flush=True)

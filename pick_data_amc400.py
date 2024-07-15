#!/usr/bin/env python

import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import KDTree
from qcnico.plt_utils import setup_tex, get_cm

ring_stats =  np.load('volker_full_dataset_ring_stats.npy')
p6i = ring_stats[:,3] / ring_stats.sum(axis=1)
p6c = ring_stats[:,4] / ring_stats.sum(axis=1)

tree = KDTree(np.vstack((p6i,p6c)).T)

target = np.array([0.445,0.279])
tols = np.linspace(0.08,0.11,10) # max allowed deviation from AMC400 ring stats
tols = tols[::-1]
clrs = get_cm(tols,max_val=0.8,cmap_str='inferno')
avgs = np.zeros((tols.shape[0],2))
nums = np.zeros(tols.shape[0],dtype='int')

zz = 1

setup_tex()
fig, ax = plt.subplots()


for k, tolerance in enumerate(tols):
    r = tolerance * np.sqrt(2) #distance from AMC-400 in (p6i,p6c) space
    iselected = tree.query_ball_point(target,r)
    avgs[k,0] = np.mean(p6i[iselected])
    avgs[k,1] = np.mean(p6c[iselected])
    nums[k] = len(iselected)

    ax.scatter(p6i,p6c,alpha=0.7,s=20.0)
    ax.scatter(p6i[iselected],p6c[iselected],alpha=0.9,s=20.0,c=clrs[k],zorder=zz)
    ax.scatter(np.mean(p6i[iselected]),np.mean(p6c[iselected]),marker='*',s=100.0,c=clrs[k],edgecolors='white',lw=0.7,zorder=zz+10)
    zz += 1


ax.scatter(*target,marker='*',s=200.0,c='r',edgecolors='k',lw=0.7)
ax.set_xlabel('$p_{6i}$')
ax.set_ylabel('$p_{6c}$')
ax.set_aspect('equal')
plt.show()

plt.plot(tols, nums,'-o')
plt.show()
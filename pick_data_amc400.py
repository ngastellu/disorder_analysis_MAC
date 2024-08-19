#!/usr/bin/env python

import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import KDTree
from qcnico.plt_utils import setup_tex, multiple_histograms
import pickle

ring_stats =  np.load('volker_full_dataset_ring_stats.npy')
p6i = ring_stats[:,3] / ring_stats.sum(axis=1)
p6c = ring_stats[:,4] / ring_stats.sum(axis=1)

np.save('p6i_volker.npy', p6i)
np.save('p6c_volker.npy', p6c)

tree = KDTree(np.vstack((p6i,p6c)).T)

ring_stats /= ring_stats.sum(axis=1)[:,None]
ring_names = ['3', '4', '5', '6i', '6c', '7', '8']
ring_stats_dict = {rn:rd for rn, rd in zip(ring_names, ring_stats.T)}
with open('volker_ring_data_full_set.pkl', 'wb') as fo: #save data for Ata
    pickle.dump(ring_stats_dict,fo)

target = np.array([0.445,0.279])


tolerance = 0.08 # allow 8% deviation from AMC-400 ring stats
r = tolerance * np.sqrt(2) #distance from AMC-400 in (p6i,p6c) space
iselected = tree.query_ball_point(target,r)
nums = len(iselected)

s1 = set(iselected)
sel_points1 = np.vstack((p6i[iselected],p6c[iselected])).T
avg1 = np.mean(sel_points1,axis=0)

# lower the tolerance but pick only points with more 6c
tolerance2 = 0.30
r2 = tolerance2 * np.sqrt(2)
iselected2 = tree.query_ball_point(target,r2)
s2 = set(iselected2)
s_added = np.array(list(s2 - s1))
nums_added = len(s_added)
sel_points2 = np.vstack((p6i[np.array(list(s_added))],p6c[np.array(list(s_added))])).T
sel_points2_old = sel_points2.copy()
mask = (sel_points2[:,1] > 0.27) * (sel_points2[:,0] > 0.27)
sel_points2 = sel_points2[mask]
avg2 = np.mean(sel_points2,axis=0)

sel_points_del = sel_points2_old[~mask]

s_added2 = np.array(list(set(s_added[mask]) | s1))
sel_points2 = np.vstack((p6i[np.array(list(s_added2))],p6c[np.array(list(s_added2))])).T 
avg2 = np.mean(sel_points2,axis=0)

s_added2 = set(s_added2)

big_p6i = set((p6i > 0.4).nonzero()[0])

# We want to add high-p6c points who also have p6i values, we do this by
# picking all points who lie above a certain slanted line in the (p6i,p6c) plane
# slope1 = 1.607
slope2 = 2.1
big_p6c = set(((p6c > 0.5) * (p6i > 0.12)).nonzero()[0])
big_p6c = set((p6c > (0.97 - slope2*p6i)).nonzero()[0])


s_added3 = s_added2 | big_p6i
sel_points3 = np.vstack((p6i[np.array(list(s_added3))],p6c[np.array(list(s_added3))])).T 
avg3 = np.mean(sel_points3,axis=0)

s_added4 = s_added3 | big_p6c
sel_points4 = np.vstack((p6i[np.array(list(s_added4))],p6c[np.array(list(s_added4))])).T 
avg4 = np.mean(sel_points4,axis=0)

high_crystalline = set((p6c > 0.6).nonzero()[0])
s_added_final = s_added4 | high_crystalline
sel_points_final = np.vstack((p6i[np.array(list(s_added_final))],p6c[np.array(list(s_added_final))])).T 
avg_final = np.mean(sel_points_final,axis=0)




setup_tex()
fig, ax = plt.subplots()

ax.scatter(p6i,p6c,alpha=0.7,s=20.0,zorder=1)
ax.scatter(*sel_points1.T,c='orange',alpha=0.9,zorder=2)
ax.scatter(*sel_points2.T,c='red',alpha=0.9,zorder=2)
ax.scatter(*sel_points_del.T,c='violet',alpha=0.3,zorder=2)


ax.scatter(*target,marker='*',s=200.0,c='limegreen',edgecolors='k',lw=0.7)
ax.scatter(*avg2,marker='*',s=200.0,c='r',edgecolors='white',lw=0.7,zorder=3)
ax.scatter(*avg1,marker='*',s=200.0,c='orange',edgecolors='white',lw=0.7,zorder=3)
ax.set_xlabel('$p_{6i}$')
ax.set_ylabel('$p_{6c}$')
ax.set_aspect('equal')
plt.show()

print('Size of first set: ', nums)
print('Nb of final points: ', sel_points_final.shape[0])
print(avg1)
print(avg_final)
print(target)

fig, ax = plt.subplots()
ax.scatter(p6i,p6c,alpha=0.7,s=20.0,zorder=1)
ax.scatter(*sel_points3.T,c='red',alpha=0.9,zorder=2,s=20.0)


ax.scatter(*target,marker='*',s=200.0,c='limegreen',edgecolors='k',lw=0.7)
ax.scatter(*avg1,marker='*',s=200.0,c='orange',edgecolors='white',lw=0.7,zorder=3)
ax.scatter(*avg3,marker='*',s=200.0,c='r',edgecolors='white',lw=0.7,zorder=3)
ax.set_xlabel('$p_{6i}$')
ax.set_ylabel('$p_{6c}$')
ax.set_aspect('equal')
plt.show()


all_pos = np.load('/Users/nico/Desktop/scripts/MAP_training/training/data/coords_13944p6.npy')
selected_pos = all_pos[np.array(list(s_added_final))]
np.save('new_AMC400_selected_indices.npy', np.array(list(s_added_final)))
np.save('new_AMC400_training_data_volker.npy', selected_pos)

fig, ax = plt.subplots()
ax.scatter(p6i,p6c,alpha=0.7,s=20.0,zorder=1)
ax.scatter(*sel_points_final.T,c='red',alpha=0.9,zorder=2,s=20.0)


ax.scatter(*target,marker='*',s=200.0,c='limegreen',edgecolors='k',lw=0.7)
ax.scatter(*avg1,marker='*',s=200.0,c='orange',edgecolors='white',lw=0.7,zorder=3)
ax.scatter(*avg_final,marker='*',s=200.0,c='r',edgecolors='white',lw=0.7,zorder=3)
ax.set_xlabel('$p_{6i}$')
ax.set_ylabel('$p_{6c}$')
ax.set_aspect('equal')
plt.show()


sel_p6i = sel_points_final[:,0]
sel_p6c = sel_points_final[:,1]
multiple_histograms((sel_p6c,sel_p6i), ('$p_{6c}$', '$p_{6i}$'))
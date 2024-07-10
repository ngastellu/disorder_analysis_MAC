#!/usr/bin/env python

import numpy as np
import matplotlib.pyplot as plt
from qcnico.plt_utils import setup_tex

datadir = '/Users/nico/Desktop/simulation_outputs/ring_stats_40x40_pCNN_MAC/'

avg_tdot25 = np.load(datadir + 'avg_ring_counts_tdot25_relaxed.npy')
std_tdot25 = np.load(datadir + 'std_dev_tdot25.npy')

avg_t1 = np.load(datadir + 'avg_ring_counts_t1_relaxed.npy')
std_t1 = np.load(datadir + 'std_dev_t1.npy')
avg_pCNN = np.load(datadir + 'avg_ring_counts_normalised.npy')



x = np.arange(3,12)
y_t1 = avg_t1 / avg_t1.sum()
y_tdot25 = avg_tdot25/ avg_tdot25.sum()
y_pCNN = avg_pCNN

dx = 0.25
multiplier = 0

lbls = ['$\\tilde{T}=1$ (38 samples)', '$\\tilde{T}=0.25$ (26 samples)', 'OG PixelCNN (300 samples)']
data = [y_t1, y_tdot25,y_pCNN]

setup_tex()

fig, ax = plt.subplots()

for y, lbl in zip(data, lbls):
    offset = dx * multiplier
    ax.bar(x+offset, y, width=dx,label=lbl)
    multiplier += 1
    print(f'{lbl}: {y*100}')
    print(y[-4:].sum())
    print('\n')

# for k,s in enumerate(std_t1.T):
#     n = k+3
#     cnt = y_t1[k]
#     err_pts = np.array([[n-dx/2,n-dx/2,n-dx/2], [cnt-s, cnt, cnt+s]])
#     ax.scatter(*err_pts, marker='_', color='k')
#     ax.plot(*err_pts,'k-',lw=0.8)

# for k,s in enumerate(std_tdot25.T):
#     n = k+3
#     cnt = y_tdot25[k]
#     err_pts = np.array([[n+dx/2,n+dx/2,n+dx/2], [cnt-s, cnt, cnt+s]])
#     ax.scatter(*err_pts, marker='_', color='k')
#     ax.plot(*err_pts,'k-',lw=0.8)



ax.set_xlabel('Ring types')
ax.set_xticks(x+dx, ['3', '4', '5', '6-c', '6-i', '7', '8', '9', '10'])
ax.set_ylabel('Average count (normalised)')
ax.set_title('Ring distribution in 40nm $\\times$ 40nm MAC')

# ax.errorbar(x,y,yerr=std[1,:])
plt.legend()
plt.show()

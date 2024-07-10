#!/usr/bin/env python

import numpy as np
import matplotlib.pyplot as plt
from qcnico.plt_utils import setup_tex

datadir = '/Users/nico/Desktop/simulation_outputs/ring_stats_40x40_pCNN_MAC/'

avg_tdot25 = np.load(datadir + 'avg_ring_counts_tdot25.npy')
std_tdot25 = np.load(datadir + 'std_dev_tdot25.npy')

avg_t1 = np.load(datadir + 'avg_ring_counts_t1.npy')
std_t1 = np.load(datadir + 'std_dev_t1.npy')




x = np.arange(3,12)
y_t1 = avg_t1
y_tdot25 = avg_tdot25
#y /= y.sum()
dx = 0.4

# Combine 6-c and 6-i
# y_tdotV25 = np.hstack((y_tdot25[0:3], [y_tdot25[3] + y_tdot25[4]], y_tdot25[5:]))
# y_t1 = np.hstack((y_t1[0:3], [y_t1[3] + y_t1[4]], y_t1[5:]))

y_t1 /= y_t1.sum()
y_tdot25 /= y_tdot25.sum()


setup_tex()

fig, ax = plt.subplots()


ax.bar(x, y_t1, align='edge', width=dx,label='$T=1$ (38 samples)')
ax.bar(x, y_tdot25, align='edge', width=-dx,label='$T=0.25$ (8 samples)')

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
ax.set_xticks(x, ['3', '4', '5', '6-c', '6-i', '7', '8', '9', '10'])
ax.set_ylabel('Average count (normalised)')
ax.set_title('Ring distribution in 40nm $\\times$ 40nm MAC')

# ax.errorbar(x,y,yerr=std[1,:])
plt.legend()
plt.show()

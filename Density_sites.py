from qcnico.coords_io import read_xyz
import sys
from os import path
import numpy as np
import math
import numba as nb
# from matplotlib import pyplot as plt



structype = sys.argv[1]
nn = int(sys.argv[2])

if structype == '40x40':
    xyz_prefix = 'bigMAC-'
else:
    xyz_prefix = structype + 'n'

dot = read_xyz(path.expanduser(f'~/scratch/clean_bigMAC/{structype}/relaxed_structures_no_dangle/{xyz_prefix}{nn}_relaxed_no-dangle.xyz'))
dot = dot[:,:2]
N = dot.shape[0]

visited=np.zeros(N)

@nb.jit(nopython=True)
def dist(i,j):
    return math.sqrt((dot[i][0]-dot[j][0])**2+(dot[i][1]-dot[j][1])**2)

@nb.jit(nopython=True)
def dev_dist():
    total,cnt,dev=0,0,0
    for i in range(N):
        for j in range(N):
            if dist(i,j)<2:
                total+=dist(i,j)
                cnt+=1
    aver=total/cnt
    print(aver)
    for i in range(N):
        for j in range(N):
            if dist(i,j)<2:
                dev+=(dist(i,j)-aver)**2
    return dev/cnt

def dfs(i,lst):
    temp=lst[:]
    for j in range(N):
        if dist(i,j)<2 and visited[j]==0:
            visited[j]=1
            temp.append(j)
            temp=dfs(j,temp)
    return temp

def cluster():
    clusters=[]
    for i in range(N):
        if visited[i]==0:
            visited[i]=1
            clusters.append(dfs(i,[i])) 
    return clusters

islands=cluster()

#not quite useful...
def island_average_dist(clusters):
    lst=[]
    for cluster in clusters:
        x_bar,y_bar=0,0
        for i in cluster:
            x_bar+=dot[i][0]
            y_bar+=dot[i][1]
        x_bar=x_bar/len(cluster)
        y_bar=y_bar/len(cluster)
        lst.append([x_bar,y_bar])
    M=len(lst)
    result,cnt=0,0
    for i in range(M):
        for j in range(M):
            result+=math.sqrt((lst[i][0]-lst[j][0])**2+(lst[i][1]-lst[j][1])**2)
            cnt+=1
    return result/cnt

cnt=0
colors=['green','red','darkorange','blue','black','purple','saddlebrown']
cnt=0

def density(num):
    x=[dot[i][0] for i in range(N)]
    y=[dot[i][1] for i in range(N)]
    S=(max(x)-min(x))*(max(y)-min(y))
    #print(S)
    return num/S

print('rho_sites: '+str(density(len(islands))*100))

# plt.xticks([])
# plt.yticks([])
# x=[dot[i][0] for i in range(N)]
# y=[dot[i][1] for i in range(N)]
# plt.xlim(min(x),max(x))
# plt.ylim(min(y),max(y))

# #plt.figure(figsize=(int(max(x)-min(x))/4,int(max(y)-min(y))/4))

# for island in islands:
#     color=colors[cnt%len(colors)]
#     x=[dot[island[i]][0] for i in range(len(island))]
#     y=[dot[island[i]][1] for i in range(len(island))]

#     plt.scatter(x,y,c=color,s=0.5)
#     cnt+=1


# plt.show()

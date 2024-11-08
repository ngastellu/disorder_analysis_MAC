#!/usr/bin/env python

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams
from qcnico.plt_utils import setup_tex, MAC_ensemble_colours


def compare_ring_stats(datadir,avg_files,labels,std_files=None,normalised=None, distinguished_hexagons=True,
                        colors=None, usetex=True, plt_objs=None, show=True, title=None, fontsize=20, fontsize_axes=20, reproduce_nature_tian=False):
    """Creates bar plot with ring distributions contained in the files in `avg_files`."""

    # avgs = np.array([np.load(datadir+avgf) for avgf in avg_files])
    avgs = [np.load(datadir+avgf) for avgf in avg_files]
    min_len = min([dat.shape[0] for dat in avgs])
    avgs = np.array([avg[:min_len] for avg in avgs]) #deal with uneven array lenghts 

    show_errorbars = False
    if std_files is not None:
        print('List of stderr files detected: skipping normalisation.')
        stds = np.array([np.load(datadir+stdf) for stdf in std_files])
        show_errorbars = True
    else:
        if normalised is not None:
            normalised = np.array(normalised)
            for n in (~normalised).nonzero()[0]:
                avgs[n] = avgs[n]/avgs[n].sum()
    
    # swap 6c and 6i to match Nature figure
    old_avgs = avgs.copy()
    for n in range(avgs.shape[0]):
        avgs[n,:2] = old_avgs[n,:2]
        avgs[n,3] = old_avgs[n,4]
        avgs[n,4] = old_avgs[n,3]
        avgs[n,5:] = old_avgs[n,5:]
    
    if reproduce_nature_tian: # reproduce fig 2h from https://doi.org/10.1038/s41586-022-05617-w
        avgs_copy = np.copy(avgs)
        avgs = avgs_copy[:,:6]
        avgs[:,5] += avgs_copy[:,6] # last entry will contain proportion of 7- and 8-membered rings
        avgs = avgs[:,2:] # omit triangles and squares
        
    
    avgs *= 100.0
    print(avgs)


    if plt_objs is None:
        fig, ax = plt.subplots()
    else:
        fig, ax = plt_objs

    if usetex:
        setup_tex()
    rcParams['font.size'] = fontsize
    rcParams['figure.figsize'] = [12.8,9.6]

    nrings = avgs.shape[1]
    x = np.arange(3,3+nrings)
    
    ndat = len(avg_files) 
    if colors is None:
        cyc = rcParams['axes.prop_cycle'] #default plot colours are stored in this `cycler` type object
        colors = [d['color'] for d in list(cyc[0:ndat])]
    
    if ndat == 2: 
        dx = 0.4
        ax.bar(x,avgs[0],align='edge', width=dx,label=labels[0], color=colors[0], edgecolor='k',lw=0.9)
        ax.bar(x,avgs[1],align='edge', width=-dx,label=labels[1], color=colors[1],edgecolor='k',lw=0.9)
        
        if show_errorbars:
           multiplier = +1
           for n, std in enumerate(stds):
                offset = (dx/2) * multiplier
                for k, s in enumerate(std):
                    y = avgs[n,k]
                    err_pts = np.array([[x[k]+offset, x[k]+offset, x[k]+offset], [y-s, y, y+s]])
                    ax.scatter(*err_pts, marker='_', color='k')
                    ax.plot(*err_pts,'k-',lw=0.8)
                multiplier = -1 
        dx = 0 # do not shift the xticks when ndat=2
    
    else:
        dx = 1/(1+ndat)
    
        multiplier = 0 
        
        for y, lbl, c in zip(avgs, labels, colors):
            offset = dx * multiplier
            ax.bar(x+offset, y, width=dx,label=lbl, color=c, edgecolor='k', lw=4.0)
            multiplier += 1
            print(f'{lbl}: {y*100}')
            print(y[-4:].sum())
            print('\n')
        
        if show_errorbars:
            multiplier = 0
            for n, std in enumerate(stds):
                offset = dx * multiplier
                for k, s in enumerate(std):
                    y = avgs[n,k]
                    err_pts = np.array([[x[k]+offset, x[k]+offset, x[k]+offset], [y-s, y, y+s]])
                    ax.scatter(*err_pts, marker='_', color='k')
                    ax.plot(*err_pts,'k-',lw=0.8)

                multiplier += 1

    ax.set_xlabel('Ring types',fontsize=fontsize_axes)
    ax.tick_params(axis='y', labelsize=fontsize)
    
    if distinguished_hexagons:
        if reproduce_nature_tian:
            ax.set_xticks(x+dx, [str(5), str(6) + '-c', str(6) + '-i',str(7) + '/' + str(8)],fontsize=fontsize)
        else:
            ax.set_xticks(x+dx, ['3', '4', '5', '6-c', '6-i'] + [str(n) for n in range(7,nrings+2)],fontsize=fontsize)
    else:
        ax.set_xticks(x+dx, [str(n) for n in range(3,nrings+3)],fontsize=fontsize)
    
    
    ax.tick_params(axis='both', length=10, width=2.5)

    if normalised is not None:
        ax.set_ylabel('Percentage (%)',fontsize=fontsize_axes)
    else:
        ax.set_ylabel('Average count',fontsize=fontsize_axes)

    if title is not None:
        ax.set_title(title,fontsize=fontsize)

    plt.legend()

    if show:
        plt.show()
    else: 
        return (fig, ax)


def plot_ring_stats(datadir,avg_file,std_file=None,normalise=True, distinguished_hexagons=True,
                        color='r', usetex=True, plt_objs=None, show=True, title=None):
    """Same as above but just for one ensemble."""
    
    avg = np.load(datadir+avg_file)
    show_errorbars = False
    if std_file is not None:
        print('Stdev file detected: skipping normalisation.')
        std = np.load(datadir+std_file)
        show_errorbars = True
    else:
        if normalise:
            avg = avg/avg.sum()
    
    # swap 6c and 6i to match Nature figure
    old_avgs = avg.copy()
    avg[:2] = old_avgs[:2]
    avg[3] = old_avgs[4]
    avg[4] = old_avgs[3]
    avg[5:] = old_avgs[5:]
    if usetex: setup_tex()

    if plt_objs is None:
        fig, ax = plt.subplots()
    else:
        fig, ax = plt_objs

    nrings = avg.shape[0]
    x = np.arange(3,3+nrings)
    dx = 0.8

    ax.bar(x,avg,align='center', width=dx, color=color)

    if show_errorbars:
        for k, s in enumerate(std):
            y = avg[k] 
            err_pts = np.array([[x[k], x[k], x[k]], [y-s, y, y+s]])
            ax.scatter(*err_pts, marker='_', color='k')
            ax.plot(*err_pts,'k-',lw=0.8)

    ax.set_xlabel('Ring types')
    
    if distinguished_hexagons:
        ax.set_xticks(x, ['3', '4', '5', '6-c', '6-i'] + [str(n) for n in range(7,nrings+2)])
    else:
        ax.set_xticks(x, [str(n) for n in range(3,nrings+3)])
    
    
    if normalise:
        ax.set_ylabel('Average count (normalised)')
    else:
        ax.set_ylabel('Average count')

    if title is not None:
        ax.set_title(title)
    
    if show:
        plt.show()
    else: 
        return (fig, ax)



if __name__ == "__main__":

    datadir = '/Users/nico/Desktop/simulation_outputs/ring_stats_40x40_pCNN_MAC/'

    avgfiles =['avg_ring_counts_tempdot5_MROfiltered.npy','avg_ring_counts_tempdot6_MROfiltered.npy','avg_ring_counts_40x40.npy']
    # labels = ['equil', 'both', 'vanilla']
    labels = ['sAMC-300','sAMC-q400','sAMC-500']


    clrs = ['darkorange', 'darkviolet', 'forestgreen']
    # clrs = MAC_ensemble_colours(clr_type='two_ensembles')
    # clrs += ['#4cc94b']
    fontsize = 45
    fontsize_axes = 55
    rcParams['font.size'] = fontsize # define font size BEFORE instantiating figure
    rcParams['figure.figsize'] = [6,5.7]
    rcParams['mathtext.fontset'] = 'cm'
    rcParams['font.family'] = 'sans-serif'

    fig, ax = plt.subplots()
    fig.subplots_adjust(bottom=0.185,top=0.99,left=0.17,right=0.885)


    compare_ring_stats(datadir, avgfiles, labels, normalised=[False, False, False], fontsize=fontsize,fontsize_axes=fontsize_axes,colors=clrs, reproduce_nature_tian=True,plt_objs=(fig,ax))

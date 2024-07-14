#!/usr/bin/env python

import numpy as np
from os import path
from matplotlib import rcParams
import matplotlib.pyplot as plt
from qcnico.graph_tools import adjacency_matrix_sparse
from qcnico.plt_utils import setup_tex
from qcnico.qcplots import plot_atoms_w_bonds


setup_tex(fontsize=40)
rcParams['figure.figsize'] = [8,8]
rCC = 1.8

for k in range(1,15):
    pos = np.load(path.expanduser(f'~/Desktop/simulation_outputs/MAC_structures/AMC-400_exptl/{k}/carbon_pos.npy')) * 10
    M = adjacency_matrix_sparse(pos,rCC)
    fig, ax = plot_atoms_w_bonds(pos,M,dotsize=5.0,show=False)
    plt.savefig(path.expanduser(f'~/Desktop/figures_worth_saving/tian_AMC400_exptl_structures/struc-{k}.pdf'))
    plt.show()

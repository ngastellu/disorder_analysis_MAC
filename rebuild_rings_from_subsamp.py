#!/usr/bin/env python

import sys
from ring_analysis_MPI import rebuild_rings

nn = sys.argv[1]
rebuild_rings(nn)

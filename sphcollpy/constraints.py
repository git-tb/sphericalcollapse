#%%

import numpy as np
import grids as gr
import operators as op

#%%

def hamiltonian_constraint_sph(r, A, KA, KB, DA, DB, lam, rho,nghost=1):
    dr_DB = gr.ghost_pad(op.D1(DB, r, nghost=nghost), nghost=nghost)

    H = -dr_DB - lam / r + A * KB * (2*KA + KB) \
        + (DA - 3*DB) / r + DA * DB /2 - 3 * DB * DB / 4 - 8 * np.pi * A * rho
    
    return H

def momentum_constraint_sph(r, KA, KB, DB, jA, nghost=1):
    dr_KB = gr.ghost_pad(op.D1(KB, r, nghost=nghost), nghost=nghost)

    M = -dr_KB + (KA - KB)*(DB / 2 + 1 / r) - 4 * np.pi * jA
    return M
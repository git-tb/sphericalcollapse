#%%

import numpy as np
import operators as op
import grids as gr
import integrators as it
import functools as ft
import field_conditions as fc

#%%

def oplog(alpha, KA, KB, r, nghost=1):
    # assumes parity conditions applied
    '''
        dt_alpha = -alpha^2 * f * (K1 + 2*K2) = -2 * alpha * (K1 + 2*K2)
        dt_Dalpha = dt(dr(ln(alpha))) 
                = dt(1/alpha * dr(alpha))
                = - 1/alpha^2 * dr(alpha) * dt(alpha) + dr(dt(alpha))/alpha
                = 2 * dr(alpha) * (K1 + K2)/alpha - 2*dr(alpha * (K1 + 2*K2))/alpha
                = -2 * dr(K1 + 2*K2) 
    '''
    
    dt_alpha = -2 * alpha * (KA + 2 * KB)
    dt_Dalpha = -2 * gr.ghost_pad(op.D1(
        (KA + 2 * KB), 
        r, nghost=nghost), nghost=nghost)
    return dt_alpha, dt_Dalpha

# def max_slicing_sph(r, KA, KB, rho, S, nghost=1):
#     '''
#     max slicing condition is
#     D^2 alpha = alpha (K_ij K^ij + 4pi*(S+rho)) =: F*alpha

#     The Laplacian in spherical coordinates is D^2 alpha = alpha'' + alpha'/r    
#     '''

#     Farr = KA ** 2 + 2 * KB**2 + 4*np.pi * (rho + S)
#     Fhandle = scipy.interpolate.interp1d(r, Farr)

#     N = r.size
#     N_noghost = N - 2 * nghost
#     dr = r[1] - r[0]

#     alpha = np.zeros(N_noghost)
#     dr_alpha = np.zeros(N_noghost)

#     alpha[-1] = 1
#     dr_alpha[-1] = 0

#     return alpha, dr_alpha/alpha


# def max_slicing_sph_D2alpha(alpha, KA, KB, rho, S):
#     return alpha*(KA ** 2 + 2 * KB**2 + 4*np.pi * (rho + S))

# def maxslc_ode(tau, alpha, r, KA, KB, rho, S, nghost=1):
#     D2alpha = (gr.ghost_pad(op.D2(alpha, r, nghost=nghost), nghost=nghost)
#             + 2 * gr.ghost_pad(op.D1(alpha,r,nghost=nghost),nghost=nghost)/r)
#     return D2alpha - alpha * (KA ** 2 + 2 * KB**2 + 4*np.pi * (rho + S))

# def solve_maxslc(alpha_init, r, KA, KB, rho, S, nghost=1):
#     odehandle = ft.partial(maxslc_ode, r=r, KA=KA, KB=KB, rho=rho, S=S, nghost=nghost)
#     dtau = r[1] - r[0]

#     result = np.copy(alpha_init)
#     for i in range(100):
#         result = it.ev_step_RK4(0, result, dtau, odehandle)
#         fc.BC_parity(result, is_even=True, left_side=True, nghost=nghost)
#         fc.BC_outflow(result, order=1, nghost=1)

#     return result




    

# %%

import numpy as np
import operators as op
import grids as gr
import slicing as slc
import integrators as igr
import field_conditions as fc
import functools as ft

# %%


def ADM_sph_sys(alpha, Dalpha, A, B, DA, DB, KA, KB, lam, SA, SB, rho, jA,
                r, nghost=1):
    # this assumes parity is already suitably imposed on fields

    # ===================================
    # ds^2 = A*dr^2 + r^2 * B * d\Omega^2
    # DA = dr(ln(A))
    # DB = dr(ln(B))
    # Dalpha = dr(ln(alpha))
    # lam = 1/r * (1 - A/B)

    # calculate all necessary derivatives
    dr_KA = gr.ghost_pad(op.D1(KA, r, nghost=nghost), nghost=nghost)
    dr_KB = gr.ghost_pad(op.D1(KB, r, nghost=nghost), nghost=nghost)
    dr_Dalpha = gr.ghost_pad(op.D1(Dalpha, r, nghost=nghost), nghost=nghost)
    dr_DB = gr.ghost_pad(op.D1(DB, r, nghost=nghost), nghost=nghost)

    fc.BC_parity(dr_KA, is_even=False, left_side=True, nghost=nghost)
    fc.BC_parity(dr_KB, is_even=False, left_side=True, nghost=nghost)
    fc.BC_parity(dr_Dalpha, is_even=True, left_side=True, nghost=nghost)
    fc.BC_parity(dr_DB, is_even=True, left_side=True, nghost=nghost)

    # matter terms
    MA = 2 * SB - SA - rho
    MB = SA - rho

    # prepare storage
    dt_A = np.zeros_like(r)
    dt_B = np.zeros_like(r)
    dt_DA = np.zeros_like(r)
    dt_DB = np.zeros_like(r)
    dt_KA = np.zeros_like(r)
    dt_KB = np.zeros_like(r)
    dt_lam = np.zeros_like(r)

    # implement ADM equations as stated in Alcubierre p.372, eq. 10.3.1 - 10.3.20
    dt_A -= 2 * alpha * A * KA
    dt_B -= 2 * alpha * B * KB

    dt_DA -= 2 * alpha * (KA * Dalpha + dr_KA)
    dt_DB -= 2 * alpha * (KB * Dalpha + dr_KB)

    dt_KA -= alpha / A * (
        dr_Dalpha + dr_DB
        + Dalpha ** 2
        - (Dalpha * DA) / 2
        + (DB ** 2) / 2
        - (DA * DB) / 2
        - A * KA * (KA + 2 * KB)
        - (1 / r) * (DA - 2 * DB)
    ) + 4 * np.pi * alpha * MA

    dt_KB -= alpha / (2 * A) * (
        dr_DB
        + Dalpha * DB
        + DB ** 2
        - DA * DB / 2
        - (1 / r) * (DA - 2 * Dalpha - 4 * DB)
        + 2 * lam / r
    ) + alpha * KB * (KA + 2 * KB) + 4 * np.pi * alpha * MB

    dt_lam += 2 * alpha * A / B * (
        dr_KB - (DB / 2) * (KA - KB) + 4 * np.pi * jA
    )

    return dt_A, dt_B, dt_DA, dt_DB, dt_KA, dt_KB, dt_lam


def ADM_matter_masslessscalar_evolution(alpha, A, B, Psi, Pi,
                                        r, nghost=1):
    # what are we computing
    dt_phi = np.zeros_like(r)
    dt_Psi = np.zeros_like(r)
    dt_Pi = np.zeros_like(r)

    # lets compute
    dt_phi = alpha * Pi / (np.sqrt(A) * B)
    dt_Psi = gr.ghost_pad(op.D1(
        alpha * Pi / (np.sqrt(A) * B),
        r, nghost=nghost), nghost=nghost)

    # the next calculation might cause instability close to the origin
    #   according to Alcubierre, for a 2nd order scheme I think this is fine
    dt_Pi = (1/r**2) * gr.ghost_pad(op.D1(
        alpha * B * r**2 * Psi / np.sqrt(A),
        r, nghost=nghost), nghost=nghost)
    # alternatively us
    # dt_Pi = gr.ghost_pad(op.Dspec(
    #     alpha * B * Psi / np.sqrt(A),
    #     r, nghost=nghost), nghost=nghost)

    return dt_phi, dt_Psi, dt_Pi


def ADM_matter_masslessscalar_conversion(A, B, Psi, Pi):
    rho = (1 / (2 * A)) * ((Pi ** 2)/(B ** 2) + Psi ** 2)
    jA = - Pi * Psi / (np.sqrt(A) * B)
    SA = (1 / (2 * A)) * ((Pi ** 2)/(B ** 2) + Psi ** 2)
    SB = (1 / (2 * A)) * ((Pi ** 2)/(B ** 2) - Psi ** 2)
    return SA, SB, rho, jA


def ADM_matter_masslessscalar_initial(r, Psifunc, nghost=1):
    B = np.ones_like(r)
    KA = np.zeros_like(r)
    KB = np.zeros_like(r)
    DB = np.zeros_like(r)

    # # compute A by solving ode with RK
    # Nr = r.size
    # y = np.zeros((Nr,2))
    # y[nghost] = np.array([1,Psi[nghost]]) # initial condition
    # dr = r[1]-r[0]

    # for it in range(nghost+1, Nr-nghost):
    #   y[it] = igr.ev_step_RK4(r[it-1], y[it-1], dr, ADM_matter_masslessscalar_initial_ode)
    #   y[it,1] = Psi[it]

    # A = y[:,0]

    # compute A
    # since the staggered grid does not contain the origin where we want to apply
    #   BC's we double the grid and sample A on the staggered grid later
    A, r_double = ADM_matter_masslessscalar_initialA(
        r, Psifunc=Psifunc, nghost=nghost)
    dr_A = np.zeros_like(A)
    dr_A[0] = ADM_matter_masslessscalar_initial_ode_v2(0, A[0], Psifunc)
    dr_A[1:] = np.array([
        ADM_matter_masslessscalar_initial_ode_v2(ri, Ai, Psifunc)
        for ri, Ai in list(zip(r_double[1:], A[1:]))
    ])

    Apad = gr.ghost_pad(A, nghost=nghost)
    fc.BC_parity(Apad, is_even=True, left_side=True, nghost=nghost)
    fc.BC_outflow(Apad, order=1, nghost=nghost)
    dr_Anum = op.D1(Apad, r_double, nghost=nghost)

    print("IVP local integration error: ",
          np.max(np.abs(dr_Anum-dr_A)[:-nghost]))
    print("IVP global integration error: ",
          (r_double[1] - r_double[0]) * np.sum(np.abs(dr_Anum-dr_A)[:-nghost]))

    # sample A on the staggered grid
    A_staggered = A[1::2]

    # add ghost nodes
    A_staggered = gr.ghost_pad(A_staggered, nghost=nghost)

    # impose parity conditions on A
    fc.BC_parity(A_staggered, is_even=True, left_side=True, nghost=nghost)
    fc.BC_outflow(A_staggered, order=1, nghost=nghost)

    # compute DA numerically from A
    DA = gr.ghost_pad(op.D1(np.log(A_staggered), r,
                      nghost=nghost), nghost=nghost)

    # finally lambda
    lam = (1/r) * (1 - A_staggered/B)

    return A_staggered, B, DA, DB, KA, KB, lam


# def ADM_matter_masslessscalar_initial_ode(r, y):
#   # unpack y
#   A, Psi = y

#   dA = A*((1 - A) / r + 4 * np.pi * r * Psi**2)
#   dPsi = 0

#   return np.array([dA, dPsi])

def ADM_matter_masslessscalar_initial_ode_v2(r, A, Psifunc=None):
    if (A == 1):
        dA = (4 * np.pi * r * Psifunc(r)**2)
        return dA
    else:
        dA = A*((1 - A) / r + 4 * np.pi * r * Psifunc(r)**2)
        return dA


def ADM_matter_masslessscalar_initialA(r, Psifunc, nghost):
    Nr_noghost = r.size - 2*nghost
    dr_half = (r[1] - r[0])/2

    r_double = np.zeros(2 * Nr_noghost)
    A = np.zeros(2 * Nr_noghost)

    # initial condition
    r_double[0] = 0
    A[0] = 1

    A_ode_start = ft.partial(
        ADM_matter_masslessscalar_initial_ode_v2, Psifunc=Psifunc)
    A_ode = ft.partial(
        ADM_matter_masslessscalar_initial_ode_v2, Psifunc=Psifunc)

    r_double[1] = dr_half
    A[1] = igr.ev_step_RK4(0, A[0], dr_half, A_ode_start)
    for it in range(2, 2*Nr_noghost):
        r_double[it] = it * dr_half
        A[it] = igr.ev_step_RK4(r_double[it-1], A[it-1], dr_half, A_ode)

    return A, r_double

# %%

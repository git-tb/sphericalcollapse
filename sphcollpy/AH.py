
import operators as op
import grids as gr
import field_conditions as fc
import numpy as np
import scipy.interpolate
import scipy.optimize


def findAH(r, A, B, KB, nghost=1):

    # some derivatives first
    dr_B = gr.ghost_pad(op.D1(B, r, nghost=nghost), nghost=nghost)
    fc.BC_parity(dr_B, is_even=False, left_side=True, nghost=nghost)

    # expansion
    H = (1/np.sqrt(A) * (2/r + dr_B/B) - 2*KB)[nghost:]

    # find roots of H
    aux = H[1:]*H[:-1]  # when H changes sign ([-...,-...,+...,+...])
    # then auxh will contain negative values

    if (np.any(aux < 0)):
        startsearch = np.min(np.where(aux < 0))

        intvl = 10
        idcs = np.arange(np.max([0, startsearch-intvl]),
                         np.min([r.size, startsearch+intvl]))

        Hinterp = scipy.interpolate.CubicSpline(r[idcs],
                                                H[idcs],
                                                extrapolate=True)

        sol = scipy.optimize.root_scalar(Hinterp,
                                         method="toms748",
                                         bracket=[r[idcs[0]], r[idcs[-1]]],
                                         x0=r[startsearch])

        if (sol.converged):
            r0 = sol.root

            # interpolate the gam2 function values (for area computation) ---------------
            det_term_A = scipy.interpolate.interp1d(
                r[idcs], (B)[idcs],
                kind="cubic"
            )(r0)

            # compute horizon area
            A_H = 4 * np.pi * det_term_A * r0 ** 2

            return True, r0, A_H

    return False, -1e10, -1e10

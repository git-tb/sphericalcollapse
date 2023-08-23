#ifndef ADM_SPHERICAL_H
#define ADM_SPHERICAL_H

#include <vector>
#include <Eigen/Dense>

#include "vec.h"
#include "grid.h"
#include "operators.h"

#define pow2(x) x *x

state_vec ADM_1plog_statpunct(state_vec u, vec r_grid, int n_ghost)
{
    // extract and label state vector
    vec alpha = u[0],
        Dalpha = u[1],
        gamtil_1 = u[2],
        gamtil_2 = u[3],
        Gamtil_1 = u[4],
        Gamtil_2 = u[5],
        kapb_1 = u[6],
        kapb_2 = u[7],
        lam = u[8],
        psi4 = u[9],
        Dpsi = u[10];

    // prepare storage
    unsigned n_size = alpha.size();
    vec dt_alpha = vec::Zero(n_size),
        dt_Dalpha = vec::Zero(n_size),
        dt_gamtil_1 = vec::Zero(n_size),
        dt_gamtil_2 = vec::Zero(n_size),
        dt_Gamtil_1 = vec::Zero(n_size),
        dt_Gamtil_2 = vec::Zero(n_size),
        dt_kapb_1 = vec::Zero(n_size),
        dt_kapb_2 = vec::Zero(n_size),
        dt_lam = vec::Zero(n_size),
        dt_psi4 = vec::Zero(n_size),
        dt_Dpsi = vec::Zero(n_size);

    // some useful derivatives
    vec d1r_kapb_1 = deriv1(kapb_1, r_grid, n_ghost),
        d1r_kapb_2 = deriv1(kapb_2, r_grid, n_ghost),
        d1r_Dalpha = deriv1(Dalpha, r_grid, n_ghost),
        d1r_Gamtil_2 = deriv1(Gamtil_2, r_grid, n_ghost),
        d1r_alpha = deriv1(alpha, r_grid, n_ghost),
        d1r_Dpsi = deriv1(Dpsi, r_grid, n_ghost);
    ghost_pad(d1r_kapb_1, n_ghost);
    ghost_pad(d1r_kapb_2, n_ghost);
    ghost_pad(d1r_Dalpha, n_ghost);
    ghost_pad(d1r_Gamtil_2, n_ghost);
    ghost_pad(d1r_alpha, n_ghost);
    ghost_pad(d1r_Dpsi, n_ghost);

    // term by term update rules
    dt_alpha = -2 * alpha * (kapb_1 + 2 * kapb_2);
    dt_Dalpha = -d1r_kapb_1 - 2 * d1r_kapb_2;
    dt_gamtil_1 = -2 * alpha * gamtil_1 * kapb_1;
    dt_gamtil_2 = -2 * alpha * gamtil_2 * kapb_2;
    dt_Gamtil_1 = -2 * alpha * (kapb_1 * Dalpha + d1r_kapb_1);
    dt_Gamtil_2 = -2 * alpha * (kapb_2 * Dalpha + d1r_kapb_2);
    dt_lam = 2 * alpha * (gamtil_1 / gamtil_2) *
             (d1r_kapb_2 - (Gamtil_2 + 4 * Dpsi) * (kapb_1 - kapb_2) / 2);
    dt_kapb_1 = (-alpha / (gamtil_1 * psi4) *
                     (d1r_Dalpha +
                      d1r_Gamtil_2 +
                      4 * d1r_Dpsi +
                      pow2(Dalpha) -
                      Dalpha / 2 * (Gamtil_1 + 4 * Dpsi) +
                      (pow2(Gamtil_2 + 4 * Dpsi)) / 2 -
                      (Gamtil_1 + 4 * Dpsi) * (Gamtil_2 + 4 * Dpsi) / 2 -
                      ((Gamtil_1 + 4 * Dpsi) -
                       2 * (Gamtil_2 + 4 * Dpsi)) /
                          r_grid) +
                 alpha * kapb_1 * (kapb_1 + 2 * kapb_2));
    dt_kapb_2 = (-alpha / (2 * gamtil_1 * psi4) *
                     (d1r_Gamtil_2 +
                      4 * d1r_Dpsi +
                      Dalpha * (Gamtil_2 + 4 * Dpsi) +
                      pow2(Gamtil_2 + 4 * Dpsi) -
                      (Gamtil_1 + 4 * Dpsi) * (Gamtil_2 + 4 * Dpsi) / 2 -
                      ((Gamtil_1 + 4 * Dpsi) -
                       4 * (Gamtil_2 + 4 * Dpsi) -
                       2 * Dalpha) /
                          r_grid +
                      2 * lam / r_grid) +
                 alpha * kapb_2 * (kapb_1 + 2 * kapb_2));

    state_vec dt_u({std::vector<vec>({dt_alpha,
                                      dt_Dalpha,
                                      dt_gamtil_1,
                                      dt_gamtil_2,
                                      dt_Gamtil_1,
                                      dt_Gamtil_2,
                                      dt_kapb_1,
                                      dt_kapb_2,
                                      dt_lam,
                                      vec::Zero(n_size),
                                      vec::Zero(n_size)})});
    return dt_u;
}

state_vec ADM_initial_Minkwoski(vec r_grid)
{
    /*
        creates initial data for Minkowski,
        ds^2 = gam_1 dr^2 +
                    r^2 * gam_2 * dOmega^2
    */
    unsigned int n_size = r_grid.size();

    vec gam_1 = vec::Ones(n_size),
        gam_2 = vec::Ones(n_size),
        Gam_1 = vec::Zero(n_size),
        Gam_2 = vec::Zero(n_size),
        kapb_1 = vec::Zero(n_size),
        kapb_2 = vec::Zero(n_size),
        lam = vec::Zero(n_size);

    return state_vec({std::vector<vec>({gam_1,
                                        gam_2,
                                        Gam_1,
                                        Gam_2,
                                        kapb_1,
                                        kapb_2,
                                        lam})});
}

// dt_alpha = -2 * alpha * (kapb_1 + 2 * kapb_2);
//     dt_Dalpha = (2 * d1r_alpha * (kapb_1 + 2 * kapb_2) / alpha -
//                  4 * d1r_alpha * (kapb_1 + 2 * kapb_2) -
//                  2 * alpha * d1r_alpha * (kapb_1 + 2 * kapb_2) -
//                  2 * alpha * (d1r_kapb_1 + 2 * d1r_kapb_2));

//     dt_gam_1 = -2 * alpha * gam_1 * kapb_1;
//     dt_gam_2 = -2 * alpha * gam_2 * kapb_2;

//     dt_Gam_1 = -2 * alpha * (kapb_1 * Dalpha + d1r_kapb_1);
//     dt_Gam_2 = -2 * alpha * (kapb_2 * Dalpha + d1r_kapb_2);

//     dt_kapb_1 = -alpha / gam_1 *
//                 (d1r_Dalpha + d1r_Gam_2 + Dalpha * Dalpha -
//                  (Dalpha * Gam_1) / 2 +
//                  Gam_2 * Gam_2 / 2 -
//                  (Gam_1 * Gam_2) / 2 -
//                  gam_1 * kapb_1 * (kapb_1 + 2 * kapb_2) -
//                  (Gam_1 - 2 * Gam_2) / r_grid);

//     dt_kapb_2 = -alpha / (2 * gam_1) *
//                 (d1r_Gam_2 + Dalpha * Gam_2 + Gam_2 * Gam_2 -
//                  Gam_1 * Gam_2 / 2 -
//                  (Gam_1 - 2 * Dalpha - 4 * Gam_2) / r_grid +
//                  2 * lam / r_grid);
//     dt_kapb_2 = alpha * kapb_2 * (kapb_1 + 2 * kapb_2);

//     dt_lam = 2 * alpha * gam_1 / gam_2 *
//              (d1r_kapb_2 -
//               Gam_2 / 2 * (kapb_1 - kapb_2));

#endif
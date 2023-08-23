#ifndef GRID_H
#define GRID_H

#include <Eigen/Dense>
#include <cmath>

#include "vec.h"

int ghost_pad(vec &fcn, int n_ghost)
{
    int n_size = fcn.size();
    vec fcn_new = vec::Zero(n_size + 2 * n_ghost);
    fcn_new(Eigen::seqN(n_ghost, n_size)) = fcn;
    fcn.resize(n_size + 2 * n_ghost);
    fcn = fcn_new;

    return 0;
}

vec create_grid_CC(unsigned int N_r, double r_a, double r_b, int n_ghost)
{
    assert(r_a <= r_b);
    double dr = (r_b - r_a) / N_r;
    int N_ext = N_r + 2 * n_ghost;
    return vec::LinSpaced(N_ext, r_a + dr / 2, r_b - dr / 2);
}

int N_t_from_CFL(double CFL, double dx, double ti, double tf, double v = 1.0)
{
    return int(std::ceil(v * ((tf - ti) / CFL) / dx));
}

double CFL_from_N_t(int N_t, double dx, double ti, double tf, double v = 1.0)
{
    return v * (tf - ti) / (dx * N_t);
}

#endif
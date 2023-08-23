#ifndef FIELD_CONDITIONS_H
#define FIELD_CONDITIONS_H

#include <Eigen/Dense>

#include "vec.h"

int impose_BC_parity(vec &fcn, bool is_even, bool cell_centered, int n_ghost = 1)
{
    /*
        imposes parity condition on left side of the array fcn
    */
    int sign = is_even ? 1 : -1;
    if (cell_centered)
        fcn(Eigen::seqN(0, n_ghost)) = sign * fcn(Eigen::seqN(n_ghost, n_ghost).reverse());
    else
        fcn(Eigen::seqN(0, n_ghost)) = sign * fcn(Eigen::seqN(n_ghost + 1, n_ghost).reverse());
    return 0;
}

int impose_BC_outflow(vec &fcn, int order = 1, int n_ghost = 1)
{
    if (order = 1)
        fcn(Eigen::seq(Eigen::last + 1 - n_ghost, Eigen::last)) = fcn(Eigen::seq(Eigen::last - n_ghost, Eigen::last - 1));
    else
        fcn(Eigen::seq(Eigen::last + 1 - n_ghost, Eigen::last)) = 2 * fcn(Eigen::seq(Eigen::last - n_ghost, Eigen::last)) -
                                                                  fcn(Eigen::seq(Eigen::last - 1 - n_ghost, Eigen::last));
    return 0;
}

#endif
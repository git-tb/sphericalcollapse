#ifndef OPERATORS_H
#define OPERATORS_H

#include <Eigen/Dense>

#include "vec.h"

vec deriv1(vec fcn, vec r_grid, int n_ghost = 1)
{
    assert(n_ghost == 1);

    float dr = r_grid[1] - r_grid[0];

    vec result = (fcn(Eigen::seq(0, Eigen::last - 2)) -
                  fcn(Eigen::seq(2, Eigen::last))) /
                 (2.0 * dr);

    return result;
}

vec deriv2(vec fcn, vec r_grid, int n_ghost = 1)
{
    assert(n_ghost == 1);

    float dr = r_grid[1] - r_grid[0];

    return (fcn(Eigen::seq(0, Eigen::last - 2)) -
            2 * fcn(Eigen::seq(1, Eigen::last - 1)) +
            fcn(Eigen::seq(2, Eigen::last))) /
           (dr * dr);
}

#endif
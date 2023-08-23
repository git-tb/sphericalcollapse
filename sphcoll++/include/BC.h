#ifndef BC_H
#define BC_H

#include <Eigen/Dense>
namespace BC
{
    // boundary conditions
    void parity(Eigen::ArrayXd &arr, bool even, int n_ghost = 1)
    {
        if (n_ghost == 0)
            return;

        arr(Eigen::seq(0, n_ghost - 1)) = (even ? +1 : -1) * arr(Eigen::seq(2 * n_ghost - 1, n_ghost, -1));
    }

    void outflow(Eigen::ArrayXd &arr, int n_ghost = 1)
    {
        if (n_ghost == 0)
            return;

        int lastidx = arr.size() - 1;
        double delta = arr[lastidx - n_ghost] - arr[lastidx - n_ghost - 1];
        arr(Eigen::seq(Eigen::last - n_ghost + 1, Eigen::last)) = arr[lastidx - n_ghost] + delta * Eigen::ArrayXd::LinSpaced(n_ghost, 1, n_ghost);
    }
}

#endif
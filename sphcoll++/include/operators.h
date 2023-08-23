#ifndef OPERATORS_H
#define OPERATORS_H

#include <iostream>
#include <Eigen/Dense>

namespace OP
{
    // derivative operator
    Eigen::ArrayXd D1(const Eigen::ArrayXd &func, const Eigen::ArrayXd &r, int n_ghost)
    {
        double ds = r[1] - r[0];
        Eigen::ArrayXd result = Eigen::ArrayXd::Zero(func.size());
        if (n_ghost == 2)
            result(Eigen::seq(2, Eigen::last - 2)) = (1.0 / 12.0 * func(Eigen::seq(0, Eigen::last - 4))  //
                                                      - 2.0 / 3.0 * func(Eigen::seq(1, Eigen::last - 3)) //
                                                      + 2.0 / 3.0 * func(Eigen::seq(3, Eigen::last - 1)) //
                                                      - 1.0 / 12.0 * func(Eigen::seq(4, Eigen::last))) /
                                                     ds;
        else
            std::cout << "WARNING: derivative only implemented for n_ghost=2" << std::endl;

        return result;
    }

    Eigen::ArrayXd D4_stencil(const Eigen::ArrayXd &func, int n_ghost)
    {
        Eigen::ArrayXd result = Eigen::ArrayXd::Zero(func.size());
        if (n_ghost == 2)
            result(Eigen::seq(2, Eigen::last - 2)) = (1.0 * func(Eigen::seq(0, Eigen::last - 4))   //
                                                      - 4.0 * func(Eigen::seq(1, Eigen::last - 3)) //
                                                      + 6.0 * func(Eigen::seq(2, Eigen::last - 2)) //
                                                      - 4.0 * func(Eigen::seq(3, Eigen::last - 1)) //
                                                      + 1.0 * func(Eigen::seq(4, Eigen::last)));
        else
            std::cout << "WARNING: derivative only implemented for n_ghost=2" << std::endl;

        return result;
    }
}

#endif
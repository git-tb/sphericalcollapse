#ifndef INTEGRATORS_H
#define INTEGRATORS_H

#include <Eigen/Dense>
#include <functional>

#include "vec.h"

state_vec step_rk4(double t, state_vec u, double dt, std::function<state_vec(double, state_vec)> fcn)
{
    state_vec v_1 = fcn(t, u),
              v_2 = fcn(t + dt / 2, u + 1 / 2 * dt * v_1 * dt),
              v_3 = fcn(t + dt / 2, u + 1 / 2 * dt * v_2 * dt),
              v_4 = fcn(t + dt, u + dt * v_3 * dt);

    return u + dt * (v_1 / 6 + v_2 / 3 + v_3 / 3 + v_4 / 6);
}

#endif
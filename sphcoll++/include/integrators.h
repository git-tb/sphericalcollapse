#ifndef INTEGRATORS_H
#define INTEGRATORS_H

namespace ITG
{
  // rk4 stepper
  template <typename T>
  T step_rk4(double t, T u, double dt, std::function<T(double, T)> fcn)
  {
    // std::cout << "rkstep" << std::endl;

    T v_1 = fcn(t, u),
      v_2 = fcn(t + dt / 2, u + 1 / 2 * dt * v_1),
      v_3 = fcn(t + dt / 2, u + 1 / 2 * dt * v_2),
      v_4 = fcn(t + dt, u + dt * v_3);

    return u + dt * (v_1 / 6 + v_2 / 3 + v_3 / 3 + v_4 / 6);
  }

  template <typename T>
  T step_rk4_callback(double t, T u, double dt, std::function<T(double, T)> fcn, std::function<T(T)> callback)
  {
    // std::cout << "rkstep" << std::endl;

    T v_1 = fcn(t, u),
      v_2 = fcn(t + dt / 2, callback(u + 1 / 2 * dt * v_1)),
      v_3 = fcn(t + dt / 2, callback(u + 1 / 2 * dt * v_2)),
      v_4 = fcn(t + dt, callback(u + dt * v_3));

    return callback(u + dt * (v_1 / 6 + v_2 / 3 + v_3 / 3 + v_4 / 6));
  }

  template <typename T>
  T step_ICN3_callback(double t, T u, double dt, std::function<T(double, T)> fcn, std::function<T(T)> callback)
  {
    // std::cout << "rkstep" << std::endl;

    T v_1 = callback(u + dt * fcn(t, u));
    T v_2 = callback(u + 0.5 * dt * (fcn(t, u) + fcn(t + dt, v_1)));
    T v_3 = callback(u + 0.5 * dt * (fcn(t, u) + fcn(t + dt, v_2)));

    return v_3;
  }
}

#endif
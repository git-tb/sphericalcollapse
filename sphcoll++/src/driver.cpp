#include <iostream>
#include <iomanip>
#include <vector>
#include <functional>
#include <cassert>
#include <Eigen/Dense>
#include <fstream>
#include <ctime>   // std::time, std::local_time
#include <iomanip> // std::put_time
#include <cmath>

// #include "interpolate.h"
#include "spline.h"
#include "initialfamilies.h"
#include "integrators.h"
#include "BC.h"
#include "util.h"
#include "operators.h"

// ode for solving constraints
Eigen::Vector2d dy(double r,
                   Eigen::Vector2d y,
                   std::function<double(double)> Psifunc,
                   std::function<double(double)> Pifunc)
{
    // ===============
    // y is (alpha, a)
    // ===============

    double alpha(y[0]), a(y[1]);

    if (r == 0)
        return Eigen::Vector2d({0, 0});
    else
    {
        double Psi(Psifunc(r)), Pi(Pifunc(r));
        double dalpha = alpha * (2 * M_PI * r * (Pi * Pi + Psi * Psi) - (1 - a * a) / (2 * r));
        double da = a * (2 * M_PI * r * (Pi * Pi + Psi * Psi) + (1 - a * a) / (2 * r));

        return Eigen::Vector2d({dalpha, da});
    }
}

// ...constraint solver
void solveconstraints_doubler(Eigen::ArrayXd &r,
                              Eigen::ArrayXd &Psi,
                              Eigen::ArrayXd &Pi,
                              Eigen::ArrayXd &alpha,
                              Eigen::ArrayXd &a,
                              int n_ghost)
{
    assert(r.size() == Psi.size() && r.size() == Pi.size() && r.size() == alpha.size() && r.size() == a.size());

    // generate lookup functions from discrete arrays in order to double the stepsize
    std::vector<double> Psistdvec, Pistdvec, rstdvec;
    eigenvec_to_stdvec(Psi, Psistdvec);
    eigenvec_to_stdvec(Pi, Pistdvec);
    eigenvec_to_stdvec(r, rstdvec);
    tk::spline Psifunc(rstdvec, Psistdvec);
    tk::spline Pifunc(rstdvec, Pistdvec);

    // initializations
    int n_noghost = r.size() - 2 * n_ghost;
    double dr_half = (r[1] - r[0]) / 2.0;
    Eigen::ArrayXd alpha_doubles(2 * n_noghost), a_doubles(2 * n_noghost), r_doubles(2 * n_noghost);

    Eigen::Vector2d ynext, ycur;
    // ===============
    // y is (alpha, a)
    // ===============

    // initial conditions
    alpha_doubles[0] = 1;
    a_doubles[0] = 1;
    r_doubles[0] = 0;

    ycur = Eigen::Vector2d({alpha_doubles[0], a_doubles[0]});

    // cast integration function into suitable form
    std::function<Eigen::Vector2d(double, Eigen::Vector2d)> dy_ode = [&Psifunc, &Pifunc](double r, Eigen::Vector2d y)
    {
        return dy(r, y, Psifunc, Pifunc);
    };

    // first step
    r_doubles[1] = dr_half;
    ynext = ITG::step_rk4(0, ycur, dr_half, dy_ode);
    alpha_doubles[1] = ynext[0];
    a_doubles[1] = ynext[1];

    ycur = ynext;

    // all other steps
    for (int i = 2; i < 2 * n_noghost; i++)
    {
        ynext = ITG::step_rk4(r_doubles[i - 1], ycur, dr_half, dy_ode);

        r_doubles[i] = i * dr_half;
        // ===============
        // y is (alpha, a)
        // ===============
        alpha_doubles[i] = ynext[0];
        a_doubles[i] = ynext[1];

        ycur = ynext;
    }

    // write into alphas and as array
    alpha(Eigen::seq(n_ghost, Eigen::last - n_ghost)) = alpha_doubles(Eigen::seq(1, Eigen::last, +2));
    a(Eigen::seq(n_ghost, Eigen::last - n_ghost)) = a_doubles(Eigen::seq(1, Eigen::last, +2));

    BC::parity(alpha, true, n_ghost);
    BC::parity(a, true, n_ghost);
    BC::outflow(alpha, n_ghost);
    BC::outflow(a, n_ghost);
}

// evolution equations
Eigen::ArrayXd dPsi(const Eigen::ArrayXd &r,
                    const Eigen::ArrayXd &a,
                    const Eigen::ArrayXd &alpha,
                    const Eigen::ArrayXd &Pi,
                    int n_ghost)
{
    if (n_ghost != 2)
        std::cout << "WARNING: dPi only valid for n_ghost=2" << std::endl;

    Eigen::ArrayXd d_Psi = OP::D1(alpha * Pi / a, r, n_ghost);

    BC::parity(d_Psi, false, n_ghost);
    BC::outflow(d_Psi, n_ghost);

    return d_Psi;
}

Eigen::ArrayXd dPi(const Eigen::ArrayXd &r,
                   const Eigen::ArrayXd &a,
                   const Eigen::ArrayXd &alpha,
                   const Eigen::ArrayXd &Psi,
                   int n_ghost)
{
    if (n_ghost != 2)
        std::cout << "WARNING: dPi only valid for n_ghost=2" << std::endl;

    Eigen::ArrayXd r2 = r * r;
    Eigen::ArrayXd d_Pi = 1 / r2 * OP::D1(alpha * Psi * r2 / a, r, n_ghost); // this calculation
                                                                             // should be fine for NGHOST == 2

    BC::parity(d_Pi, true, n_ghost);
    BC::outflow(d_Pi, n_ghost);

    return d_Pi;
}

// with dissipation
Eigen::ArrayXd dPsidiss(const Eigen::ArrayXd &r,
                        const Eigen::ArrayXd &a,
                        const Eigen::ArrayXd &alpha,
                        const Eigen::ArrayXd &Pi,
                        int n_ghost,
                        double dr, double epsilon,
                        const Eigen::ArrayXd &Psi)
{
    if (n_ghost != 2)
        std::cout << "WARNING: dPi only valid for n_ghost=2" << std::endl;

    Eigen::ArrayXd d_Psi = OP::D1(alpha * Pi / a, r, n_ghost)                   //
                           - epsilon * (1 / dr) * OP::D4_stencil(Psi, n_ghost); //

    BC::parity(d_Psi, false, n_ghost);
    BC::outflow(d_Psi, n_ghost);

    return d_Psi;
}

Eigen::ArrayXd dPidiss(const Eigen::ArrayXd &r,
                       const Eigen::ArrayXd &a,
                       const Eigen::ArrayXd &alpha,
                       const Eigen::ArrayXd &Psi,
                       int n_ghost,
                       double dr, double epsilon,
                       const Eigen::ArrayXd &Pi)
{
    if (n_ghost != 2)
        std::cout << "WARNING: dPi only valid for n_ghost=2" << std::endl;

    Eigen::ArrayXd r2 = r * r;
    Eigen::ArrayXd d_Pi = 1 / r2 * OP::D1(alpha * Psi * r2 / a, r, n_ghost)   // this calculation
                          - epsilon * (1 / dr) * OP::D4_stencil(Pi, n_ghost); // should be fine for NGHOST == 2

    BC::parity(d_Pi, true, n_ghost);
    BC::outflow(d_Pi, n_ghost);

    return d_Pi;
}

struct state_vec
{
    Eigen::ArrayXd alpha, a, Pi, Psi;

    state_vec operator+(const state_vec &other) const
    {
        state_vec result(other);
        result.alpha = alpha + other.alpha;
        result.a = a + other.a;
        result.Pi = Pi + other.Pi;
        result.Psi = Psi + other.Psi;

        return result;
    }

    state_vec operator*(const double &scalar) const
    {
        state_vec result(*this);
        result.a *= scalar;
        result.alpha *= scalar;
        result.Psi *= scalar;
        result.Pi *= scalar;

        return result;
    }
    friend state_vec operator*(const double &scalar, const state_vec &v)
    {
        return v * scalar;
    }

    state_vec operator/(const double &scalar) const
    {
        assert(scalar != 0);
        state_vec result(*this);
        result.a /= scalar;
        result.alpha /= scalar;
        result.Psi /= scalar;
        result.Pi /= scalar;

        return result;
    }

    int hasNan()
    {
        if (a.hasNaN())
            return 1;
        if (alpha.hasNaN())
            return 2;
        if (Pi.hasNaN())
            return 3;
        if (Psi.hasNaN())
            return 4;

        return -1;
    }
};

int main(int argc, char **argv)
{
    /* call signature: ./bin/driver N a0 tmax */

    // Input-Output-Format of Eigen::Matrix
    // Eigen::IOFormat::IOFormat(int _precision = StreamPrecision, int _flags = 0,
    //                           const std::string &_coeffSeparator = " ", const std::string &_rowSeparator = "\n",
    //                           const std::string &_rowPrefix = "", const std::string &_rowSuffix = "",
    //                           const std::string &_matPrefix = "", const std::string &_matSuffix = "",
    //                           const char _fill = ' '  )
    Eigen::IOFormat format(-1, 0, "", ",", "", "", "", "");

    // files
    std::time_t tptr = std::time(nullptr);
    std::tm tm = *std::localtime(&tptr);
    // std::stringstream filename;
    // filename << "file"; //<< std::put_time(&tm, "%d%m%Y_%H%M%S");

    std::fstream fstream_alpha, fstream_a, fstream_psi, fstream_pi;
    fstream_alpha.open("file_alpha.dat", std::fstream::out);
    fstream_a.open("file_a.dat", std::fstream::out);
    fstream_psi.open("file_psi.dat", std::fstream::out);
    fstream_pi.open("file_pi.dat", std::fstream::out);

    // make grid
    int Nr = std::stoi(argv[1]);
    double ra(0), rb(30);
    double dr = (rb - ra) / Nr;
    int N_GHOST = 2;

    Eigen::ArrayXd r = ra + dr / 2.0 + dr * Eigen::ArrayXd::LinSpaced(Nr + 2 * N_GHOST, -N_GHOST, Nr - 1 + N_GHOST); // (size, xmin(incl), xmax(incl))

    // initial fields
    double a0 = std::stod(argv[2]);
    std::cout << "initial amplitude a0 = " << a0 << std::endl;
    Eigen::ArrayXd phi = r.unaryExpr([&a0](double r)
                                     { return family1_phi(r, a0); });
    Eigen::ArrayXd Psi = r.unaryExpr([&a0](double r)
                                     { return family1_psi(r, a0); });
    Eigen::ArrayXd Pi = 0 * r;
    BC::parity(Psi, false, N_GHOST);
    BC::parity(Pi, true, N_GHOST);

    Eigen::ArrayXd a(0 * r), alpha(0 * r);
    solveconstraints_doubler(r, Psi, Pi, alpha, a, N_GHOST);
    BC::parity(alpha, true, N_GHOST);
    BC::parity(a, true, N_GHOST);
    BC::outflow(alpha, N_GHOST);
    BC::outflow(a, N_GHOST);

    // set up time steps
    double t = 0, tmax = std::stod(argv[3]);
    double CFL = 0.4;
    int Nt = int(std::ceil(((tmax - t) / CFL) / dr));
    double dt = (tmax - t) / Nt;

    // write initial condition to file
    fstream_a << "# a0 = " << a0 << std::endl;
    fstream_a << "\t," << r.format(format) << std::endl;
    fstream_a << t << "," << a.format(format) << std::endl;
    fstream_alpha << "# a0 = " << a0 << std::endl;
    fstream_alpha << "\t," << r.format(format) << std::endl;
    fstream_alpha << t << "," << alpha.format(format) << std::endl;
    fstream_psi << "# a0 = " << a0 << std::endl;
    fstream_psi << "\t," << r.format(format) << std::endl;
    fstream_psi << t << "," << Psi.format(format) << std::endl;
    fstream_pi << "# a0 = " << a0 << std::endl;
    fstream_pi << "\t," << r.format(format) << std::endl;
    fstream_pi << t << "," << Pi.format(format) << std::endl;

    state_vec curstate;
    curstate.a = a;
    curstate.alpha = alpha;
    curstate.Pi = Pi;
    curstate.Psi = Psi;

    // integration function and constraint solver
    std::function<state_vec(double t, state_vec)> ode = [&N_GHOST, &r, &dr](double t, state_vec u)
    {
        state_vec du;
        du.a = 0 * u.a;
        du.alpha = 0 * u.alpha;
        // du.Pi = dPi(r, u.a, u.alpha, u.Psi, N_GHOST);
        // du.Psi = dPsi(r, u.a, u.alpha, u.Pi, N_GHOST);
        du.Pi = dPidiss(r, u.a, u.alpha, u.Psi, N_GHOST, dr, 0.1, u.Pi);
        du.Psi = dPsidiss(r, u.a, u.alpha, u.Pi, N_GHOST, dr, 0.1, u.Psi);

        return du;
    };

    std::function<state_vec(state_vec)> solve_constraints_wrapper = [&N_GHOST, &r](state_vec u)
    {
        state_vec u_solved(u);
        solveconstraints_doubler(r, u_solved.Psi, u_solved.Pi, u_solved.alpha, u_solved.a, N_GHOST);
        return u_solved;
    };

    for (int i = 0; i < Nt; i++)
    {
        t += dt;
        std::cout << "t=" << t << std::endl;

        // curstate = ITG::step_rk4_callback(t, curstate, dt, ode, solve_constraints_wrapper);
        curstate = ITG::step_ICN3_callback(t, curstate, dt, ode, solve_constraints_wrapper);

        fstream_a << t << "," << curstate.a.format(format) << std::endl;
        fstream_alpha << t << "," << curstate.alpha.format(format) << std::endl;
        fstream_psi << t << "," << curstate.Psi.format(format) << std::endl;
        fstream_pi << t << "," << curstate.Pi.format(format) << std::endl;

        if (curstate.a.hasNaN())
        {
            std::cout << "NaN occured - abort" << std::endl;

            auto isnan_wrap = [](const double &x)
            { return std::isnan(x); };
            Eigen::ArrayXd::iterator it = std::find_if(curstate.a.begin(), curstate.a.end(), isnan_wrap);
            unsigned int k = std::distance(curstate.a.begin(), it);
            std::cout << "a0 = " << a0 << " | RH = " << r[k] - dr / 2 << std::endl;
            break;
        }
    }

    // close files
    fstream_alpha.close();
    fstream_a.close();
    fstream_psi.close();
    fstream_pi.close();

    return 0;
}

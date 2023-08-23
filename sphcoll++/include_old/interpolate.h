#include <cmath>
#include <vector>
#include <algorithm>
#include <cassert>

double h00(double t)
{
    return 2 * t * t * t - 3 * t * t + 1;
}

double h10(double t)
{
    return t * t * t - 2 * t * t + t;
}

double h01(double t)
{
    return -2 * t * t * t + 3 * t * t;
}

double h11(double t)
{
    return t * t * t - t * t;
}

template <typename T>
double cubic_interpolate(double x,
                         T &xarr,
                         T &yarr)
{
    assert(x <= xarr[xarr.size() - 1] && x >= xarr[0]);
    assert(xarr.size() == yarr.size());

    if (x == xarr[0])
        return yarr[0];
    if (x == xarr[xarr.size() - 1])
        return yarr[yarr.size() - 1];

    auto gtreq = [&x](const double &v) { // pass by reference necessary for whatever reason
        return (v >= x ? true : false);
    };
    typename T::iterator it = std::find_if(xarr.begin(), xarr.end(), gtreq); // find first iterator where v >= x
                                                                             // if no v >= x is found, iterator will be set to container.end()
    unsigned int k = std::distance(xarr.begin(), it);

    double delta = (xarr[k + 1] - xarr[k]);
    double t = (x - xarr[k]) / delta;

    // ===
    double mk_0(0), mk_1(0);
    double deltak__1(0), deltak_0(0), deltak_1(0);

    if (k == 0)
    {
        deltak_0 = (yarr[k + 1] - yarr[k]) / (xarr[k + 1] - xarr[k]);
        deltak_1 = (yarr[k + 1 + 1] - yarr[k + 1]) / (xarr[k + 1 + 1] - xarr[k + 1]);

        mk_0 = deltak_0;
        if (deltak_0 * deltak_1 > 0)
            mk_1 = (deltak_0 + deltak_1) / 2.0;
        else
            mk_1 = 0;
    }
    else if (k == xarr.size() - 2)
    {
        deltak__1 = (yarr[k] - yarr[k - 1]) / (xarr[k] - xarr[k - 1]);
        deltak_0 = (yarr[k + 1] - yarr[k]) / (xarr[k + 1] - xarr[k]);

        if (deltak__1 * deltak_0 > 0)
            mk_0 = (deltak__1 + deltak_0) / 2.0;
        else
            mk_0 = 0;
        mk_1 = deltak_0;
    }
    else
    {
        deltak__1 = (yarr[k] - yarr[k - 1]) / (xarr[k] - xarr[k - 1]);
        deltak_0 = (yarr[k + 1] - yarr[k]) / (xarr[k + 1] - xarr[k]);
        deltak_1 = (yarr[k + 1 + 1] - yarr[k + 1]) / (xarr[k + 1 + 1] - xarr[k + 1]);

        if (deltak__1 * deltak_0 > 0)
            mk_0 = (deltak__1 + deltak_0) / 2.0;
        else
            mk_0 = 0;
        if (deltak_0 * deltak_1 > 0)
            mk_1 = (deltak_0 + deltak_1) / 2.0;
        else
            mk_1 = 0;
    }

    if (deltak_0 == 0)
    {
        mk_0 = 0;
        mk_1 = 0;
    }
    else
    {
        double alphak(mk_0 / deltak_0), betak(mk_1 / deltak_0);
        if (alphak * alphak + betak * betak > 9)
        {
            double tauk = 3 / std::sqrt(alphak * alphak + betak * betak);
            mk_0 = tauk * alphak * deltak_0;
            mk_1 = tauk * betak * deltak_0;
        }
    }

    return yarr[k] * h00(t)        //
           + delta * mk_0 * h10(t) //
           + yarr[k + 1] * h01(t)  //
           + delta * mk_1 * h11(t);
}
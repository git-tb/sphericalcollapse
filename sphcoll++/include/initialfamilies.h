#ifndef INITIALFAMILIES_H
#define INITIALFAMILIES_H

#include <cmath>

double family1_phi(double r, double a)
{
    return a * r * r * std::exp(-(r - 5) * (r - 5));
}

double family1_psi(double r, double a)
{
    return a * (2 * r - 2 * r * r * (r - 5)) *
           std::exp(-(r - 5) * (r - 5)); //
}

#endif
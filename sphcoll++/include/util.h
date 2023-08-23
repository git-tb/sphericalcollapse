#ifndef UTIL_H
#define UTIL_H

#include <Eigen/Dense>

// utility functions
inline void eigenvec_to_stdvec(const Eigen::ArrayXd &eigenvec, std::vector<double> &stdvec)
{
    stdvec.resize(eigenvec.size());
    Eigen::ArrayXd::Map(&stdvec[0], stdvec.size()) = eigenvec;
}

inline void stdvec_to_eigenvec(Eigen::ArrayXd &eigenvec, const std::vector<double> &stdvec)
{
    eigenvec.resize(stdvec.size());
    eigenvec = Eigen::ArrayXd::Map(&stdvec[0], stdvec.size());
}

#endif
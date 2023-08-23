#ifndef VEC_H
#define VEC_H

#include <Eigen/Dense>
#include <vector>
#include <cassert>

#define vec Eigen::ArrayXf

struct state_vec
{
    std::vector<vec> fields;

    state_vec operator+(const state_vec &other) const
    {
        assert(other.fields.size() == fields.size());
        state_vec result(other);
        for (int i = 0; i < fields.size(); i++)
            result.fields[i] = fields[i] + other.fields[i];

        return result;
    }

    state_vec operator*(const double &scalar) const
    {
        state_vec result(*this);
        for (int i = 0; i < fields.size(); i++)
            result.fields[i] = scalar * fields[i];

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
        for (int i = 0; i < fields.size(); i++)
            result.fields[i] = fields[i] / scalar;

        return result;
    }

    vec &operator[](unsigned int i)
    {
        assert(i < fields.size());
        return fields[i];
    }

    int hasNan()
    {
        for (int i = 0; i < fields.size(); i++)
            if (fields[i].hasNaN())
                return i;
        return -1;
    }
};

#endif
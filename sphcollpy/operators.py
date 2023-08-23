#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
 ,-*
(_) Created on <Tue May 25 2021>

@author: Boris Daszuta
@function:
Collect discretized operators here.
"""
import numpy as np

# -----------------------------------------------------------------------------
# derivative operators


def D1(ext_fcn, gr, nghost=1):
    # discretized, centered first degree derivative operator
    ds = gr[1] - gr[0]

    if nghost == 1:
        return (-1 / 2 * ext_fcn[0:-2, ...] + 1 / 2 * ext_fcn[2:, ...]) / ds
    elif nghost == 2:
        return (1 / 12 * ext_fcn[0:-4] - 2 / 3 * ext_fcn[1:-3]
                + 2 / 3 * ext_fcn[3:-1] - 1 / 12 * ext_fcn[4:]) / ds
    elif nghost == 3:
        return (
            - 1 / 60 * ext_fcn[0:-6]
            + 3 / 20 * ext_fcn[1:-5]
            - 3 / 4 * ext_fcn[2:-4]
            + 3 / 4 * ext_fcn[4:-2]
            - 3 / 20 * ext_fcn[5:-1]
            + 1 / 60 * ext_fcn[6:]
        ) / ds
    elif nghost == 4:
        return (
            + 1 / 280 * ext_fcn[0:-8]
            - 4 / 105 * ext_fcn[1:-7]
            + 1 / 5 * ext_fcn[2:-6]
            - 4 / 5 * ext_fcn[3:-5]
            + 4 / 5 * ext_fcn[5:-3]
            - 1 / 5 * ext_fcn[6:-2]
            + 4 / 105 * ext_fcn[7:-1]
            - 1 / 280 * ext_fcn[8:]
        ) / ds


def Dspec(ext_fcn, gr, nghost=1):

    if nghost >= 1:
        red_fcn = ext_fcn[nghost-1:-(nghost-1)]
        red_r = gr[nghost-1:-(nghost-1)]

        # 2nd order centered derivative stencil
        return 3 * (
            red_r[2:]**2 * red_fcn[2:] - red_r[:-2]**2 * red_fcn[:-2]
        ) / (red_r[2:]**3 - red_r[:-2]**3)


def D2(ext_fcn, gr, nghost=1):
    # discretized, centered second degree derivative operator
    ds = gr[1] - gr[0]

    if nghost == 1:
        return (1 * ext_fcn[0:-2, ...] - 2 * ext_fcn[1:-1, ...]
                + 1 * ext_fcn[2:, ...]) / (ds ** 2)
    elif nghost == 2:
        return (-1 / 12 * ext_fcn[0:-4] + 4 / 3 * ext_fcn[1:-3]
                - 5 / 2 * ext_fcn[2:-2]
                + 4 / 3 * ext_fcn[3:-1] - 1 / 12 * ext_fcn[4:]) / (ds ** 2)
    elif nghost == 3:
        return (
            + 1 / 90 * ext_fcn[0:-6]
            - 3 / 20 * ext_fcn[1:-5]
            + 3 / 2 * ext_fcn[2:-4]
            - 49 / 18 * ext_fcn[3:-3]
            + 3 / 2 * ext_fcn[4:-2]
            - 3 / 20 * ext_fcn[5:-1]
            + 1 / 90 * ext_fcn[6:]
        ) / (ds ** 2)
    elif nghost == 4:
        return (
            - 1 / 560 * ext_fcn[0:-8]
            + 8 / 315 * ext_fcn[1:-7]
            - 1 / 5 * ext_fcn[2:-6]
            + 8 / 5 * ext_fcn[3:-5]
            - 205 / 72 * ext_fcn[4:-4]
            + 8 / 5 * ext_fcn[5:-3]
            - 1 / 5 * ext_fcn[6:-2]
            + 8 / 315 * ext_fcn[7:-1]
            - 1 / 560 * ext_fcn[8:]
        ) / (ds ** 2)


def D4_stencil(ext_fcn, nghost=1):
    # discretized, centered fourth degree derivative operator

    if nghost == 2:
        return (1 * ext_fcn[0:-4, ...]
                - 4 * ext_fcn[1:-3, ...]
                + 6 * ext_fcn[2:-2, ...]
                - 4 * ext_fcn[3:-1, ...]
                + 1 * ext_fcn[4:, ...]
                )
    else:
        return None

#
# :D
#

import numpy as np
from numpy.testing import assert_array_almost_equal
import pytest

from randomvars._continuous import Cont
from randomvars._discrete import Disc
from randomvars._utilsgrid import _y_from_xp, _p_from_xy


def test__y_from_xp():
    x = np.array([-1, 1, 2, 4])
    p = np.array([0.1, 0.2, 0.3, 0.4])
    rv_disc = Disc(x, p)

    # For L1 CDF values of xp-grid should be medians of CDF of xy-grid
    y_l1 = _y_from_xp(x, p, "L1")
    rv_cont_l1 = Cont(x, y_l1)
    assert_array_almost_equal(
        rv_cont_l1.cdf(0.5 * (x[:-1] + x[1:])), rv_disc.cdf(x[:-1]), decimal=14
    )

    # For L2 CDF values of xp-grid should be means of CDF of xy-grid
    y_l2 = _y_from_xp(x, p, "L2")
    rv_cont_l2 = Cont(x, y_l2)
    assert_array_almost_equal(
        [rv_cont_l2.integrate_cdf(a, b) / (b - a) for a, b in zip(x[:-1], x[1:])],
        rv_disc.cdf(x[:-1]),
        decimal=14,
    )


def test__p_from_xy():
    x = np.array([-1, 1, 2, 4])
    y = np.array([1, 1, 0.5, 0]) / 3.25
    rv_cont = Cont(x, y)

    # For L1 CDF values of xp-grid should be medians of CDF of xy-grid
    p_l1 = _p_from_xy(x, y, "L1")
    rv_disc_l1 = Disc(x, p_l1)
    assert_array_almost_equal(
        rv_disc_l1.cdf(x[:-1]), rv_cont.cdf(0.5 * (x[:-1] + x[1:])), decimal=14
    )

    # For L2 CDF values of xp-grid should be means of CDF of xy-grid
    p_l2 = _p_from_xy(x, y, "L2")
    rv_disc_l2 = Disc(x, p_l2)
    assert_array_almost_equal(
        rv_disc_l2.cdf(x[:-1]),
        [rv_cont.integrate_cdf(a, b) / (b - a) for a, b in zip(x[:-1], x[1:])],
        decimal=14,
    )

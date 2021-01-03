import numpy as np
from numpy.testing import assert_array_almost_equal
import pytest

from randomvars._continuous import Cont
from randomvars._discrete import Disc
from randomvars._utilsgrid import _y_from_xp, _p_from_xy


def test__y_from_xp():
    x = np.array([-1, 1, 2, 4])
    p = np.array([0.1, 0.2, 0.3, 0.4])
    y = _y_from_xp(x, p)
    # Output should be a part of a proper xy-grid
    assert np.all(y >= 0)
    assert np.sum(0.5 * (x[1:] - x[:-1]) * (y[1:] + y[:-1])) == 1

    # Zero probabilities affect output xy-grid
    x2 = np.array([-1, 1, 1.5, 2, 2.5, 3.5, 4])
    p2 = np.array([0.1, 0.2, 0.0, 0.3, 0.0, 0.0, 0.4])
    y2 = _y_from_xp(x2, p2)
    assert np.all(y >= 0)
    assert np.sum(0.5 * (x[1:] - x[:-1]) * (y[1:] + y[:-1])) == 1
    assert np.any(np.interp(x, x2, y2) != y)
    assert np.all(y2[p2 == 0] == 0)


def test__p_from_xy():
    x = np.array([-1, 1, 2, 4])
    y = np.array([1, 1, 0.5, 0]) / 3.25
    p = _p_from_xy(x, y)
    assert np.sum(p) == 1
    # Zero y-elements result into zero p-elements
    assert np.all(p[y == 0] == 0)

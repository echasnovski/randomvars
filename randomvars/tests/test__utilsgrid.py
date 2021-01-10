import numpy as np
import pytest

from randomvars._continuous import Cont
from randomvars._discrete import Disc
from randomvars._utilsgrid import _y_from_xp, _p_from_xy, _ground_xy
import randomvars.options as op
from randomvars.tests.commontests import DECIMAL, _test_equal_seq


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


class TestGroundXY:
    def test_basic(self):
        w = op.get_option("small_width")
        xy = np.array([0.0, 1.0]), np.array([1.0, 1.0])

        # By default and with `direction=None` should return input
        _test_equal_seq(_ground_xy(xy), xy)
        _test_equal_seq(_ground_xy(xy, direction=None), xy)

        # Basic usage
        _test_equal_seq(
            _ground_xy(xy, direction="both"),
            ([-w, 0, w, 1 - w, 1, 1 + w], [0, 0.5, 1, 1, 0.5, 0]),
        )
        _test_equal_seq(
            _ground_xy(xy, direction="left"), ([-w, 0, w, 1], [0, 0.5, 1, 1])
        )
        _test_equal_seq(
            _ground_xy(xy, direction="right"), ([0, 1 - w, 1, 1 + w], [1, 1, 0.5, 0])
        )

    def test_close_neighbor(self):
        # Big "small_width" is due to numerical representation issues
        with op.option_context({"small_width": 0.1}):
            w = op.get_option("small_width")
            neigh = 0.25 * w
            # Output xy-grids should have the same total square as input xy-grid
            edge_val = neigh / (neigh + w)

            xy = np.array([0.0, neigh, 1 - neigh, 1.0]), np.array([1.0, 1.0, 1.0, 1.0])

            # No extra inner point should be added if there is close neighbor
            _test_equal_seq(
                _ground_xy(xy, direction="both"),
                ([-w, 0, neigh, 1 - neigh, 1, 1 + w], [0, edge_val, 1, 1, edge_val, 0]),
                decimal=DECIMAL,
            )
            _test_equal_seq(
                _ground_xy(xy, direction="left"),
                ([-w, 0, neigh, 1 - neigh, 1], [0, edge_val, 1, 1, 1]),
                decimal=DECIMAL,
            )
            _test_equal_seq(
                _ground_xy(xy, direction="right"),
                ([0, neigh, 1 - neigh, 1, 1 + w], [1, 1, 1, edge_val, 0]),
                decimal=DECIMAL,
            )

    def test_zero_edge(self):
        xy = np.array([0.0, 1.0, 2.0]), np.array([0.0, 1.0, 0.0])

        # No grounding should be done if edge has zero y-value
        _test_equal_seq(_ground_xy(xy, direction="both"), xy)
        _test_equal_seq(_ground_xy(xy, direction="left"), xy)
        _test_equal_seq(_ground_xy(xy, direction="right"), xy)

    def test_options(self):
        with op.option_context({"small_width": 0.1}):
            w = op.get_option("small_width")
            xy = np.array([0.0, 1.0]), np.array([1.0, 1.0])
            _test_equal_seq(
                _ground_xy(xy, direction="both"),
                ([-w, 0, w, 1 - w, 1, 1 + w], [0, 0.5, 1, 1, 0.5, 0]),
            )

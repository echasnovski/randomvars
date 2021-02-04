import numpy as np
from numpy.testing import assert_array_equal, assert_array_almost_equal
import pytest

from randomvars._continuous import Cont
from randomvars._discrete import Disc
from randomvars._utilsgrid import (
    _y_from_xp,
    _p_from_xy,
    _stack_xp,
    _stack_xy,
    _compute_stack_ground_info,
    _ground_xy,
)
from randomvars.options import options
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


def test__stack_xp():
    xp_seq = [([0, 1], [0.1, 0.2]), ([0, 2], [0.5, 0.7])]
    assert_array_equal(_stack_xp(xp_seq), ([0, 1, 2], [0.6, 0.2, 0.7]))


class TestStackXY:
    def test_basic(self):
        w = options.small_width

        xy_seq1 = [([0, 1], [1, 1]), ([0.25, 1.25], [1, 1]), ([0.5, 1.5], [1, 1])]
        _test_equal_seq(
            _stack_xy(xy_seq1),
            (
                [
                    0,
                    0.25 - w,
                    0.25,
                    0.25 + w,
                    0.5 - w,
                    0.5,
                    0.5 + w,
                    1 - w,
                    1,
                    1 + w,
                    1.25 - w,
                    1.25,
                    1.25 + w,
                    1.5,
                ],
                [1, 1, 1.5, 2, 2, 2.5, 3, 3, 2.5, 2, 2, 1.5, 1, 1],
            ),
        )

        xy_seq2 = [([0, 1], [1, 1]), ([0.25, 0.75], [1, 1])]
        _test_equal_seq(
            _stack_xy(xy_seq2),
            (
                [0, 0.25 - w, 0.25, 0.25 + w, 0.75 - w, 0.75, 0.75 + w, 1],
                [1, 1, 1.5, 2, 2, 1.5, 1, 1],
            ),
        )

        xy_seq3 = [([0, 1], [1, 1]), ([0, 0.5], [1, 1]), ([0.5, 1], [1, 1])]
        _test_equal_seq(
            _stack_xy(xy_seq3), ([0, 0.5 - w, 0.5, 0.5 + w, 1], [2, 2, 2, 2, 2])
        )

        # Case when there should be no grounding
        xy_seq4 = [([0, 1], [1, 1]), ([0, 1], [1, 1])]
        _test_equal_seq(_stack_xy(xy_seq4), ([0, 1], [2, 2]))

    def test_width(self):
        w = options.small_width

        # Width should be computed as maximum value not bigger than
        # `small_width` option that ensures identical jumping slopes
        xy_seq = [([0, 0.5 * w, 1 - 0.25 * w, 1], [1, 1, 1, 1]), ([-1, 2], [1, 1])]
        W = 0.25 * w
        _test_equal_seq(
            _stack_xy(xy_seq),
            (
                [-1, -W, 0, W, 0.5 * w, 1 - W, 1, 1 + W, 2],
                [1, 1, 1.5, 2, 2, 2, 1.5, 1, 1],
            ),
            decimal=DECIMAL,
        )

        # Neighboring points for edges where grounding doesn't happen shouldn't
        # affect the output width
        xy_seq = [([0, 0.1 * w, 1], [1, 1, 1]), ([0.25, 0.75], [1, 1])]
        W = w
        _test_equal_seq(
            _stack_xy(xy_seq),
            (
                [0, 0.1 * w, 0.25 - W, 0.25, 0.25 + W, 0.75 - W, 0.75, 0.75 + W, 1],
                [1, 1, 1, 1.5, 2, 2, 1.5, 1, 1],
            ),
        )

    def test_touching_supports(self):
        w = options.small_width

        # Touching supports should lead to linear change between y-values
        xy_seq = [([0, 1], [1, 1]), ([1, 2], [2, 2])]
        _test_equal_seq(_stack_xy(xy_seq), ([0, 1 - w, 1, 1 + w, 2], [1, 1, 1.5, 2, 2]))

        # Even in case of close edge neighbors
        xy_seq = [([0, 1 - 0.25 * w, 1], [1, 1, 1]), ([1, 2], [2, 2])]
        _test_equal_seq(
            _stack_xy(xy_seq),
            ([0, 1 - 0.25 * w, 1, 1 + 0.25 * w, 2], [1, 1, 1.5, 2, 2]),
        )

    def test_zero_edge(self):
        w = options.small_width
        xy_seq = [([0, 1, 2], [0, 1, 0]), ([1, 2, 3], [0, 1, 1])]
        _test_equal_seq(_stack_xy(xy_seq), ([0, 1, 2, 3], [0.0, 1.0, 1.0, 1.0]))

    def test_options(self):
        xy_seq = [([0, 1], [1, 1]), ([0.25, 0.75], [1, 1])]
        with options.context({"small_width": 0.01}):
            w = options.small_width
            _test_equal_seq(
                _stack_xy(xy_seq),
                (
                    [0, 0.25 - w, 0.25, 0.25 + w, 0.75 - w, 0.75, 0.75 + w, 1],
                    [1, 1, 1.5, 2, 2, 1.5, 1, 1],
                ),
            )


class TestGroundInfo:
    def test_direction(self):
        _test_equal_seq(
            _compute_stack_ground_info([([0, 1], [1, 1]), ([0.5, 1.5], [1, 1])])[0],
            ["right", "left"],
        )
        _test_equal_seq(
            _compute_stack_ground_info([([0, 1], [1, 1]), ([0.5, 0.75], [1, 1])])[0],
            ["none", "both"],
        )
        _test_equal_seq(
            _compute_stack_ground_info(
                [
                    ([0, 1], [1, 1]),
                    ([0, 0.5], [1, 1]),
                    ([0.25, 0.75], [1, 1]),
                    ([0.5, 1], [1, 1]),
                ]
            )[0],
            ["none", "right", "both", "left"],
        )
        _test_equal_seq(
            _compute_stack_ground_info([([0, 1], [1, 1]), ([0, 1], [1, 1])])[0],
            ["none", "none"],
        )

    def test_width(self):
        w = options.small_width

        # Basic usage should return `small_width`
        assert_array_equal(
            _compute_stack_ground_info([([0, 1], [1, 1]), ([0.5, 1.5], [1, 1])])[1], w
        )

        # In case of close neighbors, minimum distance should be returned but
        # only using edges where grounding actually happens
        base_xy = ([0, 0.1 * w, 1 - 0.2 * w, 1], [1, 1, 1, 1])
        assert_array_almost_equal(
            _compute_stack_ground_info([base_xy, ([-1, 2], [1, 1])])[1],
            0.1 * w,
            decimal=DECIMAL,
        )
        assert_array_almost_equal(
            _compute_stack_ground_info([base_xy, ([0, 2], [1, 1])])[1],
            0.2 * w,
            decimal=DECIMAL,
        )
        assert_array_almost_equal(
            _compute_stack_ground_info([base_xy, ([0, 1], [1, 1])])[1],
            w,
            decimal=DECIMAL,
        )


class TestGroundXY:
    def test_basic(self):
        w = options.small_width
        xy = np.array([0.0, 1.0]), np.array([1.0, 1.0])

        # Basic usage
        _test_equal_seq(
            _ground_xy(xy, w, direction="both"),
            ([-w, 0, w, 1 - w, 1, 1 + w], [0, 0.5, 1, 1, 0.5, 0]),
        )
        _test_equal_seq(
            _ground_xy(xy, w, direction="left"), ([-w, 0, w, 1], [0, 0.5, 1, 1])
        )
        _test_equal_seq(
            _ground_xy(xy, w, direction="right"), ([0, 1 - w, 1, 1 + w], [1, 1, 0.5, 0])
        )
        _test_equal_seq(_ground_xy(xy, w, direction="none"), xy)

    def test_close_neighbor(self):
        # Using big "small_width" to mitigate possible floating points issues
        with options.context({"small_width": 0.1}):
            w = options.small_width
            neigh = 0.25 * w
            # Output xy-grids should have the same total square as input xy-grid
            edge_val = neigh / (neigh + w)

            xy = np.array([0.0, neigh, 1 - neigh, 1.0]), np.array([1.0, 1.0, 1.0, 1.0])

            # No inner point should be added if there is close neighbor
            _test_equal_seq(
                _ground_xy(xy, w, direction="both"),
                ([-w, 0, neigh, 1 - neigh, 1, 1 + w], [0, edge_val, 1, 1, edge_val, 0]),
                decimal=DECIMAL,
            )
            _test_equal_seq(
                _ground_xy(xy, w, direction="left"),
                ([-w, 0, neigh, 1 - neigh, 1], [0, edge_val, 1, 1, 1]),
                decimal=DECIMAL,
            )
            _test_equal_seq(
                _ground_xy(xy, w, direction="right"),
                ([0, neigh, 1 - neigh, 1, 1 + w], [1, 1, 1, edge_val, 0]),
                decimal=DECIMAL,
            )

    def test_zero_edge(self):
        w = options.small_width
        xy = np.array([0.0, 1.0, 2.0]), np.array([0.0, 1.0, 0.0])

        # No grounding should be done if edge has zero y-value
        _test_equal_seq(_ground_xy(xy, w, direction="both"), xy)
        _test_equal_seq(_ground_xy(xy, w, direction="left"), xy)
        _test_equal_seq(_ground_xy(xy, w, direction="right"), xy)

    def test_options(self):
        with options.context({"small_width": 0.1}):
            w = options.small_width
            xy = np.array([0.0, 1.0]), np.array([1.0, 1.0])
            _test_equal_seq(
                _ground_xy(xy, w, direction="both"),
                ([-w, 0, w, 1 - w, 1, 1 + w], [0, 0.5, 1, 1, 0.5, 0]),
            )

# pylint: disable=missing-function-docstring
"""Tests for 'options.py' file"""
import numpy as np
from numpy.testing import assert_array_equal
import pytest

from randomvars.options import *
from randomvars.options import _default_options


def test_default_discrete_estimator():
    # Normal usage
    out = default_discrete_estimator([3, 1, 2, 1, 3])
    assert len(out) == 2
    assert_array_equal(out[0], np.array([1, 2, 3]))
    assert_array_equal(out[1], np.array([0.4, 0.2, 0.4]))

    # Error if no finite values
    with pytest.raises(ValueError, match="doesn't have finite values"):
        default_discrete_estimator([-np.inf, np.nan, np.inf])

    # Warning if there are some non-finite values
    with pytest.warns(UserWarning, match="non-finite values"):
        out = default_discrete_estimator([1, np.nan, np.inf])
        assert_array_equal(out[0], np.array([1]))
        assert_array_equal(out[1], np.array([1]))


def test_get_option():
    # Normal usage
    assert isinstance(get_option("n_grid"), int)

    # Error on wrong option
    with pytest.raises(OptionError, match="no option 'aaa'"):
        get_option("aaa")


def test_set_option():
    prev_opt = get_option("n_grid")
    new_opt = prev_opt + 1

    try:
        # Normal usage
        set_option("n_grid", new_opt)
        assert get_option("n_grid") == new_opt

        # Error on wrong option
        with pytest.raises(OptionError, match="no option 'aaa'"):
            set_option("aaa", 0)
    finally:
        set_option("n_grid", prev_opt)


def test_reset_option():
    prev_opt = get_option("n_grid")
    def_opt = _default_options["n_grid"]
    new_opt = def_opt + 1

    try:
        # Normal usage
        set_option("n_grid", new_opt)
        assert get_option("n_grid") == new_opt

        reset_option("n_grid")
        assert get_option("n_grid") == def_opt

        # Error on wrong option
        with pytest.raises(OptionError, match="no option 'aaa'"):
            reset_option("aaa")
    finally:
        set_option("n_grid", prev_opt)


def test_option_context():
    prev_opt = get_option("n_grid")
    new_opt = prev_opt + 1

    try:
        # Normal usage
        with option_context({"n_grid": new_opt}):
            assert get_option("n_grid") == new_opt

        assert get_option("n_grid") == prev_opt

        # Error while setting option shouldn't affect option undo
        with pytest.raises(OptionError, match="no option 'aaa'"):
            with option_context({"n_grid": new_opt, "aaa": 0}):
                pass

        assert get_option("n_grid") == prev_opt
    finally:
        set_option("n_grid", prev_opt)

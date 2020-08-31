# pylint: disable=missing-function-docstring
"""Tests for 'options.py' file"""
import pytest

from randomvars.options import *
from randomvars.options import _default_options


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

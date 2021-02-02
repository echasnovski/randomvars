# pylint: disable=missing-function-docstring
"""Tests for 'options.py' file"""
import numpy as np
from numpy.testing import assert_array_equal
from scipy.stats.kde import gaussian_kde
import pytest

from randomvars.options import (
    OptionError,
    _SingleOption,
    _default_options,
    _docstring_options_list,
    _docstring_paragraph,
    _docstring_relevant_options,
    _options_list,
    estimator_bool_default,
    estimator_cont_default,
    estimator_disc_default,
    estimator_mixt_default,
    get_option,
    option_context,
    reset_option,
    set_option,
)

from randomvars.tests.commontests import _test_equal_seq


def test_estimator_bool_default():
    # Normal usage
    out = estimator_bool_default([True, False, False, True, True])
    assert out == 0.6

    # Usage with other input types
    out = estimator_bool_default([0, 0, 0, 1, 1])
    assert out == 0.4

    out = estimator_bool_default([0.0, 0.0, 0.0, 1.0, 0.0])
    assert out == 0.2

    out = estimator_bool_default(["a", "b", "c"])
    assert out == 1.0


def test_estimator_cont_default():
    # Normal usage
    sample = [1, 1, 1, 2, 3, 4]
    x_ref = np.linspace(-10, 10, 1001)
    assert_array_equal(
        estimator_cont_default(sample)(x_ref), gaussian_kde(sample)(x_ref)
    )

    # Error on small sample
    with pytest.raises(ValueError, match="two"):
        estimator_cont_default(np.array([1]))


def test_estimator_disc_default():
    # Normal usage
    out = estimator_disc_default([3, 1, 2, 1, 3])
    assert len(out) == 2
    assert_array_equal(out[0], np.array([1, 2, 3]))
    assert_array_equal(out[1], np.array([0.4, 0.2, 0.4]))

    # Error if no finite values
    with pytest.raises(ValueError, match="doesn't have finite values"):
        estimator_disc_default([-np.inf, np.nan, np.inf])

    # Warning if there are some non-finite values
    with pytest.warns(UserWarning, match="non-finite values"):
        out = estimator_disc_default([1, np.nan, np.inf])
        assert_array_equal(out[0], np.array([1]))
        assert_array_equal(out[1], np.array([1]))


def test_estimator_mixt_default():
    # Basic usage
    _test_equal_seq(
        estimator_mixt_default(np.array([10, 2, 0.1, -100, 0.1, 2, 0.2, 2, 2, 3])),
        (np.array([10, -100, 0.2, 3]), np.array([2, 0.1, 0.1, 2, 2, 2])),
    )

    # Degenerate cases
    _test_equal_seq(estimator_mixt_default(np.arange(10)), (np.arange(10), None))
    _test_equal_seq(
        estimator_mixt_default(np.array([0, 1, 0, 1])), (None, np.array([0, 1, 0, 1]))
    )


def test__docstring_paragraph():
    big_string = " ".join(["Newly added documentation"] * 5)

    @_docstring_paragraph(ccc=big_string, ddd="Extra help.")
    def f():
        """Function

        Documentation.

            {ccc}

        {ddd}

        More documentation.
        """
        return 1

    # Paragraph should be added with identation
    assert f.__doc__.find("    Newly added documentation") > -1
    assert f.__doc__.find("Extra help.") > -1

    # Paragraph should be wrapped by default
    assert all(len(s) < 79 for s in f.__doc__.splitlines())

    # Wrapping shouldn't be done if `wrap=False`
    @_docstring_paragraph(wrap=False, ccc=big_string)
    def f2():
        """Function

        {ccc}
        """
        return 1

    assert not all([len(s) < 79 for s in f2.__doc__.splitlines()])


def test__docstring_relevant_options():
    @_docstring_relevant_options(["aaa", "bbb"])
    def f():
        """Function

        Documentation.

        {relevant_options}

        More documentation.
        """

    assert f.__doc__.find("Relevant package options: `aaa`, `bbb`") > -1


def test__docstring_options_list():
    @_docstring_options_list
    def f():
        """Function

        Documentation.

        {options_list}

        More documentation.
        """

    assert f.__doc__.find(_options_list) > -1


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


class Test_SingleOption:
    def test_basic(self):
        class A:
            opt = _SingleOption(default=0.1, validator=(lambda x: x > 0, "positive"))

        a = A()

        # Get
        assert a.opt == 0.1

        # Set
        a.opt = 1
        assert a.opt == 1

        # Default
        assert type(a).__dict__["opt"].default == 0.1

    def test_errors(self):
        class A:
            opt = _SingleOption(default=0.1, validator=(lambda x: x > 0, "positive"))

        a = A()

        with pytest.raises(OptionError, match="verifying validity of value for `opt`"):
            a.opt = "a"

        with pytest.raises(OptionError, match="`opt` should be positive"):
            a.opt = 0.0

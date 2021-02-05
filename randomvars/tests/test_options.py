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
    config,
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


class TestConfig:
    """Tests for `config` object"""

    def test_basic(self):
        # Get option as attribute
        assert isinstance(config.base_tolerance, float)

        # Set option as attribute
        val = config.base_tolerance
        config.base_tolerance = 0.1
        assert config.base_tolerance == 0.1

        ## Cleanup
        config.base_tolerance = val

        # Validate option value
        with pytest.raises(OptionError, match="non-negative"):
            config.base_tolerance = -0.1
        with pytest.raises(OptionError, match="float"):
            config.base_tolerance = "a"

    def test_available_options(self):
        # `base_tolerance`
        config.base_tolerance == 1e-12

        with pytest.raises(OptionError, match="float"):
            config.base_tolerance = "0.1"
        with pytest.raises(OptionError, match="non-negative"):
            config.base_tolerance = -0.1

        # `cdf_tolerance`
        config.cdf_tolerance == 1e-4

        with pytest.raises(OptionError, match="float"):
            config.cdf_tolerance = "0.1"
        with pytest.raises(OptionError, match="non-negative"):
            config.cdf_tolerance = -0.1

        # `density_mincoverage`
        config.density_mincoverage == 0.9999

        with pytest.raises(OptionError, match="float"):
            config.density_mincoverage = "0.1"
        with pytest.raises(OptionError, match=r"inside \[0; 1\)"):
            config.density_mincoverage = -0.1
        with pytest.raises(OptionError, match=r"inside \[0; 1\)"):
            config.density_mincoverage = 1.1

        ## Zero is allowed, one is not allowed
        val = config.density_mincoverage
        config.density_mincoverage = 0.0
        config.density_mincoverage = val
        with pytest.raises(OptionError):
            config.density_mincoverage = 1.0

        # `estimator_bool`
        assert config.estimator_bool([0, 1, 0]) == estimator_bool_default([0, 1, 0])

        with pytest.raises(OptionError, match="callable"):
            config.estimator_bool = 0.0

        # `estimator_cont`
        assert np.all(
            config.estimator_cont([0, 1, 0])([-1, 0, 1])
            == estimator_cont_default([0, 1, 0])([-1, 0, 1])
        )

        with pytest.raises(OptionError, match="callable"):
            config.estimator_cont = 0.0

        # `estimator_disc`
        _test_equal_seq(
            config.estimator_disc([0, 1, 0]), estimator_disc_default([0, 1, 0])
        )

        with pytest.raises(OptionError, match="callable"):
            config.estimator_disc = 0.0

        # `estimator_mixt`
        _test_equal_seq(
            config.estimator_mixt([0, 1, 0]), estimator_mixt_default([0, 1, 0])
        )

        with pytest.raises(OptionError, match="callable"):
            config.estimator_mixt = 0.0

        # `metric`
        assert config.metric == "L2"
        with pytest.raises(OptionError, match="one of"):
            config.metric = 0.0
        with pytest.raises(OptionError, match="one of"):
            config.metric = "aaa"

        # `n_grid`
        assert config.n_grid == 1001
        with pytest.raises(OptionError, match="integer"):
            config.n_grid = "1001"
        with pytest.raises(OptionError, match="more than 1"):
            config.n_grid = 1

        # `small_prob`
        config.small_prob == 1e-6

        with pytest.raises(OptionError, match="float"):
            config.small_prob = "0.1"
        with pytest.raises(OptionError, match=r"inside \(0; 1\)"):
            config.small_prob = -0.1
        with pytest.raises(OptionError, match=r"inside \(0; 1\)"):
            config.small_prob = 1.1
        with pytest.raises(OptionError, match=r"inside \(0; 1\)"):
            config.small_prob = 0.0
        with pytest.raises(OptionError, match=r"inside \(0; 1\)"):
            config.small_prob = 1.0

        # `small_width`
        config.small_width == 1e-8

        with pytest.raises(OptionError, match="float"):
            config.small_width = "0.1"
        with pytest.raises(OptionError, match="positive"):
            config.small_width = -0.1
        with pytest.raises(OptionError, match="positive"):
            config.small_width = 0.0

    def test_list(self):
        l = config.list
        assert isinstance(l, list)
        assert all(isinstance(val, str) for val in l)
        assert all(isinstance(type(config).__dict__[val], _SingleOption) for val in l)

    def test_dict(self):
        d = config.dict
        assert isinstance(d, dict)
        assert list(d.keys()) == config.list

        # Should return current option values
        val = config.base_tolerance
        config.base_tolerance = 0.1
        assert config.dict["base_tolerance"] == 0.1

        ## Cleanup
        config.base_tolerance = val

    def test_defaults(self):
        # Should return default option values
        val = config.base_tolerance
        config.base_tolerance = 0.1
        assert config.defaults["base_tolerance"] == val

        ## Cleanup
        config.base_tolerance = val

    def test_get_single(self):
        assert config.get_single("base_tolerance") == config.base_tolerance

        # Error on non-existent option
        with pytest.raises(OptionError, match="no option `aaa`"):
            config.get_single("aaa")

    def test_get(self):
        assert config.get(["base_tolerance", "estimator_bool"]) == [
            config.base_tolerance,
            config.estimator_bool,
        ]

        # Should return list even with one-element input
        assert config.get(["base_tolerance"]) == [config.base_tolerance]

        # Error on non-existent option
        with pytest.raises(OptionError, match="no option `aaa`"):
            config.get(["base_tolerance", "aaa"])

    def test_set_single(self):
        val = config.base_tolerance
        config.set_single("base_tolerance", 0.1)
        assert config.base_tolerance == 0.1

        ## Cleanup
        config.base_tolerance = val

        # Error on non-existent option
        with pytest.raises(OptionError, match="no option `aaa`"):
            config.set_single("aaa", 0.1)

    def test_set(self):
        vals = [config.base_tolerance, config.estimator_bool]
        config.set({"base_tolerance": 0.1, "estimator_bool": lambda x: np.mean(x)})
        assert config.base_tolerance == 0.1
        assert config.estimator_bool([1, 2]) == 1.5

        ## Cleanup
        config.base_tolerance = vals[0]
        config.estimator_bool = vals[1]

        # Error on non-existent option
        with pytest.raises(OptionError, match="no option `aaa`"):
            config.set({"aaa": 0.1})

    def test_reset_single(self):
        # Should set value to default option, not the previous one
        val = config.base_tolerance
        config.base_tolerance = 0.1
        config.base_tolerance = 0.2
        config.reset_single("base_tolerance")
        assert config.base_tolerance == val

        ## Cleanup
        config.base_tolerance = val

        # Error on non-existent option
        with pytest.raises(OptionError, match="no option `aaa`"):
            config.reset_single("aaa")

    def test_reset(self):
        # Should set value to default option, not the previous one
        vals = [config.base_tolerance, config.estimator_bool]
        config.base_tolerance = 0.1
        config.base_tolerance = 0.2
        config.estimator_bool = lambda x: np.mean(x)
        config.estimator_bool = lambda x: np.median(x)
        config.reset(["base_tolerance", "estimator_bool"])
        assert config.base_tolerance == vals[0]
        assert config.estimator_bool == vals[1]

        ## Cleanup
        config.base_tolerance = vals[0]
        config.estimator_bool = vals[1]

        # Error on non-existent option
        with pytest.raises(OptionError, match="no option `aaa`"):
            config.reset(["base_tolerance", "aaa"])

    def test_context(self):
        # It shouldn't be possible to use raw `options` as context manager
        with pytest.raises(OptionError, match=r"Use `context\(\)`"):
            with config:
                pass

        # Usage of `context()`
        val = config.base_tolerance
        assert val != 0.1
        with config.context({"base_tolerance": 0.1}):
            assert config.base_tolerance == 0.1
        assert config.base_tolerance == val

        ## Cleanup
        config.base_tolerance = val

        # It shouldn't be possible to use raw `options` as context manager even
        # after using `context()` (deals with some possible implementation
        # detail)
        with pytest.raises(OptionError, match=r"Use `context\(\)`"):
            with config:
                pass

        # Error on non-existent option
        with pytest.raises(OptionError, match="no option `aaa`"):
            with config.context({"aaa": 0.1}):
                pass

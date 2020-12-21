# pylint: disable=missing-function-docstring
"""Tests for '_random.py' file"""
import numpy as np
from numpy.testing import assert_array_equal
import pytest

from randomvars._random import Rand
from randomvars._utils import _test_one_value_input


class TestRand:
    """Regression tests for `Rand` class"""

    def test_properties(self):
        rv = Rand()

        with pytest.raises(NotImplementedError):
            rv.a
        with pytest.raises(NotImplementedError):
            rv.b

    def test_support(self):
        class TmpRand(Rand):
            @property
            def a(self):
                return 0

            @property
            def b(self):
                return 1

        tmp_rv = TmpRand()
        assert tmp_rv.support() == (0, 1)

    def test_from_rv(self):
        with pytest.raises(NotImplementedError):
            Rand().from_rv("a")

        # Object of class `Rand` should be returned untouched
        rv = Rand()
        rv.aaa = "Extra method"
        rv2 = Rand.from_rv(rv)
        assert rv2.aaa == "Extra method"

    def test_from_sample(self):
        with pytest.raises(NotImplementedError):
            Rand().from_sample("a")

    def test_pdf(self):
        with pytest.raises(NotImplementedError):
            Rand().pdf(0)

    def test_logpdf(self):
        class TmpRand(Rand):
            def pdf(self, x):
                return x

        tmp_rv = TmpRand()

        # Regular checks
        assert_array_equal(tmp_rv.logpdf(np.exp([1, 2])), [1, 2])

        # One-value input
        _test_one_value_input(tmp_rv.logpdf, 1)

        # Giving zero pdf values to `np.log` shouldn't result into `RuntimeWarning`
        with pytest.warns(None):
            assert_array_equal(tmp_rv.logpdf([0]), np.array([-np.inf]))

        # Giving negative pdf values (for any reason) should result into
        # warning
        with pytest.warns(RuntimeWarning):
            assert_array_equal(tmp_rv.logpdf([-1]), np.nan)

    def test_pmf(self):
        with pytest.raises(NotImplementedError):
            Rand().pmf(0)

    def test_logpdf(self):
        class TmpRand(Rand):
            def pmf(self, x):
                return x

        tmp_rv = TmpRand()

        # Regular checks
        assert_array_equal(tmp_rv.logpmf(np.exp([1, 2])), [1, 2])

        # One-value input
        _test_one_value_input(tmp_rv.logpmf, 1)

        # Giving zero pmf values to `np.log` shouldn't result into `RuntimeWarning`
        with pytest.warns(None):
            assert_array_equal(tmp_rv.logpmf([0]), np.array([-np.inf]))

        # Giving negative pmf values (for any reason) should result into
        # warning
        with pytest.warns(RuntimeWarning):
            assert_array_equal(tmp_rv.logpmf([-1]), np.nan)

    def test_cdf(self):
        with pytest.raises(NotImplementedError):
            Rand().cdf(0)

    def test_ppf(self):
        with pytest.raises(NotImplementedError):
            Rand().ppf(0)

    def test_rvs(self):
        rv = Rand()
        rv.ppf = lambda x: x
        smpl = rv.rvs(size=10)
        assert smpl.shape == (10,)

        smpl2 = rv.rvs(size=10, random_state=101)
        assert smpl2.shape == (10,)

    def test_integrate_cdf(self):
        with pytest.raises(NotImplementedError):
            Rand().integrate_cdf(0, 1)

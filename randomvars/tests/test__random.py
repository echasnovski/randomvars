# pylint: disable=missing-function-docstring
"""Tests for '_random.py' file"""
import numpy as np
from numpy.testing import assert_array_equal, assert_array_almost_equal
import pytest

from randomvars._random import Rand
from randomvars.tests.commontests import _test_one_value_input


class TestRand:
    """Regression tests for `Rand` class"""

    def test_eq(self):
        class Tmp(Rand):
            def __init__(self, loc, scale):
                self.loc = loc
                self.scale = scale
                super().__init__()

            @property
            def params(self):
                return {"loc": self.loc, "scale": self.scale}

        # Equality
        ## Simple params
        rv1 = Tmp(0, "a")
        rv2 = Tmp(0, "a")
        assert (rv1 == rv2) and (rv1 is not rv2)

        ## Numpy array params
        rv1 = Tmp(np.array([0, 1]), np.array([0, 1]))
        rv2 = Tmp(np.array([0, 1]), np.array([[0, 1]]))
        assert (rv1 == rv2) and (rv1 is not rv2)

        ## Exotic params
        rv1 = Tmp(loc=(0, "a"), scale={"alpha": 1, "beta": 2})
        rv2 = Tmp(loc=(0, "a"), scale={"alpha": 1, "beta": 2})
        assert (rv1 == rv2) and (rv1 is not rv2)

        # Not equality
        ## Different classes
        rv1 = Tmp(0, 1)
        rv2 = Rand()
        assert (rv1 == rv2) is False

        ## Absence of params
        class Tmp2(Tmp):
            def __init__(self, loc, scale):
                super().__init__(loc=loc, scale=scale)
                self._params = super().params

            @property
            def params(self):
                return self._params

            @params.setter
            def params(self, new_params):
                self._params = new_params

            @params.deleter
            def params(self):
                del self._params

        rv1 = Tmp2(0, 1)
        rv2 = Tmp2(0, 1)
        del rv2.params
        assert (rv1 == rv2) is False

        ## Absence of dictionary params
        rv1 = Tmp2(0, 1)
        rv2 = Tmp2(0, 1)
        rv2.params = "a"
        assert (rv1 == rv2) is False

        ## Dictionary params with different keys
        rv1 = Tmp2(0, 1)
        rv2 = Tmp2(0, 1)
        rv2.params = {"a": 1}
        assert (rv1 == rv2) is False

        ## Not equal params
        rv1 = Tmp(0, 1 + 1e-8)
        rv2 = Tmp(0, 1)
        assert (rv1 == rv2) is False

    def test_properties(self):
        rv = Rand()

        assert rv.params == dict()
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

    def test_compress(self):
        rv = Rand()
        assert rv.compress() is rv

    def test_from_rv(self):
        # Objects of `Rand` class should be `convert()`ed
        rv = Rand()
        assert Rand.from_rv(rv) is rv.convert("Rand")

        with pytest.raises(NotImplementedError):
            Rand().from_rv("a")

    def test_from_sample(self):
        with pytest.raises(NotImplementedError):
            Rand().from_sample("a")

    def test_pdf(self):
        with pytest.raises(NotImplementedError):
            Rand().pdf(0)

    def test_logpdf(self):
        class TmpRand(Rand):
            def pdf(self, x):
                return np.asarray(x)

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

    def test_logpmf(self):
        class TmpRand(Rand):
            def pmf(self, x):
                return np.asarray(x)

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

    def test_logcdf(self):
        class TmpRand(Rand):
            def cdf(self, x):
                return np.asarray(x)

        tmp_rv = TmpRand()

        # Regular checks
        assert_array_equal(tmp_rv.logcdf(np.exp([1, 2])), [1, 2])

        # One-value input
        _test_one_value_input(tmp_rv.logcdf, 1)

        # Giving zero cdf values to `np.log` shouldn't result into `RuntimeWarning`
        with pytest.warns(None):
            assert_array_equal(tmp_rv.logcdf([0]), np.array([-np.inf]))

        # Giving negative cdf values (for any reason) should result into
        # warning
        with pytest.warns(RuntimeWarning):
            assert_array_equal(tmp_rv.logcdf([-1]), np.nan)

    def test_sf(self):
        class TmpRand(Rand):
            def cdf(self, x):
                return np.asarray(x)

        tmp_rv = TmpRand()

        # Regular checks
        assert_array_equal(tmp_rv.sf([1, 2]), [0, -1])

        # One-value input
        _test_one_value_input(tmp_rv.sf, 1)

    def test_logsf(self):
        class TmpRand(Rand):
            def sf(self, x):
                return np.asarray(x)

        tmp_rv = TmpRand()

        # Regular checks
        assert_array_equal(tmp_rv.logsf(np.exp([1, 2])), [1, 2])

        # One-value input
        _test_one_value_input(tmp_rv.logsf, 1)

        # Giving zero sf values to `np.log` shouldn't result into `RuntimeWarning`
        with pytest.warns(None):
            assert_array_equal(tmp_rv.logsf([0]), np.array([-np.inf]))

        # Giving negative sf values (for any reason) should result into warning
        with pytest.warns(RuntimeWarning):
            assert_array_equal(tmp_rv.logsf([-1]), np.nan)

    def test_ppf(self):
        with pytest.raises(NotImplementedError):
            Rand().ppf(0)

    def test_isf(self):
        class TmpRand(Rand):
            def cdf(self, x):
                return np.asarray(x ** 2)

            def ppf(self, q):
                return np.asarray(np.sqrt(q))

        tmp_rv = TmpRand()

        # Regular checks
        q_ref = np.linspace(0, 1, 11)
        assert_array_almost_equal(tmp_rv.sf(tmp_rv.isf(q_ref)), q_ref, decimal=15)

        # One-value input
        _test_one_value_input(tmp_rv.isf, 1)

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

    def test_convert(self):
        rv = Rand()

        # By default and supplying `None` should return self
        assert rv.convert() is rv
        assert rv.convert(None) is rv

        # Converting to own class should return self
        assert rv.convert("Rand") is rv

        # Otherwise method is not implemented
        with pytest.raises(NotImplementedError):
            rv.convert("Disc")

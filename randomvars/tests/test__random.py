# pylint: disable=missing-function-docstring
"""Tests for '_random.py' file"""
import pytest

from randomvars._random import Rand


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

    def test_pmf(self):
        with pytest.raises(NotImplementedError):
            Rand().pmf(0)

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

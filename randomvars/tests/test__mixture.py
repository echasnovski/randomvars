# pylint: disable=missing-function-docstring
"""Tests for '_mixture.py' file"""
import numpy as np
from numpy.testing import assert_array_equal
import scipy.stats.distributions as distrs
import pytest

from randomvars._continuous import Cont
from randomvars._discrete import Disc
from randomvars._mixture import Mixt
from randomvars._utils import _assert_equal_seq


def assert_equal_mixt(rv_1, rv_2):
    # Check weights
    assert rv_1.weight_cont == rv_2.weight_cont
    assert rv_1.weight_disc == rv_2.weight_disc

    # Check continuous parts
    if rv_1.cont is not None:
        if rv_2.cont is not None:
            grid_1 = rv_1.cont.x, rv_1.cont.y, rv_1.cont.p
            grid_2 = rv_2.cont.x, rv_2.cont.y, rv_2.cont.p
            _assert_equal_seq(grid_1, grid_2)
        else:
            raise ValueError("`rv_2.cont` is `None` while `rv_1.cont` is not.")
    else:
        if rv_2.cont is not None:
            raise ValueError("`rv_1.cont` is `None` while `rv_2.cont` is not.")

    # Check discrete parts
    if rv_1.disc is not None:
        if rv_2.disc is not None:
            grid_1 = rv_1.disc.x, rv_1.disc.prob, rv_1.disc.p
            grid_2 = rv_2.disc.x, rv_2.disc.prob, rv_2.disc.p
            _assert_equal_seq(grid_1, grid_2)
        else:
            raise ValueError("`rv_2.disc` is `None` while `rv_1.disc` is not.")
    else:
        if rv_2.disc is not None:
            raise ValueError("`rv_1.disc` is `None` while `rv_2.disc` is not.")


class TestMixt:
    """Regression tests for `Mixt` class"""

    def test_init_errors(self):
        cont = Cont([0, 1], [1, 1])
        disc = Disc([-1, 0.5], [0.25, 0.75])

        # Checking `cont`
        ## `cont` can be None but only with `weight_cont` equal to 0
        with pytest.raises(ValueError, match="`cont`.*`None`.*`weight_cont`"):
            Mixt(cont=None, disc=disc, weight_cont=0.5)

        ## `cont` can't be any other than `Cont` or `None`
        with pytest.raises(ValueError, match="`cont`.*`Cont`.*`None`"):
            Mixt(cont=distrs.norm(), disc=disc, weight_cont=0.5)

        # Checking `disc`
        ## `disc` can be None but only with `weight_cont` equal to 1
        with pytest.raises(ValueError, match="`disc`.*`None`.*`weight_cont`"):
            Mixt(cont=cont, disc=None, weight_cont=0.5)

        ## `disc` can't be any other than `Disc` or `None`
        with pytest.raises(ValueError, match="`disc`.*`Disc`.*`None`"):
            Mixt(cont=cont, disc=distrs.bernoulli(p=0.1), weight_cont=0.5)

        # Both `cont` and `disc` being `None` should also throw error
        with pytest.raises(ValueError):
            Mixt(cont=None, disc=None, weight_cont=0.5)

        # Checkint `weight_cont`
        with pytest.raises(ValueError, match="number"):
            Mixt(cont=cont, disc=disc, weight_cont="a")
        with pytest.raises(ValueError, match="0"):
            Mixt(cont=cont, disc=disc, weight_cont=-0.1)
        with pytest.raises(ValueError, match="1"):
            Mixt(cont=cont, disc=disc, weight_cont=1.1)

    def test_init(self):
        cont = Cont([0, 1], [1, 1])
        disc = Disc([-1, 0.5], [0.25, 0.75])

        # Basic usage
        rv_out = Mixt(cont=cont, disc=disc, weight_cont=0.875)
        assert_array_equal(rv_out.cont.x, [0, 1])
        assert_array_equal(rv_out.cont.y, [1, 1])
        assert_array_equal(rv_out.disc.x, [-1, 0.5])
        assert_array_equal(rv_out.disc.prob, [0.25, 0.75])
        assert rv_out.weight_cont == 0.875
        assert rv_out.weight_disc == 0.125

        # Checks for special input for continuous part
        ## `cont` can be `None` with `weight_cont` equal to 0
        rv_out = Mixt(cont=None, disc=disc, weight_cont=0)
        assert rv_out.cont is None

        # Checks for special input for discrete part
        ## `disc` can be `None` with `weight_cont` equal to 1
        rv_out = Mixt(cont=cont, disc=None, weight_cont=1)
        assert rv_out.disc is None

    def test_str(self):
        cont = Cont([0, 1], [1, 1])
        disc = Disc([-1, 0.5], [0.25, 0.75])

        rv = Mixt(cont=cont, disc=disc, weight_cont=0.75)

        out = (
            "Mixture RV:\n"
            f"Cont (weight = {rv.weight_cont}): {rv.cont}\n"
            f"Disc (weight = {rv.weight_disc}): {rv.disc}"
        )
        assert str(rv) == out

    def test_properties(self):
        """Tests for properties"""
        cont = Cont([0, 1], [1, 1])
        disc = Disc([-1, 0.5], [0.25, 0.75])
        weight_cont = 0.75
        rv = Mixt(cont=cont, disc=disc, weight_cont=weight_cont)

        assert isinstance(rv.cont, Cont)
        assert isinstance(rv.disc, Disc)
        assert_array_equal(rv.weight_disc, 1 - weight_cont)
        assert_array_equal(rv.weight_cont, weight_cont)

    def test_cdf(self):
        """Tests for `.cdf()` method"""
        cont = Cont([0, 1], [1, 1])
        disc = Disc([-1, 0.5], [0.25, 0.75])
        weight_cont = 0.75
        rv = Mixt(cont=cont, disc=disc, weight_cont=weight_cont)
        h = 1e-12

        # Regular checks
        x = np.array([-1.1, -1 - h, -1, 0, 0.25, 0.5 - h, 0.5, 0.75, 1, 1.1])
        assert_array_equal(
            rv.cdf(x), rv.weight_cont * rv.cont.cdf(x) + rv.weight_disc * rv.disc.cdf(x)
        )

        # Bad input
        x = np.array([-np.inf, np.nan, np.inf])
        assert_array_equal(rv.cdf(x), np.array([0, np.nan, 1]))

        # Broadcasting
        x = np.array([[-1, 0.5], [-1.1, 0.75]])
        assert_array_equal(
            rv.cdf(x),
            np.array(
                [
                    [0.75 * 0 + 0.25 * 0.25, 0.75 * 0.5 + 0.25 * 1],
                    [0.75 * 0 + 0.25 * 0, 0.75 * 0.75 + 0.25 * 1],
                ]
            ),
        )

        # Dirac-like continuous random variable
        cont_dirac = Cont([10 - 1e-8, 10, 10 + 1e-8], [0, 1, 0])
        rv_dirac = Mixt(cont=cont_dirac, disc=disc, weight_cont=0.75)
        x = np.array([10 - 1e-8, 10 - 0.5e-8, 10, 10 + 0.5e-8, 10 + 1e-8])
        assert_array_equal(
            rv_dirac.cdf(x),
            rv_dirac.weight_cont * rv_dirac.cont.cdf(x)
            + rv_dirac.weight_disc * rv_dirac.disc.cdf(x),
        )

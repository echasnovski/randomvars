# pylint: disable=missing-function-docstring
"""Tests for '_mixture.py' file"""
import numpy as np
from numpy.testing import assert_array_equal, assert_array_almost_equal
import scipy.stats.distributions as distrs
import pytest

from randomvars._continuous import Cont
from randomvars._discrete import Disc
from randomvars._mixture import Mixt
from randomvars.tests.commontests import (
    DECIMAL,
    h,
    _test_equal_seq,
    _test_input_coercion,
    _test_from_rv_rand,
    _test_log_fun,
    _test_one_value_input,
    _test_rvs_method,
)
import randomvars.options as op


def assert_equal_cont(rv_1, rv_2, decimal=None):
    xy1 = rv_1.x, rv_1.y
    xy2 = rv_2.x, rv_2.y
    _test_equal_seq(xy1, xy2, decimal=decimal)


def assert_equal_disc(rv_1, rv_2, decimal=None):
    xp1 = rv_1.x, rv_1.p
    xp2 = rv_2.x, rv_2.p
    _test_equal_seq(xp1, xp2, decimal=decimal)


def assert_equal_mixt(rv_1, rv_2, decimal=None):
    # Check weights
    assert rv_1.weight_cont == rv_2.weight_cont
    assert rv_1.weight_disc == rv_2.weight_disc

    # Check continuous parts
    if rv_1.cont is not None:
        if rv_2.cont is not None:
            assert_equal_cont(rv_1.cont, rv_2.cont, decimal)
        else:
            raise ValueError("`rv_2.cont` is `None` while `rv_1.cont` is not.")
    else:
        if rv_2.cont is not None:
            raise ValueError("`rv_1.cont` is `None` while `rv_2.cont` is not.")

    # Check discrete parts
    if rv_1.disc is not None:
        if rv_2.disc is not None:
            assert_equal_disc(rv_1.disc, rv_2.disc, decimal)
        else:
            raise ValueError("`rv_2.disc` is `None` while `rv_1.disc` is not.")
    else:
        if rv_2.disc is not None:
            raise ValueError("`rv_1.disc` is `None` while `rv_2.disc` is not.")


def assert_ppf(cont, disc, weight_cont):
    rv = Mixt(cont=cont, disc=disc, weight_cont=weight_cont)
    q = np.linspace(0, 1, 1001)[1:-1]
    x = np.linspace(rv.a, rv.b, 1001)[1:-1]

    # Check general inequalities for quantile function. Used form accounts for
    # floating point representation.
    ppf_q = rv.ppf(q)
    cdf_x = rv.cdf(x)

    ## `F(Q(q)-) <= q` (`F(Q(q)-)` - limit from left to `Q(q)` of `F`)
    assert np.all(rv.cdf(ppf_q - h) - q <= h)

    ## `q <= F(Q(q))`
    assert np.all(q - rv.cdf(ppf_q) <= h)

    ## `Q(F(x)) <= x`
    assert np.all(rv.ppf(cdf_x) - x <= h)

    ## `x <= Q(F(x)+)`
    assert np.all(x - rv.ppf(cdf_x + h) <= h)

    # Check values of cumulative probability "after jumps"
    assert_array_equal(rv.ppf(rv.cdf(disc.x)), disc.x)

    # Check values of cumulative probability "before" jumps
    ## Here ` + h` is added to ensure that input of `rv.ppf()` stays "inside jump".
    ## This accounts for possible case of two consecutive jumps inside
    ## zero-density interval of continuous part
    assert_array_equal(rv.ppf(rv.cdf(disc.x) - rv.weight_disc * disc.p + h), disc.x)

    # Check equality at cumulative probability intervals coming from continuous
    # part
    x_cont = np.setdiff1d(np.linspace(cont.a, cont.b, 1001), disc.x)[1:-1]
    assert_array_almost_equal(rv.ppf(rv.cdf(x_cont)), x_cont, decimal=DECIMAL)

    # Check extreme cumulative probabilities
    assert_array_equal(rv.ppf([0, 1]), list(rv.support()))


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
        with pytest.raises(TypeError, match="`cont`.*`Cont`.*`None`"):
            Mixt(cont=distrs.norm(), disc=disc, weight_cont=0.5)

        # Checking `disc`
        ## `disc` can be None but only with `weight_cont` equal to 1
        with pytest.raises(ValueError, match="`disc`.*`None`.*`weight_cont`"):
            Mixt(cont=cont, disc=None, weight_cont=0.5)

        ## `disc` can't be any other than `Disc` or `None`
        with pytest.raises(TypeError, match="`disc`.*`Disc`.*`None`"):
            Mixt(cont=cont, disc=distrs.bernoulli(p=0.1), weight_cont=0.5)

        # Both `cont` and `disc` being `None` should also throw error
        with pytest.raises(ValueError):
            Mixt(cont=None, disc=None, weight_cont=0.5)

        # Checking `weight_cont`
        with pytest.raises(TypeError, match="number"):
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
        assert_array_equal(rv_out.disc.p, [0.25, 0.75])
        assert rv_out.weight_cont == 0.875
        assert rv_out.weight_disc == 0.125

        # Degenerate cases
        ## `None` part. This is allowed when opposite weight is full.
        rv_none_cont = Mixt(cont=None, disc=disc, weight_cont=0)
        assert rv_none_cont.cont is None

        rv_none_disc = Mixt(cont=cont, disc=None, weight_cont=1)
        assert rv_none_disc.disc is None

        ## Extreme weight. Part with zero weight should still be present
        rv_weight_0 = Mixt(cont=cont, disc=disc, weight_cont=0)
        assert rv_weight_0.cont is cont

        rv_weight_1 = Mixt(cont=cont, disc=disc, weight_cont=1)
        assert rv_weight_1.disc is disc

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

        # Degenerate cases
        ## `None` part
        rv_none_cont = Mixt(cont=None, disc=disc, weight_cont=0)
        assert str(rv_none_cont).find("Cont (weight = 0.0): None") > -1

        rv_none_disc = Mixt(cont=cont, disc=None, weight_cont=1)
        assert str(rv_none_disc).find("Disc (weight = 0.0): None") > -1

        ## Extreme weight
        rv_weight_0 = Mixt(cont=cont, disc=disc, weight_cont=0)
        assert str(rv_weight_0).find(f"Cont (weight = 0.0): {cont}") > -1

        rv_weight_1 = Mixt(cont=cont, disc=disc, weight_cont=1)
        assert str(rv_weight_1).find(f"Disc (weight = 0.0): {disc}") > -1

    def test_properties(self):
        cont = Cont([0, 1], [1, 1])
        disc = Disc([-1, 0.5], [0.25, 0.75])
        weight_cont = 0.75
        rv = Mixt(cont=cont, disc=disc, weight_cont=weight_cont)

        assert list(rv.params.keys()) == ["cont", "disc", "weight_cont"]
        assert rv.params["cont"] == cont
        assert rv.params["disc"] == disc
        assert rv.params["weight_cont"] == weight_cont

        assert isinstance(rv.cont, Cont)
        assert isinstance(rv.disc, Disc)
        assert_array_equal(rv.weight_disc, 1 - weight_cont)
        assert_array_equal(rv.weight_cont, weight_cont)
        assert rv.a == -1.0
        assert rv.b == 1.0

        # Degenerate cases
        ## `None` part
        rv_none_cont = Mixt(cont=None, disc=disc, weight_cont=0)
        assert rv_none_cont.a == disc.a
        assert rv_none_cont.b == disc.b

        rv_none_disc = Mixt(cont=cont, disc=None, weight_cont=1)
        assert rv_none_disc.a == cont.a
        assert rv_none_disc.b == cont.b

        ## Extreme weight
        rv_weight_0 = Mixt(cont=cont, disc=disc, weight_cont=0)
        assert rv_weight_0.a == disc.a
        assert rv_weight_0.b == disc.b

        rv_weight_1 = Mixt(cont=cont, disc=disc, weight_cont=1)
        assert rv_weight_1.a == cont.a
        assert rv_weight_1.b == cont.b

    def test_support(self):
        cont = Cont([0, 1], [1, 1])
        disc = Disc([-1, 0.5], [0.25, 0.75])
        weight_cont = 0.75
        rv = Mixt(cont=cont, disc=disc, weight_cont=weight_cont)
        assert rv.support() == (-1.0, 1.0)

        # Degenerate cases
        ## `None` part
        rv_none_cont = Mixt(cont=None, disc=disc, weight_cont=0)
        assert rv_none_cont.support() == disc.support()

        rv_none_disc = Mixt(cont=cont, disc=None, weight_cont=1)
        assert rv_none_disc.support() == cont.support()

        ## Extreme weight
        rv_weight_0 = Mixt(cont=cont, disc=disc, weight_cont=0)
        assert rv_weight_0.support() == disc.support()

        rv_weight_1 = Mixt(cont=cont, disc=disc, weight_cont=1)
        assert rv_weight_1.support() == cont.support()

    def test_compress(self):
        cont = Cont(np.arange(7), [0, 0, 1, 2, 1, 0, 0])
        disc = Disc([0, 1, 2, 3], [0, 0.5, 0, 0.5])

        # Basic usage
        assert_equal_mixt(
            Mixt(cont, disc, 0.5).compress(),
            Mixt(cont.compress(), disc.compress(), 0.5),
        )

        # Degenerate cases
        assert_equal_cont(Mixt(cont, None, 1.0).compress(), cont.compress())
        assert_equal_cont(Mixt(cont, disc, 1.0).compress(), cont.compress())

        assert_equal_disc(Mixt(None, disc, 0.0).compress(), disc.compress())
        assert_equal_disc(Mixt(cont, disc, 0.0).compress(), disc.compress())

        # If nothing to compress, self should be returned
        rv = Mixt(cont.compress(), disc.compress(), 0.5)
        assert rv.compress() is rv

    def test_from_rv_basic(self):
        cont = Cont([0, 1], [1, 1])
        disc = Disc([-1, 0.5], [0.25, 0.75])

        cont_scipy = distrs.norm()
        disc_scipy = distrs.bernoulli(p=0.5)
        weight_cont_scipy = 0.75

        # Normal usage
        out = Mixt.from_rv((cont_scipy, disc_scipy), weight_cont_scipy)
        out_ref = Mixt(
            Cont.from_rv(cont_scipy), Disc.from_rv(disc_scipy), weight_cont_scipy
        )
        assert_equal_mixt(out, out_ref)

        # Objects of `Rand` class should be `convert()`ed
        _test_from_rv_rand(cls=Mixt, to_class="Mixt", assert_equal=assert_equal_mixt)

        # Degenerate cases
        ## Allow degenerate cases with tuple `rv`
        mixt_nonedisc_ref = Mixt(Cont.from_rv(cont_scipy), None, 1)
        mixt_nonecont_ref = Mixt(None, Disc.from_rv(disc_scipy), 0)

        ### `weight_cont` can be `None`
        assert_equal_mixt(Mixt.from_rv((cont_scipy, None)), mixt_nonedisc_ref)
        assert_equal_mixt(Mixt.from_rv((None, disc_scipy)), mixt_nonecont_ref)

        ### `weight_cont` can represent full weight of non-`None` part
        assert_equal_mixt(Mixt.from_rv((cont_scipy, None), 1), mixt_nonedisc_ref)
        assert_equal_mixt(Mixt.from_rv((None, disc_scipy), 0), mixt_nonecont_ref)

    def test_from_rv_errors(self):
        cont = Cont([0, 1], [1, 1])
        disc = Disc([-1, 0.5], [0.25, 0.75])

        cont_scipy = distrs.norm()
        disc_scipy = distrs.bernoulli(p=0.5)

        # `rv` should be tuple
        with pytest.raises(TypeError, match="`rv`.*tuple"):
            Mixt.from_rv(cont_scipy, weight_cont=0.5)
        with pytest.raises(TypeError, match="`rv`.*tuple"):
            Mixt.from_rv([cont_scipy, disc_scipy], weight_cont=0.5)

        # `rv` should have exactly two elements
        with pytest.raises(ValueError, match="`rv`.*two"):
            Mixt.from_rv((cont_scipy,), weight_cont=0.5)
        with pytest.raises(ValueError, match="`rv`.*two"):
            Mixt.from_rv((cont_scipy, disc_scipy, cont_scipy), weight_cont=0.5)

        # `rv` can't have both elements to be `None`
        with pytest.raises(ValueError, match="`rv`.*two `None`"):
            Mixt.from_rv((None, None), weight_cont=0.5)

        # `weight_cont` can't be `None` if both elements of tuple are not `None`
        with pytest.raises(ValueError, match="`weight_cont` can't be `None`"):
            Mixt.from_rv((cont_scipy, disc_scipy), weight_cont=None)

        # Errors for degenerate cases
        with pytest.raises(ValueError, match="`weight_cont`.*1"):
            Mixt.from_rv((cont_scipy, None), weight_cont=0.5)
        with pytest.raises(ValueError, match="`weight_cont`.*0"):
            Mixt.from_rv((None, disc_scipy), weight_cont=0.5)

    def test_from_sample_basic(self):
        # Normal usage
        sample = ([0, 0.25, 0.5, 0.75, 1], [0, 1, 1, 1])
        weight_cont = 0.75
        rv = Mixt.from_sample(sample=sample, weight_cont=weight_cont)
        rv_ref = Mixt(
            Cont.from_sample(sample[0]), Disc.from_sample(sample[1]), weight_cont
        )
        assert_equal_mixt(rv, rv_ref)

        # Degenerate cases
        rv_none_cont = Mixt.from_sample((None, sample[1]), weight_cont=0)
        rv_none_cont_ref = Mixt(None, Disc.from_sample(sample[1]), 0)
        assert_equal_mixt(rv_none_cont, rv_none_cont_ref)

        rv_none_disc = Mixt.from_sample((sample[0], None), weight_cont=1)
        rv_none_disc_ref = Mixt(Cont.from_sample(sample[0]), None, 1)
        assert_equal_mixt(rv_none_disc, rv_none_disc_ref)

    def test_from_sample_errors(self):
        # `sample` should be tuple
        with pytest.raises(TypeError, match="`sample`.*tuple"):
            Mixt.from_sample([[0, 1, 2], [0, 1, 1, 1]], weight_cont=0.5)

        # `sample` should have exactly two elements
        with pytest.raises(ValueError, match="`sample`.*two"):
            Mixt.from_sample(([0, 1, 2],), weight_cont=0.5)
        with pytest.raises(ValueError, match="`sample`.*two"):
            Mixt.from_sample(([0, 1, 2], [3, 4, 5], [6, 7]), weight_cont=0.5)

        # `sample` can't have both elements to be `None`
        with pytest.raises(ValueError, match="`sample`.*two `None`"):
            Mixt.from_sample((None, None), weight_cont=0.5)

        # Errors for degenerate cases
        with pytest.raises(ValueError, match="`weight_cont`.*1"):
            Mixt.from_sample(([0, 1], None), weight_cont=0.5)
        with pytest.raises(ValueError, match="`weight_cont`.*0"):
            Mixt.from_sample((None, [0, 1]), weight_cont=0.5)

    def test_pdf(self):
        cont = Cont([0, 1], [1, 1])
        disc = Disc([-1, 0.5], [0.25, 0.75])
        weight_cont = 0.75
        rv = Mixt(cont=cont, disc=disc, weight_cont=weight_cont)

        with pytest.raises(AttributeError, match="doesn't have.*density"):
            rv.pdf(0)

    def test_logpdf(self):
        cont = Cont([0, 1], [1, 1])
        disc = Disc([-1, 0.5], [0.25, 0.75])
        weight_cont = 0.75
        rv = Mixt(cont=cont, disc=disc, weight_cont=weight_cont)

        with pytest.raises(AttributeError, match="doesn't have.*density"):
            rv.logpdf(0)

    def test_pmf(self):
        cont = Cont([0, 1], [1, 1])
        disc = Disc([-1, 0.5], [0.25, 0.75])
        weight_cont = 0.75
        rv = Mixt(cont=cont, disc=disc, weight_cont=weight_cont)

        with pytest.raises(AttributeError, match="doesn't have.*mass"):
            rv.pmf(0)

    def test_logpmf(self):
        cont = Cont([0, 1], [1, 1])
        disc = Disc([-1, 0.5], [0.25, 0.75])
        weight_cont = 0.75
        rv = Mixt(cont=cont, disc=disc, weight_cont=weight_cont)

        with pytest.raises(AttributeError, match="doesn't have.*mass"):
            rv.logpmf(0)

    def test_cdf(self):
        cont = Cont([0, 1], [1, 1])
        disc = Disc([-1, 0.5], [0.25, 0.75])
        weight_cont = 0.75
        rv = Mixt(cont=cont, disc=disc, weight_cont=weight_cont)
        ref_x = np.array([-1.1, -1 - h, -1, 0, 0.25, 0.5 - h, 0.5, 0.75, 1, 1.1])

        # Regular checks
        assert_array_equal(
            rv.cdf(ref_x),
            rv.weight_cont * rv.cont.cdf(ref_x) + rv.weight_disc * rv.disc.cdf(ref_x),
        )

        # Coercion of not ndarray input
        _test_input_coercion(rv.cdf, ref_x)

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

        # One value input
        _test_one_value_input(rv.cdf, 0.5)
        _test_one_value_input(rv.cdf, -1)
        _test_one_value_input(rv.cdf, np.nan)

        # Dirac-like continuous random variable
        cont_dirac = Cont([10 - h, 10, 10 + h], [0, 1, 0])
        rv_dirac = Mixt(cont=cont_dirac, disc=disc, weight_cont=0.75)
        x = np.array([10 - h, 10 - 0.5e-8, 10, 10 + 0.5e-8, 10 + h])
        assert_array_equal(
            rv_dirac.cdf(x),
            rv_dirac.weight_cont * rv_dirac.cont.cdf(x)
            + rv_dirac.weight_disc * rv_dirac.disc.cdf(x),
        )

        # Degenerate cases
        ## `None` part
        rv_none_cont = Mixt(cont=None, disc=disc, weight_cont=0)
        assert_array_equal(rv_none_cont.cdf(ref_x), disc.cdf(ref_x))

        rv_none_disc = Mixt(cont=cont, disc=None, weight_cont=1)
        assert_array_equal(rv_none_disc.cdf(ref_x), cont.cdf(ref_x))

        ## Extreme weight
        rv_weight_0 = Mixt(cont=cont, disc=disc, weight_cont=0)
        assert_array_equal(rv_weight_0.cdf(ref_x), disc.cdf(ref_x))

        rv_weight_1 = Mixt(cont=cont, disc=disc, weight_cont=1)
        assert_array_equal(rv_weight_1.cdf(ref_x), cont.cdf(ref_x))

    def test_logcdf(self):
        cont = Cont([0, 1], [1, 1])
        disc = Disc([-1, 0.5], [0.25, 0.75])
        weight_cont = 0.75
        rv = Mixt(cont=cont, disc=disc, weight_cont=weight_cont)

        _test_log_fun(
            rv.logcdf,
            rv.cdf,
            x_ref=np.array([-1.1, -1 - h, -1, 0, 0.25, 0.5 - h, 0.5, 0.75, 1, 1.1]),
        )

    def test_sf(self):
        cont = Cont([0, 1], [1, 1])
        disc = Disc([-1, 0.5], [0.25, 0.75])
        weight_cont = 0.75
        rv = Mixt(cont=cont, disc=disc, weight_cont=weight_cont)
        x_ref = np.array([-1.1, -1 - h, -1, 0, 0.25, 0.5 - h, 0.5, 0.75, 1, 1.1])

        assert_array_equal(rv.sf(x_ref), 1 - rv.cdf(x_ref))

    def test_logsf(self):
        cont = Cont([0, 1], [1, 1])
        disc = Disc([-1, 0.5], [0.25, 0.75])
        weight_cont = 0.75
        rv = Mixt(cont=cont, disc=disc, weight_cont=weight_cont)

        _test_log_fun(
            rv.logsf,
            rv.sf,
            x_ref=np.array([-1.1, -1 - h, -1, 0, 0.25, 0.5 - h, 0.5, 0.75, 1, 1.1]),
        )

    def test_ppf(self):
        # `ppf()` method should be inverse to `cdf()` for every sensible input
        cont = Cont([0, 1], [1, 1])
        disc = Disc([-1, 0.5], [0.25, 0.75])
        weight_cont = 0.75
        rv = Mixt(cont=cont, disc=disc, weight_cont=weight_cont)
        ref_x = np.array([-1.1, -1 - h, -1, 0, 0.25, 0.5 - h, 0.5, 0.75, 1, 1.1])
        ref_q = rv.cdf(ref_x)
        prob = [0.25, 0.75]

        # Regular checks
        # Due to nature of quantile function it is safer to check every
        # combination of overlapping between supports of continuous and
        # discrete parts. Probably an overkill, but this on a safer side.
        # Here: `a_cont` and `b_cont` - edges of continuous part, `a_disc` and
        # `b_disc` - edges of discrete part
        ## a_disc < b_disc < a_cont < b_cont
        assert_ppf(cont, Disc([-1, -0.5], prob), weight_cont)
        ## a_disc < b_disc = a_cont < b_cont
        assert_ppf(cont, Disc([-1, 0], prob), weight_cont)
        ## a_disc < a_cont < b_disc < b_cont
        assert_ppf(cont, Disc([-1, 0.5], prob), weight_cont)
        ## a_disc < a_cont < b_disc = b_cont
        assert_ppf(cont, Disc([-1, 1], prob), weight_cont)
        ## a_disc < a_cont < b_cont < b_disc
        assert_ppf(cont, Disc([-1, 1.5], prob), weight_cont)
        ## a_disc = a_cont < b_disc < b_cont
        assert_ppf(cont, Disc([0, 0.5], prob), weight_cont)
        ## a_disc = a_cont < b_disc = b_cont
        assert_ppf(cont, Disc([0, 1], prob), weight_cont)
        ## a_disc = a_cont < b_cont < b_disc
        assert_ppf(cont, Disc([0, 1.5], prob), weight_cont)
        ## a_cont < a_disc < b_disc < b_cont
        assert_ppf(cont, Disc([0.25, 0.5], prob), weight_cont)
        ## a_cont < a_disc < b_disc = b_cont
        assert_ppf(cont, Disc([0.5, 1], prob), weight_cont)
        ## a_cont < a_disc < b_cont < b_disc
        assert_ppf(cont, Disc([0.5, 1.5], prob), weight_cont)
        ## a_cont < a_disc = b_cont < b_disc
        assert_ppf(cont, Disc([1, 1.5], prob), weight_cont)
        ## a_cont < b_cont < a_disc < b_disc
        assert_ppf(cont, Disc([1.5, 2], prob), weight_cont)

        ## Checks with one value in discrete part
        assert_ppf(cont, Disc([-1], [1]), weight_cont)
        assert_ppf(cont, Disc([0.5], [1]), weight_cont)
        assert_ppf(cont, Disc([1.5], [1]), weight_cont)

        # Coercion of not ndarray input
        _test_input_coercion(rv.ppf, ref_q)

        # Bad input
        q = np.array([-np.inf, -h, np.nan, 1 + h, np.inf])
        assert_array_equal(
            rv.ppf(q), np.array([np.nan, np.nan, np.nan, np.nan, np.nan])
        )

        # Broadcasting
        q = np.array([[0, 0.5], [0.0, 1.0]])
        assert_array_equal(rv.ppf(q), np.array([[-1, 0.5], [-1, 1]]))

        # One value input
        _test_one_value_input(rv.ppf, 0.25)
        _test_one_value_input(rv.ppf, -1)
        _test_one_value_input(rv.ppf, np.nan)

        # Should return the smallest x-value in case of zero-density interval
        # in continuous part
        cont_zero_density = Cont([0, 1, 2, 3, 4, 5], [0, 0.5, 0, 0, 0.5, 0])
        rv_dens_zero = Mixt(cont_zero_density, Disc([2.5], [1]), 0.5)
        assert rv_dens_zero.ppf(0.25) == 2

        # Degenerate cases
        ## `None` part
        rv_none_cont = Mixt(cont=None, disc=disc, weight_cont=0)
        assert_array_equal(rv_none_cont.ppf(ref_q), disc.ppf(ref_q))

        rv_none_disc = Mixt(cont=cont, disc=None, weight_cont=1)
        assert_array_equal(rv_none_disc.ppf(ref_q), cont.ppf(ref_q))

        ## Extreme weight
        rv_weight_0 = Mixt(cont=cont, disc=disc, weight_cont=0)
        assert_array_equal(rv_weight_0.ppf(ref_q), disc.ppf(ref_q))

        rv_weight_1 = Mixt(cont=cont, disc=disc, weight_cont=1)
        assert_array_equal(rv_weight_1.ppf(ref_q), cont.ppf(ref_q))

    def test_isf(self):
        cont = Cont([0, 1], [1, 1])
        disc = Disc([-1, 0.5], [0.25, 0.75])
        weight_cont = 0.75
        rv = Mixt(cont=cont, disc=disc, weight_cont=weight_cont)
        x_ref = np.array([-1.1, -1 - h, -1, 0, 0.25, 0.5 - h, 0.5, 0.75, 1, 1.1])
        q_ref = rv.cdf(x_ref)
        assert_array_equal(rv.isf(q_ref), rv.ppf(1 - q_ref))

    def test_rvs(self):
        cont = Cont([0, 1], [1, 1])
        disc = Disc([-1, 0.5], [0.25, 0.75])
        weight_cont = 0.75
        rv = Mixt(cont=cont, disc=disc, weight_cont=weight_cont)

        _test_rvs_method(rv)

    def test_integrate_cdf(self):
        cont = Cont([0, 1], [1, 1])
        disc = Disc([-1, 0.5], [0.25, 0.75])
        weight_cont = 0.75
        rv = Mixt(cont=cont, disc=disc, weight_cont=weight_cont)
        a = -10
        b = 10

        # Regular checks
        assert_array_equal(
            rv.integrate_cdf(a, b),
            rv.weight_cont * rv.cont.integrate_cdf(a, b)
            + rv.weight_disc * rv.disc.integrate_cdf(a, b),
        )

        # Degenerate cases
        ## `None` part
        rv_none_cont = Mixt(cont=None, disc=disc, weight_cont=0)
        assert_array_equal(rv_none_cont.integrate_cdf(a, b), disc.integrate_cdf(a, b))

        rv_none_disc = Mixt(cont=cont, disc=None, weight_cont=1)
        assert_array_equal(rv_none_disc.integrate_cdf(a, b), cont.integrate_cdf(a, b))

        ## Extreme weight
        rv_weight_0 = Mixt(cont=cont, disc=disc, weight_cont=0)
        assert_array_equal(rv_weight_0.integrate_cdf(a, b), disc.integrate_cdf(a, b))

        rv_weight_1 = Mixt(cont=cont, disc=disc, weight_cont=1)
        assert_array_equal(rv_weight_1.integrate_cdf(a, b), cont.integrate_cdf(a, b))

    def test_convert(self):
        import randomvars._boolean as bool

        w = op.get_option("small_width")

        cont = Cont([0, 1], [1, 1])
        disc = Disc([-1, 0.5], [0.25, 0.75])
        weight_cont = 0.75
        rv = Mixt(cont=cont, disc=disc, weight_cont=weight_cont)

        x_ref = np.linspace(rv.a, rv.b, 10001)

        # By default and supplying `None` should return self
        assert rv.convert() is rv
        assert rv.convert(None) is rv

        # Converting to Bool should result into consecutive conversion to Disc
        # and Bool
        out_bool = rv.convert("Bool")
        assert isinstance(out_bool, bool.Bool)
        assert out_bool.prob_true == rv.convert("Disc").convert("Bool").prob_true

        # Converting to Cont should result into mixture of continuous part and
        # discrete part converted to continuous
        out_cont = rv.convert("Cont")
        assert isinstance(out_cont, Cont)

        ## Remove reference points that are close to input x-grids where slight
        ## difference is expected due to "grounding" of xy-grids
        dist_to_set = lambda x, x_set: np.min(np.abs(x - x_set.reshape(-1, 1)), axis=0)
        x_ref2 = x_ref[
            (dist_to_set(x_ref, cont.x) > w) & (dist_to_set(x_ref, disc.x) > w)
        ]
        assert_array_almost_equal(
            out_cont.cdf(x_ref2),
            rv.weight_cont * rv.cont.cdf(x_ref2)
            + rv.weight_disc * rv.disc.convert("Cont").cdf(x_ref2),
            decimal=DECIMAL,
        )

        # Converting to Disc should result into mixture of continuous part
        # converted to discrete and discrete part
        out_disc = rv.convert("Disc")
        assert isinstance(out_disc, Disc)
        assert_array_equal(
            out_disc.cdf(x_ref),
            rv.weight_cont * rv.cont.convert("Disc").cdf(x_ref)
            + rv.weight_disc * rv.disc.cdf(x_ref),
        )

        # Converting to own class should return self
        out_mixt = rv.convert("Mixt")
        assert out_mixt is rv

        # Any other target class should result into error
        with pytest.raises(ValueError, match="one of"):
            rv.convert("aaa")

    def test_convert_degenerate(self):
        cont = Cont([0, 1], [1, 1])
        cont_todisc = cont.convert("Disc")
        disc = Disc([-1, 0.5], [0.25, 0.75])
        disc_tocont = disc.convert("Cont")

        # `None` part
        rv_nocont = Mixt(cont=None, disc=disc, weight_cont=0)
        rv_nocont_tocont = rv_nocont.convert("Cont")
        _test_equal_seq(
            (rv_nocont_tocont.x, rv_nocont_tocont.y), (disc_tocont.x, disc_tocont.y)
        )
        rv_nocont_todisc = rv_nocont.convert("Disc")
        _test_equal_seq((rv_nocont_todisc.x, rv_nocont_todisc.p), (disc.x, disc.p))

        rv_nodisc = Mixt(cont=cont, disc=None, weight_cont=1.0)
        rv_nodisc_tocont = rv_nodisc.convert("Cont")
        _test_equal_seq((rv_nodisc_tocont.x, rv_nodisc_tocont.y), (cont.x, cont.y))
        rv_nodisc_todisc = rv_nodisc.convert("Disc")
        _test_equal_seq(
            (rv_nodisc_todisc.x, rv_nodisc_todisc.p), (cont_todisc.x, cont_todisc.p)
        )

        # Extreme weight
        rv_weight0 = Mixt(cont=cont, disc=disc, weight_cont=0)
        rv_weight0_tocont = rv_weight0.convert("Cont")
        _test_equal_seq(
            (rv_weight0_tocont.x, rv_weight0_tocont.y), (disc_tocont.x, disc_tocont.y)
        )
        rv_weight0_todisc = rv_weight0.convert("Disc")
        _test_equal_seq((rv_weight0_todisc.x, rv_weight0_todisc.p), (disc.x, disc.p))

        rv_weight1 = Mixt(cont=cont, disc=disc, weight_cont=1.0)
        rv_weight1_tocont = rv_weight1.convert("Cont")
        _test_equal_seq((rv_weight1_tocont.x, rv_weight1_tocont.y), (cont.x, cont.y))
        rv_weight1_todisc = rv_weight1.convert("Disc")
        _test_equal_seq(
            (rv_weight1_todisc.x, rv_weight1_todisc.p), (cont_todisc.x, cont_todisc.p)
        )

    def test_convert_options(self):
        cont = Cont([0, 1], [1, 1])
        disc = Disc([-1, 0.5], [0.25, 0.75])
        weight_cont = 0.75
        rv = Mixt(cont=cont, disc=disc, weight_cont=weight_cont)

        # Grounding is done respecting `small_width` package option
        with op.option_context({"small_width": 0.1}):
            w = op.get_option("small_width")
            rv_tocont = rv.convert("Cont")
            assert_array_equal(rv_tocont.x, [-1, -w, 0, w, 0.5 - w, 0.5, 0.5 + w, 1])

        # No grounding is done if edge is close to output edge
        with op.option_context({"base_tolerance": 0.1}):
            tol = op.get_option("base_tolerance")
            rv = Mixt(
                cont=Cont([0, 1], [1, 1]),
                disc=Disc([0.5 * tol, 1 - 0.5 * tol], [0.5, 0.5]),
                weight_cont=weight_cont,
            )
            rv_tocont = rv.convert("Cont")
            assert_array_equal(rv_tocont.x, [0, 0.5 * tol, 1 - 0.5 * tol, 1])

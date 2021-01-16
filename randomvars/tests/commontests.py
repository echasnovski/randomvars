import numpy as np
from numpy.testing import assert_array_equal, assert_array_almost_equal

import pytest

import randomvars.options as op


base_tol = op.get_option("base_tolerance")
# Value that is used to test almost exact equality of two numpy arrays
DECIMAL = np.ceil(-np.log10(base_tol)).astype("int64")

# Difference of x-values that is considered "small" during approximations
h = op.get_option("small_width")


def _test_equal_seq(first, second, decimal=None):
    assert len(first) == len(second)
    for el1, el2 in zip(first, second):
        if decimal is None:
            assert_array_equal(el1, el2)
        else:
            assert_array_almost_equal(el1, el2, decimal=decimal)


def _test_input_coercion(func, arr):
    out = func(list(arr))
    out_ref = func(arr)
    assert_array_equal(out, out_ref)
    assert type(out) == type(out_ref)


def _test_from_rv_rand(cls, to_class, assert_equal):
    import randomvars._boolean as bool
    import randomvars._continuous as cont
    import randomvars._discrete as disc
    import randomvars._mixture as mixt

    rv_bool = bool.Bool(0.5)
    assert_equal(cls.from_rv(rv_bool), rv_bool.convert(to_class))
    rv_cont = cont.Cont([0, 1], [1, 1])
    assert_equal(cls.from_rv(rv_cont), rv_cont.convert(to_class))
    rv_disc = disc.Disc([-1, 2], [0.25, 0.75])
    assert_equal(cls.from_rv(rv_disc), rv_disc.convert(to_class))
    rv_mixt = mixt.Mixt(rv_cont, rv_disc, 0.5)
    assert_equal(cls.from_rv(rv_mixt), rv_mixt.convert(to_class))


def _test_log_fun(logfun, fun, x_ref):
    with np.errstate(divide="ignore"):
        logval_ref = np.log(fun(x_ref))

    # No warnings should be thrown
    with pytest.warns(None):
        assert_array_equal(logfun(x_ref), logval_ref)


def _test_one_value_input(func, value):
    # Scalar
    out = func(value)
    assert out.ndim == 0
    assert type(out) == np.ndarray

    # One-value array
    out = func([value])
    assert out.ndim == 1
    assert type(out) == np.ndarray


def _test_rvs_method(obj):
    # Regular checks
    smpl = obj.rvs(size=10)
    assert smpl.shape == (10,)
    assert type(smpl) == np.ndarray
    assert np.all((obj.a <= smpl) & (smpl <= obj.b))

    # Treats default `size` as 1
    assert obj.rvs().shape == tuple()

    # Accepts `size` to be empty tuple
    assert obj.rvs(size=()).shape == tuple()

    # Supplying number results into array with positive shape
    assert obj.rvs(size=1).shape == (1,)

    # Broadcasting
    smpl_array = obj.rvs(size=(10, 2))
    assert smpl_array.shape == (10, 2)

    # Usage of `random_state`
    smpl_1 = obj.rvs(size=100, random_state=np.random.RandomState(101))
    smpl_2 = obj.rvs(size=100, random_state=np.random.RandomState(101))
    assert_array_equal(smpl_1, smpl_2)

    # Usage of integer `random_state` as a seed
    smpl_1 = obj.rvs(size=100, random_state=101)
    smpl_2 = obj.rvs(size=100, random_state=101)
    assert_array_equal(smpl_1, smpl_2)

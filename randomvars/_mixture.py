""" Code for mixture random variable
"""
import warnings

import numpy as np

from randomvars._continuous import Cont
from randomvars._discrete import Disc
from randomvars._random import Rand
import randomvars._utils as utils


class Mixt(Rand):
    """Mixture random variable

    Class for mixture random variable which has a mixture distribution of
    continuous (of class `Cont`) and discrete (of class `Dict`) random
    variables with some predefined weights. Basically, this might be treated as
    a random variable which generates one sample value in two steps:

    - First, select a part from which sample will be drawn. Continuous part
      might be selected with `weight_cont` probability, discrete - with
      `weight_disc` (equal to `1 - weight_cont`) probability.
    - Second, sample a value from selected distribution.

    There are three ways to create instance of `Mixt` class:

    1. Directly supply continuous (`cont`) and discrete (`disc`) random
    variables along with weight of **continuous** part (`weight_cont`):
    ```
        from randomvars import Cont, Disc
        my_cont = Cont(x=[0, 1], y=[1, 1])
        my_disc = Disc(x=[0, 1], prob=[0.25, 0.75])

        # This is an equal weighted mixture of continuous uniform and bernoulli
        # random variables
        my_mixt = Mixt(cont=my_cont, disc=my_disc, weight_cont=0.5)
        print(my_mixt)

        # `Mixt` can hold only one part when it has full weight and the other
        # part is `None`.
        my_mixt_2 = Mixt(cont=my_cont, disc=None, weight_cont=1.0)
        print(my_mixt_2)
    ```
    2. Use `Mixt.from_rv()`:
    ```

    ```
    3. Use `Mixt.from_sample()` to create estimation based on existing two
    samples (one for continuous part and another for discrete) and weight of
    continuous part:
    ```
        # Two samples should be supplied in tuple
        my_mixt = Mixt.from_sample(sample=([0, 0.5, 1], [0, 1]), weight_cont=0.5)
        print(my_mixt)

        # One of samples can be `None` but only if other part has full weight
        my_mixt_2 = Mixt.from_sample(([0, 0.5, 1], None), weight_cont=1.0)

    ```
    """

    def __init__(self, cont, disc, weight_cont):
        # User-facing attributes
        self._cont, self._disc, self._weight_cont = self._impute_init_args(
            cont, disc, weight_cont
        )
        self._weight_disc = 1.0 - self._weight_cont

        if self._missing_cont():
            self._a, self._b = self._disc.a, self._disc.b
        elif self._missing_disc():
            self._a, self._b = self._cont.a, self._cont.b
        else:
            self._a = min(self._cont.a, self._disc.a)
            self._b = max(self._cont.b, self._disc.b)

        # Private attributes
        self._cum_p_tuple = self._compute_cum_p_tuple()

        super().__init__()

    @staticmethod
    def _impute_init_args(cont, disc, weight_cont):
        # Impute `weight_cont`
        try:
            weight_cont = float(weight_cont)
        except ValueError:
            raise ValueError("`weight_cont` should be a number.")

        if (weight_cont < 0) or (weight_cont > 1):
            raise ValueError("`weight_cont` should be between 0 and 1 (inclusively).")

        # Impute `cont`
        if cont is None:
            if weight_cont > 0:
                raise ValueError("`cont` can't be `None` if `weight_cont` is above 0.")
        elif not isinstance(cont, Cont):
            raise ValueError("`cont` should be object of `Cont` or `None`.")

        # Impute `disc`
        if disc is None:
            if weight_cont < 1:
                raise ValueError("`disc` can't be `None` if `weight_cont` is below 1.")
        elif not isinstance(disc, Disc):
            raise ValueError("`disc` should be object of `Disc` or `None`.")

        return cont, disc, weight_cont

    def _compute_cum_p_tuple(self):
        """Compute tuple defining cumulative probability grid

        Main purpose is to be used inside quantile (`.ppf()`) function. Its
        output defines intervals on "cumulative probability line" inside which
        quantile function of mixture has single nature: continuous or discrete.

        Returns
        -------
        cum_p_tuple : Tuple with three numpy arrays
            Elements are:
            - Cum_p-grid (cumulative probability grid), that starts at 0 and
              ends at 1.
            - X-grid which is values corresponding to both cum_p-grid and
              interval type.
            - Identifiers of intervals' nature. Has length less by one than
              previous two. Its value at position `i` is "c" if interval
              defined by `i`th and `i+1`th elements of previous grids has
              continuous nature. Value "d" - if discrete. Its values are
              interchanging:
                - First value depends on if there is a non-zero tail of
                  continuous part to the left of first value of discrete part.
                - "Inner" values are different to the previous ones.
                - Last value depends on if there is a non-zero tail of
                  continuous part to the right of last value of discrete part.
            Note:
            - Both cum_p-grid and x-grid can have consecutive duplicating elements:
                - Cum_p-grid - in case there are values of discrete part inside
                  zero density region of continuous part. Those define
                  "interval" of continuous nature, which in fact isn't used.
                - X-grid - for every value of discrete part in case there is
                  continuous part. First one corresponds to the left limit of
                  mixture CDF at that point ("before jump"), second - to the
                  value of mixture CDF at that point ("after jump")
        """
        # Case of one discrete part
        if self._missing_cont():
            disc_x = self._disc.x
            cum_p = np.concatenate([[0], self._disc.cdf(disc_x)])
            x = np.concatenate([[self._disc.a], disc_x])
            ids = np.repeat("d", len(x) - 1)
            return cum_p, x, ids

        # Case of one continuous part
        if self._missing_disc():
            cont_x = self._cont.x
            cum_p = self._cont.cdf(cont_x)
            x = cont_x
            ids = np.repeat("c", len(x) - 1)
            return cum_p, x, ids

        # Compute "inner" x-grid
        disc_x = self._disc.x
        x = np.repeat(disc_x, 2)

        # Compute "inner" cum_p-grid
        ## Values of mixture cdf
        cum_p_right = self.cdf(disc_x)
        ## Values of left limit of mixture cdf
        cum_p_left = cum_p_right - self._weight_disc * self._disc.p
        ## Interchange elements of `cum_p_left` and `cum_p_right`
        cum_p = np.array([cum_p_left, cum_p_right], order="C").flatten(order="F")

        # Compute "inner" interval identifiers
        ids = np.tile(["d", "c"], len(disc_x))[:-1]

        # Add possibly missing intervals because of continuous part tails
        if cum_p[0] > 0:
            cum_p = np.concatenate([[0], cum_p])
            x = np.concatenate([[self._cont.a], x])
            ids = np.concatenate([["c"], ids])
        if cum_p[-1] < 1:
            cum_p = np.concatenate([cum_p, [1]])
            x = np.concatenate([x, [self._cont.b]])
            ids = np.concatenate([ids, ["c"]])

        return cum_p, x, ids

    def __str__(self):
        return (
            "Mixture RV:\n"
            f"Cont (weight = {self._weight_cont}): {self._cont}\n"
            f"Disc (weight = {self._weight_disc}): {self._disc}"
        )

    @property
    def cont(self):
        """Return continuous part of mixture"""
        return self._cont

    @property
    def disc(self):
        """Return discrete part of mixture"""
        return self._disc

    @property
    def weight_cont(self):
        """Return weight of continuous part in mixture"""
        return self._weight_cont

    @property
    def weight_disc(self):
        """Return weight of discrete part in mixture"""
        return self._weight_disc

    @property
    def a(self):
        """Return left edge of support"""
        return self._a

    @property
    def b(self):
        """Return right edge of support"""
        return self._b

    # `support()` is inherited from `Rand`

    @classmethod
    def from_rv(cls, rv, weight_cont=None):
        """Create mixture RV from two general Rvs

        This is mostly a wrapper for `Mixt(cont=Cont.from_rv(rv[0]),
        disc=Disc.from_rv(rv[1]), weight_cont=weight_cont)`. It has special
        treatment for degenerate cases (when one part of mixture is `None`):
        - If input `rv` is object of `Cont` or `Disc` it is transformed into
          tuple with other part being `None`.
        - If one of `rv` element is `None` and other part has full weight (or
          weight is `None`), mixture random variable with only one part is
          created.

        **Note** that if `rv` is already an object of class `Mixt`, it is
        returned untouched.

        Parameters
        ----------
        rv : tuple with two elements or object of `Cont`, `Disc`, or `Mixt`
            First element should be a valid input for `Cont.from_rv()` or
            `None`. Second - for `Disc.from_rv()` or `None`. Objects of `Cont`
            and `Disc` are transformed into tuples with other part being
            `None`. If object of `Mixt`, returned untouched.
        weight_cont : number or `None` (default)
            Weight of continuous part. Can be `None` if one of input tuple's
            element is `None`.

        Returns
        -------
        rv_out : Mixt
            Mixture random variable with parts created from input random
            variables.
        """
        if isinstance(rv, Mixt):
            # `Mixt` should be returned untouched
            return rv
        elif isinstance(rv, Cont):
            # Allow `Mixt.from_rv(rv)` if `rv` is object of `Cont`
            rv = (rv, None)
        elif isinstance(rv, Disc):
            # Allow `Mixt.from_rv(rv)` if `rv` is object of `Disc`
            rv = (None, rv)

        _assert_two_tuple(rv, "rv")
        if (weight_cont is None) and (rv[0] is not None) and (rv[1] is not None):
            raise ValueError(
                "`weight_cont` can't be `None` if both elements of `rv` are not `None`."
            )

        if rv[0] is None:
            if weight_cont is None:
                # Allow `Mixt.from_rv((None, disc))`
                weight_cont = 0.0
            return cls(cont=None, disc=Disc.from_rv(rv[1]), weight_cont=weight_cont)
        if rv[1] is None:
            if weight_cont is None:
                # Allow `Mixt.from_rv((cont, None))`
                weight_cont = 1.0
            return cls(cont=Cont.from_rv(rv[0]), disc=None, weight_cont=weight_cont)

        return cls(
            cont=Cont.from_rv(rv[0]),
            disc=Disc.from_rv(rv[1]),
            weight_cont=weight_cont,
        )

    @classmethod
    def from_sample(cls, sample, weight_cont):
        """Create mixture RV from two samples

        This is mostly a wrapper for for
        `Mixt(cont=Cont.from_sample(sample[0]),
        disc=Disc.from_sample(sample[1]), weight_cont=weight_cont)`. If one of
        samples is `None` and other part has full weight, mixture random
        variable with only one part is created.

        Parameters
        ----------
        sample : tuple with two elements
            First element should be a valid input for `Cont.from_sample()` or
            `None`. Second - for `Disc.from_sample()` or `None`.
        weight_cont : number
            Weight of continuous part.

        Returns
        -------
        rv_out : Mixt
            Mixture random variable with parts created from sample.
        """
        _assert_two_tuple(sample, "sample")

        if sample[0] is None:
            return cls(
                cont=None, disc=Disc.from_sample(sample[1]), weight_cont=weight_cont
            )
        if sample[1] is None:
            return cls(
                cont=Cont.from_sample(sample[0]), disc=None, weight_cont=weight_cont
            )

        return cls(
            cont=Cont.from_sample(sample[0]),
            disc=Disc.from_sample(sample[1]),
            weight_cont=weight_cont,
        )

    def _missing_cont(self):
        return (self._cont is None) or (self._weight_cont == 0)

    def _missing_disc(self):
        return (self._disc is None) or (self._weight_disc == 0)

    def pdf(self, x):
        raise AttributeError("`Mixt` doesn't have probability density function.")

    def pmf(self, x):
        raise AttributeError("`Mixt` doesn't have probability mass function.")

    def cdf(self, x):
        """Cumulative distribution function

        Return values of cumulative distribution function at points `x`.

        Parameters
        ----------
        x : array_like with numeric values

        Returns
        -------
        cdf_vals : ndarray with shape inferred from `x`
        """
        x = np.asarray(x, dtype="float64")

        if self._missing_cont():
            return self._disc.cdf(x)
        if self._missing_disc():
            return self._cont.cdf(x)

        res = self._weight_cont * self._cont.cdf(
            x
        ) + self._weight_disc * self._disc.cdf(x)

        # Using `np.asarray()` to ensure ndarray output in case of `x`
        # originally was scalar
        return np.asarray(res, dtype="float64")

    def ppf(self, q):
        """Percent point (quantile, inverse of cdf) function

        Return values of percent point (quantile, inverse of cdf) function at
        cumulative probabilities `q`.

        Parameters
        ----------
        q : array_like with numeric values

        Returns
        -------
        ppf_vals : ndarray with shape inferred from `q`
        """
        # Make early return in degenerate cases
        if self._missing_cont():
            return self._disc.ppf(q)
        if self._missing_disc():
            return self._cont.ppf(q)

        q = np.asarray(q, dtype="float64")
        res = np.zeros_like(q, dtype="float64")
        cum_p, x, ids = self._cum_p_tuple

        # Locate intervals for `q` elements
        q_ind = utils._searchsorted_wrap(cum_p, q, side="left", edge_inside=True) - 1
        ind_is_good = (q_ind >= 0) & (q_ind < len(cum_p) - 1) & (q != 0.0) & (q != 1.0)

        if np.any(ind_is_good):
            # Process "good" values of `q` separately
            q_good = q[ind_is_good]
            q_ind_good = q_ind[ind_is_good]
            res_good = res[ind_is_good]

            # Process intervals resulted from discrete part
            ## Value of quantile function at "discrete" intervals are the left
            ## value of corresponding x-grid.
            is_in_disc = ids[q_ind_good] == "d"
            disc_inds = q_ind_good[is_in_disc]
            res_good[is_in_disc] = x[disc_inds]

            # Process intervals resulted from continuous part
            is_in_cont = ids[q_ind_good] == "c"
            cont_inds = q_ind_good[is_in_cont]
            ## We need to solve `w_c * F_c(x) + w_d * F_d(x) = q` for `x`, if it is
            ## known that `q` is such that it lies inside interval (possibly zero
            ## length) created by continuous part.
            ## We know that for the whole interval discrete term `w_d * F_d(x)` is
            ## constant. So it can be found by taking `x` as left edge of interval
            ## (`x_left`), for which we know mixture CDF `F_m(x_left)` (value of
            ## `cum_p`) and can compute value of `F_c(x_left)`. Then `disc_term =
            ## F_m(x_left) - w_c * F_c(x_left)`.
            ## This reduces original problem to `F_c(x) = (q - disc_term) / w_c`,
            ## which can solved by directly using quantile function of continuous
            ## part.
            disc_term = cum_p[cont_inds] - self._weight_cont * self._cont.cdf(
                x[cont_inds]
            )
            q_mod = (q_good[is_in_cont] - disc_term) / self._weight_cont
            res_good[is_in_cont] = self._cont.ppf(q_mod)

            res[ind_is_good] = res_good

        # All values of `q` outside of [0; 1] and equal to `nan` should result
        # into `nan`
        res[np.invert(ind_is_good)] = np.nan

        # Values 0.0 and 1.0 should be treated separately due to floating point
        # representation issues during `utils._searchsorted_wrap()`
        # application. In some extreme cases last `_p` can be smaller than 1 by
        # value of 10**(-16) magnitude, which will result into "bad" value of
        # `q_ind` (that is why this should also be done after assigning `nan`
        # to "bad" values)
        res[q == 0.0] = self._a
        res[q == 1.0] = self._b

        return np.asarray(res, dtype="float64")

    # `rvs()` is inherited from `Rand`


def _assert_two_tuple(x, x_name):
    if type(x) != tuple:
        raise ValueError(f"`{x_name}` should be a tuple.")
    if len(x) != 2:
        raise ValueError(f"`{x_name}` should have exactly two elements.")
    if (x[0] is None) and (x[1] is None):
        raise ValueError(f"`{x_name}` can't have two `None` elements.")

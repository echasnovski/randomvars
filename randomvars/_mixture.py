""" Code for mixture random variable
"""
import warnings

import numpy as np

from randomvars._continuous import Cont
from randomvars._discrete import Disc
from randomvars._random import Rand
from randomvars.options import options, _docstring_relevant_options
import randomvars._utils as utils
import randomvars._utilsgrid as utilsgrid


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
    2. Use `Mixt.from_rv()` to create approximation of some existing mixture
    random variable (object with methods `cdf()` and `ppf()`):
    ```
        from randomvars import Cont, Disc
        rv_ref = Mixt(Cont([0, 1], [1, 1]), Disc([-0.5, 1.5], [0.5, 0.5]), 0.75)
        class TmpRV:
            def __init__(self, rv):
                self.cdf = rv.cdf
                self.ppf = rv.ppf

        rv = TmpRV(rv_ref)
        rv_mixt = Mixt.from_rv(rv)
        (rv_mixt.cont.params, rv_mixt.disc.params, rv_mixt.weight_cont)
    ```
    3. Use `Mixt.from_sample()` to create estimation based on some existing sample:
    ```
        from scipy.stats import binom, norm
        sample_cont = norm().rvs(size=300, random_state=101)
        sample_disc = binom(n=10, p=0.1).rvs(size=100, random_state=102)
        sample = np.concatenate((sample_cont, sample_disc))
        ## There is no need for sample to have any structure. It is assumed to
        ## be a sample from mixture distribution.
        rng = np.random.default_rng(103)
        sample = rng.permutation(sample)
        rv_mixt = Mixt.from_sample(sample)
        (rv_mixt.cont.params, rv_mixt.disc.params, rv_mixt.weight_cont)
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
        self._cump_tuple = self._compute_cump_tuple()

        super().__init__()

    @staticmethod
    def _impute_init_args(cont, disc, weight_cont):
        # Impute `weight_cont`
        try:
            weight_cont = float(weight_cont)
        except ValueError:
            raise TypeError("`weight_cont` should be a number.")

        if (weight_cont < 0) or (weight_cont > 1):
            raise ValueError("`weight_cont` should be between 0 and 1 (inclusively).")

        # Impute `cont`
        if cont is None:
            if weight_cont > 0:
                raise ValueError("`cont` can't be `None` if `weight_cont` is above 0.")
        elif not isinstance(cont, Cont):
            raise TypeError("`cont` should be object of `Cont` or `None`.")

        # Impute `disc`
        if disc is None:
            if weight_cont < 1:
                raise ValueError("`disc` can't be `None` if `weight_cont` is below 1.")
        elif not isinstance(disc, Disc):
            raise TypeError("`disc` should be object of `Disc` or `None`.")

        return cont, disc, weight_cont

    def _compute_cump_tuple(self):
        """Compute tuple defining cumulative probability grid

        Main purpose is to be used inside quantile (`.ppf()`) function. Its
        output defines intervals on "cumulative probability line" inside which
        quantile function of mixture has single nature: continuous or discrete.

        Returns
        -------
        cump_tuple : Tuple with three numpy arrays
            Elements are:
            - Cump-grid (cumulative probability grid), that starts at 0 and
              ends at 1.
            - X-grid which is values corresponding to both cump-grid and
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
            - Both cump-grid and x-grid can have consecutive duplicating elements:
                - Cump-grid - in case there are values of discrete part inside
                  zero density region of continuous part. Those define
                  "interval" of continuous nature, which in fact isn't used.
                - X-grid - for every value of discrete part in case there is
                  continuous part. First one corresponds to the left limit of
                  mixture CDF at that point ("before jump"), second - to the
                  value of mixture CDF at that point ("after jump")
        """
        # Case of one discrete part
        if self._missing_cont():
            cump = np.array([0, 1])
            x = np.array([self._a, self._b])
            ids = np.array(["d"])
            return cump, x, ids

        # Case of one continuous part
        if self._missing_disc():
            cump = np.array([0, 1])
            x = np.array([self._a, self._b])
            ids = np.array(["c"])
            return cump, x, ids

        # Compute "inner" x-grid
        disc_x = self._disc.x
        x = np.repeat(disc_x, 2)

        # Compute "inner" cump-grid
        ## Values of mixture cdf
        cump_right = self.cdf(disc_x)
        ## Values of left limit of mixture cdf
        cump_left = cump_right - self._weight_disc * self._disc.p
        ## Interchange elements of `cump_left` and `cump_right`
        cump = np.array([cump_left, cump_right], order="C").flatten(order="F")

        # Compute "inner" interval identifiers
        ids = np.tile(["d", "c"], len(disc_x))[:-1]

        # Add possibly missing intervals because of continuous part tails
        if cump[0] > 0:
            cump = np.concatenate([[0], cump])
            x = np.concatenate([[self.a], x])
            ids = np.concatenate([["c"], ids])
        if cump[-1] < 1:
            cump = np.concatenate([cump, [1]])
            x = np.concatenate([x, [self.b]])
            ids = np.concatenate([ids, ["c"]])

        return cump, x, ids

    def __str__(self):
        return (
            "Mixture RV:\n"
            f"Cont (weight = {self._weight_cont}): {self._cont}\n"
            f"Disc (weight = {self._weight_disc}): {self._disc}"
        )

    @property
    def params(self):
        return {
            "cont": self._cont,
            "disc": self._disc,
            "weight_cont": self._weight_cont,
        }

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

    def compress(self):
        """Compress random variable

        Here the meaning of "compress" is to return a random variable (possibly
        of different class) which numerically has the same CDF values and uses
        minimum amount of parameters.

        Compressing of mixture RV is done by the following algorithm:
        - If `weight_cont` is zero, compressed version of discrete part is returned.
        - If `weight_disc` is zero, compressed version of continuous part is
          returned.
        - Otherwise, return mixture of compressed discrete and continuous
          parts.

        Returns
        -------
        rv_compressed : compressed RV
            If nothing to compress, self is returned.
        """
        if self._missing_cont():
            return self._disc.compress()
        if self._missing_disc():
            return self._cont.compress()

        cont_compressed = self._cont.compress()
        disc_compressed = self._disc.compress()
        if (cont_compressed is self._cont) and (disc_compressed is self._disc):
            return self
        else:
            return type(self)(
                cont=cont_compressed,
                disc=disc_compressed,
                weight_cont=self._weight_cont,
            )

    @classmethod
    @_docstring_relevant_options(["base_tolerance", "small_prob"])
    def from_rv(cls, rv):
        """Create mixture RV from general RV

        Mixture RV with discrete and continuous parts is created by the
        following algorithm:
        - **Detect discrete part**:
            - Compute possible values of input random variable `rv` as
              quantiles (values of `rv.ppf`) from equidistant grid between 0
              and 1 with step `small_prob` package option. **Note** that this
              step might take some time because usually `small_prob` is quite
              small which results into big length of grid array.
            - Estimate values to come from discrete part if they appear at
              least twice as possible values. That is, value is guaranteed to
              be detected if its probability multiplied by weight is at least
              `2 * small_prob`. **If no values are detected, then mixture with
              only continuous part `Cont.from_rv(rv)` is returned**.
            - Estimate jumps of `rv.cdf` at detected points `x` as `rv.cdf(x) -
              rv.cdf(x_left)`, where `x_left` is the smallest value that is
              close to `x` (controlled by `base_tolerance` package option).
            - Estimate weight of discrete part as sum of jumps at detected
              points. **If it is approximately equal to one (controlled by
              `base_tolerance` package option), mixture with only detected
              discrete part is returned**.
            - Estimate probabilities of detected values as jumps normalized to
              unity sum.
        - **Construct continuous part**. Having detected discrete part and its
          weight, construct continuous part by `Cont.from_rv(rv_cont)` where
          `rv_cont` has appropriate `cdf` and `ppf` methods.

        **Note** that if `rv` is an object of class `Rand`, it is converted to
        `Mixt` via `rv.convert("Mixt")`.

        {relevant_options}

        Parameters
        ----------
        rv : Object with methods `cdf()` and `ppf()`
            Methods `cdf()` and `ppf()` should implement functions for
            cumulative distribution and quantile functions respectively.
            Recommended to be an object of class
            `scipy.stats.distributions.rv_frozen` (`rv_discrete` with all
            hyperparameters defined).

        Returns
        -------
        rv_out : Mixt
            Mixture random variable estimated from input.
        """
        if isinstance(rv, Rand):
            return rv.convert("Mixt")

        # Check input
        rv_dir = dir(rv)
        if not all(method in rv_dir for method in ["cdf", "ppf"]):
            raise ValueError("`rv` should have methods `cdf()` and `ppf()`.")

        # Detect discrete part
        disc, weight_cont = _detect_disc_part(rv)

        # Make early return for degenerate cases
        if weight_cont == 0.0:
            return Mixt(cont=None, disc=disc, weight_cont=0.0)
        if weight_cont == 1.0:
            return Mixt(cont=Cont.from_rv(rv), disc=None, weight_cont=1.0)

        # Construct continuous part
        cont = _construct_cont_part(rv, disc, weight_cont)

        return Mixt(cont=cont, disc=disc, weight_cont=weight_cont)

    @classmethod
    @_docstring_relevant_options(["estimator_cont", "estimator_disc", "estimator_mixt"])
    def from_sample(cls, sample):
        """Create mixture RV from sample

        Mixture RV is created by the following algorithm:
        - **Estimate samples from parts** with mixture estimator (taken from
          package option "estimator_mixt") in the form `estimate =
          estimator_mixt(sample)`. If `estimate` is an object of class `Rand`
          it is forwarded to `Mixt.from_rv()`.
        - **Estimate parts** via `cont=Cont.from_sample(estimate[0])` and
          `disc=Disc.from_sample(estimate[1])`. If some estimate part is `None`
          or has zero length, mixture with only other part is created.
        - **Create random variable** with `Mixt(cont=cont, disc=disc,
          weight_cont=weight_cont)`, where `weight_cont` is estimated as
          fraction of continuous sample length relative to sum of both
          estimated samples' lengths.

        {relevant_options}

        Parameters
        ----------
        sample : 1d array-like
            This should be a valid input to `np.asarray()` so that its output
            is numeric and has single dimension.

        Returns
        -------
        rv_out : Mixt
            Mixture random variable with parts estimated from sample.
        """
        # Check and prepare input
        sample = utils._as_1d_numpy(sample, "sample", chkfinite=False, dtype="float64")

        # Get options
        estimator_mixt = options.estimator_mixt

        # Estimate distribution
        estimate = estimator_mixt(sample)

        # Make early return if `estimate` is random variable
        if isinstance(estimate, Rand):
            return Mixt.from_rv(estimate)

        _assert_two_tuple(estimate, "estimate")

        # Construct random variable
        if (estimate[0] is None) or (len(estimate[0]) == 0):
            return cls(cont=None, disc=Disc.from_sample(estimate[1]), weight_cont=0.0)
        if (estimate[1] is None) or (len(estimate[1]) == 0):
            return cls(cont=Cont.from_sample(estimate[0]), disc=None, weight_cont=1.0)

        return cls(
            cont=Cont.from_sample(estimate[0]),
            disc=Disc.from_sample(estimate[1]),
            # Having sum of two `estimate` lengths is important because it is
            # allowed for `estimate` parts to have different total number of
            # values than in input `sample`
            weight_cont=len(estimate[0]) / (len(estimate[0]) + len(estimate[1])),
        )

    def _missing_cont(self):
        return (self._cont is None) or (self._weight_cont == 0)

    def _missing_disc(self):
        return (self._disc is None) or (self._weight_disc == 0)

    def pdf(self, x):
        raise AttributeError("`Mixt` doesn't have probability density function.")

    def logpdf(self, x):
        raise AttributeError("`Mixt` doesn't have probability density function.")

    def pmf(self, x):
        raise AttributeError("`Mixt` doesn't have probability mass function.")

    def logpmf(self, x):
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

    # `logcdf()` is inherited from `Rand`

    # `sf()` is inherited from `Rand`

    # `logsf()` is inherited from `Rand`

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
        cump, x, ids = self._cump_tuple

        # Locate intervals for `q` elements
        q_ind = utils._searchsorted_wrap(cump, q, side="left", edge_inside=True) - 1
        ind_is_good = (q_ind >= 0) & (q_ind < len(cump) - 1) & (q != 0.0) & (q != 1.0)

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
            ## `cump`) and can compute value of `F_c(x_left)`. Then `disc_term =
            ## F_m(x_left) - w_c * F_c(x_left)`.
            ## This reduces original problem to `F_c(x) = (q - disc_term) / w_c`,
            ## which can solved by directly using quantile function of continuous
            ## part.
            disc_term = cump[cont_inds] - self._weight_cont * self._cont.cdf(
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

    # `isf()` is inherited from `Rand`

    # `rvs()` is inherited from `Rand`

    def integrate_cdf(self, a, b):
        """Efficient version of CDF integration"""
        if self._missing_cont():
            return self._disc.integrate_cdf(a, b)
        if self._missing_disc():
            return self._cont.integrate_cdf(a, b)

        return self._weight_cont * self._cont.integrate_cdf(
            a, b
        ) + self._weight_disc * self._disc.integrate_cdf(a, b)

    @_docstring_relevant_options(["base_tolerance", "small_width"])
    def convert(self, to_class=None):
        """Convert to different RV class

        Conversion is done by the following logic depending on the value of
        `to_class`:
        - If it is `None` or `"Mixt"`, `self` is returned.
        - If it is `"Bool"`, boolean RV is returned by converting to to Disc
          and Bool consecutively.
        - If it is `"Disc"`, discrete RV is returned. Its xp-grid is computed
          by the following algorithm:
            - Convert continuous part (if present) to discrete.
            - Return mixture of discrete variables: x-grid is union of two
              x-grids, p-grid is a weighted sum of probabilities at points of
              output x-grid.
        - If it is `"Cont"`, continuous RV is returned. Its xy-grid is computed
          by the following algorithm:
            - Convert discrete part (if present) to continuous.
            - Return mixture of continuous variables: x-grid is union of two
              x-grids, y-grid is a weighted sum of densities at points of
              output x-grid.
          **Note** that before creating mixture of continuous variables, they
          are "grounded" (see `Cont.ground()`) to create a proper mixture.
          Also:
            - Grounding is not done at points close (in terms of "closeness
              with tolerance", see `base_tolerance` package option) to what
              will be edges of mixture.
            - Grounding width is chosen to be the minimum of `small_width`
              package option and neighbor distances (distance between edge and
              nearest point in xy-grid) for all edges where grounding actually
              happens. This ensures smooth behavior in case of "touching
              supports".

        {relevant_options}

        Parameters
        ----------
        to_class : string or None, optional
            Name of target class. Can be one of: `"Bool"`, `"Cont"`, `"Disc"`,
            `"Mixt"`, or `None`.

        Raises
        ------
        ValueError:
            In case not supported `to_class` is given.
        """
        # Use inline `import` statements to avoid circular import problems
        if to_class == "Bool":
            return self.convert("Disc").convert("Bool")
        elif to_class == "Cont":
            if self._missing_cont():
                return self.disc.convert("Cont")
            if self._missing_disc():
                return self.cont

            disc_converted = self.disc.convert("Cont")
            xy_disc = disc_converted.x, disc_converted.y * self._weight_disc
            xy_cont = self.cont.x, self.cont.y * self._weight_cont

            x, y = utilsgrid._stack_xy((xy_disc, xy_cont))
            return Cont(x=x, y=y)
        elif to_class == "Disc":
            if self._missing_cont():
                return self.disc
            if self._missing_disc():
                return self.cont.convert("Disc")

            xp_disc = self.disc.x, self.disc.p * self._weight_disc
            cont_converted = self.cont.convert("Disc")
            xp_cont = cont_converted.x, cont_converted.p * self._weight_cont

            x, p = utilsgrid._stack_xp((xp_disc, xp_cont))
            return Disc(x=x, p=p)
        elif (to_class == "Mixt") or (to_class is None):
            return self
        else:
            raise ValueError(
                '`metric` should be one of "Bool", "Cont", "Disc", or "Mixt".'
            )


def _assert_two_tuple(x, x_name):
    if type(x) != tuple:
        raise TypeError(f"`{x_name}` should be a tuple.")
    if len(x) != 2:
        raise ValueError(f"`{x_name}` should have exactly two elements.")
    if (x[0] is None) and (x[1] is None):
        raise ValueError(f"`{x_name}` can't have two `None` elements.")


def _detect_disc_part(rv):
    # Get options
    small_prob = options.small_prob

    # Detect x-values coming from discrete part
    q_try = np.concatenate((np.arange(np.floor(1 / small_prob)) * small_prob, [1.0]))
    ## Usually (with small `small_prob`) takes a long time to compute
    x_try = rv.ppf(q_try)
    ## Value is estimated to come from discrete part if it appeared at least
    ## twice (because of a discontinuity jump in CDF)
    vals, counts = np.unique(x_try, return_counts=True)
    x = vals[counts >= 2]

    # Return early if there are no detected discrete values
    if len(x) == 0:
        return None, 1.0

    # Estimate p-grid
    ## Mixture probabilities are discrete probabilities multiplied by weight of
    ## discrete part
    ## **Important note**: this approach slightly overestimates probability of
    ## discrete value if there is a continuous part to its left. This leads to:
    ## - Overestimating weight of discrete part.
    ## - Later having CDF estimation of continuous part to be strictly
    ##   decreasing in the small right neighborhood of this discrete value.
    ##   This (possible small negative density values) is expected to be
    ##   handled inside `Cont.from_rv()`.
    mixt_probs = rv.cdf(x) - rv.cdf(x - utils._tolerance(x))
    weight_disc = np.sum(mixt_probs)

    ## Return early if mixture consists only from discrete part
    if utils._is_close(weight_disc, 1.0):
        return Disc(x=x, p=mixt_probs), 0.0

    return Disc(x=x, p=mixt_probs / weight_disc), 1 - weight_disc


def _construct_cont_part(rv_in, disc, weight_cont):
    weight_disc = 1 - weight_cont

    # Reconstruct CDF (here it is assumed `0 < weight_cont < 1`)
    cont_cdf = lambda t: (rv_in.cdf(t) - weight_disc * disc.cdf(t)) / weight_cont

    # Reconstruct quantile function
    q_grid_cont = np.concatenate(([0], cont_cdf(disc.x)))
    q_grid_disc = np.concatenate(([0], disc.cdf(disc.x)))

    class ContEstimate:
        def cdf(self, x):
            return cont_cdf(x)

        def ppf(self, q):
            q_ind = (
                utils._searchsorted_wrap(q_grid_cont, q, side="left", edge_inside=True)
                - 1
            )
            return rv_in.ppf(weight_cont * q + weight_disc * q_grid_disc[q_ind])

    return Cont.from_rv(ContEstimate())

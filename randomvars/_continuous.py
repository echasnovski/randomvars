""" Code for continuous random variable
"""
import warnings

import numpy as np
from scipy.stats.distributions import rv_frozen

import randomvars._utils as utils
from randomvars.downgrid_maxtol import downgrid_maxtol
from randomvars.options import get_option


class Cont:
    """Continuous random variable

    Class for continuous random variable **defined by piecewise-linear
    density**. It has **finite support**, **on which it is continuous**, and
    has **finite density values**. Defined by xy-grid of density: values in
    x-grid and their density in y-grid. In points inside support (segment from
    minimum to maximum values of x-grid) density is a linear interpolation of
    xy-grid. Outside of support density is equal to 0.

    There are three ways to create instance of `Cont` class:

    1. Directly supply parts of xy-grid (`x` for x-grid and `y` for y-grid):
    ```
        # "Triangular" distribution
        my_cont = Cont(x=[0, 1, 2], y=[0, 1, 0])
        my_cont.pdf([0.5, 1, 2.5])
    ```
    2. Use `Cont.from_rv()` to create approximation of some existing continuous
    random variable (object with methods `cdf()` and `ppf()`):
    ```
        from scipy.stats import norm
        rv_norm = norm()
        my_norm = Cont.from_rv(rv_norm)

        # Approximations are designed to be a compromise between high accuracy
        # and low number of grid points in piecewise-linear density
        rv_norm.pdf([-4, -0.1, 0, 0.1, 4])
        my_norm.pdf([-4, -0.1, 0, 0.1, 4])
    ```
    3. Use `Cont.from_sample()` to create estimation based on some existing sample:
    ```
        from scipy.stats import norm
        sample = norm().rvs(size=100, random_state=101)
        my_rv = Cont.from_sample(sample)
        my_rv.pdf([-0.1, 0, 0.1])
    ```
    """

    def __init__(self, x, y):
        x, y = self._impute_init_args(x, y)

        self._x = x
        self._y = y
        self._cum_p = utils._trapez_integral_cum(self._x, self._y)
        self._a = x[0]
        self._b = x[-1]

    @staticmethod
    def _impute_init_args(x, y):
        x = utils._as_1d_numpy(x, "x", chkfinite=True, dtype="float64")
        y = utils._as_1d_numpy(y, "y", chkfinite=True, dtype="float64")

        x, y = utils._sort_parallel(x, y, warn=True)

        if not np.all(np.diff(x) > 0):
            x, y = utils._unique_parallel(x, y, warn=True)

        if (len(x) < 2) or (len(y) < 2):
            raise ValueError("Both `x` and `y` should have at least two elements.")

        utils._assert_positive(y, "y")

        y = y / utils._trapez_integral(x, y)

        return x, y

    def __str__(self):
        x_len = len(self.x)
        s = "s" if x_len > 2 else ""
        return (
            f"Continuous RV with {x_len-1} interval{s} (support: [{self.a}, {self.b}])"
        )

    @property
    def x(self):
        """Return x-grid (`x` component of piecewise-linear density)"""
        return self._x

    @property
    def y(self):
        """Return y-grid (`y` component of piecewise-linear density)"""
        return self._y

    @property
    def cum_p(self):
        """Return cumulative probability grid of piecewise-linear density"""
        return self._cum_p

    @property
    def a(self):
        """Return left edge of support"""
        return self._a

    @property
    def b(self):
        """Return right edge of support"""
        return self._b

    def support(self):
        """Return support of random variable"""
        return (self.a, self.b)

    def _coeffs_by_ind(self, ind=None):
        """Compute density linear coefficients based on index of interval.

        If `ind` is `None`, then coefficients for all intervals are returned.

        Index `i` corresponds to coefficients from interval with endpoints
        `self._x[i-1]` and `self._x[i]`. Which intervals `self._x` values
        represent should be decided before calling this function during
        computation of `ind`.
        Indexes 0 and `len(self._x)` result in zero coefficients. Indexes
        outside `[0, len(self._x)]` result into `np.nan` coefficients.

        Parameters
        ----------
        ind : numpy integer array, optional
            Describes index of interval, coefficients of which should be
            returned.

        Returns
        -------
        coeffs : tuple with 2 float arrays with lengths equal to length of
        `ind`.
            First element represents intercept, second - slope.

        Examples
        --------
        >>> rv = Cont([0, 1, 2], [0, 1, 0])
        >>> rv._coeffs_by_ind()
        (array([0., 2.]), array([ 1., -1.]))
        >>> rv._coeffs_by_ind(np.array([0, 1, 2, 3]))
        (array([0., 0., 2., 0.]), array([ 0.,  1., -1.,  0.]))
        >>> rv._coeffs_by_ind(np.array([-1, 100]))
        (array([nan, nan]), array([nan, nan]))
        """
        if ind is None:
            ind = np.arange(start=1, stop=len(self._x))

        inter = np.zeros_like(ind, dtype=np.float64)
        slope = np.zeros_like(ind, dtype=np.float64)

        ind_as_nan = (ind < 0) | (ind > len(self._x))
        inter[ind_as_nan] = np.nan
        slope[ind_as_nan] = np.nan

        ind_is_in = (ind > 0) & (ind < len(self._x))
        inds_in = ind[ind_is_in]
        if len(inds_in) > 0:
            slope[ind_is_in] = (self._y[inds_in - 1] - self._y[inds_in]) / (
                self._x[inds_in - 1] - self._x[inds_in]
            )
            inter[ind_is_in] = (
                self._y[inds_in - 1] - slope[ind_is_in] * self._x[inds_in - 1]
            )

        return (inter, slope)

    def _grid_by_ind(self, ind=None):
        """Get grid values of interval

        If `ind` is `None` (default), it is equivalent to
        `np.arange(len(self._x))`, i.e. all present grid is returned. Indexes
        outside of `(0, len(self._x)]` will produce `np.nan` in output.

        Parameters
        ----------
        ind : numpy integer array, optional
            Describes index of interval, *left* grid elements of which should
            be returned.

        Returns
        -------
        grid: tuple with 3 numpy arrays
            Elements represent `x`, `y`, and `p` *left* values of intervals.

        Examples
        --------
        >>> rv = Cont([0, 1, 2], [0, 1, 0])
        >>> x, y, p = rv._grid_by_ind(np.array([-1, 0, 1, 2, 3, 4]))
        >>> x
        array([nan, nan,  0.,  1.,  2., nan])
        >>> x, y, p = rv._grid_by_ind()
        >>> p
        array([0. , 0.5, 1. ])
        """
        if ind is None:
            return (self._x, self._y, self._cum_p)

        x = np.empty_like(ind, dtype=np.float64)
        y = np.empty_like(ind, dtype=np.float64)
        p = np.empty_like(ind, dtype=np.float64)

        # There is no grid elements to the left of interval 0, so outputs are
        # `np.nan` for it
        out_is_nan = (ind <= 0) | (ind > len(self._x))
        x[out_is_nan] = np.nan
        y[out_is_nan] = np.nan
        p[out_is_nan] = np.nan

        out_isnt_nan = ~out_is_nan
        ind_in = ind[out_isnt_nan] - 1
        x[out_isnt_nan] = self._x[ind_in]
        y[out_isnt_nan] = self._y[ind_in]
        p[out_isnt_nan] = self._cum_p[ind_in]

        return (x, y, p)

    def pdf_coeffs(self, x, side="right"):
        """Compute density linear coefficients based on `x`.

        For each entry of `x`, compute linear coefficients of piecewise-linear
        density at that point, so that `pdf(x)` is equal to `intercept +
        x*slope`.

        If it is equal to the element of density x-grid, coefficients are taken
        from right or left interval, depending on value of `side`. Exceptions
        are edge elements: if `side` is "right", coefficients of the right most
        value of x-grid is taken from its left interval; if `side` is "left" -
        from its right for left most element.

        Parameters
        ----------
        x : numpy array.
            Points at which density coefficients should be computed.
        side : str, optional
            Should be one of "right" (default) or "left".

        Returns
        -------
        coeffs : tuple with 2 float arrays with lengths equal to length of `x`.
            First element is intercept, second - slope.

        Notes
        -----
        Values `np.nan`, `-np.inf`, and `np.inf` are valid inputs. For `np.nan`
        both coefficients are `np.nan`. for infinities their are zeros.

        Examples
        --------
        >>> rv_p = Cont([0, 1, 2], [0, 1, 0])
        >>> x = np.array([-1, 0, 0.5, 1, 1.5, 2, 2.5])
        >>> rv_p.pdf_coeffs(x)
        (array([0., 0., 0., 2., 2., 2., 0.]), array([ 0.,  1.,  1., -1., -1., -1.,  0.]))
        >>> rv_p.pdf_coeffs(x, side="left")
        (array([0., 0., 0., 0., 2., 2., 0.]), array([ 0.,  1.,  1.,  1., -1., -1.,  0.]))
        >>> rv_p.pdf_coeffs(np.array([-np.inf, np.nan, np.inf]))
        (array([ 0., nan,  0.]), array([ 0., nan,  0.]))
        """
        if side not in ["left", "right"]:
            raise ValueError('`side` should be one of "right" or "left"')

        ind = utils._searchsorted_wrap(self._x, x, side=side, edge_inside=True)

        return self._coeffs_by_ind(ind)

    @classmethod
    def from_rv(cls, rv, supp=None):
        """Create continuous RV from general RV

        Continuous RV with piecewise-linear density is created by the following
        algorithm:
        - **Detect finite support**. Left and right edges are treated
          separately. If edge is supplied, it is used untouched. If not, it is
          computed by "removing" corresponding (left or right) tail which has
          probability of `small_prob` (package option).
        - **Create x-grid**. It is computed as union of equidistant (fixed
          distance between consecutive points) and equiprobable (fixed
          probability between consecutive points) grids between edges of
          detected finite support. Number of points in grids is equal to
          `n_grid` (package option). Also it is ensured that no points lie very
          close to each other (order of `1e-13` distance), because otherwise
          output will have unstable values.
        - **Create density xy-grid**. X-grid is taken from previous step, while
          corresponding y-grid is computed as derivatives (with
          `np.gradient()`) of CDF-values. Density is not used directly to
          account for its possible infinite values.
        - **Downgrid density xy-grid**. `downgrid_maxtol()` is used with
          tolerance ensuring that difference of total integrals between input
          and downgridded xy-grids is less than `integr_tol` (package option).

        **Note** that if `rv` is already an object of class `Cont`, it is
        returned untouched.

        Relevant package options: `n_grid`, `small_prob`, `integr_tol`. See
        documentation of `randomvars.options.get_option()` for more
        information. To temporarily set options use
        `randomvars.options.option_context()` context manager.

        Parameters
        ----------
        rv : Object with methods `cdf()` and `ppf()`
            Methods `cdf()` and `ppf()` should implement functions for
            cumulative distribution and quantile functions respectively.
            Recommended to be an object of class
            `scipy.stats.distributions.rv_frozen` (`rv_continuous` with all
            hyperparameters defined).
        supp : Tuple with two numbers or `None`, optional
            Forced support edges. Elements should be either finite numbers
            (returned untouched) or `None` (finite support edge is detected).
            Single `None` is equivalent to `(None, None)`, i.e. finding both edges
            of finite support.

        Returns
        -------
        rv_out : Cont
            Random variable with finite support and piecewise-linear density
            which approximates density of input `rv`.
        """
        # Make early return
        if isinstance(rv, Cont):
            return rv

        # Check input
        rv_dir = dir(rv)
        if not all(method in rv_dir for method in ["cdf", "ppf"]):
            raise ValueError("`rv` should have methods `cdf()` and `ppf()`.")

        # Get options
        n_grid = get_option("n_grid")
        small_prob = get_option("small_prob")
        integr_tol = get_option("integr_tol")

        # Detect effective support of `rv`
        x_left, x_right = _detect_finite_supp(rv, supp, small_prob)

        # Construct equidistant grid
        x_equi = np.linspace(x_left, x_right, n_grid)

        # Construct quantile grid
        prob_left, prob_right = rv.cdf([x_left, x_right])
        prob_equi = np.linspace(prob_left, prob_right, n_grid)
        x_quan = rv.ppf(prob_equi)

        # Combine equidistant and quantile grids into one sorted array
        x = _combine_grids(x_equi, x_quan)

        # Compute `y` as derivative of CDF. Not using `pdf` directly to account
        # for infinite density values.
        # Using `edge_order=2` gives better accuracy when edge has infinite
        # density
        y = np.gradient(rv.cdf(x), x, edge_order=2)
        ## Account for possible negative values of order 1e-17
        y = np.clip(y, 0, None)

        # Reduce grid size allowing such maximum difference so that
        # piecewise-linear integrals differ by not more than `integr_tol`
        x, y = downgrid_maxtol(x, y, integr_tol / (x[-1] - x[0]))

        return cls(x, y)

    @classmethod
    def from_sample(cls, sample):
        """Create continuous RV from sample

        Continuous RV with piecewise-linear density is created by the following
        algorithm:
        - **Estimate density** with density estimator (taken from package
          option "density_estimator") in the form `density =
          density_estimator(sample)`. If `density` is object of class `Cont`,
          it is returned untouched. If it is object of
          `scipy.stats.distributions.rv_frozen` (`rv_continuous` with all
          hyperparameters defined), it is forwarded to `Cont.from_rv()`.
        - **Estimate effective range of density**: interval inside which total
          integral of density is not less than `density_mincoverage` (package
          option). Specific algorithm how it is done is subject to change.
          General description of the current one:
            - Make educated guess about initial range (usually with input
              sample range).
            - Iteratively extend range in both directions until density total
              integral is above desired threshold.
        - **Create x-grid**. It is computed as union of equidistant (fixed
          distance between consecutive points) and equiprobable (fixed
          probability between consecutive points based on sample quantiles)
          grids between edges of detected density range. Number of points in
          grids is equal to `n_grid` (package option). Also it is ensured that
          no points lie very close to each other (order of `1e-13` distance),
          because otherwise output will have unstable values.
        - **Create density xy-grid**. X-grid is taken from previous step,
          y-grid is taken as values of density estimate at points of x-grid.
        - **Downgrid density xy-grid**. `downgrid_maxtol()` is used with
          tolerance ensuring that difference of total integrals between input
          and downgridded xy-grids is less than `integr_tol` (package option).
        - **Create random variable** with `Cont(x=x, y=y)`, where `x` and `y`
          are x-grid and y-grid.

        Relevant package options: `density_estimator`, `density_mincoverage`,
        `n_grid`, `integr_tol`. See documentation of
        `randomvars.options.get_option()` for more information. To temporarily
        set options use `randomvars.options.option_context()` context manager.

        Parameters
        ----------
        sample : 1d array-like
            This should be a valid input to `np.asarray()` so that its output
            is numeric and has single dimension.

        Returns
        -------
        rv_out : Cont
            Random variable with finite support and piecewise-linear density
            which approximates density estimate of input `sample`.
        """
        # Check and prepare input
        sample = utils._as_1d_numpy(sample, "sample", chkfinite=False, dtype="float64")

        # Get options
        density_estimator = get_option("density_estimator")
        density_mincoverage = get_option("density_mincoverage")
        n_grid = get_option("n_grid")
        integr_tol = get_option("integr_tol")

        # Estimate density
        density = density_estimator(sample)

        # Make early return if `density` is random variable
        if isinstance(density, Cont):
            return density
        if isinstance(density, rv_frozen):
            return Cont.from_rv(density)

        # Estimate density range
        x_left, x_right = _estimate_density_range(density, sample, density_mincoverage)

        # Construct equidistant grid
        x_equi = np.linspace(x_left, x_right, n_grid)

        # Construct quantile grid
        prob_equi = np.linspace(0, 1, n_grid)
        x_quan = np.quantile(a=sample, q=prob_equi, interpolation="linear")

        # Combine equidistant and quantile grids into one sorted array
        x_grid = _combine_grids(x_equi, x_quan)
        y_grid = density(x_grid)

        # Reduce grid size allowing such maximum difference so that
        # piecewise-linear integrals differ by not more than `integr_tol`
        x_grid, y_grid = downgrid_maxtol(
            x_grid, y_grid, tol=integr_tol / (x_grid[-1] - x_grid[0])
        )

        return cls(x_grid, y_grid)

    def pdf(self, x):
        """Probability density function

        Return values of probability density function at points `x`.

        Parameters
        ----------
        x : array_like with numeric values

        Returns
        -------
        pdf_vals : ndarray with shape inferred from `x`
        """
        return np.interp(x, self._x, self._y, left=0, right=0)

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
        x = np.asarray(x, dtype=np.float64)
        res = np.zeros_like(x, dtype=np.float64)

        x_ind = utils._searchsorted_wrap(self._x, x, side="right", edge_inside=True)
        ind_is_good = (x_ind > 0) & (x_ind < len(self._x))

        if np.any(ind_is_good):
            x_good = x[ind_is_good]
            x_ind_good = x_ind[ind_is_good]

            gr_x, _, gr_p = self._grid_by_ind(x_ind_good)
            inter, slope = self._coeffs_by_ind(x_ind_good)

            # Using `(a+b)*(a-b)` instead of `(a*a-b*b)` for better accuracy in
            # case density x-grid has really close elements
            res[ind_is_good] = (
                gr_p
                + inter * (x_good - gr_x)
                + 0.5 * slope * (x_good + gr_x) * (x_good - gr_x)
            )

        # `res` is already initialized with zeros, so taking care of `x` to the
        # left of support is not necessary
        res[x_ind == len(self.x)] = 1.0

        return utils._copy_nan(fr=x, to=res)

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
        q = np.asarray(q, dtype=np.float64)
        res = np.zeros_like(q, dtype=np.float64)

        q_ind = utils._searchsorted_wrap(self._cum_p, q, side="right", edge_inside=True)
        ind_is_good = (q_ind > 0) & (q_ind < len(self._x)) & (q != 0.0) & (q != 1.0)

        if np.any(ind_is_good):
            q_good = q[ind_is_good]
            q_ind_good = q_ind[ind_is_good]

            grid_q = self._grid_by_ind(q_ind_good)
            coeffs_q = self._coeffs_by_ind(q_ind_good)

            res[ind_is_good] = self._find_quant(q_good, grid_q, coeffs_q)

        # All values of `q` outside of [0; 1] and equal to `nan` should result
        # into `nan`
        res[np.invert(ind_is_good)] = np.nan

        # Values 0.0 and 1.0 should be treated separately due to floating point
        # representation issues during `utils._searchsorted_wrap()`
        # application. In some extreme cases last `_p` can be smaller than 1 by
        # value of 10**(-16) magnitude, which will result into "bad" value of
        # `q_ind` (that is why this should also be done after assigning `nan`
        # to "bad" values)
        res[q == 0.0] = self._x[0]
        res[q == 1.0] = self._x[-1]

        return res

    def rvs(self, size=None, random_state=None):
        """Random number generation

        Generate random numbers into array of desired size.

        Parameters
        ----------
        size : int or tuple of ints, optional
            Defining number of random variates (default is 1).
        random_state : `None`, int, or RandomState, optional
            Source of uniform random number generator. If `None`, it is
            initiated as `numpy.random.RandomState()`. If integer,
            `numpy.random.RandomState(seed=random_state)` is used.
        """
        if random_state is None:
            random_state = np.random.RandomState()
        elif isinstance(random_state, int):
            random_state = np.random.RandomState(seed=random_state)

        U = random_state.uniform(size=size)

        return self.ppf(U)

    def _find_quant(self, q, grid, coeffs):
        """Compute quantiles with data from linearity intervals

        Based on precomputed data of linearity intervals, compute actual quantiles.
        Here `grid` and `coeffs` are `(x, y, p)` and `(inter, slope)` values of
        intervals inside which `q` quantile is located.

        Parameters
        ----------
        q : numpy array
        grid : tuple with 3 numpy arrays with lengths same to `len(q)`
        coeffs : tuple with 3 numpy arrays with lengths same to `len(q)`

        Returns
        -------
        quant : numpy array with the same length as q
        """
        res = np.empty_like(q, dtype=np.float64)
        x, _, p = grid
        inter, slope = coeffs

        is_quad = ~np.isclose(slope, 0)
        is_lin = ~(is_quad | np.isclose(inter, 0))
        is_const = ~(is_quad | is_lin)

        # Case of quadratic CDF curve (density is a line not aligned to x axis)
        # The "true" quadratic curves are transformed in terms of `t = x - x_l`
        # for numerical accuracy
        # Equations have form a*t^2 + t*x + c = 0
        a = 0.5 * slope[is_quad]
        b = 2 * a * x[is_quad] + inter[is_quad]
        c = p[is_quad] - q[is_quad]
        # Theoretically, `discr` should always be >= 0. However, due to
        # numerical inaccuracies of magnitude ~10^(-15), here call to
        # `np.clip()` is needed.
        discr = np.clip(b * b - 4 * a * c, 0, None)
        res[is_quad] = (-b + np.sqrt(discr)) / (2 * a) + x[is_quad]

        # Case of linear CDF curve (density is non-zero constant)
        res[is_lin] = x[is_lin] + (q[is_lin] - p[is_lin]) / inter[is_lin]

        # Case of plateau in CDF (density equals zero)
        res[is_const] = x[is_const]

        return res


def _detect_finite_supp(rv, supp=None, small_prob=1e-6):
    """Detect finite support of random variable

    Finite support edge is detected via testing actual edge to be finite:
    - If finite, it is returned.
    - If infinite, output is a value with outer tail having probability
      `small_prob`.

    Parameters
    ----------
    rv : Random variable from 'scipy' module.
    supp : Tuple with two elements or `None`, optional
        Forced support edges. Elements should be either finite numbers
        (returned untouched) or `None` (finite support edge is detected).
        Single `None` is equivalent to `(None, None)`, i.e. finding both edges
        of finite support.
    small_prob : Tail probability, optional
        Probability value of tail that might be cutoff in order to get finite
        support.

    Returns
    -------
    supp : Tuple with 2 values for left and right edges of finite support.
    """
    if supp is None:
        supp = (None, None)

    if supp[0] is None:
        left = rv.ppf(0)
        if np.isneginf(left):
            left = rv.ppf(small_prob)
    else:
        left = supp[0]

    if supp[1] is None:
        right = rv.ppf(1)
        if np.isposinf(right):
            right = rv.ppf(1 - small_prob)
    else:
        right = supp[1]

    return left, right


def _estimate_density_range(density, sample, density_mincoverage):
    """Estimate effective range of sample density estimate

    Goal is to estimate range within which integral of density is not less than
    `density_mincoverage`.

    Iterative algorithm:
        - First iteration is to take density range as sample range. If it has
          zero width, then return range with assumption of constant density
          (forced to be 1 if actual value for some reason is zero) at sample
          point.
        - Other iterations: extend current range to increase covered
          probability. Current algorithm is as follows (subject to change):
            - Compute "covered probability": integral of density inside current
              range.
            - Compute values of density at range ends and branch:
                - **If their sum is positive**:
                    - Split non-covered probability to left and right range
                      ends proportionally with "weights" computed from density
                      values at range ends. This illustrates approach "Extend
                      more towards side with bigger density value (as it takes
                      less effort to make bigger density coverage gain)".
                    - Left and right additive increases in range are computed
                      using assumption that density will remain constant (equal
                      to value at nearest range end) outside of current range.
                - **If their sum is not positive** (but covered probability is
                  still not enough), extend both ends equally by the amount
                  `0.5 * noncov_prob * range_width`. Here `noncov_prob` -
                  non-covered probability (one minus density integral inside
                  current range), `range_width` - width of current range.
    """
    cur_range = _init_range(sample, density)
    cur_cov_prob = utils._quad_silent(density, cur_range[0], cur_range[1])

    while cur_cov_prob < density_mincoverage:
        cur_range, cur_cov_prob = _extend_range(cur_range, density, cur_cov_prob)

    return cur_range


def _init_range(sample, density):
    x_left, x_right = sample.min(), sample.max()

    if x_left == x_right:
        x, y = x_left, density(x_left)
        # Compute width with assumption of constant density
        half_width = 0.5 / y if y > 0 else 0.5
        x_left, x_right = x - half_width, x + half_width

    return x_left, x_right


def _extend_range(x_range, density, cov_prob):
    noncov_prob = 1 - cov_prob
    y_left, y_right = density(np.asarray(x_range))
    y_sum = y_left + y_right

    if y_sum > 0:
        # "Allocate" more probability to side with bigger density value
        alpha = y_left / y_sum
        prob_left, prob_right = alpha * noncov_prob, (1 - alpha) * noncov_prob

        # Compute deltas as if extension is done with constant density value
        delta_left = prob_left / y_left if y_left > 0 else 0
        delta_right = prob_right / y_right if y_right > 0 else 0
    else:
        delta_left = 0.5 * noncov_prob * (x_range[1] - x_range[0])
        delta_right = delta_left

    res_range = (x_range[0] - delta_left, x_range[1] + delta_right)

    # Update covered probability. Here not using direct approach of the form
    # `utils._quad_silent(density, x_range[0], x_range[1])` is crucial because
    # it may lead to inaccurate results with wide range.
    cov_prob_left = utils._quad_silent(density, res_range[0], x_range[0])
    cov_prob_right = utils._quad_silent(density, x_range[1], res_range[1])
    res_cov_prob = cov_prob + cov_prob_left + cov_prob_right

    return res_range, res_cov_prob


def _combine_grids(grid1, grid2, tol=1e-13):
    x = np.union1d(grid1, grid2)

    ## Ensure that minimum difference between consecutive elements isn't
    ## very small, otherwise this will make `np.gradient()` perform poorly
    x_is_good = np.concatenate([[True], np.ediff1d(x) > tol])

    return x[x_is_good]

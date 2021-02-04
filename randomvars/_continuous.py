""" Code for continuous random variable
"""
import warnings

import numpy as np
from scipy.interpolate import UnivariateSpline, splantider
from scipy.stats.distributions import rv_frozen

import randomvars._utils as utils
import randomvars._utilsgrid as utilsgrid
from randomvars.downgrid_maxtol import downgrid_maxtol
from randomvars._random import Rand
from randomvars.options import options, _docstring_relevant_options


class Cont(Rand):
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

        # User-facing attributes
        self._x = x
        self._y = y
        self._a = x[0]
        self._b = x[-1]

        # Private attributes
        self._cump = utils._trapez_integral_cum(self._x, self._y)

        super().__init__()

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
        x_len = len(self._x)
        s = "s" if x_len > 2 else ""
        return (
            f"Continuous RV with {x_len-1} interval{s} "
            f"(support: [{self._a}, {self._b}])"
        )

    @property
    def params(self):
        return {"x": self._x, "y": self._y}

    params.__doc__ = Rand.params.__doc__

    @property
    def x(self):
        """Return x-grid (`x` component of piecewise-linear density)"""
        return self._x

    @property
    def y(self):
        """Return y-grid (`y` component of piecewise-linear density)"""
        return self._y

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

        Here the meaning of "compress" is to return a random variable which
        numerically has the same CDF values and uses minimum amount of
        parameters.

        Compressing of continuous RV is done by the following algorithm:
        - Remove x-values from beginning and end which has zero y-values except
          the "most center" ones. Those describe tails with zero probability
          which don't affect the CDF values.
        - Remove x-values which provide "extra linearity": they contribute
          points on piecewise-linear density which lie on the segment
          connecting neighbor points. In other words, x-values for which slopes
          of neighbor intervals are the same. Removing them won't affect
          density structure and hence CDF values.

        Returns
        -------
        rv_compressed : compressed RV
            If nothing to compress, self is returned.
        """
        x, y = self._x, self._y

        # X-values which contribute zero probability from beginning and end
        zero_probs = 0.5 * (x[1:] - x[:-1]) * (y[1:] + y[:-1]) == 0
        zero_tail_left = np.concatenate((np.minimum.accumulate(zero_probs), [False]))
        zero_tail_right = np.concatenate(
            (np.minimum.accumulate(zero_probs[::-1]), [False])
        )[::-1]

        # X-values with extra linearity
        _, slope = self._coeffs_by_ind()
        extra_linearity = np.concatenate(([False], slope[:-1] == slope[1:], [False]))

        # Compress
        x_is_good = ~(zero_tail_left | zero_tail_right | extra_linearity)
        if np.all(x_is_good):
            return self
        else:
            return type(self)(x=x[x_is_good], y=y[x_is_good])

    @_docstring_relevant_options(["small_width"])
    def ground(self, w=None, direction="both"):
        """Update xy-grid to represent explicit piecewise-linear function

        Implicitly xy-grid represents piecewise-linear density in the following way:
        - For points inside `[x[0]; x[-1]]` (support) output is a linear
          interpolation.
        - For points outside support output is zero.

        This function returns new `Cont` object with transformed xy-grid so
        that density can be computed as a direct linear interpolation on the
        whole real line. This is done by possibly approximating "jumps" at the
        edge(s) of support. Approximation at edge `(x_e, y_e)` is performed by
        introducing a linear smoothing of a jump:
          - Add outside point (x_e +/- w, 0) (sign depends on whether edge is right
            or left). Here `w` is width of approximation controlled by input
            argument `w`.
          - Possibly add inner point `(x_e -/+ w, f(x_e -/+ w))` (`f(x)` - function
            xy-grid represents). It is done only if distance between closest to
            edge point (neighbor) and edge is strictly greater than `w`.
          - Adjust edge y-value to preserve total probability.

        Grounding can be done partially:
          - If `direction="left"` or `direction="right"`, edge jump is
            approximated only on left/right edge respectively.
          - If `direction="none"`, no grounding is done.

        Notes:
        - If edge is already zero, then no grounding is done.
        - This might lead to a very close points on x-grid: in case distance
          between edge and neighbor is `w + eps` with `eps` being very small.
        - If there is a neighbor strictly closer than `w`, slopes of jump
          approximation depend on input neighbor distance.

        {relevant_options}

        Parameters
        ----------
        w : float or None
            Width of jump approximation. If `None`, `small_width` package
            option is used.
        direction : string
            Can be one of `"both"`, `"left"`, `"right"`, or `"none"`. Controls
            which edge(s) should be grounded (if any).
        """
        if direction not in ["both", "left", "right", "none"]:
            raise ValueError(
                '`direction` should be one of "both", "left", "right", "none".'
            )
        if w is None:
            w = options.small_width

        x, y = utilsgrid._ground_xy(xy=(self._x, self._y), w=w, direction=direction)
        return Cont(x=x, y=y)

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

        inter = np.zeros_like(ind, dtype="float64")
        slope = np.zeros_like(ind, dtype="float64")

        ind_as_nan = (ind < 0) | (ind > len(self._x))
        inter[ind_as_nan] = np.nan
        slope[ind_as_nan] = np.nan

        ind_is_in = (ind > 0) & (ind < len(self._x))
        inds_in = ind[ind_is_in]
        if len(inds_in) > 0:
            slope[ind_is_in] = (self._y[inds_in] - self._y[inds_in - 1]) / (
                self._x[inds_in] - self._x[inds_in - 1]
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
            Elements represent `x`, `y`, and `_cump` *left* values of
            intervals.

        Examples
        --------
        >>> rv = Cont([0, 1, 2], [0, 1, 0])
        >>> x, y, cump = rv._grid_by_ind(np.array([-1, 0, 1, 2, 3, 4]))
        >>> x
        array([nan, nan,  0.,  1.,  2., nan])
        >>> x, y, cump = rv._grid_by_ind()
        >>> cump
        array([0. , 0.5, 1. ])
        """
        if ind is None:
            return (self._x, self._y, self._cump)

        x = np.empty_like(ind, dtype="float64")
        y = np.empty_like(ind, dtype="float64")
        cump = np.empty_like(ind, dtype="float64")

        # There is no grid elements to the left of interval 0, so outputs are
        # `np.nan` for it
        out_is_nan = (ind <= 0) | (ind > len(self._x))
        x[out_is_nan] = np.nan
        y[out_is_nan] = np.nan
        cump[out_is_nan] = np.nan

        out_isnt_nan = ~out_is_nan
        ind_in = ind[out_isnt_nan] - 1
        x[out_isnt_nan] = self._x[ind_in]
        y[out_isnt_nan] = self._y[ind_in]
        cump[out_isnt_nan] = self._cump[ind_in]

        return (x, y, cump)

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
    @_docstring_relevant_options(
        ["base_tolerance", "cdf_tolerance", "n_grid", "small_prob"]
    )
    def from_rv(cls, rv, supp=None):
        """Create continuous RV from general RV

        Continuous RV with piecewise-linear density is created by the following
        algorithm:
        - **Detect finite support**. Left and right edges are treated
          separately. If edge is supplied, it is used untouched. If not, it is
          detected based on the value supplied via `rv.ppf()` at extreme
          quantile (`rv.ppf(0.0)` for left edge and `rv.ppf(1.0)` for right
          edge):
            - If value is finite, edge is computed to be the value "closest to
              positive probability region" (most right for left edge and most
              left for right edge) while having the same extreme value of CDF.
              This is done by iterative procedure which is terminated when
              certain two numbers are considered to be approximately equal
              (controlled by `base_tolerance` package option).
            - If value is infinite, edge is computed by "removing"
              corresponding (left or right) tail which has probability of
              `small_prob` (package option).
        - **Create x-grid**. It is computed as union of equidistant (fixed
          distance between consecutive points) and equiprobable (fixed
          probability between consecutive points) grids between edges of
          detected finite support. Number of points in grids is equal to
          `n_grid` (package option). Also it is ensured that no points lie very
          close to each other (order of `1e-12` distance), because otherwise
          output density will have unstable values.
        - **Fit quadratic spline to CDF at x-grid**. This dramatically reduces
          number of points in output `Cont`'s xy-grid in exchange of (usually)
          small inaccuracy. Spline is fitted using
          `scipy.interpolate.UnivariateSpline` with the following arguments:
            - `x` is equal to x-grid from previous step.
            - `y` is equal to values of `rv.cdf()` at x-grid.
            - `k` (spline degree) is given as 2 for output to represent
              quadratic spline.
            - `s` (smoothing factor) is taken as square of `cdf_tolerance`
              package option (as `s` represents mean squared approximation
              error). Bigger values of `cdf_tolerance` lead to smaller number
              of elements in output `Cont`'s xy-grid. Smaller values (together
              with large enough values of `n_grid` option) lead to better
              approximation of `rv.cdf()`.
        - **Create density xy-grid**. Density is estimated as spline derivative
          of quadratic spline from previous step. X-grid is taken as knots of
          "density spline". Y-grid is computed as values of "density spline" at
          those knots truncating possible negative values to become zero. Here
          negative values can occur if CDF approximation is allowed to be loose
          (either `n_grid` is low or `cdf_tolerance` is high).

        **Note** that if `rv` is an object of class `Rand`, it is converted to
        `Cont` via `rv.convert("Cont")`.

        {relevant_options}

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
        if isinstance(rv, Rand):
            return rv.convert("Cont")

        # Check input
        rv_dir = dir(rv)
        if not all(method in rv_dir for method in ["cdf", "ppf"]):
            raise ValueError("`rv` should have methods `cdf()` and `ppf()`.")

        # Get options
        n_grid = options.n_grid
        small_prob = options.small_prob
        cdf_tolerance = options.cdf_tolerance

        # Detect effective support of `rv`
        finite_supp = _detect_finite_supp(rv, supp, small_prob)

        # Compute combination of equidistant and quantile grids
        x = _compute_union_grid(
            x_range=finite_supp,
            prob_range=rv.cdf(list(finite_supp)),
            quantile_fun=rv.ppf,
            n_grid=n_grid,
        )

        # Fit quadratic spline to points on CDF at computed grid.
        # Although "the user is strongly dissuaded from choosing k
        # even, together with a small s-value" (see 'curfit.f' file in
        # 'scipy/interpolate/fitpack'), this currently proved to be working
        # quite good.
        cdf_spline = UnivariateSpline(x=x, y=rv.cdf(x), k=2, s=cdf_tolerance ** 2)

        # Construct xy-grid as knots and values of derivative spline
        x, y = _xy_from_cdf_spline(cdf_spline)

        return cls(x, y)

    @classmethod
    @_docstring_relevant_options(
        ["cdf_tolerance", "density_mincoverage", "estimator_cont", "n_grid"]
    )
    def from_sample(cls, sample):
        """Create continuous RV from sample

        Continuous RV with piecewise-linear density is created by the following
        algorithm:
        - **Estimate distribution** with continuous estimator (taken from package
          option "estimator_cont") in the form `estimate =
          estimator_cont(sample)`. If `estimate` is an object of class `Rand`
          or `scipy.stats.distributions.rv_frozen` (`rv_continuous` with all
          hyperparameters defined), it is forwarded to `Cont.from_rv()`.
          Otherwise it should represent density.
        - **Estimate effective range of density**: interval inside which total
          integral of density is not less than `density_mincoverage` (package
          option). Specific algorithm how it is done is subject to change.
          General description of the current one:
            - Make educated guess about initial range (usually with input
              sample range).
            - Iteratively extend range in both directions until density total
              integral is above desired threshold.
        - **Create x-grid**. It is computed as union of equidistant (fixed
          distance between consecutive points inside detected density range)
          and equiprobable (fixed probability between consecutive points based
          on sample quantiles) grids. Number of points in grids is equal to
          `n_grid` (package option). Also it is ensured that no points lie very
          close to each other (order of `1e-12` distance), because otherwise
          output density will have unstable values.
        - **Approximate CDF at x-grid**. X-grid is taken from previous step.
          Using values of density estimate at x-grid, CDF values are computed
          as cumulative integrals (normalized for the widest one to be equal to
          one) of piecewise-linear density defined by xy-grid. This is a fast
          approximation to integrals of density estimate from minus infinity to
          values of x-grid.
        - **Fit quadratic spline to cdf-grid**. This dramatically reduces
          number of points in output `Cont`'s xy-grid in exchange of (usually)
          small inaccuracy. **Note** that it is possible to fit linear spline
          to density xy-grid, however approximating CDF: showed better
          reduction of output grid size while introducing adequate inaccuracy
          to CDF function; is consistent with `Cont.from_rv()` and utilizes the
          same `cdf_tolerance` package option. Spline is fitted using
          `scipy.interpolate.UnivariateSpline` with the following arguments:
            - `x` is equal to x-grid from previous step.
            - `y` is equal to CDF values at x-grid.
            - `k` (spline degree) is given as 2 for output to represent
              quadratic spline.
            - `s` (smoothing factor) is taken as square of `cdf_tolerance`
              package option (as `s` represents mean squared approximation
              error). Bigger values of `cdf_tolerance` lead to smaller number
              of elements in output `Cont`'s xy-grid. Smaller values (together
              with large enough values of `n_grid` option) lead to better
              approximation of random variable with estimated density.
        - **Create density xy-grid**. Density is estimated as spline derivative
          of quadratic spline from previous step. X-grid is taken as knots of
          "density spline". Y-grid is computed as values of "density spline" at
          those knots truncating possible negative values to become zero. Here
          negative values can occur if CDF approximation is allowed to be loose
          (either `n_grid` is low or `cdf_tolerance` is high).

        {relevant_options}

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
        cdf_tolerance = options.cdf_tolerance
        density_mincoverage = options.density_mincoverage
        estimator_cont = options.estimator_cont
        n_grid = options.n_grid

        # Estimate distribution
        estimate = estimator_cont(sample)

        # Make early return if `estimate` is random variable
        if isinstance(estimate, (Rand, rv_frozen)):
            return Cont.from_rv(estimate)

        # Estimate density range
        density_range = _estimate_density_range(estimate, sample, density_mincoverage)

        # Compute combination of equidistant and quantile grids
        x_grid = _compute_union_grid(
            x_range=density_range,
            prob_range=(0, 1),
            quantile_fun=lambda q: np.quantile(a=sample, q=q, interpolation="linear"),
            n_grid=n_grid,
        )
        y_grid = estimate(x_grid)

        # Fit quadratic spline to cdf-grid. This uses the same approach as `from_rv()`
        # and results into considerably fewer xy-grid elements at the cost of
        # some accuracy loss in terms of actual density approximation.
        # Although "the user is strongly dissuaded from choosing k
        # even, together with a small s-value" (see 'curfit.f' file in
        # 'scipy/interpolate/fitpack'), this currently proved to be working
        # quite good.
        cdf_grid = utils._trapez_integral_cum(x_grid, y_grid)
        cdf_grid = cdf_grid / cdf_grid[-1]
        cdf_spline = UnivariateSpline(x=x_grid, y=cdf_grid, k=2, s=cdf_tolerance ** 2)

        # Construct xy-grid as knots and values of derivative spline
        x, y = _xy_from_cdf_spline(cdf_spline)

        return cls(x, y)

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
        x = np.asarray(x, dtype="float64")

        # Using `np.asarray()` to ensure ndarray output in case of `x`
        # originally was scalar
        return np.asarray(
            np.interp(x, self._x, self._y, left=0, right=0), dtype="float64"
        )

    # `logpdf()` is inherited from `Rand`

    def pmf(self, x):
        raise AttributeError(
            "`Cont` doesn't have probability mass function. Use `pdf()` instead."
        )

    def logpmf(self, x):
        raise AttributeError(
            "`Cont` doesn't have probability mass function. Use `logpdf()` instead."
        )

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
        res = np.zeros_like(x, dtype="float64")

        x_ind = utils._searchsorted_wrap(self._x, x, side="right", edge_inside=True)
        ind_is_good = (x_ind > 0) & (x_ind < len(self._x))

        if np.any(ind_is_good):
            x_good = x[ind_is_good]
            x_ind_good = x_ind[ind_is_good]

            gr_x, gr_y, gr_cump = self._grid_by_ind(x_ind_good)
            _, slope = self._coeffs_by_ind(x_ind_good)

            res[ind_is_good] = (
                gr_cump + gr_y * (x_good - gr_x) + 0.5 * slope * (x_good - gr_x) ** 2
            )

        # `res` is already initialized with zeros, so taking care of `x` to the
        # left of support is not necessary
        res[x_ind == len(self._x)] = 1.0

        return np.asarray(utils._copy_nan(fr=x, to=res), dtype="float64")

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
        q = np.asarray(q, dtype="float64")
        res = np.zeros_like(q, dtype="float64")

        # Using `side="left"` is crucial to return the smallest value in case
        # there are more than one. For example, when there are zero-density
        # intervals (which will result into consecutive duplicated values of
        # `_cump`).
        # Using `edge_inside=True` is crucial in order to treat the left edge
        # of support as part of support.
        q_ind = utils._searchsorted_wrap(self._cump, q, side="left", edge_inside=True)
        ind_is_good = (q_ind > 0) & (q_ind < len(self._cump)) & (q != 0.0) & (q != 1.0)

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
        # application. In some extreme cases last `_cump` can be smaller than
        # 1 by value of 10**(-16) magnitude, which will result into "bad" value
        # of `q_ind` (that is why this should also be done after assigning
        # `nan` to "bad" values)
        res[q == 0.0] = self._a
        res[q == 1.0] = self._b

        return np.asarray(res, dtype="float64")

    # `isf()` is inherited from `Rand`

    # `rvs()` is inherited from `Rand`

    def _find_quant(self, q, grid, coeffs):
        """Compute quantiles with data from linearity intervals

        Based on precomputed data of linearity intervals, compute actual quantiles.
        Here `grid` and `coeffs` are `(x, y, cump)` and `(inter, slope)` values
        of intervals inside which `q` quantile is located.

        Parameters
        ----------
        q : numpy array
        grid : tuple with 3 numpy arrays with lengths same to `len(q)`
        coeffs : tuple with 3 numpy arrays with lengths same to `len(q)`

        Returns
        -------
        quant : numpy array with the same length as q
        """
        res = np.empty_like(q, dtype="float64")
        x, y, cump = grid
        _, slope = coeffs

        is_quad = ~utils._is_zero(slope)
        is_lin = ~(is_quad | utils._is_zero(y))
        is_const = ~(is_quad | is_lin)

        # Case of quadratic CDF curve (density is a line not aligned to x axis)
        # The "true" quadratic curves are transformed in terms of `t = x - x_l`
        # for numerical accuracy
        # Equations have form a*t^2 + b*t + c = 0
        a = 0.5 * slope[is_quad]
        b = y[is_quad]
        c = cump[is_quad] - q[is_quad]
        # Theoretically, `discr` should always be >= 0. However, due to
        # numerical inaccuracies of magnitude ~10^(-15), here call to
        # `np.clip()` is needed.
        discr = np.clip(b * b - 4 * a * c, 0, None)
        res[is_quad] = x[is_quad] + (-b + np.sqrt(discr)) / (2 * a)

        # Case of linear CDF curve (density is non-zero constant)
        res[is_lin] = x[is_lin] + (q[is_lin] - cump[is_lin]) / y[is_lin]

        # Case of plateau in CDF (density equals zero)
        res[is_const] = x[is_const]

        return res

    @property
    def _cdf_spline(self):
        density_tck = (
            np.concatenate(([self._x[0]], self._x, [self._x[-1]])),
            np.concatenate((self._y, [0, 0])),
            1,
        )
        cdf_tck = splantider(density_tck)
        return utils.BSplineConstExtrapolate(
            left=0, right=1, t=cdf_tck[0], c=cdf_tck[1], k=cdf_tck[2]
        )

    def integrate_cdf(self, a, b):
        """Efficient version of CDF integration"""
        return self._cdf_spline.integrate(a=a, b=b)

    def convert(self, to_class=None):
        """Convert to different RV class

        Conversion is done by the following logic depending on the value of
        `to_class`:
        - If it is `None` or `"Cont"`, `self` is returned.
        - If it is `"Bool"`, boolean RV is returned with probability of `False`
          equal to 0. That is because, following general Python agreement, the
          only numerical value converted to `False` is zero, which probability
          is exactly 0 (probability of continuous RV being exactly to some
          number is always 0).
        - If it is `"Disc"`, discrete RV is returned. Its xp-grid is computed
          by the following algorithm:
            - X-grid is taken the same as x-grid of `self`.
            - P-grid is computed so that input continuous RV is a maximum
              likelihood estimation of output discrete RV. This approach is
              taken to be inverse of discrete-to-continuous conversion.
        - If it is `"Mixt"`, mixture RV with only continuous component equal to
          `self` is returned.

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
            import randomvars._boolean as bool

            # Probability of `True` is a probability of all non-zero elements,
            # which is 1 because probability of continuous RV getting exactly
            # zero is 0.
            return bool.Bool(prob_true=1)
        elif (to_class == "Cont") or (to_class is None):
            return self
        elif to_class == "Disc":
            import randomvars._discrete as disc

            # Convert xy-grid to xp-grid
            p = utilsgrid._p_from_xy(x=self._x, y=self._y)
            return disc.Disc(x=self._x, p=p)
        elif to_class == "Mixt":
            import randomvars._mixture as mixt

            # Output is a degenerate mixture with only continuous component
            return mixt.Mixt(disc=None, cont=self, weight_cont=1.0)
        else:
            raise ValueError(
                '`metric` should be one of "Bool", "Cont", "Disc", or "Mixt".'
            )


def _detect_finite_supp(rv, supp=None, small_prob=1e-6):
    """Detect finite support of random variable

    Finite support edge is detected via testing extreme quantiles (0 for left
    and 1 for right) to be finite:
    - If finite, output is a value which has extreme value of CDF and is
      "closest to positive probability region". This accounts for a possibility
      when quantile function returns unnecessarily extreme values at extreme
      quantiles, i.e. it has "zero tails".
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
        left_try = rv.ppf(np.array([0.0, small_prob]))
        if np.isneginf(left_try[0]):
            left = left_try[1]
        else:
            left = utils._collapse_while_equal_fval(f=rv.cdf, interval=left_try, side=0)
    else:
        left = supp[0]

    if supp[1] is None:
        right_try = rv.ppf(np.array([1.0 - small_prob, 1.0]))
        if np.isposinf(right_try[1]):
            right = right_try[0]
        else:
            right = utils._collapse_while_equal_fval(
                f=rv.cdf, interval=right_try, side=1
            )
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
    x_left, x_right = utils._minmax(sample)

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
    # `utils._quad_silent(density, res_range[0], res_range[1])` is crucial
    # because it may lead to inaccurate results with wide range.
    cov_prob_left = utils._quad_silent(density, res_range[0], x_range[0])
    cov_prob_right = utils._quad_silent(density, x_range[1], res_range[1])
    res_cov_prob = cov_prob + cov_prob_left + cov_prob_right

    return res_range, res_cov_prob


def _compute_union_grid(x_range, prob_range, quantile_fun, n_grid, tol=1e-12):
    # Equidistant grid
    x_equi = np.linspace(x_range[0], x_range[1], n_grid)

    # Equiprobable grid
    ## Don't use first and last values in this grid because:
    ## - It helps overcome issue with `extra zero tails`. When `quantile_fun`
    ##   returns unnecessarily extreme values because it was constructed
    ##   assuming zero tails. Like, for example, with xy-grid `([0, 1, 2, 3,
    ##   4], [0, 0, 1, 0, 0])`.
    ## - In case of usual usage of `Cont.from_rv` it right away introduces
    ##   duplicated points, as `prob_range` is `cdf(x_range)`.
    prob_equi = np.linspace(prob_range[0], prob_range[1], n_grid)[1:-1]
    x_quan = quantile_fun(prob_equi)

    # Raw union grid
    x = np.union1d(x_equi, x_quan)

    ## Ensure that minimum difference between consecutive elements isn't
    ## very small, otherwise this will make `np.gradient()` perform poorly
    x_is_good = np.concatenate([[True], np.ediff1d(x) > tol])

    return x[x_is_good]


def _xy_from_cdf_spline(cdf_spline):
    spline_deriv = cdf_spline.derivative()

    x = spline_deriv.get_knots()
    # Here `y` can have negative values (and not so small ones in case of small
    # `n_grid` or big `cdf_tolerance`)
    y = np.clip(spline_deriv(x), 0, None)
    # Renormalize xy-grid to have integral 1
    y = y / utils._trapez_integral(x, y)

    return x, y

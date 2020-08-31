""" Code for random variable with piecewise-linear density
"""
import warnings

import numpy as np
from scipy.stats.distributions import rv_continuous

from randomvars.downgrid_maxtol import downgrid_maxtol
from randomvars.options import get_option


def _searchsorted_wrap(a, v, side="left", edge_inside=True):
    """Wrapper for `np.searchsorted()` which respects `np.nan`

    Output index for every `np.nan` value in `v` is `-1`.

    Parameters
    ----------
    edge.inside: boolean, optional
        Should the most extreme edge (left for `side="left"`, right for
        `side="right"`) be treated as inside its adjacent interior interval?
        Default is `True`.

    Examples
    --------
    >>> _searchsorted_wrap([0, 1], [-np.inf, -1, 0, 0.5, 1, 2, np.inf, np.nan])
    array([ 0,  0,  1,  1,  1,  2,  2, -1])
    >>> _searchsorted_wrap([0, 1], [0, 1], side="right", edge_inside=False)
    array([1, 2])
    """
    # Here call to `np.asarray()` is needed to ensure that output is not a scalar
    # (which can happen in certain cases)
    res = np.asarray(np.searchsorted(a, v, side=side))
    a = np.asarray(a)
    v = np.asarray(v)

    if edge_inside:
        if side == "right":
            # For right-most edge return coefficients from last interval
            res[v == a[-1]] = len(a) - 1
        elif side == "left":
            # For left-most edge return coefficients from first interval
            res[v == a[0]] = 1

    # Return `-1` index if element of `v` is `np.nan`
    res[np.isnan(v)] = -1

    return res


class rv_piecelin(rv_continuous):
    """Random variable with piecewise-linear density"""

    def __init__(self, x, y, *args, **kwargs):
        x, y = self._impute_xy(x, y)

        self._x = x
        self._y = y
        self._p = _trapez_integral_cum(self._x, self._y)

        # Set support
        kwargs["a"] = self.a = self._x[0]
        kwargs["b"] = self.b = self._x[-1]

        super(rv_piecelin, self).__init__(*args, **kwargs)

    @staticmethod
    def _impute_xy(x, y):
        try:
            x = np.asarray_chkfinite(x, dtype=np.float64)
        except:
            raise ValueError(
                "`x` is not convertable to numeric numpy array with finite values."
            )
        if len(x.shape) > 1:
            raise ValueError("`x` is not a 1d array.")

        try:
            y = np.asarray_chkfinite(y, dtype=np.float64)
        except:
            raise ValueError(
                "`y` is not convertable to numeric numpy array with finite values."
            )
        if len(y.shape) > 1:
            raise ValueError("`y` is not a 1d array.")

        if len(x) != len(y):
            raise ValueError("Lengths of `x` and `y` do not match.")

        if not np.all(np.diff(x) > 0):
            warnings.warn(
                "`x` is not sorted. `x` and `y` are rearranged so as `x` is sorted."
            )
            x_argsort = np.argsort(x)
            x = x[x_argsort]
            y = y[x_argsort]

        if np.any(y < 0):
            raise ValueError("`y` has negative values.")
        if not np.any(y > 0):
            raise ValueError("`y` has no positive values.")

        y = y / _trapez_integral(x, y)

        return x, y

    @property
    def x(self):
        """Return `x` component of piecewise-linear density"""
        return self._x

    @property
    def y(self):
        """Return `y` component of piecewise-linear density"""
        return self._y

    @property
    def p(self):
        """Return cumulative probability grid of piecewise-linear density"""
        return self._p

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
        >>> rv = rv_piecelin([0, 1, 2], [0, 1, 0])
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
        >>> rv = rv_piecelin([0, 1, 2], [0, 1, 0])
        >>> x, y, p = rv._grid_by_ind(np.array([-1, 0, 1, 2, 3, 4]))
        >>> x
        array([nan, nan,  0.,  1.,  2., nan])
        >>> x, y, p = rv._grid_by_ind()
        >>> p
        array([0. , 0.5, 1. ])
        """
        if ind is None:
            return (self._x, self._y, self._p)

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
        p[out_isnt_nan] = self._p[ind_in]

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
        >>> rv_p = rv_piecelin([0, 1, 2], [0, 1, 0])
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

        ind = _searchsorted_wrap(self._x, x, side=side, edge_inside=True)

        return self._coeffs_by_ind(ind)

    @classmethod
    def from_rv(cls, rv, supp=None, tail_prob=None, n_grid=None, integr_tol=None):
        """Create piecewise-linear RV from general RV

        Piecewise-linear RV is created by the following algorithm:
        - **Detect finite support**. Left and right edges are treated
          separately. If edge is supplied, it is used untouched. If not, it is
          computed by "removing" corresponding (left or right) tail which has
          probability of `tail_prob` (package option).
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

        Relevant package options: `n_grid`, `tail_prob`, `integr_tol`. See
        documentation of `randomvars.options.get_option()` for more
        infromation. To temporarily set options use
        `randomvars.options.option_context()` context manager.

        Parameters
        ----------
        rv : rv_frozen
            Object of class `rv_continuous` with all hyperparameters defined.
        supp : Tuple with two numbers or `None`, optional
            Forced support edges. Elements should be either finite numbers
            (returned untouched) or `None` (finite support edge is detected).
            Single `None` is equivalent to `(None, None)`, i.e. finding both edges
            of finite support.
        tail_prob : float, optional
            Probability value of tail that might be cutoff in order to get finite
            support.
        n_grid : int, optional
            Number of points in initial equidistant and quantile xy-grids, by
            default 1001.
        integr_tol : float, optional
            Integral tolerance for maximum tolerance downgridding, by default 1e-4.

        Returns
        -------
        rv_out : rv_piecelin
            Random variable with finite support and piecewise-linear density
            which approximates density of input `rv`.
        """
        # Ensure settings are set
        if n_grid is None:
            n_grid = get_option("n_grid")
        if tail_prob is None:
            tail_prob = get_option("tail_prob")
        if integr_tol is None:
            integr_tol = get_option("integr_tol")

        # Detect effective support of `rv`
        x_left, x_right = _detect_finite_supp(rv, supp, tail_prob)

        # Construct equidistant grid
        x_equi = np.linspace(x_left, x_right, n_grid)

        # Construct quantile grid
        prob_left, prob_right = rv.cdf([x_left, x_right])
        prob_equi = np.linspace(prob_left, prob_right, n_grid)
        x_quan = rv.ppf(prob_equi)

        # Combine equidistant and quantile grids into one sorted array
        x = np.union1d(x_equi, x_quan)
        ## Ensure that minimum difference between consecutive elements isn't
        ## very small, otherwise this will make `np.gradient()` perform poorly
        x_is_good = np.concatenate([[True], np.ediff1d(x) > 1e-13])
        x = x[x_is_good]

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

    def _pdf(self, x, *args):
        """Implementation of probability density function"""
        return np.interp(x, self._x, self._y, left=0, right=0)

    def _cdf(self, x, *args):
        """Implementation of cumulative distribution function

        Notes
        -----
        Dealing with `x` values outside of support is supposed to be done in
        `rv_continuous`.
        """
        x = np.asarray(x, dtype=np.float64)
        x_ind = _searchsorted_wrap(self._x, x, side="right", edge_inside=True)

        gr_x, _, gr_p = self._grid_by_ind(x_ind)
        inter, slope = self._coeffs_by_ind(x_ind)

        # Using `(a+b)*(a-b)` instead of `(a*a-b*b)` for better accuracy in
        # case density x-grid has really close elements
        return gr_p + inter * (x - gr_x) + 0.5 * slope * (x + gr_x) * (x - gr_x)

    def _ppf(self, q, *args):
        """Implementation of Percent point function

        Notes
        -----
        Dealing with `q` values outside of `[0; 1]` is supposed to be done in
        `rv_continuous`.
        """
        q_ind = _searchsorted_wrap(self._p, q, side="right", edge_inside=True)
        grid_q = self._grid_by_ind(q_ind)
        coeffs_q = self._coeffs_by_ind(q_ind)

        return self._find_quant(q, grid_q, coeffs_q)

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


def _trapez_integral(x, y):
    """Compute integral with trapezoidal formula.

    >>> _trapez_integral(np.array([0, 1]), np.array([1, 1]))
    1.0
    >>> _trapez_integral(np.array([-1, 0, 1]), np.array([0, 10, 5]))
    12.5
    """
    return np.sum(0.5 * np.diff(x) * (y[:-1] + y[1:]))


def _trapez_integral_cum(x, y):
    """Compute cumulative integral with trapezoidal formula.

    Element of output represents cumulative probability **before** its left "x"
    edge.

    >>> _trapez_integral_cum(np.array([0, 1]), np.array([1, 1]))
    array([0., 1.])
    >>> _trapez_integral_cum(np.array([-1, 0, 1]), np.array([0, 10, 5]))
    array([ 0. ,  5. , 12.5])
    """
    res = np.cumsum(0.5 * np.diff(x) * (y[:-1] + y[1:]))
    return np.concatenate([[0], res])


def _detect_finite_supp(rv, supp=None, tail_prob=1e-6):
    """Detect finite support of random variable

    Finite support edge is detected via testing actual edge to be finite:
    - If finite, it is returned.
    - If infinite, output is a value with outer tail having probability
      `tail_prob`.

    Parameters
    ----------
    rv : Random variable from 'scipy' module.
    supp : Tuple with two elements or `None`, optional
        Forced support edges. Elements should be either finite numbers
        (returned untouched) or `None` (finite support edge is detected).
        Single `None` is equivalent to `(None, None)`, i.e. finding both edges
        of finite support.
    tail_prob : Tail probability, optional
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
            left = rv.ppf(tail_prob)
    else:
        left = supp[0]

    if supp[1] is None:
        right = rv.ppf(1)
        if np.isposinf(right):
            right = rv.ppf(1 - tail_prob)
    else:
        right = supp[1]

    return left, right

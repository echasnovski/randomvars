""" Code for random variable with piecewise-linear density
"""
import warnings

import numpy as np
from scipy.stats.distributions import rv_continuous


def _searchsorted_wrap(a, v, side="left", edge_inside=True):
    """Wrapper for `np.searchsorted()` which respects `np.nan`

    Output index for every `np.nan` value in `v` is `-1`.

    Parameters
    ----------
    edge.inside: Boolean, optional
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
    """ Random variable with piecewise-linear density
    """

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
        x : numpy numeric array.
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

    def _pdf(self, x, *args):
        """Implementation of probability density function
        """
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
        q : numpy numeric array
        grid : tuple with 3 numpy numeric arrays with lengths same to `len(q)`
        coeffs : tuple with 3 numpy numeric arrays with lengths same to `len(q)`

        Returns
        -------
        quant : numpy numeric array with the same length as q
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
    """ Compute integral with trapezoidal formula.

    >>> _trapez_integral(np.array([0, 1]), np.array([1, 1]))
    1.0
    >>> _trapez_integral(np.array([-1, 0, 1]), np.array([0, 10, 5]))
    12.5
    """
    return np.sum(0.5 * np.diff(x) * (y[:-1] + y[1:]))


def _trapez_integral_cum(x, y):
    """ Compute cumulative integral with trapezoidal formula.

    Element of output represents cumulative probability **before** its left "x"
    edge.

    >>> _trapez_integral_cum(np.array([0, 1]), np.array([1, 1]))
    array([0., 1.])
    >>> _trapez_integral_cum(np.array([-1, 0, 1]), np.array([0, 10, 5]))
    array([ 0. ,  5. , 12.5])
    """
    res = np.cumsum(0.5 * np.diff(x) * (y[:-1] + y[1:]))
    return np.concatenate([[0], res])


if __name__ == "__main__":
    import doctest

    doctest.testmod()

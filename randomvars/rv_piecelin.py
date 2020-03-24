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
    res = np.searchsorted(a, v, side=side)
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
        self._cumprob = _trapez_integral_cum(self._x, self._y)

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

    def pdf_grid(self):
        """Get density grid

        Returns
        -------
        grid : tuple with 2 numeric arrays of same length.
            Two components are `x` and `y` arrays defining piecewise-linear density.
        """
        return (self._x, self._y)

    def _coeffs_by_ind(self, ind):
        """Compute density linear coefficients based on index of interval.

        Index `i` corresponds to coefficients from interval with endpoints
        `self._x[i-1]` and `self._x[i]`. Which intervals `self._x` values
        represent should be decided before calling this function during
        computation of `ind`.
        Indexes 0 and `len(self._x)` result in zero coefficients. Indexes
        outside `[0, len(self._x)]` result into `np.nan` coefficients.

        Parameters
        ----------
        ind : numpy integer.
            Describes index of interval, coefficients of which should be returned.

        Returns
        -------
        coeffs : tuple with 2 float arrays with lengths equal to length of
        `ind`.
            First element represents intercept, second - slope.

        Examples
        --------
        >>> rv_p = rv_piecelin([0, 1, 2], [0, 1, 0])
        >>> rv_p._coeffs_by_ind(np.array([0, 1, 2, 3]))
        (array([0., 0., 2., 0.]), array([ 0.,  1., -1.,  0.]))
        >>> rv_p._coeffs_by_ind(np.array([-1, 100]))
        (array([nan, nan]), array([nan, nan]))
        """
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
        """ Implementation of probability density function
        """
        return np.interp(x, self._x, self._y)


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

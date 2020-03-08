""" Code for random variable with piecewise-linear density
"""
import numpy as np
from scipy.stats.distributions import rv_continuous


class rv_piecelin(rv_continuous):
    """ Random variable with piecewise-linear density
    """

    def __init__(self, density, *args, **kwargs):
        if len(density) != 2:
            raise ValueError("Expected length 2 for parameter `den`")
        if len(density[0]) != len(density[1]):
            raise ValueError(
                "Number of elements of `x` and `y` components do not match"
            )

        self._x = np.asarray(density[0])
        self._y = np.asarray(density[1]) / trapez_integral(density[0], density[1])

        # Set support
        kwargs["a"] = self.a = self._x[0]
        kwargs["b"] = self.b = self._x[-1]

        super(rv_piecelin, self).__init__(*args, **kwargs)

    def _pdf(self, x, *args):
        return np.interp(x, self._x, self._y)


def trapez_integral(x, y):
    """ Compute integral with trapezoidal formula
    """
    return 0.5 * np.diff(x) * (y[:-1] + y[1:])

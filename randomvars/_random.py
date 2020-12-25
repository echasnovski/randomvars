""" Code for general random variable
"""
import numpy as np


class Rand:
    """General random variable

    This class implements methods common for all random variables in this
    package. It is intended to be only used for subclassing. All random
    variable classes in this package inherit from this class.
    """

    @property
    def a(self):
        """Return left edge of support"""
        raise NotImplementedError

    @property
    def b(self):
        """Return right edge of support"""
        raise NotImplementedError

    def support(self):
        """Return support of random variable"""
        return (self.a, self.b)

    @classmethod
    def from_rv(cls, rv):
        """Create RV from different RV"""
        if isinstance(rv, Rand):
            return rv

        raise NotImplementedError

    @classmethod
    def from_sample(cls, sample):
        """Create RV from sample"""
        raise NotImplementedError

    def pdf(self, x):
        """Probability density function"""
        raise NotImplementedError

    def logpdf(self, x):
        """Logarithm of probability density function"""
        # Do not throw warning if pdf at `x` is zero
        with np.errstate(divide="ignore"):
            # Using `np.asarray()` to ensure ndarray output in case of `x`
            # originally was scalar
            return np.asarray(np.log(self.pdf(x)))

    def pmf(self, x):
        """Probability mass function"""
        raise NotImplementedError

    def logpmf(self, x):
        """Logarithm of probability mass function"""
        # Do not throw warning if pmf at `x` is zero
        with np.errstate(divide="ignore"):
            # Using `np.asarray()` to ensure ndarray output in case of `x`
            # originally was scalar
            return np.asarray(np.log(self.pmf(x)))

    def cdf(self, x):
        """Cumulative distribution function"""
        raise NotImplementedError

    def logcdf(self, x):
        """Logarithm of cumulative distribution function"""
        # Do not throw warning if cdf at `x` is zero
        with np.errstate(divide="ignore"):
            # Using `np.asarray()` to ensure ndarray output in case of `x`
            # originally was scalar
            return np.asarray(np.log(self.cdf(x)))

    def sf(self, x):
        """Survival function"""
        return np.asarray(1.0 - self.cdf(x))

    def logsf(self, x):
        """Logarithm of survival function"""
        # Do not throw warning if cdf at `x` is zero
        with np.errstate(divide="ignore"):
            # Using `np.asarray()` to ensure ndarray output in case of `x`
            # originally was scalar
            return np.asarray(np.log(self.sf(x)))

    def ppf(self, q):
        """Percent point (quantile, inverse of cdf) function"""
        raise NotImplementedError

    def rvs(self, size=None, random_state=None):
        """Random number generation

        Generate random numbers into array of desired size.

        Parameters
        ----------
        size : int or tuple of ints, optional
            Defining number of random variates (default is one output). Integer
            is treated as tuple with single element. `None` and empty tuple
            result into numpy array with zero dimensions.
        random_state : `None`, int, or RandomState, optional
            Source of uniform random number generator. If `None`, it is
            initiated as `numpy.random.RandomState()`. If integer,
            `numpy.random.RandomState(seed=random_state)` is used.

        Returns
        -------
        smpl : ndarray with shape defined in `size`
        """
        if random_state is None:
            random_state = np.random.RandomState()
        elif isinstance(random_state, int):
            random_state = np.random.RandomState(seed=random_state)

        U = random_state.uniform(size=size)

        return self.ppf(U)

    def integrate_cdf(self, a, b):
        """Efficient version of CDF integration"""
        raise NotImplementedError

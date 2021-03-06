""" Code for general random variable
"""
import numpy as np


class Rand:
    """General random variable

    This class implements methods common for all random variables in this
    package. It is intended to be only used for subclassing. All random
    variable classes in this package inherit from this class.
    """

    def __eq__(self, other):
        if type(self) is not type(other):
            return False

        # Guard for possible absence of `params` property and if they are not
        # dictionary
        try:
            self_params = self.params
            other_params = other.params

            if self_params.keys() != other_params.keys():
                return False
        except AttributeError:
            return False

        return all(
            np.all(self_params[key] == other_params[key]) for key in self_params.keys()
        )

    @property
    def params(self):
        """Parameters of random variable

        Dictionary of parameters which completely describes this object. Its
        structure should reflect input arguments of `__init__()` method. As a
        rule of thumb, it should be enough to create a copy of this object via:
        `type(self)(**self.params)`.

        Its primary usage is to check equality of two RVs via `rv1 == rv2`:
        they are equal, if they are of the same class and have the same
        parameters (checked per element with `np.all(element1 == element2)`).
        """
        return dict()

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

    def compress(self):
        """Compress random variable

        Here the meaning of "compress" is to return a random variable (possibly
        of different class) which numerically has the same CDF values and uses
        minimum amount of parameters.

        Returns
        -------
        rv_compressed : self
            This class of random variable already has the minimum amount of
            parameters to achieve its CDF, so self is returned.
        """
        return self

    @classmethod
    def from_rv(cls, rv):
        """Create RV from different RV"""
        if isinstance(rv, Rand):
            return rv.convert("Rand")

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

    def isf(self, q):
        """Inverse survival function"""
        return np.asarray(self.ppf(1.0 - np.asarray(q)))

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

    def convert(self, to_class=None):
        """Convert to different RV class"""
        if (to_class == "Rand") or (to_class is None):
            return self

        raise NotImplementedError

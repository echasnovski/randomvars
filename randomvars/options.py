import textwrap
import warnings

import numpy as np
from scipy.stats.kde import gaussian_kde


__all__ = [
    "OptionError",
    "config",
    "estimator_bool_default",
    "estimator_cont_default",
    "estimator_disc_default",
    "estimator_mixt_default",
]


# %% Default estimators
def estimator_bool_default(sample):
    """Default estimator of boolean distribution

    This estimator returns proportion of `True` values after converting input
    to boolean Numpy array.

    Parameters
    ----------
    sample : array_like
        This should be a valid input to `np.asarray()` so that its output is
        boolean.

    Returns
    -------
    prob_true : number
    """
    sample = np.asarray(sample, dtype="bool")
    return np.mean(sample)


def estimator_cont_default(sample):
    """Default estimator of continuous distribution

    This estimator is a direct wrapper of scipy.stats.kde.gaussian_kde.
    """
    if len(sample) < 2:
        raise ValueError(
            "Sample should have at least two elements for `gaussian_kde` to work."
        )
    return gaussian_kde(sample)


def estimator_disc_default(sample):
    """Default estimator of discrete distribution

    This estimator returns unique values of input as distributions values.
    Their probabilities are proportional to number of their occurrences in input.

    Parameters
    ----------
    sample : array_like
        This should be a valid input to `np.asarray()` so that its output is
        numeric.

    Returns
    -------
    x, p : tuple with two elements
        Here `x` represents estimated values of distribution and `p` -
        estimated probabilities.
    """
    sample = np.asarray(sample, dtype="float64")

    sample_is_finite = np.isfinite(sample)
    if not np.all(sample_is_finite):
        if not np.any(sample_is_finite):
            raise ValueError(
                "Input sample in discrete estimator doesn't have finite values."
            )
        else:
            warnings.warn("Input sample in discrete estimator has non-finite values.")
            sample = sample[sample_is_finite]

    vals, counts = np.unique(sample, return_counts=True)
    return vals, counts / np.sum(counts)


def estimator_mixt_default(sample):
    """Default estimator of mixture distribution

    This estimator returns tuple with two non-overlaping parts of `sample`
    which are estimated to come from continuous and discrete parts of mixture
    distribution. Estimation is done by deciding sample element to be from
    discrete part if it is present at least twice in input `sample`.

    If some part of estimation has no elements, it is represented as `None` in
    output.

    Parameters
    ----------
    sample : array_like
        This should be a valid input to `np.asarray()` so that its output is
        numeric.

    Returns
    -------
    sample_cont, sample_disc : tuple with two elements
        Elements can be `None` if estimation showed no elements from
        corresponding mixture part.
    """
    # Detect sample from discrete part
    sample = np.asarray(sample)
    vals, inverse, counts = np.unique(sample, return_inverse=True, return_counts=True)
    disc_inds = np.nonzero(counts >= 2)[0]
    sample_is_disc = np.isin(inverse, disc_inds)

    # Return separation
    if np.all(sample_is_disc):
        return (None, sample)
    elif np.all(~sample_is_disc):
        return (sample, None)
    else:
        return (sample[~sample_is_disc], sample[sample_is_disc])


# %% Configuration
class OptionError(KeyError):
    """
    Exception describing error in interaction with package options.
    """


class _Option:
    def __init__(self, default, validator):
        self.default = default
        self.option = default
        self.validator_f, self.validator_str = validator

    def __set_name__(self, owner, name):
        self.name = name

    def __get__(self, obj, objtype=None):
        return self.option

    def __set__(self, obj, value):
        try:
            value_is_valid = self.validator_f(value)
        except BaseException as e:
            raise OptionError(
                "There was an error verifying validity of value for "
                f"`{self.name}`: {str(e)}"
            )

        if not value_is_valid:
            raise OptionError(f"`{self.name}` should be {self.validator_str}")

        self.option = value


_validator_nonneg = (lambda x: isinstance(x, float) and x >= 0, "a non-negative float")
_validator_callable = (lambda x: callable(x), "a callable")


class _Config:
    """Package configuration

    List of available options:
    - base_tolerance : float, default 1e-12. Tolerance to be used for testing
      approximate equality of two numbers. It is used to compute tolerance
      associated with any number `x`:
        - If `abs(x) <= 1`, tolerance is equal to `base_tolerance`. Based on
          this, `base_tolerance` can be viewed as an absolute tolerance for
          "small" numbers.
        - If `abs(x) > 1`, tolerance increases proportionally to the spacing
          between floating point numbers at `x` (see `numpy.spacing()`). This
          approach is chosen in order to find compromise between relative and
          absolute tolerance.
    - cdf_tolerance: float, default 1e-4. Tolerance for CDF approximation.
      Usually meant as mean approximation error. Smaller values lead to better
      approximation, larger values lead to less number of grid elements (knots)
      in output approximation. However, using large values (bigger than 0.01)
      is discouraged because this might lead to unexpected properties of
      approximation (like increasing density in tails where it should
      originally decrease, etc.).
    - density_mincoverage : float, default 0.9999. Minimum value of integral
      within output of density range estimation.
    - estimator_bool : callable, default
      randomvars.options.estimator_bool_default. Estimator for
      `Bool.from_sample()`. Function which takes sample as input and returns
      one of:
        - Number representing probability of `True` for boolean random
          variable.
        - Object of class `Rand` or `rv_frozen` (`rv_discrete` with all
          hyperparameters defined).
    - estimator_cont : callable, default
      randomvars.options.estimator_cont_default.  Estimator for
      `Cont.from_sample()`. Function which takes sample as input and returns
      one of:
        - Callable object for density estimate (takes points as input and
          returns density values).
        - Object of class `Rand` or `rv_frozen` (`rv_continuous` with all
          hyperparameters defined).
      **Notes**:
        - Theoretical integral of density over whole real line should be 1.
        - Output density callable should be vectorized: allow numpy array as
          input and return numpy array with the same length.
        - There is worse performance if output density callable has
          discontinuity.
    - estimator_disc : callable, default
      randomvars.options.estimator_disc_default. Estimator for
      `Disc.from_sample()`. Function which takes sample as input and returns
      one of:
        - Tuple with two elements representing `x` and `prob` of discrete
          distribution.
        - Object of class `Rand` or `rv_frozen` (`rv_discrete` with all
          hyperparameters defined).
    - estimator_mixt : callable, default
      randomvars.options.estimator_mixt_default. Estimator for
      `Mixt.from_sample()`. Function which takes sample as input and returns
      one of:
        - Tuple with two elements representing samples from continuous and
          discrete parts. Absence of sample from certain part should be
          indicated by `None` element of output tuple: there will be no
          corresponding part in output `Mixt`.
        - Object of class `Rand`.
    - metric : string, default "L2". Type of metric which measures distance
      between functions. Used in internal computations. Possible values:
        - "L1": metric is defined as integral of absolute difference between
          functions. Usually corresponds to using some kind of "median values".
        - "L2": metric is defined as square root of integral of square
          difference between functions. Usually corresponds to using some kind
          of "mean values".
    - n_grid : int, default 1001. Number of points in initial xy-grids when
      creating object of class `Cont`.
    - small_prob : float, default 1e-6. Probability value that can be
      considered "small" during approximations.
    - small_width : float, default 1e-8. Difference between x-values that can
      be considered "small" during approximations.
    """

    # Available options
    base_tolerance = _Option(1e-12, _validator_nonneg)
    cdf_tolerance = _Option(1e-4, _validator_nonneg)
    density_mincoverage = _Option(
        0.9999,
        (lambda x: isinstance(x, float) and 0 <= x and x < 1, "a float inside [0; 1)"),
    )
    estimator_bool = _Option(estimator_bool_default, _validator_callable)
    estimator_cont = _Option(estimator_cont_default, _validator_callable)
    estimator_disc = _Option(estimator_disc_default, _validator_callable)
    estimator_mixt = _Option(estimator_mixt_default, _validator_callable)
    metric = _Option(
        "L2",
        (lambda x: isinstance(x, str) and x in ["L1", "L2"], 'one of "L1" or "L2"'),
    )
    n_grid = _Option(
        1001, (lambda x: isinstance(x, int) and x > 1, "an integer more than 1")
    )
    small_prob = _Option(
        1e-6,
        (lambda x: isinstance(x, float) and 0 < x and x < 1, "a float inside (0; 1)"),
    )
    small_width = _Option(
        1e-8, (lambda x: isinstance(x, float) and x > 0, "a positive float")
    )

    # Methods
    def __init__(self):
        self._list = [
            key for key, val in type(self).__dict__.items() if isinstance(val, _Option)
        ]
        self._defaults = {opt: getattr(self, opt) for opt in self._list}

    def _validate_option(self, opt):
        if opt not in self._list:
            raise OptionError(f"There is no option `{opt}`")

    @property
    def list(self):
        return self._list

    @property
    def defaults(self):
        return self._defaults

    @property
    def dict(self):
        return {opt: getattr(self, opt) for opt in self._list}

    def get_single(self, opt):
        self._validate_option(opt)
        return getattr(self, opt)

    def get(self, opt_list):
        return [self.get_single(opt) for opt in opt_list]

    def set_single(self, opt, value):
        self._validate_option(opt)
        setattr(self, opt, value)

    def set(self, opt_dict):
        # Ensure that all options are valid before setting
        for opt in opt_dict.keys():
            self._validate_option(opt)

        for key, val in opt_dict.items():
            self.set_single(key, val)

    def reset_single(self, opt):
        self._validate_option(opt)
        setattr(self, opt, type(self).__dict__[opt].default)

    def reset(self, opt_list):
        # Ensure that all options are valid before resetting
        for opt in opt_list:
            self._validate_option(opt)

        for opt in opt_list:
            self.reset_single(opt)

    def context(self, opt_dict):
        self._context_input = opt_dict
        return self

    def __enter__(self):
        if not hasattr(self, "_context_input"):
            raise OptionError("Use `context()` method to create context manager")

        self._context_undo = {opt: self.get_single(opt) for opt in self._context_input}
        self.set(self._context_input)

    def __exit__(self, *args):
        self.set(self._context_undo)

        del self._context_input
        del self._context_undo


config = _Config()


# %% Documentation helpers
def _docstring_paragraph(wrap=True, **kwargs):
    def decorator(f):
        doc = f.__doc__

        # Ensure paragraph indentation and width wrap
        kwargs_new = {}
        for marker, string in kwargs.items():
            marker_full = f"{{{marker}}}"
            # Assuming marker is placed on a separate line
            target_line = [s for s in doc.splitlines() if s.find(marker_full) > -1][0]
            indentation = target_line.replace(marker_full, "")

            if wrap:
                lines = textwrap.wrap(string, width=79 - len(indentation))
            else:
                lines = [string]

            paragraph = f"\n{indentation}".join(lines)
            kwargs_new[marker] = paragraph

        f.__doc__ = doc.format(**kwargs_new)

        return f

    return decorator


def _docstring_relevant_options(opt_list):
    opt_list_string = f'`{"`, `".join(opt_list)}`'
    opt_paragraph = (
        f"Relevant package options: {opt_list_string}. See documentation of "
        "`randomvars.options.get_option()` for more information. To temporarily set "
        "options use `randomvars.options.option_context()` context manager."
    )
    return _docstring_paragraph(relevant_options=opt_paragraph)

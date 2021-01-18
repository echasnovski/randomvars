import textwrap
import warnings

import numpy as np
from scipy.stats.kde import gaussian_kde


__all__ = [
    "OptionError",
    "estimator_bool_default",
    "estimator_cont_default",
    "estimator_disc_default",
    "get_option",
    "option_context",
    "reset_option",
    "set_option",
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
    x, prob : tuple with two elements
        Here `x` represents estimated values of distribution and `prob` -
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


# %% Options
_default_options = {
    "base_tolerance": 1e-12,
    "cdf_tolerance": 1e-4,
    "density_mincoverage": 0.9999,
    "estimator_bool": estimator_bool_default,
    "estimator_cont": estimator_cont_default,
    "estimator_disc": estimator_disc_default,
    "metric": "L2",
    "n_grid": 1001,
    "small_prob": 1e-6,
    "small_width": 1e-8,
}
_options = _default_options.copy()

_options_list = """
- base_tolerance : float, default 1e-12. Tolerance to be used for testing
  approximate equality of two numbers. It is used to compute tolerance
  associated with any number `x`:
    - If `abs(x) <= 1`, tolerance is equal to `base_tolerance`. Based on this,
      `base_tolerance` can be viewed as an absolute tolerance for "small"
      numbers.
    - If `abs(x) > 1`, tolerance increases proportionally to the spacing
      between floating point numbers at `x` (see `numpy.spacing()`). This
      approach is chosen in order to find compromise between relative and
      absolute tolerance.
- cdf_tolerance: float, default 1e-4. Tolerance for CDF approximation. Usually
  meant as mean approximation error. Smaller values lead to better
  approximation, larger values lead to less number of grid elements (knots) in
  output approximation. However, using large values (bigger than 0.01) is
  discouraged because this might lead to unexpected properties of approximation
  (like increasing density in tails where it should originally decrease, etc.).
- density_mincoverage : float, default 0.9999. Minimum value of integral within
  output of density range estimation.
- estimator_bool : callable, default randomvars.options.estimator_bool_default.
  Estimator for `Bool.from_sample()`. Function which takes sample as input and
  returns one of:
    - Number representing probability of `True` for boolean random variable.
    - Object of class `Rand` or `rv_frozen` (`rv_discrete` with all
      hyperparameters defined).
- estimator_cont : callable, default randomvars.options.estimator_cont_default.
  Estimator for `Cont.from_sample()`. Function which takes sample as input and
  returns one of:
    - Callable object for density estimate (takes points as input and returns
      density values).
    - Object of class `Rand` or `rv_frozen` (`rv_continuous` with all
      hyperparameters defined).
  **Notes**:
    - Theoretical integral of density over whole real line should be 1.
    - Output density callable should be vectorized: allow numpy array as input
      and return numpy array with the same length.
    - There is worse performance if output density callable has discontinuity.
- estimator_disc : callable, default randomvars.options.estimator_disc_default.
  Estimator for `Disc.from_sample()`. Function which takes sample as input and
  returns one of:
    - Tuple with two elements representing `x` and `prob` of discrete distribution.
    - Object of class `Rand` or `rv_frozen` (`rv_discrete` with all
      hyperparameters defined).
- metric : string, default "L2". Type of metric which measures distance between
  functions. Used in internal computations. Possible values:
    - "L1": metric is defined as integral of absolute difference between
      functions. Usually corresponds to using some kind of "median values".
    - "L2": metric is defined as square root of integral of square difference
      between functions. Usually corresponds to using some kind of "mean
      values".
- n_grid : int, default 1001. Number of points in initial xy-grids when
  creating object of class `Cont`.
- small_prob : float, default 1e-6. Probability value that can be considered
  "small" during approximations.
- small_width : float, default 1e-8. Difference between x-values that can be
  considered "small" during approximations.
"""


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


def get_option(opt):
    try:
        return _options[opt]
    except KeyError:
        raise OptionError(f"There is no option '{opt}'.")


get_option.__doc__ = f"""
Get package option

List of available options:
{_options_list}

Parameters
----------
opt : str
    Option name.

Raises
------
OptionError : if no such option exists.
"""


def set_option(opt, val):
    # Ensure that `opt` option can be accessed, raising relevant error
    # otherwise
    get_option(opt)

    # Set option
    _options[opt] = val


set_option.__doc__ = f"""
Set package option

List of available options:
{_options_list}

Parameters
----------
opt : str
    Option name.
val : any
    Option value.

Raises
------
OptionError : if no such option exists.
"""


def reset_option(opt):
    # Ensure that `opt` option can be accessed, raising relevant error
    # otherwise
    get_option(opt)

    # Reset option
    set_option(opt, _default_options[opt])


reset_option.__doc__ = f"""
Reset package option to default

List of available options:
{_options_list}

Parameters
----------
opt : str
    Option name.

Raises
------
OptionError : if no such option exists.
"""


class option_context:
    def __init__(self, opt_dict):
        self.opt_dict = opt_dict

    def __enter__(self):
        self.undo = {opt: get_option(opt) for opt in self.opt_dict}

        for opt, val in self.opt_dict.items():
            set_option(opt, val)

    def __exit__(self, *args):
        if self.undo:
            for opt, val in self.undo.items():
                set_option(opt, val)


option_context.__doc__ = f"""
Context manager to temporarily set options in the `with` statement context.

List of available options:
{_options_list}

Parameters
----------
opt_dict : dict
    Dictionary with option names as keys and option values as values.
"""


class OptionError(KeyError):
    """
    Exception describing error in interaction with package options.
    """

from scipy.stats.kde import gaussian_kde

from randomvars._utils import default_discrete_estimator, default_boolean_estimator


__all__ = [
    "OptionError",
    "default_boolean_estimator",
    "default_discrete_estimator",
    "get_option",
    "option_context",
    "reset_option",
    "set_option",
]

_default_options = {
    "boolean_estimator": default_boolean_estimator,
    "cdf_tolerance": 1e-4,
    "density_estimator": gaussian_kde,
    "density_mincoverage": 0.9999,
    "discrete_estimator": default_discrete_estimator,
    "metric": "L2",
    "n_grid": 1001,
    "small_prob": 1e-6,
    "small_width": 1e-8,
    "tolerance": (0.0, 1e-12),
}
_options = _default_options.copy()

_options_list = """
- boolean_estimator : callable, default
  randomvars.options.default_boolean_estimator. Function which takes sample as
  input and returns one of:
    - Number representing probability of `True` for boolean random variable.
    - Object of class `Bool, `Disc`, or `rv_frozen` (`rv_discrete` with all
      hyperparameters defined).
- cdf_tolerance: float, default 1e-4. Tolerance for CDF approximation. Usually
  meant as mean approximation error. Smaller values lead to better
  approximation, larger values lead to less number of grid elements (knots) in
  output approximation. However, using large values (bigger than 0.01) is
  discouraged because this might lead to unexpected properties of approximation
  (like increasing density in tails where it should originally decrease, etc.).
- density_estimator : callable, default scipy.stats.kde.gaussian_kde. Function
  which takes sample as input and returns one of:
    - Callable object for density estimate (takes points as input and returns
      density values).
    - Object of class `Cont` or `rv_frozen` (`rv_continuous` with all
      hyperparameters defined).
  **Notes**:
    - Theoretical integral of density over whole real line should be 1.
    - Output density callable should be vectorized: allow numpy array as input
      and return numpy array with the same length.
    - There is worse performance if output density callable has discontinuity.
- density_mincoverage : float, default 0.9999. Minimum value of integral within
  output of density range estimation.
- discrete_estimator : callable, default
  randomvars.options.default_discrete_estimator. Function which takes sample as
  input and returns one of:
    - Tuple with two elements representing `x` and `prob` of discrete distribution.
    - Object of class `Disc` or `rv_frozen` (`rv_discrete` with all
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
- tolerance : tuple with two elements, default (0.0, 1e-12). Tuple with
  relative and absolute tolerance to be used for testing approximate equality
  of two numbers. Testing is done following logic of builtin `math.isclose()`:
    - `rtol` is a maximum difference for being considered "close", relative to
      the magnitude of the input values.
    - `atol` is a maximum difference for being considered "close", regardless
      of the magnitude of the input values (needed to )
    - For the values to be considered close, the difference between them must
      be smaller than at least one of the tolerances.
  **Notes**:
    - Default values are different than in `math.isclose()`. This is done to
      account for the possibility to do approximate equality to zero.
    - The `numpy.isclose()` is not used due to its lack of symmetry:
      `numpy.isclose(a, b)` is not necessary equal to `numpy.isclose(b, a)`.
"""


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

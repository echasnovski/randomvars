import numpy as np

# %% Conversion
def _y_from_xp(x, p):
    """Compute y-grid from xp-grid

    Compute y-grid which together with input x-grid is dual to input xp-grid.
    Duality is defined in terms of maximum likelihood estimation. Output
    xy-grid maximizes weighted log-likelihood `sum(p * log(y))` subject to
    integration constraint on xy-grid (`0.5 * sum((x[1:] - x[:-1]) * (y[1:] +
    y[:-1])) = 1`).

    Notes:
    - Points with zero p-elements affect the output y-grid: they indicate that
      in that region probability should be low (corresponding elements of
      y-grid will be zero). This is somewhat counterintuitive, as presence of
      zero probabilities doesn't change input discrete variable, but affects
      output continuous one.
    """
    return p / _convert_coeffs(x)


def _p_from_xy(x, y):
    """Compute p-grid from xy-grid

    Compute p-grid which together with input x-grid is dual to input xy-grid.
    Duality is defined in terms of maximum likelihood estimation of xy-grid.
    Output xp-grid is the one, for which input xy-grid maximizes weighted
    log-likelihood `sum(p * log(y))` subject to integration constraint on
    xy-grid (`0.5 * sum((x[1:] - x[:-1]) * (y[1:] + y[:-1])) = 1`). This
    approach is taken to be inverse of y-from-p conversion.

    Notes:
    - Points with zero y-elements result into zero p-elements.
    """
    return y * _convert_coeffs(x)


def _convert_coeffs(x):
    """These are coefficients of y-grid when computing integral using
    trapezoidal rule"""
    x_ext = np.concatenate(([x[0]], x, [x[-1]]))
    return 0.5 * (x_ext[2:] - x_ext[:-2])

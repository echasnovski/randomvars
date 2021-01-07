import numpy as np

# %% Conversion
# There were different other approaches to Cont-Disc conversion, which were
# decided to be less appropriate:
# - In Cont-Disc construct discrete distribution with the same x-grid to be the
#   closest to input continuous CDF in terms of some metric ("L1" or "L2").
#   These were discarded because they were not invertible and hence not really
#   possible to create appropriate Disc-Cont conversion. The problem was that
#   during inverse conversion there were negative values in y-grid, which is an
#   additional problem. For example, `x = [0, 1]`, `p = [0.9, 0.1]`.
# - Another idea of Cont-Disc conversion was along the following lines:
#     - Assume there are many elements sampled from input distribution.
#     - For every sample element find the closest one among input x-grid.
#     - Take sample probability of x-grid elements as ratio of number of times
#       it was the closest and number of all points.
#     - Probability of element in x-grid is a limit of sample probabilities. Those
#       can be computed directly by computing probability of Voronoi intervals
#       (with ends at midpoints of adjacent intervals).
#   This turned out to be a previous approach with "L1" metric, which is not
#   invertible.


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

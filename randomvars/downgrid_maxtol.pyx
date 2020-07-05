#cython: language_level=3, boundscheck=False
import numpy as np
cimport numpy as np
from libc.stdint cimport uint8_t

cdef bint is_segment_inside_cone(
    double base_x,
    double base_y,
    double slope_min,
    double slope_max,
    double seg1_x,
    double seg1_y,
    double seg2_x,
    double seg2_y
):
    """Compute if segment lies inside 2d closed cone

    Two-dimensional closed cone is defined as all rays from point `(base_x,
    base_y)` and  with slopes inside `[slope_min, slope_max]` range (rays are
    directed to the right of origin). Segment connects point `(seg1_x, seg1_y)`
    and `(seg2_x, seg2_y)`.

    This function computes if whole segment lies inside cone (even if it
    touches some edge).

    Parameters
    ----------
    base_x, base_y : double
        Numbers for x and y coordinates of 2d cone origin point.
    slope_min, slope_max : double
        Numbers for minimum and maximum values of slope (edges of 2d cone).
    seg1_x, seg1_y : double
        Numbers for x and y coordinates of segment start
    seg2_x, seg2_y : double
        Numbers for x and y coordinates of segment end

    Returns
    -------
    is_inside : bint
        Boolean value indicating if whole segment lies inside cone.
    """
    cdef double seg_slope_1 = (seg1_y - base_y) / (seg1_x - base_x)
    cdef double seg_slope_2 = (seg2_y - base_y) / (seg2_x - base_x)

    # Segment lies inside cone if its both ends' slopes (computed with respect to
    # cone's base point) lie inside `[slope_min, slope_max]`
    if (
        (seg_slope_1 >= slope_min)
        and (seg_slope_1 <= slope_max)
        and (seg_slope_2 >= slope_min)
        and (seg_slope_2 <= slope_max)
    ):
        return True
    else:
        return False


cdef (double, double) tolerance_slope_window(
    double base_x, double base_y, double point_x, double point_y, double tol
):
    """ Compute slope window for rays to be within tolerance of supplied point

    Computes slope window of 2d cone with base point `(base_x, base_y)` and
    which passes through points `(point_x, point_y-tol)` and `(point_x,
    point_y+tol)`.
    """
    cdef double slope_min = (point_y - base_y - tol) / (point_x - base_x)
    cdef double slope_max = (point_y - base_y + tol) / (point_x - base_x)

    return (slope_min, slope_max)


cdef (double, double) intersect_intervals(
    double inter1_min, double inter1_max, double inter2_min, double inter2_max
):
    """Compute intersection of intervals

    Computes intersections of intervals `(inter1_min, inter1_max)` and
    `(inter2_min, inter2_max)`. Basically, the output is `(max(inter1_min,
    inter2_min), min(inter1_max, inter2_max))` but optimized for better
    execution speed.
    """
    cdef double res_min, res_max

    if inter1_min <= inter2_min:
        res_min = inter2_min
    else:
        res_min = inter1_min

    if inter1_max <= inter2_max:
        res_max = inter1_max
    else:
        res_max = inter2_max

    return (res_min, res_max)


def downgrid_maxtol(x, y, tol, double_pass=True):
    """Downgrid with maximum tolerance

    Downgrid input xy-grid so that maximum difference between points on output
    piecewise-linear function and input xy-grid is not more than `tol`. Output
    xy-grid is a subset of input xy-grid. **Note** that first and last point is
    always inside output xy-grid.

    There are two variations of downgriddings: single and double (default)
    pass. Single pass is performed by iteratively (from left to right)
    determining if grid element should be in output. Output of double pass is a
    union of single passes from left to right and from right to left.

    Parameters
    ----------
    x : numpy array
    y : numpy array
    tol : scalar
        Tolerance. If zero, points that lie between colinear segments will be
        removed without precision loss of piecewise-linear function.
    double_pass : boolean scalar, optional
        Whether to do a double pass (default `True`): one from left to right
        and one from right to left. Output grid is a union of single passes.

    Returns
    -------
    xy_grid : Tuple with two numpy arrays with same lengths
        Subset of input xy-grid which differs from it by no more than `tol`.
    """
    x = x.astype(np.double)
    y = y.astype(np.double)
    tol = float(tol)

    # Using `np.asarray()` here to turn memoryview into an array
    res_isin = np.asarray(downgrid_maxtol_isin(x, y, tol))

    if double_pass:
        rev_x = x[-1] - x[::-1]
        rev_y = y[::-1]
        second_pass = np.asarray(downgrid_maxtol_isin(rev_x, rev_y, tol))[::-1]
        # Output should be a union of passes, i.e. point should be present in
        # output if it equals to 1 in at least one of first or second passes
        res_isin = np.maximum(res_isin, second_pass)

    output_inds = res_isin.nonzero()[0]
    return x[output_inds], y[output_inds]


cdef uint8_t[:] downgrid_maxtol_isin(
    double[:] x, double[:] y, double tol=0.001
):
    if len(x) <= 2:
        return np.ones(x.shape[0], dtype=np.uint8)

    cdef int n_x = x.shape[0]
    res_boolint = np.zeros(n_x, dtype=np.uint8)
    cdef uint8_t[:] res_boolint_view = res_boolint

    # First point is always inside output grid
    res_boolint_view[0] = 1

    # Initialize base point and slope window
    cdef double base_x = x[0], base_y = y[0]

    cdef double slope_min, slope_max
    slope_min, slope_max = tolerance_slope_window(
        base_x, base_y, x[1], y[1], tol
    )

    # Initialize variables to be used inside loop
    cdef double seg_start_x, seg_start_y, seg_end_x, seg_end_y
    cdef double seg_end_slope_min, seg_end_slope_max
    cdef int cur_i = 2

    while cur_i < len(x):
        seg_end_x = x[cur_i]
        seg_end_y = y[cur_i]

        # Compute if segment lies inside current base cone. If it does, then it
        # can be skipped. It it goes out of the current base cone, it means
        # that skipping will introduce error strictly more than `tol`, so
        # adding current segment start to output xy-grid is necessary.
        segment_is_inside = is_segment_inside_cone(
            base_x=base_x,
            base_y=base_y,
            slope_min=slope_min,
            slope_max=slope_max,
            seg1_x=x[cur_i - 1],
            seg1_y=y[cur_i - 1],
            seg2_x=seg_end_x,
            seg2_y=seg_end_y,
        )

        if segment_is_inside:
            # Update slope window by using intersection of current slope window
            # and slope window of segment end. Intersection is used because in
            # order to maintain maximum error within tolerance rays should pass
            # inside both current 2d cone and 2d cone defined by end of current
            # segment
            seg_end_slope_min, seg_end_slope_max = tolerance_slope_window(
                base_x, base_y, seg_end_x, seg_end_y, tol
            )
            slope_min, slope_max = intersect_intervals(
                slope_min, slope_max, seg_end_slope_min, seg_end_slope_max
            )
        else:
            # Write segment start as new point in output
            res_boolint_view[cur_i - 1] = 1

            # Update base point to be current segment start
            base_x, base_y = x[cur_i - 1], y[cur_i - 1]

            # Update slope window to be slope window of current segment end
            slope_min, slope_max = tolerance_slope_window(
                base_x, base_y, seg_end_x, seg_end_y, tol
            )

        cur_i += 1

    # Last point is always inside output grid
    res_boolint_view[n_x-1] = 1

    return res_boolint

import numpy as np
import scipy.stats as ss
from scipy.integrate import quad
from scipy.interpolate import BSpline, UnivariateSpline
import scipy.interpolate as interpol
import matplotlib.pyplot as plt


def approx_cdf_piececont(
    f, f_inv, x_condition, cump_condition, method="mean", max_steps=100, thres=1e-8
):
    x_is_free = np.isnan(x_condition)
    x_fixed_val = x_condition[~x_is_free]
    cump_is_free = np.isnan(cump_condition)
    cump_fixed_val = cump_condition[~cump_is_free]

    center_fun = construct_center_fun(f, method)

    cump_prev = init_cump(cump_condition)
    x_prev = x_from_cump(cump_prev, f_inv, x_is_free, x_fixed_val)
    steps = [(x_prev, cump_prev)]

    for i in range(1, max_steps):
        print(i, end=": ")
        cump_cur = cump_from_x(x_prev, center_fun, cump_is_free, cump_fixed_val)
        x_cur = x_from_cump(cump_cur, f_inv, x_is_free, x_fixed_val)
        steps.append((x_cur, cump_cur))

        if stop_rule(
            step_cur=(x_cur, cump_cur), step_prev=(x_prev, cump_prev), thres=thres
        ):
            break

        x_prev = x_cur
        cump_prev = cump_cur

    return steps


def init_cump(cump_condition):
    """Initialize p-vector
    Every non-conditioned value is initialized to be proportional probability
    apart from nearest supplied conditions (after adding 0 and 1 to edges).
    """
    cump_ext = np.concatenate([[0], cump_condition, [1]])
    x_equi = np.arange(len(cump_ext))

    inds = np.isfinite(cump_ext)

    return np.interp(x_equi[1:-1], x_equi[inds], cump_ext[inds])


def x_from_cump(cump, f_inv, x_is_free, x_fixed_val):
    res = np.zeros(len(cump) + 1)

    cump_ext = np.concatenate([[0], cump, [1]])
    cump_mid = 0.5 * (cump_ext[:-1] + cump_ext[1:])

    res[x_is_free] = f_inv(cump_mid[x_is_free])
    res[~x_is_free] = x_fixed_val

    return res


def cump_from_x(x, center_fun, cump_is_free, cump_fixed_val):
    res = np.zeros(len(cump_is_free))

    x_left = x[np.concatenate([cump_is_free, [False]])]
    x_right = x[np.concatenate([[False], cump_is_free])]

    res[cump_is_free] = center_fun(x_left, x_right)
    res[~cump_is_free] = cump_fixed_val

    return res


def construct_center_fun(f, method):
    if method == "mean":
        # Use faster integration method if available
        if "integrate" in dir(f):
            integr_f = f.integrate
        else:
            integr_f = lambda a, b: quad(f, a, b)[0]

        def center_fun(a, b):
            res = np.zeros(len(a))

            a_less = a < b

            a1 = a[a_less]
            b1 = b[a_less]
            res[a_less] = np.array([integr_f(l, r) for l, r in zip(a1, b1)]) / (b1 - a1)

            res[~a_less] = f(a[~a_less])

            return res

    elif method == "mean_approx":
        # Compute mean based on quadratic approximation which should be good
        # enough because CDFs will be piecewise-quadratic. This is essentially
        # Simpson's rule.
        def center_fun(a, b):
            res = np.zeros(len(a))

            a_less = a < b

            a1 = a[a_less]
            b1 = b[a_less]
            mid1 = 0.5 * (a1 + b1)
            res[a_less] = (f(a1) + 4 * f(mid1) + f(b1)) / 6

            res[~a_less] = f(a[~a_less])

            return res

    elif method == "median":

        def center_fun(a, b):
            return f(0.5 * (a + b))

    return center_fun


def stop_rule(step_cur, step_prev, thres):
    x_error = relative_error(step_cur[0], step_prev[0])
    cump_error = relative_error(step_cur[1], step_prev[1])

    print(f"{x_error=:10.9f}; {cump_error=:10.9f}")

    return (x_error < thres) and (cump_error < thres)


def relative_error(x, y):
    return np.sum(np.abs(x - y)) / np.sum(np.abs(y))


def p_from_cump(cump):
    return np.diff(cump, prepend=0, append=1)


def make_cdf_spline(rv, n_grid=1001, s=1e-8, prob_left=1e-6, prob_right=1e-6):
    x_equi = np.linspace(rv.ppf(prob_left), rv.ppf(1 - prob_right), n_grid)
    x_quan = rv.ppf(np.linspace(prob_left, 1 - prob_right, n_grid))
    x = np.union1d(x_equi, x_quan)
    cdf_spline_raw = UnivariateSpline(x=x, y=rv.cdf(x), k=2, s=s)
    density_spline = cdf_spline_raw.derivative()

    x_grid = density_spline.get_knots()
    y_grid = np.clip(density_spline(x_grid), 0, None)
    density_tck = interpol.splrep(x=x_grid, y=y_grid, k=1, s=0)
    cdf_tck = interpol.splantider(density_tck)
    return BSplineCDF(*cdf_tck)


def fun_distance(f, step, method="mean"):
    x_grid, cump_grid = step
    g = make_piececonst(x_grid, cump_grid)

    if (method == "mean") or (method == "mean_approx"):
        left_integral = quad(lambda x: f(x) ** 2, -np.inf, x_grid[0])[0]
        right_integral = quad(lambda x: (1 - f(x)) ** 2, x_grid[-1], np.inf)[0]

        integrals = np.array(
            [
                quad(lambda x: (f(x) - val) ** 2, x_l, x_r)[0]
                for x_l, x_r, val in zip(x_grid[:-1], x_grid[1:], cump_grid)
            ]
        )
        return np.sqrt(left_integral + np.sum(integrals) + right_integral)
    elif method == "median":
        left_integral = quad(lambda x: f(x), -np.inf, x_grid[0])[0]
        right_integral = quad(lambda x: 1 - f(x), x_grid[-1], np.inf)[0]

        integrals = np.array(
            [
                quad(lambda x: np.abs(f(x) - val), x_l, x_r)[0]
                for x_l, x_r, val in zip(x_grid[:-1], x_grid[1:], cump_grid)
            ]
        )
        return left_integral + np.sum(integrals) + right_integral


def make_piececonst(x, cump):
    return interpol.interp1d(
        x=x,
        y=np.concatenate([cump, [1]]),
        kind="previous",
        fill_value=(0, 1),
        bounds_error=False,
    )


class BSplineCDF(BSpline):
    def __call__(self, x):
        res = super().__call__(x)
        res[x < self.t[0]] = 0
        res[x > self.t[-1]] = 1

        return res


# rv = ss.beta(a=3, b=3)
# rv = ss.norm()
rv = ss.binom(n=10, p=0.5)

f = rv.cdf
# f = make_cdf_spline(rv, prob_left=1e-6, prob_right=1e-6)
f_inv = rv.ppf

cump_condition = np.repeat(np.nan, 10)
x_condition = np.repeat(np.nan, 11)
# cump_condition = np.array([0.1, np.nan, 0.75, 0.95, 0.99])
# x_condition = np.concatenate([[-4], np.repeat(np.nan, len(cump_condition) - 1), [4]])

## Method "mean_approx" works weel with continuous cdf in the form of quadratic
## spline.
## However, it works poorly for discrete CDF, which is currently its main target.
## Moreover, it works poorly in general with discrete CDF, as it doesn't return all
## values when it can (thus resulting into perfect approximation).
method = "mean_approx"
# method = "median"

steps = approx_cdf_piececont(
    f, f_inv, x_condition, cump_condition, max_steps=1001, thres=1e-6, method=method
)

x, cump = steps[-1]

if (method == "mean") or (method == "mean_approx"):
    print(
        np.array(
            [quad(f, x_l, x_r)[0] / (x_r - x_l) for x_l, x_r in zip(x[:-1], x[1:])]
        )
        - cump
    )
elif method == "median":
    print(np.array([f(0.5 * (x_l + x_r)) for x_l, x_r in zip(x[:-1], x[1:])]) - cump)

[fun_distance(f, step, method=method) for step in steps[:10]]
fun_distance(f, steps[-1], method=method)

x_plotgrid = np.linspace(f.t[0], f.t[-1], 1001)
plt.plot(x_plotgrid, f(x_plotgrid), label="f")
plt.plot(x_plotgrid, make_piececonst(x, cump)(x_plotgrid), label="approx")
plt.legend()
plt.show()


# %% Experiments with approximating piecewise-constant function
def const_grid_to_lin_grid(x, cump, h=1e-8):
    x_ext = np.array([x - h, x], order="c").ravel(order="f")
    y_ext = np.array(
        [np.concatenate(([0], cump)), np.concatenate((cump - h, [1]))], order="c"
    ).ravel(order="f")

    return x_ext, y_ext


x_const = np.array([0, 1, 2, 10, 11])
cump_const = np.cumsum([0.1, 0.3, 0.1, 0.3])

# rv = ss.binom(n=10, p=0.1)
# x_const = np.arange(0, 11)
# cump_const = rv.cdf(x_const[:-1])

x_lin, y_lin = const_grid_to_lin_grid(x_const, cump_const)

plt.plot(x_lin, y_lin, label="linear")
plt.legend()
plt.show()

f = interpol.interp1d(
    x_lin, y_lin, kind="linear", bounds_error=False, fill_value=(0, 1)
)
f_inv = interpol.interp1d(
    y_lin, x_lin, kind="linear", bounds_error=False, fill_value=(x_lin[0], x_lin[-1])
)

x_len = 5
cump_condition = np.repeat(np.nan, x_len - 1)
x_condition = np.repeat(np.nan, x_len)

# method = "mean_approx"
method = "mean"

steps = approx_cdf_piececont(
    f, f_inv, x_condition, cump_condition, max_steps=1001, thres=1e-6, method=method
)

x, cump = steps[-1]
x_plotgrid = np.linspace(x_const[0], x_const[-1], 1001)
plt.plot(x_plotgrid, f(x_plotgrid), label="f")
plt.plot(x_plotgrid, make_piececonst(x, cump)(x_plotgrid), label="approx")
plt.legend()
plt.show()

# Still doesn't work
## Proposed solution
fun_distance(f, (x, cump), method=method)
## Actual solution in case of decreasing probabilities
fun_distance(f, (x_const[:x_len], cump_const[: (x_len - 1)]), method=method)

## Maybe try to initialize `x` with the biggest probabilities
x_const = np.array([0, 1, 2, 10, 11, 12])
cump_const = np.cumsum([0.1, 0.2, 0.1, 0.1, 0.5])
p_const = p_from_cump(cump_const)

x_len = 4
p_order = np.argsort(p_const)
inds = p_order[-x_len:]
p = p_const[inds]
x_init = x_const[inds]
inds2 = np.argsort(x_init)
p = p[inds2]
x_init = x_init[inds2]

cump_condition = np.repeat(np.nan, x_len - 1)
x_condition = np.repeat(np.nan, x_len)
cump_is_free = np.isnan(cump_condition)
cump_fixed_val = cump_condition[~cump_is_free]

cump_init = cump_from_x(
    x_init, construct_center_fun(f, method), cump_is_free, cump_fixed_val
)

x_plotgrid = np.linspace(x_const[0], x_const[-1], 1001)
plt.plot(x_plotgrid, make_piececonst(x_const, cump_const)(x_plotgrid), label="f")
plt.plot(x_plotgrid, make_piececonst(x_init, cump_init)(x_plotgrid), label="approx")
plt.legend()
plt.show()

"""
Code about best uniform approximation (based on distance between CDFs) and its
comparison with spline approaches.
"""
import numpy as np
import scipy.stats as ss
from scipy.integrate import quad
from scipy.interpolate import UnivariateSpline
from scipy.optimize import fsolve, least_squares, minimize
import matplotlib.pyplot as plt

from randomvars import Cont

# rv = Cont.from_rv(ss.norm())
# rv = Cont.from_rv(ss.beta(a=2, b=5))
# rv = Cont.from_rv(ss.expon())
# rv = Cont.from_rv(ss.laplace())
# rv = Cont.from_rv(ss.cauchy())
# rv = Cont.from_rv(ss.chi2(df=1))
rv = Cont.from_rv(ss.beta(a=0.4, b=0.7))
init_ab = rv.ppf([1e-6, 1 - 1e-6])
a_bounds = rv.support()
b_bounds = rv.support()

## Non-linear equation approach
def make_f_system_spline(rv):
    f = rv.cdf
    x = rv.x
    cdf_vals = rv.cdf(x)
    cdf_spline = UnivariateSpline(x=x, y=cdf_vals, k=2, s=0)
    cdf_spline_mom1 = UnivariateSpline(x=x, y=cdf_vals * x, k=3, s=0)

    def f_system(x):
        a, b = x
        moment_0 = cdf_spline.integral(a, b)
        moment_1 = cdf_spline_mom1.integral(a, b)

        # return [moment_0 - 0.5 * (b - a), moment_1 - (b - a) * (a + 2 * b) / 6]
        return [moment_0 / (b - a) - 0.5, moment_1 / (b - a) - (a + 2 * b) / 6]

    def jacobian(x):
        a, b = x
        return [
            [-f(a) + 0.5, f(b) - 0.5],
            [-f(a) * a + (2 * a + b) / 6, f(b) * b + (a - 4 * b) / 6],
        ]

    return f_system, jacobian


## Line spline fitting apporach
def unif_approx_spline(rv, n_grid=1001):
    cdf_line = UnivariateSpline(x=rv.x, y=rv.cdf(rv.x), k=1, s=np.inf)
    t, c = cdf_line.get_knots(), cdf_line.get_coeffs()
    slope = np.diff(c) / np.diff(t)
    inter = c[0] - slope * t[0]
    left = max(rv.a, -inter[0] / slope[0])
    right = min(rv.b, (1 - inter[0]) / slope[0])

    x = np.array([left, right])
    y = np.repeat(1 / (right - left), 2)

    return x, y


def general_approx_spline(rv, n_grid=1001):
    cdf_spline = UnivariateSpline(x=rv.x, y=rv.cdf(rv.x), k=2, s=np.inf)
    pdf_spline = cdf_spline.derivative()
    x, y = pdf_spline.get_knots(), pdf_spline.get_coeffs()
    y = np.clip(y, 0, None)
    y = y / np.trapz(y, x)

    return x, y


## Computations
f_system_spline, f_prime_spline = make_f_system_spline(rv)

# # Non-linear equations solution
f_solution_spline = fsolve(func=f_system_spline, x0=init_ab, fprime=f_prime_spline)
a_optim, b_optim = f_solution_spline

# # Least squares constrained minimization
# f_solution_spline_min = least_squares(
#     fun=f_system_spline,
#     x0=init_ab,
#     jac=f_prime_spline,
#     bounds=list(zip(a_bounds, b_bounds)),
# )
# a_optim, b_optim = f_solution_spline_min.x

a_optim = np.clip(a_optim, *a_bounds)
b_optim = np.clip(b_optim, *b_bounds)
unif_optim = Cont(x=[a_optim, b_optim], y=np.repeat(1 / (b_optim - a_optim), 2))

unif_spline = Cont(*unif_approx_spline(rv))

general_spline = Cont(*general_approx_spline(rv))

x_plot = np.linspace(init_ab[0], init_ab[1], 1001)

plt.plot(x_plot, rv.cdf(x_plot), label="input")
plt.plot(x_plot, unif_optim.cdf(x_plot), label="non-linear equation")
plt.plot(x_plot, unif_spline.cdf(x_plot), label="uniform spline")
plt.plot(x_plot, general_spline.cdf(x_plot), label="general spline")
plt.legend()
plt.show()

plt.plot(x_plot, rv.pdf(x_plot), label="input")
plt.plot(x_plot, unif_optim.pdf(x_plot), label="non-linear equation")
plt.plot(x_plot, unif_spline.pdf(x_plot), label="uniform spline")
plt.plot(x_plot, general_spline.pdf(x_plot), label="general spline")
plt.show()

## Square of distances
quad(lambda t: (rv.cdf(t) - unif_spline.cdf(t)) ** 2, rv.x[0], rv.x[-1])[0]
quad(lambda t: (rv.cdf(t) - unif_optim.cdf(t)) ** 2, rv.x[0], rv.x[-1])[0]
quad(lambda t: (rv.cdf(t) - general_spline.cdf(t)) ** 2, rv.x[0], rv.x[-1])[0]

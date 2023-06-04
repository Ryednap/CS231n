from random import randrange
from typing import Callable
import numpy as np


def eval_numerical_gradient(f: Callable[[np.ndarray], float], x: np.ndarray, verbose=True, h=0.00001) -> np.ndarray:
    """
    A naive implementation of numerical gradient of **f** at **x**

    :param f: Callable function that takes a single parameter a numpy array and returns the
        value of the function at that parameter as float.
    :param x: Is the point (numpy array) to evaluate the gradient at.
    :param verbose: If True, then the gradient at each point is printed as logs. Default True
    :param h: Step size for computing numerical gradient. Default 1e-5
    :return: A numpy array of the same size as that of **x** containing gradient of function **f** at point **x**.
    """

    fx = f(x)  # evaluate function value at original point
    grad = np.zeros_like(x)

    # iterate over all indexes of x.
    for ix in np.ndindex(x.shape[0]):

        oldval = x[ix]
        x[ix] = oldval + h  # increment by h
        fxph = f(x)  # evaluate f(x + h)
        x[ix] = oldval - h
        fxmh = f(x)  # evaluate f(x - h)
        x[ix] = oldval  # restore

        # compute the partial derivative with centered formula
        grad[ix] = (fxph - fxmh) / (2 * h)  # the slope

        if verbose:
            print(ix, grad[ix])

        return grad


def eval_numerical_gradient_array(f: Callable[[np.ndarray], np.ndarray], x: np.ndarray, df, h=0.00001):
    """
    Evaluate a numeric gradient for a function that accepts a numpy array and returns a numpy array.
    Specifically, this method calculated the gradient of the input x with respect to a function that
    has n-dimensional output. To calculate the gradient, we are using the chain rule and upstream derivative
    is given to us.

    :param f: A function that takes a numpy ndarray and returns the value of the function at that point,
            which is also a numpy array.
    :param x: A numpy ndarray representing a specific point in higher dimensional.
    :param df: An upstream gradient array that needs to be multiple with numerical gradient.
    :param h: A float representing step size for numerical gradient calculation
    :return:
    """

    grad = np.zeros_like(x)
    for ix in np.ndindex(x.shape[0]):

        oldval = x[ix]
        x[ix] = oldval + h
        pos = f(x).copy()
        x[ix] = oldval - h
        neg = f(x).copy()
        x[ix] = oldval

        grad[ix] = np.sum((pos - neg) * df) / (2 * h)
    return grad


def grad_check_sparse(f: Callable[[np.ndarray], float], x: np.ndarray, analytic_grad: np.ndarray, num_checks=10, h=1e-5)\
        -> None:
    """
    This method implements gradient check in sparse way (it's approximate).
    For sparse checking, this method samples a few random elements
    and only returns numerical in this dimension
    :param f: A function that takes a numpy ndarray and returns the value of the function at that point.
    :param x: A numpy ndarray representing a specific point in higher dimensional.
    :param analytic_grad: A numpy ndarray representing analytic gradient at point `x`.
    :param num_checks: Number of times to check the gradient sparsely.
    :param h: A float representing step size for numerical gradient calculation
    :return: None
    """

    for i in range(num_checks):
        ix = tuple([randrange(m) for m in x.shape])

        oldval = x[ix]
        x[ix] = oldval + h  # increment by h
        fxph = f(x)  # evaluate f(x + h)
        x[ix] = oldval - h  # decrement by h
        fxmh = f(x)  # evaluate f(x - h)
        x[ix] = oldval  # reset

        grad_numerical = (fxph - fxmh) / (2 * h)
        grad_analytic = analytic_grad[ix]
        rel_error = abs(grad_numerical - grad_analytic) / (abs(grad_numerical) + abs(grad_analytic))
        print(
            "numerical: %f analytic: %f, relative error: %e"
            % (grad_numerical, grad_analytic, rel_error)
        )

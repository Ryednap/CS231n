import numpy as np

from src.layers import *


def affine_relu_forward(x: np.ndarray, w: np.ndarray, b: np.ndarray) -> tuple[np.ndarray, tuple[AffineCache, ReluCache]]:
    """
    Convenience layer that performs an affine transform followed a ReLU activation.

    :param x: A numpy array as input to the affine layer
    :param w: A numpy array as weights for affine layer
    :param b: A numpy array as biases for affine layer.
    :return: A tuple of:
        - out: A numpy array as output from relu layer.
        - cache: A tuple representing the `AffineCache` and `ReluCache`.
    """

    a, fc_cache = affine_forward(x, w, b)
    out, relu_cache = relu_forward(a)

    cache = (fc_cache, relu_cache)
    return out, cache


def affine_relu_backward(dout: np.ndarray, cache: tuple[AffineCache, ReluCache]) -> AffineGrad:
    """
    This layer performs backward-pass for the *affine-relu* convenience layer.

    :param dout: A numpy array representing upstream gradient with respect to output from `affine_relu_forward`.
    :param cache: A tuple containing:
        - fc_cache: An `AffineCache` object containing cache for affine-function (full-connected).
        - relu_cache: A `ReluCache` object containing cache for relu-function.
    :return: An `AffineGrad` object which represents gradient with respect to input of affine-function.
    """

    fc_cache, relu_cache = cache
    da = relu_backward(dout, relu_cache)
    dx, dw, db = affine_backward(da, fc_cache)
    return dx, dw, db


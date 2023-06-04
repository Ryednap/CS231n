"""
This file implements various first-order update rules that are commonly used for training neural networks.
Each update rule accepts current weights and the gradient of the loss with respect to those weights and produces
the next set of weights. Each update rule has the same interface:

    def update(w, dw, config=None):

Inputs:
    - w: A numpy array giving the current weights.
    - dw: A numpy array of the same shape as W giving the gradient of the loss with respect w.
    - config: A dictionary containing hyperparameter values such as learning rate, momentum, etc.
      If the update rule requires caching values over many iterations, then config will also hold these values.

Returns:
    - next_w: The next weights after the update.
    - config: The config dictionary to be passed to the next iteration for the update rule.

Note: For most update rules, the default learning rate will probably not perform well;
however, the default values of the other hyperparameters should work well for a variety of different problems.

For efficiency, update rules may perform in-place updates, mutating w and setting next_w equal to w.
"""
import numpy as np


def sgd(w: np.ndarray, dw: np.ndarray, config=None):
    """
    Performs vanilla stochastic gradient descent

    config_format:
        - learning_rate: Scalar learning rate.
    """

    if config is None:
        config = {}
    config.setdefault("learning_rate", 1e-2)

    w -= config['learning_rate'] * dw
    return w, config


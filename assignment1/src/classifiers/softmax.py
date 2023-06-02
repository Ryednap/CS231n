from typing import Tuple

import numpy as np


def softmax_loss_vectorized(W: np.ndarray, X: np.ndarray, y: np.ndarray, reg: float) -> Tuple[float, np.ndarray]:
    """
    Structured SVM loss function, vectorized implementation (optimal).

    Inputs and outputs are the same as `svm_loss_naive`.

    :param W: A numpy array of shape (D, C) containing weights.
    :param X: A numpy array of shape (N, D) containing a mini-batch of data.
    :param y: A numpy array of shape (N, ) containing labels; y[i] = c means
        that X[i] has label, c where 0 <= c < C
    :param reg: A floating number representing the regularization strength.

    :return: A tuple of:
        - loss as single float
        - gradient with respect to weights W; a numpy array of the same shape as W.
    """
    pass


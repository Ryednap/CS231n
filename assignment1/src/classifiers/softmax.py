from typing import Tuple

import numpy as np


def softmax_loss_naive(W: np.ndarray, X: np.ndarray, y: np.ndarray, reg: float) -> Tuple[float, np.ndarray]:
    """
    This method contains Softmax loss function, naive implementation (with loops)

    Inputs have dimension D, there are C classes, and we operate on mini-batches of N examples.

    :param W: A numpy array of shape (D, C) containing weights.
    :param X: A numpy array of shape (N, D) containing inputs.
    :param y: A numpy array of shape (N,) containing labels;
        y[i] = c means that X[i] has label c, where 0 <= c < C.
    :param reg: A floating number representing regularization strength.
    :return: A tuple of:
        - loss as single float
        - A numpy array of gradient with respect to weight W; an array of shape as W.
    """

    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)

    #############################################################################
    # TODO: Compute the softmax loss and its gradient using explicit loops.     #
    # Store the loss in loss and the gradient in dW. If you are not careful     #
    # here, it is easy to run into numeric instability. Don't forget the        #
    # regularization!                                                           #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    pass

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    return loss, dW


def softmax_loss_vectorized(W: np.ndarray, X: np.ndarray, y: np.ndarray, reg: float) -> Tuple[float, np.ndarray]:
    """
    Thsi method contains Softmax loss function, vectorized implementation (optimal).

    Inputs and outputs are the same as `softmax_loss_naive`.

    :param W: A numpy array of shape (D, C) containing weights.
    :param X: A numpy array of shape (N, D) containing a mini-batch of data.
    :param y: A numpy array of shape (N, ) containing labels; y[i] = c means
        that X[i] has label, c where 0 <= c < C
    :param reg: A floating number representing the regularization strength.

    :return: A tuple of:
        - loss as single float
        - gradient with respect to weights W; a numpy array of the same shape as W.
    """

    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)

    #############################################################################
    # TODO: Compute the softmax loss and its gradient using explicit loops.     #
    # Store the loss in loss and the gradient in dW. If you are not careful     #
    # here, it is easy to run into numeric instability. Don't forget the        #
    # regularization!                                                           #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    pass

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    return loss, dW

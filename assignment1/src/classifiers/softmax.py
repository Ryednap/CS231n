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

    num_train = X.shape[0]
    num_classes = W.shape[1]

    # implementing the log expanded version of the softmax
    # i.e. loss[i] = -f[i] + log(sum(f[j]))  for ith example, and j represents classes.
    for i in range(num_train):
        scores = X[i].dot(W)
        scores -= np.max(scores)  # numerical stability trick

        log_sum = 0
        for j in range(num_classes):
            if j == y[i]:
                loss -= scores[j]
                dW[np.arange(W.shape[0]), j] -= X[i]

            log_sum += np.exp(scores[j])

        loss += np.log(log_sum)
        for j in range(num_classes):
            dW[np.arange(W.shape[0]), j] += (np.exp(scores[j]) * X[i]) / log_sum

    loss /= num_train
    dW /= num_train

    loss += reg * np.sum(W * W)
    dW += 2 * reg * W

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    return loss, dW


def softmax_loss_vectorized(W: np.ndarray, X: np.ndarray, y: np.ndarray, reg: float) -> Tuple[float, np.ndarray]:
    """
    This method contains Softmax loss function, vectorized implementation (optimal).

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
    num_train = X.shape[0]

    scores = X.dot(W)  # (N, C) matrix
    scores -= np.max(scores, axis=1, keepdims=True)  # for numerical stability
    dScores = X.T

    # For the next stage, I would recommend the readers to derive the derivative for a
    # single example and then do the parallel operation as matrix.
    # Ultimately, you will get this:
    #       For classes (j) which are not target label = np.exp(xj) / sum(np.exp(x))
    #       For the class (i) which is target label = (-sum(np.exp(x)) + np.exp(xi)) / sum(np.exp(x))
    #                                           = np.exp(xi) / sum(np.exp(x)) - 1
    # So note that for all classes we have a common formula np.exp(x) / sum(np.exp(x)). For the target label
    #  class, we just need to subtract the 1.
    # Note: The above-shown derivative expression is with entire `-log()` term taken in account.
    proba = np.exp(scores[np.arange(num_train), y]) / np.sum(np.exp(scores), axis=1)
    loss = np.sum(-np.log(proba))
    dL = np.exp(scores) / np.sum(np.exp(scores), axis=1, keepdims=True)  # derivative for all classes
    dL[np.arange(num_train),  y] -= 1  # subtract 1 for the class which is target label

    dW = dScores.dot(dL)

    loss /= num_train
    dW /= num_train

    loss += reg * np.sum(W * W)
    dW += 2 * reg * W

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    return loss, dW

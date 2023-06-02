from typing import Tuple

import numpy as np


def svm_loss_naive(W: np.ndarray, X: np.ndarray, y: np.ndarray, reg: float) -> Tuple[float, np.ndarray]:
    """
    Structured SVM loss function, naive implementation (with loops).

    Inputs have dimension D, there are C classes, and we operate on mini-batches
    of N examples.

    :param W: A numpy array of shape (D, C) containing weights.
    :param X: A numpy array of shape (N, D) containing a mini-batch of data.
    :param y: A numpy array of shape (N, ) containing labels; y[i] = c means
        that X[i] has label, c where 0 <= c < C
    :param reg: A floating number representing the regularization strength.

    :return: A tuple of:
        - loss as single float
        - gradient with respect to weights W; a numpy array of the same shape as W.
    """

    dW = np.zeros_like(W)  # initialize the gradient as zero

    # compute the loss and the gradient
    num_classes = W.shape[1]
    num_train = X.shape[0]
    loss = 0.0

    for i in range(num_train):
        scores = X[i].dot(W)
        correct_class_score = scores[y[i]]
        for j in range(num_classes):
            if j == y[i]:
                continue
            margin = scores[j] - correct_class_score + 1  # Note: delta = 1
            if margin > 0:
                loss += margin
                dW[np.arange(dW.shape[0]), j] += X[i]
                dW[np.arange(dW.shape[0]), y[i]] -= X[i]

    # Right now the loss is a sum over all training examples, but we want it
    # to be an average instead, so we divide by num_train.
    loss /= num_train
    dW /= num_train

    # Add regularization to the loss.
    loss += reg * np.sum(W * W)
    dW += 2 * reg * W

    #############################################################################
    # TODO:                                                                     #
    # Compute the gradient of the loss function and store it dW.                #
    # Rather that first computing the loss and then computing the derivative,   #
    # it may be simpler to compute the derivative at the same time that the     #
    # loss is being computed. As a result you may need to modify some of the    #
    # code above to compute the gradient.                                       #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    pass

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    return loss, dW


def svm_loss_vectorized(W: np.ndarray, X: np.ndarray, y: np.ndarray, reg: float) -> Tuple[float, np.ndarray]:
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

    loss = 0.0
    dW = np.zeros_like(W)  # initialize the gradient as zero

    #############################################################################
    # TODO:                                                                     #
    # Implement a vectorized version of the structured SVM loss, storing the    #
    # result in loss.                                                           #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    num_train = X.shape[0]

    scores = X.dot(W)  # (N, C) matrix
    dScores = X.T  # derivative of scores with respect to W

    margin = np.fmax(0, scores - scores[np.arange(num_train), y].reshape(-1, 1) + 1)  # delta = 1
    margin[np.arange(num_train), y] = 0  # set the margin to 0 for the class where we have label

    dMargin = np.where(margin > 0, 1, 0)  # derivative of margin with respect to scores
    dMargin[np.arange(num_train), y] = -np.count_nonzero(margin, axis=1)

    loss = np.sum(margin)
    dW = dScores.dot(dMargin)

    loss /= num_train
    dW /= num_train

    loss += reg * np.sum(W * W)
    dW += 2 * reg * W
    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    #############################################################################
    # TODO:                                                                     #
    # Implement a vectorized version of the gradient for the structured SVM     #
    # loss, storing the result in dW.                                           #
    #                                                                           #
    # Hint: Instead of computing the gradient from scratch, it may be easier    #
    # to reuse some of the intermediate values that you used to compute the     #
    # loss.                                                                     #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    pass

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    return loss, dW

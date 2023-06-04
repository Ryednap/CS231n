import numpy as np


# Type alias that refers to cache for affine layers which is saved in forward pass.
# The cache represents Tuple of (input, weight, bias)
AffineCache = tuple[np.ndarray, np.ndarray, np.ndarray]

# Type alias that refers to gradient object that is returned from the backward
# pass from `affine_backward`. The grad represents (dx, dw, db).
AffineGrad = tuple[np.ndarray, np.ndarray, np.ndarray]

# Type alias that refers to cache from relu layers which is saved in forward pass.
# The cache represents a numpy array containing the input itself.
ReluCache = np.ndarray

# Type alias that refers to gradient object that is returned from the backward
# pass from `relu_backward`. The gradient represents `dx` gradient with respect to input.
ReluGrad = np.ndarray


def affine_forward(x: np.ndarray, w: np.ndarray, b: np.ndarray) -> tuple[np.ndarray, AffineCache]:
    """
    Computes the forward pass for an affine (fully-connected) layer.

    The in put `x` has shape (N, d_1, ..., d_k) and contains a minibatch of N examples,
    where each example x[i] has shape (d_1, ..., d_k). We will reshape each input into
    a vector of dimension D = d_1 * d_2 * .... * d_k (effectively flattening out), and
    then transform it to an output vector of dimension M.

    :param x: A numpy array containing input data of shape (N, d_1, ..., d_k)
    :param w: A numpy array of weights, of shape (D, M) where D = d_1 * d_2 * .... * d_k
    :param b: A numpy array of biases, of shape (M, )
    :return: A tuple of:
        - out: A numpy array as forward pass output of shape (N, M)
        - cache: An `AffineCache` type containing (w, x, b).
    """

    out = None
    ###########################################################################
    # TODO: Implement the affine forward pass. Store the result in out. You   #
    # will need to reshape the input into rows.                               #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    pass

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    cache = (x, w, b)
    return out, cache


def affine_backward(dout: np.ndarray, cache: AffineCache) -> AffineGrad:
    """
    Computes the backward pass for an affine layer (full-connected layer).

    :param dout: Upstream derivative, of shape (N, M)
    :param cache: `AffineCache` type containing:
        - x: Input data, of shape (N, d_1, d_2, ..., d_k)
        - w: Weights of shape (D, M) where D = d_1 * d_2 * ... * d_k
        - b: Biases of shape (M, )
    :return: An `AffineGrad` which is tuple of:
        - dx: A numpy array representing gradient of upstream with respect to x, of shape (N, d_1, ..., d_k).
        - dw: A numpy array representing gradient of upstream with respect to w, of shape (D, M).
        - db: A numpy array representing gradient of upstream with respect to b, of shape (M, ).
    """
    x, w, b = cache
    dx, dw, db = None, None, None

    ###########################################################################
    # TODO: Implement the affine backward pass.                               #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    pass

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return dx, dw, db


def relu_forward(x: np.ndarray) -> tuple[np.ndarray, ReluCache]:
    """
    Computes the forward pass for a layer of rectified linear units (ReLUs).

    :param x: A numpy array representing Inputs, of any shape
    :return: A tuple of:
        - out: A numpy array resulting from forward pass, of the same shape as x.
        - cache: A `ReluCache` containing input.
    """

    out = None
    ###########################################################################
    # TODO: Implement the ReLU forward pass.                                  #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    pass

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    cache = x
    return out, cache


def relu_backward(dout: np.ndarray, cache: ReluCache) -> ReluGrad:
    """
    Computes the backward pass for a layer of rectified linear units (ReLUs).

    :param dout: Upstream derivatives, of any shape
    :param cache: A `ReluCache` type containing Input x, of shape as dout.
    :return: A `ReluGrad` object which contains:
        - dx: A numpy array of shape same as x representing gradient of upstream with respect to x.
    """

    dx, x = None, cache
    ###########################################################################
    # TODO: Implement the ReLU backward pass.                                 #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    pass

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return dx


def svm_loss(x: np.ndarray, y: np.ndarray) -> tuple[float, np.ndarray]:
    """
    Computes the loss and gradient using multiclass SVM (Hinge Loss).
    The loss is computed for *x*, and the gradient is calculated with respect to *x*.

    :param x: Input data, of shape (N, C) where x[i, j] is the score for jth class for ith input.
    :param y: A numpy array of labels, of shape (N,) where y[i] is the label for x[i] and 0 <= y[i] < c
    :return: A tuple of
        - loss: A float scalar giving the loss
        - dx: A numpy array of the same shape as x, representing gradient of the loss with respect to x.
    """

    loss, dx = None, None

    ###########################################################################
    # TODO: Copy over your solution from A1.
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    pass

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return loss, dx


def softmax_loss(x: np.ndarray, y: np.ndarray) -> tuple[float, np.ndarray]:
    """
    Computes the loss and gradient for softmax classification.
    The loss is computed for *x*, and the gradient is calculated with respect to *x*.

    :param x: Input data, of shape (N, C) where x[i, j] is the score for jth class for ith input.
    :param y: A numpy array of labels, of shape (N,) where y[i] is the label for x[i] and 0 <= y[i] < c
    :return: A tuple of
        - loss: A float scalar giving the loss
        - dx: A numpy array of the same shape as x, representing gradient of the loss with respect to x.
    """

    loss, dx = None, None

    ###########################################################################
    # TODO: Copy over your solution from A1.
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    pass

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return loss, dx

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

    The input `x` has shape (N, d_1, ..., d_k) and contains a minibatch of N examples,
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

    flatten = np.reshape(x, (x.shape[0], -1))  # Assuming batch-first
    out = flatten.dot(w) + b

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

    flatten = np.reshape(x, (x.shape[0], -1))

    dw = flatten.T @ dout

    dx = dout @ w.T  # gradient with respect W.X + b
    dx = np.reshape(dx, x.shape)  # gradient with respect to flatten

    """
        Be aware that np.sum() and matrix multiplication shapes will not match.
        always cross check the shape of the gradient with respect to parameter.
    """
    # db = np.sum(dout, axis = 1)   # one of the other approach
    db = np.ones((1, x.shape[0])) @ dout   # this comes nicely out of chain rule.
    db = np.reshape(db, b.shape)

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

    out = np.fmax(0, x)

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

    dx = np.where(x > 0, dout, 0)

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
    N = x.shape[0]

    margin = np.fmax(0, x - x[np.arange(N), y].reshape(-1, 1) + 1)  # delta = 1
    dx = np.ones_like(x)

    margin[np.arange(N), y] = 0
    dx[np.arange(N), y] = 0

    loss = np.sum(margin)
    dx[np.arange(N), y] = -np.count_nonzero(margin, axis=1)

    loss /= N
    dx /= N

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

    N = x.shape[0]

    expo = np.exp(x - np.max(x, axis=1, keepdims=True))  # would save a lot of time

    proba = expo[np.arange(N), y].reshape(-1, 1) / np.sum(expo, axis=1, keepdims=True)

    nll = -np.log(proba)

    dx = expo / np.sum(expo, axis=1, keepdims=True)
    dx[np.arange(N), y] -= 1

    loss = np.sum(nll) / N
    dx /= N

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return loss, dx

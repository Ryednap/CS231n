import typing

import numpy as np

from src.layer_utils import *


class TwoLayerNet(object):
    """
    A two-layer fully-connected neural network with ReLU non-linearity and softmax loss
    that uses modular layer design. We assume an input dimension of D, a hidden dimension of H,
    and perform classification over C classes.

    The architecture should be : affine -> relu -> affine -> softmax.

    Note this class doesn't implement gradient descent; instead, it will interact with a
    separate `Solver` object that is responsible for running optimization.

    The learnable parameters of the model are stored in dictionary `self.params` that
    maps parameters names to numpy arrays.
    """

    def __init__(
            self, input_dim=3 * 32 * 32,
            hidden_dim=100, num_classes=10,
            weight_scale=1e-3, reg=0.0
    ):
        """
        Initialize a new network.

        :param input_dim: An integer giving the size of the input
        :param hidden_dim: An integer giving the size of hidden layer
        :param num_classes: An integer giving the number of classes to classify
        :param weight_scale: A scalar giving the standard deviation for random
               initialization of the weights.
        :param reg: A scalar giving L2 regularization strength.
        """

        self.params = {}
        self.reg = reg

        ############################################################################
        # TODO: Initialize the weights and biases of the two-layer net. Weights    #
        # should be initialized from a Gaussian centered at 0.0 with               #
        # standard deviation equal to weight_scale, and biases should be           #
        # initialized to zero. All weights and biases should be stored in the      #
        # dictionary self.params, with first layer weights                         #
        # and biases using the keys 'W1' and 'b1' and second layer                 #
        # weights and biases using the keys 'W2' and 'b2'.                         #
        ############################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

        pass

        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

    @typing.overload
    def loss(self, X: np.ndarray, y=None) -> float:
        pass

    @typing.overload
    def loss(self, X: np.ndarray, y=None) -> tuple[float, dict]:
        pass

    def loss(self, X: np.ndarray, y=None) -> typing.Union[float, tuple[float, dict]]:
        """
        Computes loss and gradient for a mini-batch of data.

        :param X: A numpy array representing input data of shape (N, d_1, ..., d_k)
        :param y: A numpy array containing labels, of shape (N, ). y[i] contains label for X[i].
        :return:
            If y is None, then run a test-time forward pass of the model and return:
                - scores: A numpy array of shape (N, C) giving the classification scores, where
                  scores[i, c] is the classification score for X[i] and class c.

            If y is not None, then run a training-time forward pass of the model and return:
                - loss: Scalar value giving the loss
                - grads: Dictionary with the same keys as self.params, mapping parameter
                  names to gradients of the loss with respect to those parameters.
        """

        scores = None
        ############################################################################
        # TODO: Implement the forward pass for the two-layer net, computing the    #
        # class scores for X and storing them in the scores variable.              #
        ############################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

        pass

        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        # If y is None, then we are in test mode so just return scores
        if y is None:
            return scores

        loss, grads = 0, {}
        ############################################################################
        # TODO: Implement the backward pass for the two-layer net. Store the loss  #
        # in the loss variable and gradients in the grads dictionary. Compute data #
        # loss using softmax, and make sure that grads[k] holds the gradients for  #
        # self.params[k]. Don't forget to add L2 regularization!                   #
        #                                                                          #
        # NOTE: To ensure that your implementation matches ours and you pass the   #
        # automated tests, make sure that your L2 regularization includes a factor #
        # of 0.5 to simplify the expression for the gradient.                      #
        ############################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

        pass

        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        return loss, grads

from typing import List, Tuple

import numpy as np

from src.classifiers.linear_svm import svm_loss_vectorized
from src.classifiers.softmax import softmax_loss_vectorized


class LinearClassifier(object):
    def __init__(self):
        self.W = None

    def __int__(self):
        self.W = None

    def train(
            self,
            X: np.ndarray,
            y: np.ndarray,
            learning_rate=1e-3,
            reg=1e-5,
            num_iters=100,
            batch_size=200,
            verbose=False

    ) -> List[float]:
        """
        Train this linear classifier using stochastic gradient descent (SGD).

        :param X: A numpy array of shape (N, D) containing training data; there are N
            training samples each of dimension D.
        :param y: A numpy array of shape (N,) containing training labels;
            y[i] = c means that X[i] has label 0 <= c < C for C classes.
        :param learning_rate: Learning rate for optimization.
        :param reg: Regularization strength.
        :param num_iters: Number of steps to take when optimizing.
        :param batch_size: Number of training examples to use at each step.
        :param verbose: If True, print progress during optimization.

        :return: A List containing the value of the loss function at each training iteration.
        """

        num_train, dim = X.shape
        num_classes = np.max(y) + 1  # assume y takes values 0...K-1 where K is the number of classes.

        if self.W is None:
            # lazily initialize W
            self.W = 0.001 * np.random.randn(dim, num_classes)  # (D, C) matrix

        # Run stochastic gradient descent to optimize W
        loss_history = []
        for it in range(num_iters):
            X_batch = None
            y_batch = None

            #########################################################################
            # TODO:                                                                 #
            # Sample batch_size elements from the training data and their           #
            # corresponding labels to use in this round of gradient descent.        #
            # Store the data in X_batch and their corresponding labels in           #
            # y_batch; after sampling X_batch should have shape (batch_size, dim)   #
            # and y_batch should have shape (batch_size,)                           #
            #                                                                       #
            # Hint: Use np.random.choice to generate indices. Sampling with         #
            # replacement is faster than sampling without replacement.              #
            #########################################################################
            # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

            pass

            # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

            # evaluate loss and gradient
            loss, grad = self.loss(X_batch, y_batch, reg)
            loss_history.append(loss)

            # perform parameter update
            #########################################################################
            # TODO:                                                                 #
            # Update the weights using the gradient and the learning rate.          #
            #########################################################################
            # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

            pass

            # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

            if verbose and it % 100 == 0:
                print(f"iteration {it} / {num_iters}: {loss}")

        return loss_history

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        This method uses the trained weights of this linear classifier to predict labels for data points.

        :param X: A numpy array of shape (N, D) containing testing data; there are N test samples each of dimension D.

        :return: A numpy array of shape (N,) containing predicted labels for the data in X. Each element is an integer
            giving the predicted class for the given data.
        """
        y_pred = np.zeros(X.shape[0])

        ###########################################################################
        # TODO:                                                                   #
        # Implement this method. Store the predicted labels in y_pred.            #
        ###########################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

        pass

        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        return y_pred

    def loss(self, X_batch: np.ndarray, y_batch: np.ndarray, reg: float) -> Tuple[float, np.ndarray]:
        """
        Compute the loss function and its derivative.
        Subclasses will override this to substitute their own implementation.

        :param X_batch: A numpy array of shape (N, D) containing a minibatch of N data points
            each point has dimension D.
        :param y_batch: A numpy array of shape (N,) containing labels for the minibatch.
        :param reg: A float denoting the regularization strength.

        :return: A tuple containing:
            - loss as a single float
            - gradient with respect to self.W; a numpy array of same shape as W.
        """
        pass


class LinearSVM(LinearClassifier):
    """A subclass that uses the Multiclass SVM loss function"""

    def loss(self, X_batch: np.ndarray, y_batch: np.ndarray, reg: float) -> Tuple[float, np.ndarray]:
        return svm_loss_vectorized(self.W, X_batch, y_batch, reg)


class Softmax(LinearClassifier):
    """A subclass that uses the Softmax + Cross-entropy loss (NLL) function."""

    def loss(self, X_batch: np.ndarray, y_batch: np.ndarray, reg: float) -> Tuple[float, np.ndarray]:
        return softmax_loss_vectorized(self.W, X_batch, y_batch, reg)

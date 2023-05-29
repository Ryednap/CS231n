import numpy as np


class KNearestNeighbor(object):

    def __int__(self):
        pass

    def train(self, X: np.ndarray, y: np.ndarray) -> None:
        """
        Train the classifier. For k-nearest neighbors, this is just
        memorizing the training data. Hence, only takes O(1) time.

        :param X: A numpy array of shape (num_train, D) containing the training data
                consisting of num_train samples each of dimension D.
        :param y: A numpy array of shape (num_train, ) containing the training labels,
                where y[i] is the label for X[i].
        :return: None
        """

        self.X_train = X
        self.y_train = y

    def predict(self, X: np.ndarray, k=1, num_loops=0) -> np.ndarray:
        """
        Predict labels for test data using this classifier.

        :param X: A numpy array of shape (num_test, D) containing the test data of
                num_test samples each of dimension D.
        :param k: The number of nearest neighbors that vote for the predicted labels.
        :param num_loops: Determines which implementation to use to compute distances
                between training points and testing points.
        :return: A numpy array of shape (num_test, ) containing predicted labels.
        """
        pass

    def compute_distances_two_loops(self, X: np.ndarray) -> np.ndarray:
        """
        Compute the distance between each test point in `X` and each training point
        in `self.X_train` using a nested loop over both the training data and the
        test data.

        :param X: A numpy array of shape (num_test, D) containing test data
        :return: `dists` - A numpy array of shape (num_test, num_train) where,
                dists[i, j] is the Euclidean distance between the ith test point and
                the jth training point.
        """

        num_test = X.shape[0]
        num_train = self.X_train.shape[0]
        dists = np.zeros((num_test, num_train))

        for i in range(num_test):
            for j in range(num_train):
                #####################################################################
                # TODO:                                                             #
                # Compute the l2 distance between the ith test point and the jth    #
                # training point, and store the result in dists[i, j]. You should   #
                # not use a loop over dimension, nor use np.linalg.norm().          #
                #####################################################################
                # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
                pass

                # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

        return dists

    def compute_distance_one_loops(self, X: np.ndarray) -> np.ndarray:
        """
        Compute the distance between each test point in `X` and each training point
        in `self.X_train` using a single loop over the test data.

        :param X: A numpy array of shape (num_test, D) containing test data
        :return: `dists` - A numpy array of shape  (num_test, num_train) where,
                dists[i, j] is the Euclidean distance between the ith test point and
                the jth training point.
        """

        num_test = X.shape[0]
        num_train = self.X_train.shape[0]
        dists = np.zeros((num_test, num_train))

        for i in range(num_test):
            #####################################################################
            # TODO:                                                             #
            # Compute the l2 distance between the ith test point and the jth    #
            # training point, and store the result in dists[i, j]. You should   #
            # not use a loop over dimension, nor use np.linalg.norm().          #
            #####################################################################
            # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
            pass

            # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

        return dists

    def compute_distance_no_loops(self, X: np.ndarray) -> np.ndarray:
        """
        Compute the distance between each test point in `X` and each training point
        in `self.X_train` using no explicit loops.

        :param X: A numpy array of shape (num_test, D) containing test data
        :return: `dists` - A numpy array of shape  (num_test, num_train) where,
                dists[i, j] is the Euclidean distance between the ith test point and
                the jth training point.
        """

        num_test = X.shape[0]
        num_train = self.X_train.shape[0]
        dists = np.zeros((num_test, num_train))

        #########################################################################
        # TODO:                                                                 #
        # Compute the l2 distance between all test points and all training      #
        # points without using any explicit loops, and store the result in      #
        # dists.                                                                #
        #                                                                       #
        # You should implement this function using only basic array operations; #
        # in particular you should not use functions from scipy,                #
        # nor use np.linalg.norm().                                             #
        #                                                                       #
        # HINT: Try to formulate the l2 distance using matrix multiplication    #
        #       and two broadcast sums.                                         #
        #########################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

        pass

        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

        return dists

    def predict_labels(self, dists: np.ndarray, k=1) -> np.ndarray:
        """
        Given a matrix `(dists)` of distances between test points and training points,
        predict a label for each test point.

        :param dists: A numpy array of shape (num_test, num_train) where dists[i, j]
                gives the distance between the ith test point and the jth training point.
        :param k: Integer representing number of k-nearest neighbours distances to consider
                to vote for label.
        :return: A numy array of shape (num_test, ) containing predicted labels for the test data,
                where y[i] is the predicted label for the test point X[i].
        """

        num_test = dists.shape[0]
        y_pred = np.zeros(num_test)

        for i in range(num_test):
            # A list of length k storing the labels of the k nearest neighbors to
            # the ith test point.
            closest_y = []
            #########################################################################
            # TODO:                                                                 #
            # Use the distance matrix to find the k nearest neighbors of the ith    #
            # testing point, and use self.y_train to find the labels of these       #
            # neighbors. Store these labels in closest_y.                           #
            # Hint: Look up the function numpy.argsort.                             #
            #########################################################################
            # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

            pass

            # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

            #########################################################################
            # TODO:                                                                 #
            # Now that you have found the labels of the k nearest neighbors, you    #
            # need to find the most common label in the list closest_y of labels.   #
            # Store this label in y_pred[i]. Break ties by choosing the smaller     #
            # label.                                                                #
            #########################################################################
            # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

            pass

            # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

        return y_pred

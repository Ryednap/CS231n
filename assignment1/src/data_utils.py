from __future__ import print_function

import os.path
from typing import Any, Tuple, Dict

import pickle
import platform
import numpy as np


def load_pickle(f) -> Dict[str, Any]:
    """
    Function to load pickle file depending on the user's python version
    :param f: pickle file to load
    :return: dictionary representing the loaded pickle file.
    """

    version = platform.python_version_tuple()
    if version[0] == "2":
        return pickle.load(f)
    elif version[0] == "3":
        return pickle.load(f, encoding="latin1")

    raise ValueError(f"invalid python version: {version}")


def load_CIFAR_batch(filename: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Function to load a single batch of CIFAR dataset. The single batch
    is of size 10000.
    :param filename: String representing the file path of the dataset.
    :return: Tuple of X and y i.e. input and output
    """

    with open(filename, "rb") as f:
        datadict = load_pickle(f)
        X = datadict['data']
        y = datadict['labels']

        # Reshape the images to have a channel last. By having the channel last property,
        # the images are by default compatible to be of using to various image libraries
        # and also more importantly, CNN.
        X = X.reshape(10000, 3, 32, 32).transpose(0, 2, 3, 1).astype("float")
        y = np.array(y)

        return X, y


def load_CIFAR10(ROOT: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Function to load all CIFAR dataset.
    :param ROOT: String representing the ROOT path from where data will be read
    :return: 4 tuples of (X_train, y_train, X_test, y_test).
    """
    xs = []
    ys = []

    for b in range(1, 6):
        f = os.path.join(ROOT, f"data_batch_{b}")
        X, y = load_CIFAR_batch(f)
        xs.append(X)
        ys.append(y)

    Xtr = np.concatenate(xs)  # X train stuff
    Ytr = np.concatenate(ys)  # y train stuff

    del X, y  # garbage collect

    Xte, Yte = load_CIFAR_batch(os.path.join(ROOT, "test_batch"))  # testing stuff
    return Xtr, Ytr, Xte, Yte


def get_CIFAR10_data(
        num_training=49000, num_validation=1000,
        num_test=1000, subtract_mean=True
) -> Dict[str, np.ndarray]:
    """
    Load CIFAR-10 dataset from disk and perform preprocessing to prepare
    it for classifiers. These are the same steps as we used for SVM, but
    condensed to a single function.
    :param num_training: Number of training data to sample. By default,
    :param num_validation: number 49000 of validation data to sample. By default, 1000
    :param num_test:  number of test data to sample. By default, 1000
    :param subtract_mean: True if you want the images to be zero centered else False. By default, True.
    :return:
    """

    # Load the raw CIFAR-10 data
    cifar10_dir = os.path.join(
        os.path.dirname(__file__),
        "datasets/cifar-10-batches_py"
    )

    X_train, y_train, X_test, y_test = load_CIFAR10(cifar10_dir)

    # Subsample the data
    mask = list(range(num_training, num_training + num_validation))
    X_val = X_train[mask]
    y_val = y_train[mask]

    mask = list(range(num_training))
    X_train = X_train[mask]
    y_train = y_train[mask]

    mask = list(range(num_test))
    X_test = X_test[mask]
    y_test = y_test[mask]

    # Normalize the data: subtract the mean image to make it zero-centered
    if subtract_mean:
        mean_image = np.mean(X_train, axis=0)
        X_train -= mean_image
        X_val -= mean_image
        X_test -= mean_image

    # Transpose so that channels come first and then copy
    # to prevent any issues with copies or references.
    X_train = X_train.transpose(0, 3, 1, 2).copy()
    X_val = X_val.transpose(0, 3, 1, 2).copy()
    X_test = X_test.transpose(0, 3, 1, 2).copy()

    # Package the data into a dictionary
    return {
        "X_train": X_train,
        "y_train": y_train,
        "X_val": X_val,
        "y_val": y_val,
        "X_test": X_test,
        "y_test": y_test
    }

import pickle
from copy import deepcopy
from src import optim

import numpy as np


class Solver(object):
    """
    A solver encapsulates all the logic necessary for training classification models.
    The solver performs stochastic gradient descent using different update rules as
    defined in `optim.py`.

    The solver accepts both training and validation data and labels, so it can
    periodically check classification accuracy on both training and validation data
    to watch out for overfitting.

    To train a model, you will first construct a `Solver` instance, passing the model,
    dataset, and various options (learning_rate, batch_size, etc.) to the constructor.
    You will then call the `train()` method to run the optimization procedure and
    train the model.

    After the `train()` method returns, `model.params` will contain the parameters
    that performed best on validation set over the course of training.
    In addition, the instance variable `solver.loss` history will contain a list of all
    losses encountered during training and the instance variable `solver.train_acc_history`
    and `solver.val_acc_history` will be lists of the accuracies of the model on the training
    and validation set at each epoch.

    Example usage might look something look like this::

        data = {
          'X_train': # training data
          'y_train': # training labels
          'X_val': # validation data
          'y_val': # validation labels
        }
        model = MyAwesomeModel(hidden_size=100, reg=10)
        solver = Solver(model, data,
                        update_rule='sgd',
                        optim_config={
                          'learning_rate': 1e-4,
                        },
                        lr_decay=0.95,
                        num_epochs=5, batch_size=200,
                        print_every=100)
        solver.train()

    A solver works on a model object that must conform to the following API:

    - model.params must be a dictionary mapping string parameter names to numpy
      arrays containing parameter values.

    - model.loss(X, y) must be a function that computes training-time loss and
      gradients, and test-time classification scores, with the following inputs
      and outputs:

      Inputs:
        - X: Array giving a minibatch of input data of shape (N, d_1, ..., d_k)
        - y: Array of labels, of shape (N,) giving labels for X where y[i] is the
            label for X[i].

      Returns:
          If y is None, run a test-time forward pass and return:
              - scores: Array of shape (N, C) giving classification scores for X where
                scores[i, c] gives the score of class c for X[i].

      If y is not None, run a training time forward and backward pass and
      return a tuple of:
          - loss: Scalar giving the loss
          - grads: Dictionary with the same keys as self.params mapping parameter
            names to gradients of the loss with respect to those parameters.

    """

    def __init__(self, model, data, **kwargs):
        """
        Construct a new Solver instance.

        Required arguments:
            - model: A model object conforming to the API described above.
            - data: A dictionary of training and validation data containing:

              `X_train`: Array of shape (N_train, d_1, ..., d_k) of training images.
              `X_val`: Array of shape (N_val, d_1, ..., d_k) for validation images.
              `y_train`: Array of shape (N_train,) of labels for training images.
              `y_val`: Array of shape(N_val,) of labels for validation images.

        Optional arguments:
            - update_rule: A string giving the name of an update rule defined in `optim.py`. Default `sgd'.
            - optim_config: A dictionary containing hyperparameters that will be passed to the chosen update rule.
              Each update rule requires a `learning_rate` parameter so that should always be present.
            - lr_decay: A scalar for learning rate decay; after each epoch this value multiplies the learning rate.
            - batch_size: Size of mini-batches used to compute loss and gradient during training.
            - num_epochs: The number of epochs to run for during training.
            - print_every: Integer; training losses will be printed every `print_every` iterations.
            - verbose: Boolean; if set to false, then no output will be printed during training.
            - num_train_samples: Number of training samples used to check training accuracy: default is 1000;
              set to None to use an entire training set.
            - num_val_samples: Number of validation samples used to check validation accuracy: default is None,
              which uses an entire validation set.
            - checkpoint_name: If not None, then save model checkpoints here every epoch.
        """

        self.best_params = None
        self.best_val_acc = None
        self.model = model
        self.X_train = data['X_train']
        self.X_val = data['X_val']
        self.y_train = data['y_train']
        self.y_val = data['y_val']

        # Unpack the Optional keyword arguments
        self.update_rule = kwargs.pop("update_rule", "sgd")
        self.optim_config = kwargs.pop("optim_config", {})
        self.lr_decay = kwargs.pop("lr_decay", 1.0)
        self.batch_size = kwargs.pop("batch_size", 100)
        self.num_epochs = kwargs.pop("num_epochs", 10)
        self.num_train_samples = kwargs.pop("num_train_samples", 1000)
        self.num_val_samples = kwargs.pop("num_val_samples", None)

        self.checkpoint_name = kwargs.pop("checkpoint_name", None)
        self.print_every = kwargs.pop("print_every", 10)
        self.verbose = kwargs.pop("verbose", True)

        # Throw an error if there are extra keyword arguments
        if len(kwargs) > 0:
            extra = ", ".join('"{k}"' for k in list(kwargs.keys()))
            raise ValueError(f"Unrecognized arguments {extra}")

        # Make sure that update rule exists, then replace the string name with the actual function
        if not hasattr(optim, self.update_rule):
            raise ValueError(f"Invalid update_rule {self.update_rule}")
        else:
            self.update_rule = getattr(optim, self.update_rule)

        self._reset()

    def _reset(self) -> None:
        """
        Set up some bookkeeping for optimization. Don't call this manually.
        :return: None
        """

        self.epoch = 0
        self.best_val_acc = 0
        self.best_params = {}
        self.loss_history = []
        self.train_acc_history = []
        self.val_acc_history = []

        # Make a deep copy of the optim_config for each parameter
        self.optim_configs = {}
        for p in self.model.params:
            d = {k: v for k, v in self.optim_config.items()}
            self.optim_configs[p] = d

    def _step(self) -> None:
        """
        Make a single gradient update. This is called by train() and should not be called manually.
        :return: None
        """
        # make a mini-batch of training data.
        num_train = self.X_train.shape[0]
        batch_mask = np.random.choice(num_train, self.batch_size, replace=False)
        X_batch = self.X_train[batch_mask]
        y_batch = self.y_train[batch_mask]

        # compute loss and gradient
        loss, grads = self.model.loss(X_batch, y_batch)
        self.loss_history.append(loss)

        # Perform a parameter update
        for p, w in self.model.params.items():
            dw = grads[p]
            config = self.optim_configs[p]
            next_w, next_config = self.update_rule(w, dw, config)
            self.model.params[p] = next_w
            self.optim_configs[p] = next_config

    def _save_checkpoint(self) -> None:
        """
        Saves the `self.model` as a checkpoint. Don't Call manually.
        :return: None
        """
        if self.checkpoint_name is not None:
            checkpoint = {
                "model": self.model,
                "update_rule": self.update_rule,
                "lr_decay": self.lr_decay,
                "optim_config": self.optim_configs,
                "batch_size": self.batch_size,
                "num_train_samples": self.num_train_samples,
                "num_val_samples": self.num_val_samples,
                "epoch": self.epoch,
                "loss_history": self.loss_history,
                "train_acc_history": self.train_acc_history,
                "val_acc_history": self.val_acc_history
            }

            filename = f"{self.checkpoint_name}_epoch_{self.epoch}.pkl"
            if self.verbose:
                print(f"Saving checkout to {filename}")

            with open(filename, "wb") as f:
                pickle.dump(checkpoint, f)

    def check_accuracy(self, X: np.ndarray, y: np.ndarray, num_samples=None, batch_size=100) -> float:
        """
        Check the accuracy of the model on the provided data

        :param X: A numpy array of data, of shape (N, d_1, ..., d_k)
        :param y: A numpy array of labels, of shape (N,)
        :param num_samples: If not None, subsample the data and only test the model on num_samples datapoints
        :param batch_size: Inter using which split X and y into batches of this size to avoid using too much memory.
        :return: A floating scalar representing the accuracy.
        """

        N = X.shape[0]
        if num_samples is not None and N > num_samples:
            mask = np.random.choice(N, num_samples)
            N = num_samples
            X = X[mask]
            y = y[mask]

        # Compute predictions in batches
        num_batches = N // batch_size
        if N % batch_size != 0:
            num_batches += 1

        y_pred = []
        for i in range(num_batches):
            start = i * batch_size
            end = (i + 1) * batch_size
            scores = self.model.loss(X[start: end])  # (N, C) matrix
            y_pred.append(np.argmax(scores, axis=1))

        y_pred = np.hstack(y_pred)  # stack them in a linear row
        acc = np.mean(y_pred == y)

        return acc

    def train(self) -> None:
        """
        Run optimization to train the model.
        :return: None
        """

        num_train = self.X_train.shape[0]
        iterations_per_epoch = max(num_train // self.batch_size, 1)
        num_iterations = self.num_epochs * iterations_per_epoch

        for t in range(num_iterations):
            self._step()

            # print training loss if verbose
            if self.verbose and (t % self.print_every == 0):
                print(
                    "(Iteration %d / %d) loss: %f"
                    % (t + 1, num_iterations, self.loss_history[-1])
                )

            # At the end of every epoch, increment the epoch counter and decay the learning rate.
            epoch_end = ((t + 1) % iterations_per_epoch == 0)
            if epoch_end:
                self.epoch += 1
                for k in self.optim_configs:
                    self.optim_configs[k]["learning_rate"] *= self.lr_decay

            # check train and val accuracy on the first iteration, the last iteration, and at the end of each epoch
            first_it = (t == 0)
            last_it = (t == num_iterations - 1)
            if first_it or last_it or epoch_end:
                train_acc = self.check_accuracy(
                    self.X_train, self.y_train, num_samples=self.num_train_samples
                )
                val_acc = self.check_accuracy(
                    self.X_val, self.y_val, num_samples=self.num_val_samples
                )
                self.train_acc_history.append(train_acc)
                self.val_acc_history.append(val_acc)
                self._save_checkpoint()

                if self.verbose:
                    print(
                        "(Epoch %d / %d) train acc: %f; val_acc: %f"
                        % (self.epoch, self.num_epochs, train_acc, val_acc)
                    )

                # Keep track of the best model
                if val_acc > self.best_val_acc:
                    self.best_val_acc = val_acc
                    self.best_params = {}
                    for k, v in self.model.params.items():
                        self.best_params[k] = v.copy()

            # At the end of training, swap the best params into the model
            self.model.params = self.best_params


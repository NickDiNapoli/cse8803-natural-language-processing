import pandas as pd
import numpy as np
from scipy.special import softmax
from sklearn.preprocessing import OneHotEncoder

from warnings import simplefilter
# ignore all future warnings
simplefilter(action='ignore', category=FutureWarning)


class Regression(object):
    def __init__(self):
        """
        Initialize instance of OneHotEncoder in self.oh for use in onehot function.
        """
        self.oh = OneHotEncoder()

    def onehot(self, labels):
        """
        Helper function to encode a labels into one hot encoding format

        Args:
                labels: list of class labels

        Return:
                onehotencoded: (N, C) numpy array where:
                        N is the number of datapoints in the list 'labels'
                        C is the number of distinct labels/classes in 'labels'

        Hint: np.array may be helpful in converting a sparse matrix into a numpy array

        """
        labels = np.asarray(labels)
        onehotencoded = self.oh.fit_transform(labels.reshape(-1, 1)).toarray()
        return onehotencoded

    def gradient(self, X, Y, W):
        """
        Apply softmax function to compute the predicted labels and calculate the gradients of the loss w.r.t the weights.

        Args:
                X: (N, D) numpy array of the TF-IDF features for the data.
                Y: (N, C) numpy array of the one-hot encoded labels.
                W: (D, C) numpy array of weights.

        Return:
                gradient: (D,C) numpy array of the computed gradients

        Hint: Use the formula in Section 1.1 of HW2.ipynb to compute the gradients
        """
        mu = 0.01  # Do not change momentum value.
        output = np.matmul(X, W)
        probs = softmax(output, axis=1)
        # preds = np.argmax(probs, axis=1)
        gradient = (-1/X.shape[0] * np.matmul(X.T, Y-probs)) + 2*mu*W

        return gradient

    def gradient_descent(self, X, Y, epochs=10, eta=0.1):
        """
        Basic gradient descent algorithm with fixed eta and mu

        Args:
                X: (N, D) numpy array of the TF-IDF features for the data.
                Y: (N, C) numpy array of the one-hot encoded labels.
                epochs: Number of epochs for the gradient descent (optional, defaults to 10).
                eta: Learning rate (optional, defaults to 0.1)

        Return:
                weight: (D,C) weight matrix

        Hint: Weight should be initialized to be zeros
        """
        D, C = X.shape[1], Y.shape[1]
        weight = np.zeros(shape=(D,C))
        for e in range(epochs):
            weight = weight -eta*self.gradient(X, Y, weight)

        return weight

    def fit(self, data, labels):
        """
        Fit function for calculating the weights using gradient descent algorithm.
        NOTE : This function is given and does not have to be implemented or changed.

        Args:
                data: (N, D) TF-IDF features for the data.
                labels: (N, ) list of class labels
        """

        X = np.asarray(data)
        Y_onehot = self.onehot(labels)
        self.W = self.gradient_descent(X, Y_onehot)

    def predict(self, data):
        """
        Predict function for predicting the class labels.

        Args:
                data: (N, D) TF-IDF features for the data.

        Return:
                predictedLabels: (N,) list of predicted classes for the data.
        """
        output = np.matmul(data, self.W)
        probs = softmax(output, axis=1)
        predictedLabels = np.argmax(probs, axis=1)

        return predictedLabels

import numpy as np
from sklearn.preprocessing import OneHotEncoder

from warnings import simplefilter
# ignore all future warnings
simplefilter(action='ignore', category=FutureWarning)

class Perceptron(object):
    def __init__(self):
        """
        Initialize instance of OneHotEncoder for use in onehot function.
        """
        self.oh = OneHotEncoder()

    def onehot(self, Y):
        """
        Helper function to encode the labels into one hot encoding format used in the one-vs-all classifier.
        Replace the class label from 0 and 1 to -1 and 1.

        Args:
                Y: list of class labels

        Return:
                onehotencoded: (N, C) numpy array where:
                                N is the number of datapoints in the list 'Y'
                                C is the number of distinct labels/classes in 'Y'

        Hint:
        1. It may be helpful to refer to sklearn documentation for the OneHotEncoder
        https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.OneHotEncoder.html
        2. After the normal onehot encoding, replace the 0's with -1's because we will use this for the one-vs-all classifier

        """
        labels = np.asarray(Y)
        onehotencoded = self.oh.fit_transform(labels.reshape(-1, 1)).toarray()
        onehotencoded[onehotencoded == 0.] = -1
        #print(onehotencoded)
        return onehotencoded


    def perceptron(self, X, Y, epochs=10):
        """
        1 vs all perceptron algorithm. We will use the regularization term (alpha) as 1.

        Args:
                X: (N, D) numpy array of the TF-IDF features for the data.
                Y: (N, C) numpy array of the one-hot encoded labels.
                epochs: Number of epochs for perceptron.

        Return:
                weight: (D, 1) weight matrix 

        Hint:
        1. Initialize weight to be 0s
        2. Read the documentation + code of the fit( ) method to understand what needs to be returned by this method.
        """
        '''
        print(X.shape)
        D = X.shape[1]
        weight = np.zeros(shape=(D, 1))
        for e in range(epochs):
            for i in range(X.shape[0]):
                print(i)
                #print(np.count_nonzero(X[i]))
                #print('--------')
                pred = np.matmul(X[i], weight)
                #print(pred, Y[i])
                print(np.matmul(Y[i].T, X[i]).shape)
                if pred*Y[i] < 0:
                    weight = weight + np.matmul(Y[i].T, X[i])
                    #print(weight)

        return weight
        '''
        '''
        print(X)
        #print(Y)
        D = X.shape[1]
        weight = np.zeros(shape=(D, 1))

        for e in range(epochs):
            print(f"epoch : {e}")
            for i in range(X.shape[0]):
                print(X[i].shape, weight.shape)
                pred = np.dot(X[i], weight)
                print(pred, pred.shape)
                #print(Y[i])
                if np.any(pred * Y[i] <= 0):
                    print('true')
                    weight += np.expand_dims(Y[i], axis=1) * np.expand_dims(X[i], axis=0)
                    #weight = weight + np.dot(X[i].T, Y[i])
        print(np.count_nonzero(weight))
        '''
        D = X.shape[1]
        C = Y.shape[1] if len(Y.shape) > 1 else 1
        weight = np.zeros((D, C))

        for _ in range(epochs):
            for i in range(X.shape[0]):
                pred = np.dot(X[i], weight)
                #print(pred)
                if np.any(pred * Y[i] <= 0):
                    weight[:, :] += np.outer(X[i], Y[i])
                    #print(np.count_nonzero(weight), weight.shape)

        return weight

    def fit(self, data, labels):
        """
        Fit function for calculating the weights using perceptron.
        NOTE : This function is given and does not have to be implemented or changed.

        Args:
                data: (N, D) TF-IDF features for the data.
                labels: (N, ) list of class labels
        """

        bias_ones = np.ones((len(data), 1))
        X = np.hstack((data, bias_ones))
        Y = self.onehot(labels)

        self.classes = Y.shape[1]
        self.weights = np.zeros((X.shape[1], Y.shape[1]))

        for i in range(Y.shape[1]):
            W = self.perceptron(X, Y[:, i])
            #print(W)
            self.weights[:, i] = W[:, 0]

    def predict(self, data):
        """
        Predict function for predicting the class labels.

        Args:
                data: (N, D) TF-IDF features for the data.

        Return:
                predictedLabels: list of predicted classes for the data.
        """
        #print(data.shape, self.weights.shape)
        #print(self.weights[0])
        #print(self.weights[-1])
        #print(self.weights)
        bias_ones = np.ones((data.shape[0], 1))
        data = np.hstack((data, bias_ones))
        #predictedLabels = np.matmul(data, self.weights).tolist()
        predictedLabels = np.argmax(np.dot(data, self.weights), axis=1).tolist()

        return predictedLabels

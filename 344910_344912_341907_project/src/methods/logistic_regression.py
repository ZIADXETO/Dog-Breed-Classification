import numpy as np

from ..utils import get_n_classes, label_to_onehot, onehot_to_label, accuracy_fn

def f_softmax(data, W):
    """
    Softmax function for multi-class logistic regression.
    
    Args:
        data (array): Input data of shape (N, D)
        W (array): Weights of shape (D, C) where C is the number of classes
    Returns:
        array of shape (N, C): Probability array where each value is in the
            range [0, 1] and each row sums to 1.
            The row i corresponds to the prediction of the ith data sample, and 
            the column j to the jth class. So element [i, j] is P(y_i=k | x_i, W)
    """
    v = np.exp(data@W)
    return v / np.sum(v, axis = 1, keepdims = True)

def gradient_logistic_multi(data, labels, W):
    """
    Compute the gradient of the entropy for multi-class logistic regression.
    
    Args:
        data (array): Input data of shape (N, D)
        labels (array): Labels of shape  (N, C)  (in one-hot representation)
        W (array): Weights of shape (D, C)
    Returns:
        grad (np.array): Gradients of shape (D, C)
    """
    v = f_softmax(data, W) - labels
    return data.T @ v

def logistic_regression_predict_multi(data, W):
    """
    Prediction the label of data for multi-class logistic regression.
    
    Args:
        data (array): Dataset of shape (N, D).
        W (array): Weights of multi-class logistic regression model of shape (D, C)
    Returns:
        array of shape (N,): Label predictions of data.
    """
    probabilities = f_softmax(data, W)
    return np.argmax(probabilities, axis = 1)

def logistic_regression_train_multi(data, labels, max_iters, lr):
    """
    Training function for multi class logistic regression.
    
    Args:
        data (array): Dataset of shape (N, D).
        labels (array): Labels of shape (N, C)
        max_iters (int): Maximum number of iterations.
        lr (int): The learning rate of  the gradient step.
    Returns:
        weights (array): weights of the logistic regression model, of shape(D, C)
    """
    D, C = data.shape[1], get_n_classes(labels)
    lab = label_to_onehot(labels)
    weights = np.random.normal(0, 0.1, (D, C))
    for _ in range(max_iters):
        gradient = gradient_logistic_multi(data, lab, weights)
        weights -= lr * gradient
        predictions = logistic_regression_predict_multi(data, weights)
        if accuracy_fn(predictions, onehot_to_label(lab)) == 100:
            break
    return weights

class LogisticRegression(object):
    """
    Logistic regression classifier.
    """
    def __init__(self, lr, max_iters = 500, task_kind = "classification"):
        """
        Initialize the new object (see dummy_methods.py)
        and set its arguments.

        Arguments:
            lr (float): learning rate of the gradient descent
            max_iters (int): maximum number of iterations
        """
        self.lr = lr
        self.max_iters = max_iters
        self.task_kind = task_kind
    
    def fit(self, training_data, training_labels):
        """
        Trains the model, returns predicted labels for training data.

        Arguments:
            training_data (array): training data of shape (N,D)
            training_labels (array): regression target of shape (N,)
        Returns:
            pred_labels (array): target of shape (N,)
        """

        self.weight = logistic_regression_train_multi(training_data, training_labels, self.max_iters, self.lr)
        pred_labels = self.predict(training_data)
        return pred_labels
    
    def predict(self, test_data):
        """
        Runs prediction on the test data.

        Arguments:
            test_data (array): test data of shape (N,D)
        Returns:
            pred_labels (array): labels of shape (N,)
        """
        return logistic_regression_predict_multi(test_data, self.weight)

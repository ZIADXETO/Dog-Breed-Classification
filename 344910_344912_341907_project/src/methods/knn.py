import numpy as np

def euclidean_dist(example, training_examples):
    """Compute the Euclidean distance between a single example 
        vector and all training_examples.

    Inputs:
        example: shape (D,)
        training_examples: shape (NxD) 
    Outputs:
        euclidean distances: shape (N,)
    """
    value = (example - training_examples)**2
    return np.sqrt(np.sum(value, axis = 1))

def find_k_nearest_neighbors(k, distances):
    """ Find the indices of the k smallest distances from a list of distances.

    Inputs:
        k: integer
        distances: shape (N,) 
    Outputs:
        indices of the k nearest neighbors: shape (k,)
    """
    indices = np.argsort(distances)
    return indices[:k]

class KNN(object):
    """
        kNN classifier object.
    """
    def kNN_one_example(self, unlabeled_example):
        """Returns the label of a single unlabeled example based on the k-nearest neighbors.
        Inputs:
            unlabeled_example: shape (D,)
        Outputs:
            predicted label
        """
        distances = euclidean_dist(unlabeled_example, self.training_data)
        nn_indices = find_k_nearest_neighbors(self.k, distances)
        neighbor_labels = self.training_labels[nn_indices]
        
        if self.task_kind == "classification":
            return np.argmax(np.bincount(neighbor_labels))
        elif self.task_kind == "regression":
            return np.mean(neighbor_labels, axis = 0)
        else:
            raise ValueError("Unsupported task kind: {}".format(self.task_kind))
        
    def __init__(self, k = 1, task_kind = "classification"):
        """
            Call set_arguments function of this class.
        """
        self.k = k
        self.task_kind = task_kind

    def fit(self, training_data, training_labels):
        """
            Trains the model, returns predicted labels for training data.
            Hint: Since KNN does not really have parameters to train, you can try saving the training_data
            and training_labels as part of the class. This way, when you call the "predict" function
            with the test_data, you will have already stored the training_data and training_labels
            in the object.

            Arguments:
                training_data (np.array): training data of shape (N,D)
                training_labels (np.array): labels of shape (N,)
            Returns:
                pred_labels (np.array): labels of shape (N,)
        """
        self.training_data = training_data
        self.training_labels = training_labels
        return self.predict(training_data)
            
    def predict(self, test_data):
        """
            Runs prediction on the test data.

            Arguments:
                test_data (np.array): test data of shape (N,D)
            Returns:
                test_labels (np.array): labels of shape (N,)
        """
         # Apply kNN_one_example to each element in test_data along axis 1
        test_labels = np.apply_along_axis(self.kNN_one_example, 1, test_data)
        return test_labels
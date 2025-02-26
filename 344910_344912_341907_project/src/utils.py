import numpy as np 
import matplotlib.pyplot as plt


# Generaly utilies
##################

def label_to_onehot(labels, C=None):
    """
    Transform the labels into one-hot representations.

    Arguments:
        labels (array): labels as class indices, of shape (N,)
        C (int): total number of classes. Optional, if not given
                 it will be inferred from labels.
    Returns:
        one_hot_labels (array): one-hot encoding of the labels, of shape (N,C)
    """
    N = labels.shape[0]
    if C is None:
        C = get_n_classes(labels)
    one_hot_labels = np.zeros([N, C])
    one_hot_labels[np.arange(N), labels.astype(int)] = 1
    return one_hot_labels

def onehot_to_label(onehot):
    """
    Transform the labels from one-hot to class index.

    Arguments:
        onehot (array): one-hot encoding of the labels, of shape (N,C)
    Returns:
        (array): labels as class indices, of shape (N,)
    """
    return np.argmax(onehot, axis=1)

def append_bias_term(data):
    """
    Append to the data a bias term equal to 1.

    Arguments:
        data (array): of shape (N,D)
    Returns:
        (array): shape (N,D+1)
    """
    N = data.shape[0]
    data = np.concatenate([np.ones([N, 1]),data], axis=1)
    return data

def normalize_fn(data, means, stds):
    """
    Return the normalized data, based on precomputed means and stds.
    
    Arguments:
        data (array): of shape (N,D)
        means (array): of shape (1,D)
        stds (array): of shape (1,D)
    Returns:
        (array): shape (N,D)
    """
    # return the normalized features
    return (data - means) / stds

def get_n_classes(labels):
    """
    Return the number of classes present in the data labels.
    
    This is approximated by taking the maximum label + 1 (as we count from 0).
    """
    return int(np.max(labels) + 1)

def shuffle_indices(N, K):
    """
    Generates shuffled indices for the dataset and determines the size of each fold for K-Fold cross-validation.

    Parameters:
    - N (int): Total number of samples in the dataset.
    - K (int): Number of folds in the cross-validation.

    Returns:
    - tuple:
        - indices (np.ndarray): Shuffled indices of the dataset, ensuring each fold is a random sample.
        - fold_size (int): Number of samples in each fold, computed as N // K.
    """
    fold_size = N // K
    indices = np.arange(N)
    np.random.shuffle(indices)
    return indices, fold_size

def find_all(X, Y, indices, fold, fold_size, model):
    """
    Performs training and validation for one fold in the K-Fold cross-validation using the specified model.

    Parameters:
    - X (np.ndarray): Feature matrix of the dataset.
    - Y (np.ndarray): Target variable or labels of the dataset.
    - indices (np.ndarray): Array of shuffled indices, provided by shuffle_indices function.
    - fold (int): The current fold index being processed.
    - fold_size (int): Number of samples in each fold.
    - model (object): An instance of a machine learning model that has a fit and predict method.

    Returns:
    - tuple:
        - Y_pred (np.ndarray): Predictions made.
        - Y_val (np.ndarray): Actual labels.
    """
    val_ind = indices[fold * fold_size: (fold + 1) * fold_size] 
    train_ind = np.setdiff1d(indices, val_ind)

    model.fit(X[train_ind], Y[train_ind])
    return model.predict(X[val_ind]), Y[val_ind]

def plot_performance_curve(x_vals, y_vals, xlabel, ylabel, title, log_scale=False):
    """
    Plot a performance curve using the provided x and y values.
    
    Parameters:
    - x_vals: The values for the x-axis (typically list of hyperparameters like lambda, learning rate, etc.)
    - y_vals: The corresponding performance metrics for the y-axis (like MSE, accuracy, etc.)
    - xlabel: Label for the x-axis.
    - ylabel: Label for the y-axis.
    - title: Title of the plot.
    - log_scale: Boolean to determine if the x-axis should be on a logarithmic scale.
    """
    plt.figure(figsize=(10, 5))
    plt.plot(x_vals, y_vals, linestyle='-', color='royalblue', linewidth=4)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True)
    if log_scale:
        plt.xscale('log')
    plt.show()

# Metrics
#########

def accuracy_fn(pred_labels, gt_labels):
    """
    Return the accuracy of the predicted labels.
    """
    return np.mean(pred_labels == gt_labels) * 100.

def macrof1_fn(pred_labels, gt_labels):
    """Return the macro F1-score."""
    class_ids = np.unique(gt_labels)
    macrof1 = 0
    for val in class_ids:
        predpos = (pred_labels == val)
        gtpos = (gt_labels==val)
        
        tp = sum(predpos*gtpos)
        fp = sum(predpos*~gtpos)
        fn = sum(~predpos*gtpos)
        if tp == 0:
            continue
        else:
            precision = tp/(tp+fp)
            recall = tp/(tp+fn)

        macrof1 += 2*(precision*recall)/(precision+recall)

    return macrof1/len(class_ids)

def mse_fn(pred,gt):
    '''
        Mean Squared Error
        Arguments:
            pred: NxD prediction matrix
            gt: NxD groundtruth values for each predictions
        Returns:
            returns the computed loss
            
    '''
    loss = (pred-gt)**2
    loss = np.mean(loss)
    return loss
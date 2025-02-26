import numpy as np
from src.methods.linear_regression import LinearRegression 
from src.methods.logistic_regression import LogisticRegression 
from src.methods.knn import KNN 
from src.utils import mse_fn, accuracy_fn, shuffle_indices, find_all, plot_performance_curve
import matplotlib.pyplot as plt

def cross_validation(X, Y, K, args_list, model_class, score_fct, param_type, fixed_params = 0):
    """
    Performs K-Fold cross-validation for a given model class across different hyperparameters or arguments.

    Parameters:
    - X (np.ndarray): Input features.
    - Y (np.ndarray): Target variable.
    - K (int): Number of folds for cross-validation.
    - args_list (list): List of arguments or hyperparameters to test.
    - model_class (class): The model class to be instantiated and evaluated.
    - score_fct (function): The scoring function to evaluate model performance (e.g., mse_fn or accuracy_fn).
    - param_type (str): Specifies how the fixed_params should be applied ("lr" for learning rate, "max_iters" for maximum iterations).
    - fixed_params (int or float): A fixed parameter value that remains constant while other parameters vary (default is 0).

    Returns:
    - performance (list): List of mean scores computed for each set of arguments across all folds.
    """
    indices, fold_size = shuffle_indices(X.shape[0], K)
    performance = []
    for arg in args_list:
        scores = []
        for fold in range(K):
            if len(param_type) != 0:
                model = model_class(fixed_params, arg) if param_type == "lr" else model_class(arg, fixed_params)
            else :
                model = model_class(arg)
            Y_pred, Y_val = find_all(X, Y, indices, fold, fold_size, model)
            scores.append(score_fct(Y_val, Y_pred))

        performance.append(np.mean(scores))
    return performance

def KFold_cross_validation_ridge(X, Y, K, args_list):
    """
    Evaluates Ridge Regression model using K-Fold cross-validation on a range of learning rates.

    Parameters:
    - X, Y, K, args_list: See cross_validation for details.

    Outputs:
    - Plots the learning rate versus average MSE.
    """
    performance = cross_validation(X, Y, K, args_list, LinearRegression, mse_fn, "")
    plot_performance_curve(args_list, performance, 'Learning Rate', 
                           'Average mse', 'Learning Rate vs Average mse', log_scale=False)
    
    

def find_best_lr(X, Y, K, args_list, fixed_max_iters):
    """
    Finds the best learning rate for Logistic Regression with a fixed number of iterations, evaluated via accuracy.

    Parameters:
    - X, Y, K, args_list: See cross_validation for details.
    - fixed_max_iters (int): The fixed number of maximum iterations for the Logistic Regression model.

    Outputs:
    - Plots learning rate versus average accuracy on a logarithmic scale.
    """
    performance = cross_validation(X, Y, K, args_list, LogisticRegression, accuracy_fn, "max_iters", fixed_max_iters)
    plot_performance_curve(args_list, performance, 'Learning Rate', 'Average accuracy', 
                           'Learning Rate vs Average Accuracy', log_scale=True)

def find_best_max_iters(X, Y, K, args_list, fixed_lr):
    """
    Identifies the best maximum iterations setting for Logistic Regression with a fixed learning rate, based on accuracy.

    Parameters:
    - X, Y, K, args_list: See cross_validation for details.
    - fixed_lr (float): The fixed learning rate for the Logistic Regression model.

    Outputs:
    - Plots max iterations versus average accuracy.
    """
    performance = cross_validation(X, Y, K, args_list, LogisticRegression, accuracy_fn, "lr", fixed_lr)
    plot_performance_curve(args_list, performance, 'Max Iterations', 
                           'Average accuracy', 'Max Iterations vs Average Accuracy', log_scale=False)

def KFold_cross_validation_KNN(X, Y, K, args_list, task_kind):
    """
    Conducts K-Fold cross-validation for K-Nearest Neighbors (KNN) models, evaluated either for classification or regression.

    Parameters:
    - X, Y, K, args_list: See cross_validation for details.
    - task_kind (str): Determines if the task is "classification" or "regression".

    Outputs:
    - Plots number of neighbors (k) versus accuracy or MSE, depending on the task.
    """
    score_fct = accuracy_fn if task_kind == "classification" else mse_fn
    performance = cross_validation(X, Y, K, args_list, KNN, score_fct, "task_kind", task_kind)
    plot_performance_curve(args_list, performance, 'Number of Neighbors (k)', 
                           'Accuracy' if task_kind == "classification" else 'MSE', 
                           f'K-NN Cross-Validation Results ({task_kind.capitalize()})', log_scale=False)

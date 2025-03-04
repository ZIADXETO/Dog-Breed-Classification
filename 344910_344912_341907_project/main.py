import argparse

import numpy as np
import matplotlib.pyplot as plt

from src.data import load_data
from src.methods.dummy_methods import DummyClassifier
from src.methods.logistic_regression import LogisticRegression
from src.methods.linear_regression import LinearRegression 
from src.methods.knn import KNN
from src.utils import normalize_fn, append_bias_term, accuracy_fn, macrof1_fn, mse_fn
from src.cross_validation import KFold_cross_validation_ridge, find_best_lr, find_best_max_iters, KFold_cross_validation_KNN

import os
np.random.seed(100)

def main(args):
    """
    The main function of the script. Do not hesitate to play with it
    and add your own code, visualization, prints, etc!

    Arguments:
        args (Namespace): arguments that were parsed from the command line (see at the end 
                          of this file). Their value can be accessed as "args.argument".
    """
    ## 1. First, we load our data and flatten the images into vectors

    ##EXTRACTED FEATURES DATASET
    if args.data_type == "features":
        feature_data = np.load('features.npz',allow_pickle=True)
        xtrain, xtest, ytrain, ytest, ctrain, ctest =feature_data['xtrain'],feature_data['xtest'],\
        feature_data['ytrain'],feature_data['ytest'],feature_data['ctrain'],feature_data['ctest']

    ##ORIGINAL IMAGE DATASET (MS2)
    elif args.data_type == "original":
        data_dir = os.path.join(args.data_path,'dog-small-64')
        xtrain, xtest, ytrain, ytest, ctrain, ctest = load_data(data_dir)

    ##TODO: xtrain and xtest are for both.
    ##TODO: ctrain and ctest are for regression task. (To be used for Linear Regression and KNN)
    ##TODO: ytrain, ytest are for classification task. (To be used for Logistic Regression and KNN)

    

    ## 2. Then we must prepare it. This is were you can create a validation set,
    #  normalize, add bias, etc.

    mean = np.mean(xtrain, axis = 0, keepdims = True)
    stds = np.std(xtrain, axis = 0, keepdims = True)

    xtrain_normalized = normalize_fn(xtrain, mean, stds)
    xtest_normalized = normalize_fn(xtest, mean, stds)

    xtrain = append_bias_term(xtrain_normalized) 
    xtest = append_bias_term(xtest_normalized)


    # add --test for no validation set 
    if not args.test:
        validation_split = 0.2  # Using 20% of the training data for validation
        num_train = int((1 - validation_split) * xtrain.shape[0])
        indices = np.random.permutation(xtrain.shape[0])
        train_indices = indices[:num_train]
        val_indices = indices[num_train:]

        xtest, ytest, ctest = xtrain[val_indices], ytrain[val_indices], ctrain[val_indices]
        xtrain, ytrain, ctrain = xtrain[train_indices], ytrain[train_indices], ctrain[train_indices]
        pass
    ### WRITE YOUR CODE HERE to do any other data processing

    
    ## 3. Initialize the method you want to use.

    # Use NN (FOR MS2!)
    if args.method == "nn":
        raise NotImplementedError("This will be useful for MS2.")

    # Follow the "DummyClassifier" example for your methods
    if args.method == "dummy_classifier":
        method_obj = DummyClassifier(arg1=1, arg2=2)
    elif args.method == "linear_regression":
        method_obj = LinearRegression(lmda = args.lmda)
    elif args.method == "logistic_regression":
        method_obj = LogisticRegression(lr = args.lr, max_iters = args.max_iters)    
    elif args.method == "knn":
            if args.task == "center_locating":
                method_obj = KNN(k = args.K, task_kind = "regression")    
            else :
                method_obj = KNN(k = args.K, task_kind = "classification")    



    ## 4. Train and evaluate the method

    if args.task == "center_locating":
        # Fit parameters on training data
        preds_train = method_obj.fit(xtrain, ctrain)

        # Perform inference for training and test data
        train_pred = method_obj.predict(xtrain)
        preds = method_obj.predict(xtest)

        ## Report results: performance on train and valid/test sets
        train_loss = mse_fn(train_pred, ctrain)
        loss = mse_fn(preds, ctest)

        print(f"\nTrain loss = {train_loss:.10f} - Test loss = {loss:.10f}")

    elif args.task == "breed_identifying":

        # Fit (:=train) the method on the training data for classification task
        preds_train = method_obj.fit(xtrain, ytrain)

        # Predict on unseen data
        preds = method_obj.predict(xtest)

        ## Report results: performance on train and valid/test sets
        acc = accuracy_fn(preds_train, ytrain)
        macrof1 = macrof1_fn(preds_train, ytrain)
        print(f"\nTrain set: accuracy = {acc:.3f}% - F1-score = {macrof1:.6f}")

        acc = accuracy_fn(preds, ytest)
        macrof1 = macrof1_fn(preds, ytest)
        print(f"Test set:  accuracy = {acc:.3f}% - F1-score = {macrof1:.6f}")
    else:
        raise Exception("Invalid choice of task! Only support center_locating and breed_identifying!")

    ### WRITE YOUR CODE HERE if you want to add other outputs, visualization, etc.
    vals = np.linspace(0, 1, 75)
    lr_list = np.logspace(-3, -1, 15)
    max_list = range(100, 1000, 400)
    k_list = range(1, 8)
    k = 5

    if args.method == "linear_regression":
        KFold_cross_validation_ridge(xtrain, ctrain, k, vals)
    elif args.method == "logistic_regression":
        find_best_lr(xtrain, ytrain, k, lr_list, args.max_iters) 
        find_best_max_iters(xtrain, ytrain, k, max_list, args.lr)
    elif args.method == "knn":
        if args.task == "center_locating":
            KFold_cross_validation_KNN(xtrain, ctrain, k, k_list, "regression")
        else :
            KFold_cross_validation_KNN(xtrain, ytrain, k, k_list, "classification")
    
if __name__ == '__main__':
    # Definition of the arguments that can be given through the command line (terminal).
    # If an argument is not given, it will take its default value as defined below.
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', default="center_locating", type=str, help="center_locating / breed_identifying")
    parser.add_argument('--method', default="dummy_classifier", type=str, help="dummy_classifier / knn / linear_regression/ logistic_regression / nn (MS2)")
    parser.add_argument('--data_path', default="data", type=str, help="path to your dataset")
    parser.add_argument('--data_type', default="features", type=str, help="features/original(MS2)")
    parser.add_argument('--lmda', type=float, default=10, help="lambda of linear/ridge regression")
    parser.add_argument('--K', type=int, default=1, help="number of neighboring datapoints used for knn")
    parser.add_argument('--lr', type=float, default=1e-5, help="learning rate for methods with learning rate")
    parser.add_argument('--max_iters', type=int, default=100, help="max iters for methods which are iterative")
    parser.add_argument('--test', action="store_true", help="train on whole training data and evaluate on the test data, otherwise use a validation set")


    # Feel free to add more arguments here if you need!

    # MS2 arguments
    parser.add_argument('--nn_type', default="cnn", help="which network to use, can be 'Transformer' or 'cnn'")
    parser.add_argument('--nn_batch_size', type=int, default=64, help="batch size for NN training")

    # "args" will keep in memory the arguments and their values,
    # which can be accessed as "args.data", for example.
    args = parser.parse_args()
    main(args)

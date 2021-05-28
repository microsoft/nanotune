from typing import Any, Dict

import sklearn as sk
from sklearn import svm
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier

param_dict_type = Dict[sk.base.ClassifierMixin, Dict[str, Dict[str, Any]]]

CLASSIFIER_PARAMS: param_dict_type = {
    RandomForestClassifier: {  # 'pinchoff': {'criterion': 'gini', 'max_features': 'auto', 'max_samples_leafs': 1,
        #                                 'min_samples_split': 6, 'n_estimators': 10, 'n_jobs': -1},
        "singledot": {
            "criterion": "entropy",
            "max_features": "sqrt",  #'max_samples_leafs': 2,
            "min_samples_split": 4,
            "n_estimators": 500,
            "n_jobs": -1,
        },
        "doubledot": {
            "criterion": "entropy",
            "max_features": "auto",  #'max_samples_leafs': 1,
            "min_samples_split": 2,
            "n_estimators": 100,
            "n_jobs": -1,
        },
        "dotregime": {
            "criterion": "gini",
            "max_features": "log2",  #'max_samples_leafs': 2,
            "min_samples_split": 2,
            "n_estimators": 100,
            "n_jobs": -1,
        },
    },
    MLPClassifier: {  #'pinchoff': {'activation': 'relu', 'alpha': 0.0001, 'batch_size': 100, 'hidden_layer_sizes': [100],
        #                         'learning_rate': 'constant', 'max_iter': 3000, 'power_t': 0.1, 'solver': 'lbfgs'},
        "singledot": {
            "activation": "relu",
            "alpha": 0.1,
            "batch_size": 300,
            "hidden_layer_sizes": [100],
            "learning_rate": "adaptive",
            "max_iter": 3000,
            "power_t": 0.6,
            "solver": "sgd",
        },
        "doubledot": {
            "activation": "logistic",
            "alpha": 0.001,
            "batch_size": 200,
            "hidden_layer_sizes": [200],
            "learning_rate": "invscaling",
            "max_iter": 3000,
            "power_t": 0.6,
            "solver": "lbfgs",
        },
        "dotregime": {
            "activation": "relu",
            "alpha": 0.001,
            "batch_size": 200,
            "hidden_layer_sizes": [300],
            "learning_rate": "constant",
            "max_iter": 3000,
            "power_t": 0.4,
            "solver": "sgd",
        },
    },
    svm.SVC: {  #'pinchoff': {'C': 1000, 'gamma': 1, 'kernel': 'rbf'},
        "singledot": {"C": 0.1, "gamma": 0.1, "kernel": "poly"},
        "doubledot": {"C": 0.1, "gamma": 0.1, "kernel": "poly"},
        "dotregime": {"C": 10, "gamma": 0.1, "kernel": "linear"},
    },
    LogisticRegression: {  #'pinchoff': {'C': 100, 'class_weight': 'balanced', 'fit_intercept': True, 'max_iter': 1000,
        #         'n_jobs': -1, 'penalty': 'l1', 'solver': 'liblinear'},
        "singledot": {
            "C": 0.1,
            "class_weight": None,
            "fit_intercept": True,
            "max_iter": 1000,
            "n_jobs": -1,
            "penalty": "l1",
            "solver": "liblinear",
        },
        "doubledot": {
            "C": 1000,
            "class_weight": "balanced",
            "fit_intercept": True,
            "max_iter": 1000,
            "n_jobs": -1,
            "penalty": "l2",
            "solver": "newton-cg",
        },
        "dotregime": {
            "C": 0.1,
            "class_weight": "balanced",
            "fit_intercept": True,
            "max_iter": 1000,
            "n_jobs": -1,
            "penalty": "l2",
            "solver": "sag",
        },
    },
    KNeighborsClassifier: {  #'pinchoff': {'algorithm': 'auto', 'leaf_size': 10, 'n_jobs': -1, 'n_neighbors': 2, 'p': 3,
        #         'weights': 'distance'},
        "singledot": {
            "algorithm": "auto",
            "leaf_size": 10,
            "n_jobs": -1,
            "n_neighbors": 2,
            "p": 2,
            "weights": "distance",
        },
        "doubledot": {
            "algorithm": "auto",
            "leaf_size": 10,
            "n_jobs": -1,
            "n_neighbors": 2,
            "p": 1,
            "weights": "distance",
        },
        "dotregime": {
            "algorithm": "auto",
            "leaf_size": 10,
            "n_jobs": -1,
            "n_neighbors": 2,
            "p": 2,
            "weights": "uniform",
        },
    },
    DecisionTreeClassifier: {  #'pinchoff': {'criterion': 'gini', 'max_features': 'auto', 'max_samples_leafs': 2, 'min_samples_split': 6, 'splitter': 'random'},
        "singledot": {
            "criterion": "entropy",
            "max_features": None,  #'max_samples_leafs': 3,
            "min_samples_split": 2,
            "splitter": "best",
        },
        "doubledot": {
            "criterion": "entropy",
            "max_features": "sqrt",  #'max_samples_leafs': 3,
            "min_samples_split": 6,
            "splitter": "best",
        },
        "dotregime": {
            "criterion": "entropy",
            "max_features": "auto",  #'max_samples_leafs': 1,
            "min_samples_split": 2,
            "splitter": "random",
        },
    },
}

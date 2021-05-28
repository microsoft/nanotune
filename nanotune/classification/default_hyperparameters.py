from typing import Any, Dict

CLF_SETTINGS: Dict[str, Dict[str, Any]] = {
    "pinchoff": {
        "classifier": "LogisticRegression",
        "data_types": ["signal"],
        "hyper_parameters": {
            "C": 0.1,
            "class_weight": "balanced",
            "fit_intercept": True,
            "max_iter": 1000,
            "penalty": "l1",
            "solver": "liblinear",
        },
    },
    "singledot": {
        "classifier": "MLPClassifier",
        "data_types": ["signal", "frequencies"],
        "hyper_parameters": {
            "activation": "relu",
            "alpha": 0.1,
            "batch_size": 300,
            "hidden_layer_sizes": [100],
            "learning_rate": "adaptive",
            "max_iter": 3000,
            "power_t": 0.6,
            "solver": "sgd",
        },
    },
    "doubledot": {
        "classifier": "MLPClassifier",
        "data_types": ["signal", "frequencies"],
        "hyper_parameters": {
            "activation": "logistic",
            "alpha": 0.001,
            "batch_size": 200,
            "hidden_layer_sizes": [200],
            "learning_rate": "invscaling",
            "max_iter": 3000,
            "power_t": 0.6,
            "shuffle": True,
            "solver": "lbfgs",
        },
    },
    "dotregime": {
        "classifier": "MLPClassifier",
        "data_types": ["signal", "frequencies"],
        "hyper_parameters": {
            "activation": "relu",
            "alpha": 0.001,
            "batch_size": 200,
            "hidden_layer_sizes": [300],
            "learning_rate": "constant",
            "max_iter": 3000,
            "power_t": 0.4,
            "shuffle": True,
            "solver": "sgd",
        },
    },
}

import json
import logging
import os
import time
from typing import Any, Dict, List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
from prettytable import PrettyTable
from scipy import interp
from sklearn import svm
from sklearn.calibration import calibration_curve
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, auc, average_precision_score,
                             brier_score_loss, confusion_matrix, roc_curve)
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

import nanotune as nt
from nanotune.data.dataset import Dataset
from nanotune.data.export_data import prep_data

logger = logging.getLogger(__name__)
ALLOWED_CATEGORIES = list(dict(nt.config["core"]["features"]).keys())
DOT_LABEL_MAPPING = dict(nt.config["core"]["dot_mapping"])

RELEVANT_FEATURE_INDEXES: Dict[str, List[int]] = {
    "pinchoff": [1, 2, 3, 4],
    "outerbarriers": [1],
    "singledot": [1],
    "doubledot": [1],
    "dotregime": [1],
}

DEFAULT_CLF_PARAMETERS: Dict[str, Any] = {
    "SVC": {"kernel": "linear", "probability": True, "gamma": "auto"},
    # 'LinearSVC': {'class_weight': 'balanced', 'max_iter': 5000},
    "LogisticRegression": {"solver": "newton-cg", "max_iter": 3000},
    "MLPClassifier": {"max_iter": 3000},
    "GaussianProcessClassifier": {},
    "DecisionTreeClassifier": {"max_depth": 5},
    "RandomForestClassifier": {"max_depth": 5, "n_estimators": 10, "max_features": 1},
    "AdaBoostClassifier": {},
    "GaussianNB": {},
    "QuadraticDiscriminantAnalysis": {},
    "KNeighborsClassifier": {"n_neighbors": 2},
}

METRIC_NAMES = ["accuracy_score", "brier_score_loss", "auc", "average_precision_recall"]

DEFAULT_DATA_FILES = {
    "pinchoff": "pinchoff.npy",
    "singledot": "dots.npy",
    "doubledot": "dots.npy",
    "dotregime": "dots.npy",
}

DEFAULT_N_ITER = {
    "pinchoff": 100,
    "singledot": 20,
    "doubledot": 25,
    "dotregime": 20,
}


class Classifier:
    """
    We assume that if no relevant_labels are supplied, single and double dot
    data have been extracted into the same file and labelled with labels
    specified in DOT_LABEL_MAPPING.
    """

    def __init__(
        self,
        data_filenames: List[str],
        category: str,
        data_types: Optional[List[str]] = None,
        folder_path: Optional[str] = None,
        test_size: float = 0.2,
        classifier: Optional[str] = None,
        hyper_parameters: Dict[str, Union[str, float, int]] = {},
        multi_class: bool = False,
        retained_variance: float = 0.99,
        name: Optional[str] = "",
        file_fractions: Optional[List[float]] = None,
        clf_params: Optional[Dict[str, Union[str, float, int]]] = None,
        feature_indexes: Optional[List[int]] = None,
        relevant_labels: Optional[List[int]] = None,
    ) -> None:
        """ """

        if folder_path is None:
            folder_path = nt.config["db_folder"]

        if category not in ALLOWED_CATEGORIES:
            logger.error(
                "Classifier category must be one " + "of {}".format(ALLOWED_CATEGORIES)
            )
            raise ValueError
        self.category = category

        if data_types is None:
            data_types = ["signal", "frequencies"]
        if "features" in data_types:
            if feature_indexes is None:
                self.feature_indexes = RELEVANT_FEATURE_INDEXES[category]
            else:
                self.feature_indexes = feature_indexes

        if relevant_labels is None:
            if category in DOT_LABEL_MAPPING.keys():
                # logger.warning('Classifier: Assuming both data of dot' +
                #                 ' regimes are saved in ' +
                #                 '{}'.format(data_filenames))
                relevant_labels = DOT_LABEL_MAPPING[category]
            else:
                relevant_labels = [0, 1]
        self.relevant_labels = sorted(relevant_labels)

        if file_fractions is None:
            file_fractions = [1.0] * len(data_filenames)
        self.file_fractions = file_fractions
        # Add default values to parameter dict in case some have not been
        # specified
        if classifier is None:
            default_params = {}
        else:
            default_params = DEFAULT_CLF_PARAMETERS[classifier]
        default_params.update(hyper_parameters)
        self.clf_params = default_params

        self.folder_path = folder_path
        self.file_paths, name_addon = self._list_paths(data_filenames)
        self.name = name_addon + category + "_"

        if classifier is not None:
            self.name += classifier
        self.clf_type = classifier

        self.retained_variance = retained_variance

        self.data_types = data_types
        self.test_size = test_size
        self.multi_class = multi_class

        if classifier == "SVC":
            self.clf = svm.SVC(**self.clf_params)
        elif classifier == "LogisticRegression":
            self.clf = LogisticRegression(**self.clf_params)
        elif classifier == "LinearSVC":
            self.clf = svm.LinearSVC(**self.clf_params)
        elif classifier == "MLPClassifier":
            self.clf = MLPClassifier(**self.clf_params)
        elif classifier == "GaussianProcessClassifier":
            self.clf = GaussianProcessClassifier(**self.clf_params)
        elif classifier == "DecisionTreeClassifier":
            self.clf = DecisionTreeClassifier(**self.clf_params)
        elif classifier == "RandomForestClassifier":
            self.clf = RandomForestClassifier(**self.clf_params)
        elif classifier == "AdaBoostClassifier":
            self.clf = AdaBoostClassifier(**self.clf_params)
        elif classifier == "GaussianNB":
            self.clf = GaussianNB(**self.clf_params)
        elif classifier == "QuadraticDiscriminantAnalysis":
            self.clf = QuadraticDiscriminantAnalysis(**self.clf_params)
        elif classifier == "KNeighborsClassifier":
            self.clf = KNeighborsClassifier(**self.clf_params)
        else:
            self.clf = None

        (self.original_data, self.labels) = self.load_data(
            self.file_paths, self.data_types, file_fractions=self.file_fractions
        )

    def load_data(
        self,
        file_paths: List[str],
        data_types: List[str],
        file_fractions: Optional[List[float]] = [1.0],
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Load data from file and separate data from labels
        """
        DATA_TYPE_MAPPING = dict(nt.config["core"]["data_types"])

        len_2d = np.prod(nt.config["core"]["standard_shapes"]["2"]) + 1
        all_the_stuff = np.empty([len(DATA_TYPE_MAPPING), 0, len_2d])
        try:
            for ip in range(len(file_paths)):
                print(file_paths[ip])
                sub_data = np.array(
                    np.load(file_paths[ip], allow_pickle=True), dtype=np.float64
                )
                frac = file_fractions[ip]  # type: ignore
                n_samples = int(round(sub_data.shape[1] * frac))
                print("n_samples: {}".format(n_samples))
                select = np.random.choice(sub_data.shape[1], n_samples, replace=False)
                sub_data = sub_data[:, select, :]

                all_the_stuff = np.concatenate([all_the_stuff, sub_data], axis=1)
                print("shape all_the_stuff: {}".format(all_the_stuff.shape))
        except ValueError:
            len_1d = np.prod(nt.config["core"]["standard_shapes"]["1"]) + 1
            all_the_stuff = np.empty([len(DATA_TYPE_MAPPING), 0, len_1d])
            for ip in range(len(file_paths)):
                sub_data = np.array(
                    np.load(file_paths[ip], allow_pickle=True), dtype=np.float64
                )
                frac = file_fractions[ip]  # type: ignore
                n_samples = int(round(sub_data.shape[1] * frac))
                select = np.random.choice(sub_data.shape[1], n_samples, replace=False)
                sub_data = sub_data[:, select, :]
                all_the_stuff = np.concatenate([all_the_stuff, sub_data], axis=1)

        labels = all_the_stuff[0, :, -1]
        data = all_the_stuff[:, :, :-1]

        relevant_data = np.empty((data.shape[1], 0))
        for data_type in data_types:
            to_append = data[DATA_TYPE_MAPPING[data_type]]

            if data_type == "features":
                to_append[to_append == nt.config["core"]["fill_value"]] = np.nan
                to_append = to_append[:, np.isfinite(to_append).any(axis=0)]

                try:
                    to_append = to_append[:, self.feature_indexes]
                except IndexError:
                    logger.warning(
                        "Some data in {} ".format(file_paths)
                        + "does not have the"
                        + "feature requested. Make sure all data "
                        + "has been fitted with appropriate"
                        + " fit classes."
                    )

            relevant_data = np.append(relevant_data, to_append, axis=1)

        # remove NaNs in labels
        mask = ~np.isnan(labels)
        relevant_data = relevant_data[mask, :]
        labels = labels[mask]
        # remove NaNs in data
        mask = np.isfinite(relevant_data).any(axis=-1)
        relevant_data = relevant_data[mask]
        labels = labels[mask].astype(int)

        relevant_data, labels = self.separate_data(relevant_data, labels)

        return relevant_data, labels

    def separate_data(
        self,
        data: np.ndarray,
        labels: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Extract sub data depending on which labels we would like to predict
        """
        shape = data.shape
        relevant_data = np.empty((0, shape[1]))
        relevant_labels = np.empty([0])

        for il, label in enumerate(self.relevant_labels):
            relevant_indx = np.where(labels == label)[0]
            relevant_data = np.concatenate(
                [relevant_data, data[relevant_indx, :]], axis=0
            )
            relevant_labels = np.concatenate(
                [relevant_labels, np.ones(len(relevant_indx)) * il], axis=0
            )

        return relevant_data, relevant_labels

    def train(
        self,
        data: Optional[np.ndarray] = None,
        labels: Optional[np.ndarray] = None,
    ) -> None:
        """"""
        if data is None:
            data = self.original_data
            labels = self.labels
        (data_to_use, labels_to_use) = self.select_equal_populations(data, labels)

        X_train, _ = self.prep_data(train_data=data_to_use)
        self.clf.fit(X_train, labels_to_use)

    def prep_data(
        self,
        train_data: Optional[np.ndarray] = None,
        test_data: Optional[np.ndarray] = None,
        perform_pca: Optional[bool] = False,
        scale_pc: Optional[bool] = False,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        scale and extract principle components
        """
        (train_data, test_data) = self.scale_raw_data(
            train_data=train_data, test_data=test_data
        )
        if perform_pca:
            (train_data, test_data) = self.get_principle_components(
                train_data=train_data, test_data=test_data
            )
        if scale_pc:
            (train_data, test_data) = self.scale_compressed_data(
                train_data=train_data, test_data=test_data
            )

        return train_data, test_data

    def select_equal_populations(
        self,
        data: np.ndarray,
        labels: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Make sure we have 50% of one and 50% of other population
        """
        # self.data_to_use = copy.deepcopy(self.original_data)
        populations_labels, population_counts = np.unique(labels, return_counts=True)

        n_each = int(np.min(population_counts))
        new_data = np.empty([n_each * len(populations_labels), data.shape[-1]])
        new_labels = np.empty(n_each * len(populations_labels), int)

        for ii, label in enumerate(populations_labels):
            idx = np.where(labels == int(label))
            idx = np.random.choice(idx[0], n_each, replace=False)
            idx = idx.astype(int)
            dat = data[idx]
            new_data[ii * n_each : (ii + 1) * n_each] = dat
            label_array = np.ones(n_each, dtype=int) * int(label)
            new_labels[ii * n_each : (ii + 1) * n_each] = label_array

        p = np.random.permutation(len(new_labels))

        return new_data[p], new_labels[p]

    def split_data(
        self,
        data: np.ndarray,
        labels: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:

        (train_data, test_data, train_labels, test_labels) = train_test_split(
            data, labels, test_size=self.test_size, random_state=0
        )

        return train_data, test_data, train_labels, test_labels

    def scale_raw_data(
        self,
        train_data: Optional[np.ndarray] = None,
        test_data: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """"""
        if train_data is not None:
            self.raw_scaler = StandardScaler()
            self.raw_scaler.fit(train_data)
            train_data = self.raw_scaler.transform(train_data)

        if test_data is not None:
            if hasattr(self, "raw_scaler"):
                test_data = self.raw_scaler.transform(test_data)
            else:
                logger.error("Scale train data before scaling test data.")
                raise AttributeError

        return train_data, test_data

    def scale_compressed_data(
        self,
        train_data: Optional[np.ndarray] = None,
        test_data: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """"""
        if train_data is not None:
            self.compressed_scaler = StandardScaler()
            self.compressed_scaler.fit(train_data)
            train_data = self.compressed_scaler.transform(train_data)

        if test_data is not None:
            if hasattr(self, "compressed_scaler"):
                test_data = self.compressed_scaler.transform(test_data)
            else:
                logger.error(
                    "Train data principle components has not been"
                    + " have to be scaled before scaling test PC."
                )
                raise AttributeError

        return train_data, test_data

    def get_principle_components(
        self,
        train_data: Optional[np.ndarray] = None,
        test_data: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """"""
        if train_data is not None:
            self.pca = PCA(self.retained_variance)
            self.pca.fit(train_data)
            train_data = self.pca.transform(train_data)

        if test_data is not None:
            if hasattr(self, "pca"):
                test_data = self.pca.transform(test_data)
            else:
                logger.error("Compress train data before compressing test" + " data.")
                raise AttributeError

        return train_data, test_data

    def score(self, test_data: np.ndarray, test_labels: np.ndarray) -> float:
        """"""
        self.clf_score = self.clf.score(test_data, test_labels)
        return self.clf_score

    def predict(
        self,
        dataid: int,
        db_name: str,
        db_folder: Optional[str] = None,
    ) -> np.ndarray:
        """"""
        if db_folder is None:
            db_folder = nt.config["db_folder"]

        DATA_TYPE_MAPPING = dict(nt.config["core"]["data_types"])

        df = Dataset(dataid, db_name)
        condensed_data_all = prep_data(df, self.category)

        predictions = []

        for condensed_data in condensed_data_all:

            relevant_data = np.empty((1, 0))
            for data_type in self.data_types:
                to_append = condensed_data[DATA_TYPE_MAPPING[data_type]]

                if data_type == "features":
                    to_append[to_append == nt.config["core"]["fill_value"]] = np.nan
                    to_append = to_append[:, np.isfinite(to_append).any(axis=0)]

                    try:
                        to_append = to_append[:, self.feature_indexes]
                    except IndexError:
                        logger.warning(
                            "Some data in {} ".format(dataid)
                            + "does not have the "
                            + "feature requested. Make sure all data "
                            + "has been "
                            + "fitted with appropriate fit "
                            + "classes."
                        )

                relevant_data = np.append(relevant_data, to_append, axis=1)

            _, relevant_data = self.prep_data(test_data=relevant_data)

            predictions.append(self.clf.predict(relevant_data))

        return predictions

    def compute_ROC(
        self,
        data_types: Optional[List[str]] = None,
        n_splits: int = 10,
        n_population_subselect: int = 10,
        save_to_file: bool = False,
        path: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Compute ROC for multiple train and test datasets
        """
        if path is None:
            path = os.path.join(nt.config["db_folder"], "ROC")
        if data_types is None:
            data_types = self.data_types

        cv = StratifiedKFold(n_splits=n_splits)

        result: Dict[str, Any] = {}

        # save additional info
        result["metadata"] = {}
        result["metadata"]["n_splits"] = n_splits
        result["metadata"]["n_population_subselect"] = n_population_subselect
        result["metadata"]["file_paths"] = self.file_paths
        result["metadata"]["clf_params"] = self.clf.get_params()

        for data_type in data_types:
            result[data_type] = {}
            # this line below could be made more efficient and not load
            # from file every time
            (self.original_data, self.labels) = self.load_data(
                self.file_paths, [data_type], file_fractions=self.file_fractions
            )
            tprs: List[List[float]] = []
            aucs: List[float] = []

            for redraw_iter in range(n_population_subselect):
                # tprs = []
                # aucs = []
                mean_fpr = np.linspace(0, 1, 100)

                X, y = self.select_equal_populations(self.original_data, self.labels)

                for train, test in cv.split(X, y):
                    # scale data as we would in real life
                    X_train = X[train]
                    y_train = y[train]

                    X_test = X[test]
                    y_test = y[test]

                    X_train, X_test = self.prep_data(
                        train_data=X_train, test_data=X_test
                    )

                    # probas = self.clf.fit(X[train], y[train]).predict_proba(X[test])
                    probas = self.clf.fit(X_train, y_train).predict_proba(X_test)

                    # Compute ROC curve and area the curve
                    fpr, tpr, thresholds = roc_curve(y_test, probas[:, 1])

                    if np.all(np.diff(fpr) >= 0):
                        # interpolate curve to np.linspace(0, 1, 100)
                        tprs.append(interp(mean_fpr, fpr, tpr))
                        tprs[-1][0] = 0.0
                        # compute area under the roc curve:
                        roc_auc = auc(fpr, tpr)
                        aucs.append(roc_auc)
                    else:
                        logger.warning(
                            "Averaging over less than the "
                            + " desired number of train-test splits."
                        )
            mean_tpr = np.mean(tprs, axis=0)
            mean_tpr[-1] = 1.0

            mean_auc = auc(mean_fpr, mean_tpr)
            std_auc = np.std(aucs)
            std_tpr = np.std(tprs, axis=0)

            result[data_type]["mean_fpr"] = mean_fpr
            result[data_type]["mean_tpr"] = mean_tpr
            result[data_type]["std_tpr"] = std_tpr
            result[data_type]["mean_auc"] = mean_auc
            result[data_type]["std_auc"] = std_auc

        if save_to_file:
            name = "ROC_" + self.name

            if not os.path.exists(path):
                os.makedirs(path)
            path = os.path.join(path, name)
            np.save(path, result)

        return result

    def compute_ROC_different_source(
        self,
        train_files: List[str],
        test_files: List[str],
        data_types: Optional[List[str]] = None,
        save_to_file: Optional[bool] = False,
        path: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Load different data for training
        """
        if path is None:
            path = os.path.join(nt.config["db_folder"], "ROC")

        if data_types is None:
            logger.warning("No data types specified. No ROC will be computed.")

        train_files_full, _ = self._list_paths(train_files)
        test_files_full, _ = self._list_paths(test_files)

        result: Dict[str, Any] = {}
        for data_type in data_types:  # type: ignore
            result[data_type] = {}

            ff = self.file_fractions
            train_data, train_labels = self.load_data(
                train_files_full, [data_type], file_fractions=ff
            )
            self.train(train_data, train_labels)

            test_data, test_labels = self.load_data(
                test_files_full, [data_type], file_fractions=ff
            )

            (test_data, test_labels) = self.select_equal_populations(
                test_data, test_labels
            )

            _, test_data = self.prep_data(test_data=test_data)
            probas = self.clf.predict_proba(test_data)

            # Compute ROC curve and area the curve
            fpr, tpr, thresholds = roc_curve(test_labels, probas[:, 1])
            mean_fpr = np.linspace(0, 1, 100)

            if np.all(np.diff(fpr) >= 0):
                roc_auc = auc(fpr, tpr)
                tpr = interp(mean_fpr, fpr, tpr)
                tpr[0] = 0.0
            else:
                logger.error(
                    "Unable to compute ROC with different test " + "and train files."
                )

            result[data_type]["mean_fpr"] = mean_fpr
            result[data_type]["mean_tpr"] = tpr
            result[data_type]["std_tpr"] = 0
            result[data_type]["mean_auc"] = roc_auc
            result[data_type]["std_auc"] = 0

        result["metadata"] = {}
        result["metadata"]["n_splits"] = 1
        result["metadata"]["n_population_subselect"] = 1
        result["metadata"]["train_files"] = train_files_full
        result["metadata"]["test_files"] = test_files_full
        result["metadata"]["file_fractions"] = self.file_fractions
        result["metadata"]["clf_params"] = self.clf.get_params()

        if save_to_file:
            self.name = "ROC" + self.name + "_train"

            for train_file in train_files:
                self.name += os.path.splitext(train_file)[0]
            self.name += "_test"
            for test_file in test_files:
                self.name += os.path.splitext(test_file)[0]

            if not os.path.exists(path):
                os.makedirs(path)
            path = os.path.join(path, self.name)
            np.save(path, result)

        return result

    def plot_ROC(
        self,
        roc_result: Optional[Dict[str, Any]] = None,
        save_to_file: bool = False,
        path: Optional[str] = None,
    ) -> str:
        """
        Plot ROC and return path where the figure was saved
        """
        if path is None:
            path = os.path.join(nt.config["db_folder"], "ROC")
        if roc_result is None:
            logger.warning(
                "No ROC result dict given. Computing one with " + "default values."
            )
            roc_result = self.compute_ROC(save_to_file=True)

        _ = plt.figure()
        for d_type, res in roc_result.items():
            if d_type != "metadata":
                mean_fpr = res["mean_fpr"]
                mean_tpr = res["mean_tpr"]
                std_tpr = res["std_tpr"]
                mean_auc = res["mean_auc"]
                std_auc = res["std_auc"]

                label = r"Mean ROC (AUC = %0.2f $\pm$ %0.2f) " % (mean_auc, std_auc)
                label += d_type
                plt.plot(mean_fpr, mean_tpr, label=label, lw=2, alpha=0.8)

                tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
                tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
                plt.fill_between(
                    mean_fpr,
                    tprs_lower,
                    tprs_upper,
                    color="grey",
                    alpha=0.2,
                )
        # plot the last std twice to have a legend entry (and only one)
        plt.fill_between(
            mean_fpr,
            tprs_lower,
            tprs_upper,
            color="grey",
            alpha=0.2,
            label=r"$\pm$ 1 std. dev.",
        )
        plt.plot(
            [0, 1], [0, 1], linestyle="--", lw=2, color="r", label="Chance", alpha=0.8
        )
        plt.xlim([-0.05, 1.05])
        plt.ylim([-0.05, 1.05])
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("Receiver Operating Characteristic " + self.name)
        plt.legend(loc="lower right", bbox_to_anchor=(1, 0))

        filename = "ROC_" + self.name
        path1 = os.path.join(path, filename + ".eps")
        plt.savefig(path1, format="eps", dpi=600)

        path2 = os.path.join(path, filename + ".png")
        plt.savefig(path2, format="png", dpi=600)

        return path2

    def compute_metrics(
        self,
        n_iter: Optional[int] = None,
        n_test: int = 100,
        save_to_file: bool = True,
        filename: str = "",
        supp_train_data: Optional[List[str]] = None,
        n_supp_train: Optional[int] = None,
        perform_pca: bool = False,
        scale_pc: bool = False,
    ) -> Tuple[Dict[str, Dict[str, Any]], np.ndarray]:
        """"""
        if n_iter is None:
            n_iter = DEFAULT_N_ITER[self.category]

        metrics = np.empty([len(METRIC_NAMES), n_iter])

        conf_matrix = []

        start_time = time.time()
        train_times = []
        test_times = []

        if supp_train_data is not None:
            supp_train_data, name_addon = self._list_paths(supp_train_data)
            f_frac = [1.0] * len(supp_train_data)
            (train_data_addon, train_labels_addon) = self.load_data(
                supp_train_data, self.data_types, file_fractions=f_frac
            )
            if n_supp_train is None:
                n_supp_train = train_data_addon.shape[0]

            mask = [1] * n_supp_train
            mask = mask + [0] * (train_data_addon.shape[0] - n_supp_train)
            mask = np.array(mask, dtype=bool)
            np.random.shuffle(mask)

            train_data_addon = train_data_addon[mask]
            train_labels_addon = train_labels_addon[mask]
        else:
            train_data_addon = None
            train_labels_addon = None

        for curr_iter in range(n_iter):
            start_time_inner = time.time()

            (data_to_use, labels_to_use) = self.select_equal_populations(
                self.original_data, self.labels
            )
            (train_data, test_data, train_labels, test_labels) = self.split_data(
                data_to_use, labels_to_use
            )
            if train_data_addon is not None:
                # print(train_data_addon.shape)
                # print(train_data.shape)
                # print(train_labels_addon.shape)
                # print(train_labels.shape)

                train_data = np.concatenate([train_data, train_data_addon], axis=0)
                train_labels = np.concatenate(
                    [train_labels, train_labels_addon], axis=0
                )

            X_train, X_test = self.prep_data(
                train_data=train_data,
                test_data=test_data,
                perform_pca=perform_pca,
                scale_pc=scale_pc,
            )

            probas = self.clf.fit(X_train, train_labels).predict_proba(X_test)

            train_times.append(time.time() - start_time_inner)

            fpr, tpr, thresholds = roc_curve(test_labels, probas[:, 1])

            start_time_inner = time.time()
            for itt in range(n_test):
                pred_labels = self.clf.predict(X_test)
            test_times.append((time.time() - start_time_inner) / n_test)

            m_in = METRIC_NAMES.index("accuracy_score")
            metrics[m_in, curr_iter] = accuracy_score(test_labels, pred_labels)

            m_in = METRIC_NAMES.index("brier_score_loss")
            metrics[m_in, curr_iter] = brier_score_loss(test_labels, pred_labels)

            m_in = METRIC_NAMES.index("auc")
            metrics[m_in, curr_iter] = auc(fpr, tpr)

            if hasattr(self.clf, "decision_function"):
                y_score = self.clf.decision_function(X_test)
            else:
                y_score = self.clf.predict_proba(X_test)[:, 1]

            m_in = METRIC_NAMES.index("average_precision_recall")
            metrics[m_in, curr_iter] = average_precision_score(test_labels, y_score)
            conf_matrix.append(confusion_matrix(test_labels, pred_labels))

        elapsed_time = (time.time() - start_time) / n_iter
        conf_matrix = np.array(conf_matrix)

        info_dict: Dict[str, Any] = {
            "n_iter": n_iter,
            "classifier": self.clf_type,
            "category": self.category,
            "data_files": self.file_paths,
            "data_types": self.data_types,
            "hyper_parameters": self.clf.get_params(),
            "metric_names": METRIC_NAMES,
            "elapsed_time [s/iter]": elapsed_time,
            "n_test": test_data.shape[0],
            "n_train": train_data.shape[0],
            "mean_train_time": np.mean(train_times),
            "std_train_time": np.std(train_times),
            "mean_test_time": np.mean(test_times),
            "std_test_time": np.std(test_times),
            "perform_pca": perform_pca,
            "scale_pc": scale_pc,
            "metadata": {},
            "supp_train_data": supp_train_data,
        }

        for im, metric_name in enumerate(METRIC_NAMES):
            info_dict[metric_name] = {
                "std": np.std(metrics[im]),
                "mean": np.mean(metrics[im]),
            }

        info_dict["confusion_matrix"] = {
            "std": np.std(conf_matrix, axis=0).tolist(),
            "mean": np.mean(conf_matrix, axis=0).tolist(),
        }

        if save_to_file:
            if not filename:
                filename = self.name + "_"
                if supp_train_data is not None:
                    filename = filename + name_addon + "_"
                filename += "_".join(self.data_types)
                if perform_pca:
                    filename += "_PCA"
                if scale_pc:
                    filename += "_scaled"
                filename += ".json"

            path = os.path.join(nt.config["db_folder"], "classifier_metrics")
            if not os.path.exists(path):
                os.makedirs(path)
            path = os.path.join(path, filename)
            with open(path, "w") as f:
                json.dump(info_dict, f)

        return info_dict, metrics

    def display_metrics(
        self,
        info_dict: Dict[str, Dict[str, float]],
        all_of_it: Optional[bool] = False,
    ) -> None:
        """"""
        inf_t = PrettyTable(["parameter", "value"])
        for key in info_dict.keys():
            if key not in METRIC_NAMES and key != "metric_names":
                inf_t.add_row([key, info_dict[key]])

        t = PrettyTable(["metric", "mean", "std"])
        for mn in METRIC_NAMES:
            t.add_row(
                [
                    mn,
                    "{0:.3f}".format(info_dict[mn]["mean"]),
                    "{0:.3f}".format(info_dict[mn]["std"]),
                ]
            )
        t.add_row(
            [
                mn,
                np.array(info_dict["confusion_matrix"]["mean"]),
                np.array(info_dict["confusion_matrix"]["std"]),
            ]
        )

        if all_of_it:
            print(inf_t)
        print(t)

    def determine_number_of_redraws(
        self,
        n_max_iter: int = 200,
        perform_pca: bool = False,
        scale_pc: bool = False,
        save_to_file: bool = True,
        data_folder: Optional[str] = None,
        figure_folder: Optional[str] = None,
        filename: Optional[str] = None,
    ) -> Dict[str, Dict[str, Any]]:
        """
        Skipping confusion matrix
        """
        # TODO: change method name to better reflect what it actually does.
        means = np.empty((len(METRIC_NAMES), n_max_iter))
        stds = np.empty((len(METRIC_NAMES), n_max_iter))

        info_dict, metrics = self.compute_metrics(
            n_iter=n_max_iter,
            n_test=1,
            save_to_file=False,
            perform_pca=perform_pca,
            scale_pc=scale_pc,
        )
        for n_eval in range(n_max_iter):
            for m_id, metric in enumerate(METRIC_NAMES):
                means[m_id, n_eval] = np.mean(metrics[m_id, 0 : n_eval + 1])
                stds[m_id, n_eval] = np.std(metrics[m_id, 0 : n_eval + 1])

        info_dict["mean_metric_variations"] = means.tolist()
        info_dict["std_metric_variations"] = stds.tolist()
        info_dict["metadata"]["metric_names"] = "Skip confusion matrix"

        if save_to_file:
            if filename is None:
                filename = "metric_fluctuations_" + self.name
                filename = filename + "_" + "_".join(self.data_types)
                if perform_pca:
                    filename += "_PCA"
                if scale_pc:
                    filename += "_scaled"

            if data_folder is None:
                data_folder = os.path.join(nt.config["db_folder"], "classifier_stats")
            if not os.path.exists(data_folder):
                os.makedirs(data_folder)
            data_file = os.path.join(data_folder, filename + ".json")
            with open(data_file, "w") as f:
                json.dump(info_dict, f)

        return info_dict

    # def tune_hyper_parameters(self,
    #                           ) -> Dict:
    #     """
    #     """
    #     return best_params

    def _list_paths(self, filenames: List[str]) -> Tuple[List[str], str]:
        """
        add path to file names
        """
        file_paths = []
        name = ""

        for filename in filenames:
            p = os.path.join(self.folder_path, filename)
            file_paths.append(p)
            name = name + os.path.splitext(filename)[0] + "_"
        return file_paths, name

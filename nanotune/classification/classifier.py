import json
import logging
import os
import time
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy.typing as npt
import numpy as np

from sklearn import svm
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, auc, average_precision_score,
                             brier_score_loss, confusion_matrix, roc_curve)
from sklearn.model_selection import train_test_split
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
    """Emulates binary classifiers implemented in scikit-learn.
    It includes methods loading labelled data saved to numpy files, splitting
    it into balanced sets, training and evaluating the classifier. The data to
    load is expected to be in the same format as outputted by
    `nanotune.data.export_data.export_data`.

    Attributes:
        hyper_parameters: hyperparameters/keyword arguments to pass to the
            binary classifier.
        category: which measurements wil be classified, e.g. 'pinchoff',
            'singledot', 'dotregime'. Supported categories correspond to
            keys of nt.config['core']['features'].
        file_fractions: fraction of all data loaded that should be used. For
            a value less than 1, data is chosen at random. It can be used to
            calculate accuracy variation when different random sub-sets of
            avaliable data are used for training data.
        folder_path: path to folder where numpy data files are located.s
        file_paths: list of paths of numpy data files.
        classifier_type: string indicating which binary classifier to use. E.g
            'SVG', 'MLPClassifier', .... For a full list see the keys of
            `DEFAULT_CLF_PARAMETERS`.
        retained_variance: variance to retain when calculating principal
            components.
        data_types: data types, one or more of: 'signal', 'gradient',
            'frequencies', 'features'.
        test_size: size of test set; it's the relative proportional of all data
            loaded.
        clf: instance of a scikit-learn binary classifier.
        original_data: all data loaded.
        labels: labels of original data.
    """
    def __init__(
        self,
        data_filenames: List[str],
        category: str,
        data_types: Optional[List[str]] = None,
        folder_path: Optional[str] = None,
        test_size: float = 0.2,
        classifier_type: Optional[str] = None,
        hyper_parameters: Optional[Dict[str, Union[str, float, int]]] = None,
        retained_variance: float = 0.99,
        file_fractions: Optional[List[float]] = None,
    ) -> None:

        if folder_path is None:
            folder_path = nt.config["db_folder"]

        if hyper_parameters is None:
            hyper_parameters = {}

        if category not in ALLOWED_CATEGORIES:
            logger.error(
                "Classifier category must be one of {}".format(ALLOWED_CATEGORIES)
            )
            raise ValueError
        self.category = category

        if data_types is None:
            data_types = ["signal", "frequencies"]
        if "features" in data_types:
            self._feature_indexes = RELEVANT_FEATURE_INDEXES[category]
        else:
            self._feature_indexes = []

        if category in DOT_LABEL_MAPPING.keys():
            relevant_labels = DOT_LABEL_MAPPING[category]
        else:
            relevant_labels = [0, 1]
        self._relevant_labels = sorted(relevant_labels)

        if file_fractions is None:
            file_fractions = [1.0] * len(data_filenames)
        self.file_fractions = file_fractions
        # Add default values to parameter dict in case some have not been
        # specified
        if classifier_type is None:
            default_params = {}
        else:
            default_params = DEFAULT_CLF_PARAMETERS[classifier_type]
        default_params.update(hyper_parameters)
        self.hyper_parameters = default_params

        self.folder_path = folder_path
        self.file_paths, name_addon = self._list_paths(data_filenames)
        self.name = name_addon + category + "_"

        if classifier_type is not None:
            self.name += classifier_type
        self.classifier_type = classifier_type

        self.retained_variance = retained_variance

        self.data_types = data_types
        self.test_size = test_size

        if classifier_type == "SVC":
            self.clf = svm.SVC(**self.hyper_parameters)
        elif classifier_type == "LogisticRegression":
            self.clf = LogisticRegression(**self.hyper_parameters)
        elif classifier_type == "LinearSVC":
            self.clf = svm.LinearSVC(**self.hyper_parameters)
        elif classifier_type == "MLPClassifier":
            self.clf = MLPClassifier(**self.hyper_parameters)
        elif classifier_type == "GaussianProcessClassifier":
            self.clf = GaussianProcessClassifier(**self.hyper_parameters)
        elif classifier_type == "DecisionTreeClassifier":
            self.clf = DecisionTreeClassifier(**self.hyper_parameters)
        elif classifier_type == "RandomForestClassifier":
            self.clf = RandomForestClassifier(**self.hyper_parameters)
        elif classifier_type == "AdaBoostClassifier":
            self.clf = AdaBoostClassifier(**self.hyper_parameters)
        elif classifier_type == "GaussianNB":
            self.clf = GaussianNB(**self.hyper_parameters)
        elif classifier_type == "QuadraticDiscriminantAnalysis":
            self.clf = QuadraticDiscriminantAnalysis(**self.hyper_parameters)
        elif classifier_type == "KNeighborsClassifier":
            self.clf = KNeighborsClassifier(**self.hyper_parameters)
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
    ) -> Tuple[npt.NDArray[np.float64], npt.NDArray[np.int64]]:
        """Loads data including labels from numpy files.

        Args:
            file_paths: list of paths of numpy data files.
            data_types: data types, one or more of: 'signal', 'gradient',
            'frequencies', 'features'.
            file_fractions: fraction of all data loaded that should be used. For
            a value less than 1, data is chosen at random. It can be used to
            calculate accuracy variation when different random sub-sets of
            avaliable data are used for training data.

        Returns:
            np.array: data
            np.array: labels
        """
        DATA_TYPE_MAPPING = dict(nt.config["core"]["data_types"])

        len_2d = np.prod(nt.config["core"]["standard_shapes"]["2"]) + 1
        all_the_stuff = np.empty([len(DATA_TYPE_MAPPING), 0, len_2d])
        try:
            for ip in range(len(file_paths)):
                sub_data = np.array(
                    np.load(file_paths[ip], allow_pickle=True),
                    dtype=np.float64
                )
                frac = file_fractions[ip]  # type: ignore
                n_samples = int(round(sub_data.shape[1] * frac))
                select = np.random.choice(sub_data.shape[1], n_samples, replace=False)
                sub_data = sub_data[:, select, :]

                all_the_stuff = np.concatenate([all_the_stuff, sub_data], axis=1)
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
                    to_append = to_append[:, self._feature_indexes]
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

        relevant_data, labels = self.select_relevant_data(relevant_data, labels)

        return relevant_data, labels

    def select_relevant_data(
        self,
        data: npt.NDArray[np.float64],
        labels: npt.NDArray[np.int64],
    ) -> Tuple[npt.NDArray[np.float64], npt.NDArray[np.int64]]:
        """Extract a subset data depending on which labels we would like to
        predict. Only relevant for dot data as the data file may contain
        four different labels.

        Args:
            data: original data
            label: original labels
        """
        shape = data.shape
        relevant_data = np.empty((0, shape[1]))
        relevant_labels = np.empty([0])

        for il, label in enumerate(self._relevant_labels):
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
        data: Optional[npt.NDArray[np.float64]] = None,
        labels: Optional[npt.NDArray[np.int64]] = None,
    ) -> None:
        """Trains binary classifier. Equal populations of data are selected
        before the sklearn classifier is fitted.

        Args:
            data: prepped data
            label: corresponding labels
        """
        if data is None:
            data = self.original_data
        if labels is None:
            labels = self.labels
        (data_to_use, labels_to_use) = self.select_equal_populations(
            data, labels)

        X_train, _ = self.prep_data(train_data=data_to_use)
        self.clf.fit(X_train, labels_to_use)

    def prep_data(
        self,
        train_data: Optional[npt.NDArray[np.float64]] = None,
        test_data: Optional[npt.NDArray[np.float64]] = None,
        perform_pca: Optional[bool] = False,
        scale_pc: Optional[bool] = False,
    ) -> Tuple[Optional[npt.NDArray[np.float64]],
            Optional[npt.NDArray[np.float64]]]:
        """Prepares data for training and testing.
        It scales the data and extracts principle components is
        desired. Transformations are applied to both train
        and test data, although transformations are first fitted on
        training data and then applied to test data.

        Args:
            train_data: dataset to be used for training
            test_data: dataset to be used for testing
            perform_pca: whether or not to perform a PCA and thus use principal
                components for training and testing.
            scale_pc: whether to scale principal components.

        Returns:
            np.array: prepped training data
            np.array: prepped testing data
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
        data: npt.NDArray[np.float64],
        labels: npt.NDArray[np.int64],
    ) -> Tuple[npt.NDArray[np.float64], npt.NDArray[np.int64]]:
        """Selects a subset of data at random so that each population/label
        appears equally often. Makes for a balanced dataset.

        Args:
            data: the data to subsample and balance
            labels: labels corresponding to `data`.
        """
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
        data: npt.NDArray[np.float64],
        labels: npt.NDArray[np.int64],
    ) -> Tuple[npt.NDArray[np.float64], npt.NDArray[np.float64],
            npt.NDArray[np.int64], npt.NDArray[np.int64]]:
        """Splits data into train and test set.
        Emulates sklearn's `train_test_split`.

        Args:
            data: the data to split
            labels: labels of `data`

        Returns:
            np.array: train data
            np.array: test data
            np.array: train labels
            np.array: test labels
        """
        (train_data, test_data, train_labels, test_labels) = train_test_split(
            data, labels, test_size=self.test_size, random_state=0
        )

        return train_data, test_data, train_labels, test_labels

    def scale_raw_data(
        self,
        train_data: Optional[npt.NDArray[np.float64]] = None,
        test_data: Optional[npt.NDArray[np.float64]] = None,
    ) -> Tuple[Optional[npt.NDArray[np.float64]],
            Optional[npt.NDArray[np.float64]]]:
        """Scales data loaded from numpy files by emulating sklearn's
        `StandardScaler`. The scaler is first fitted using the train set before
        also scaling the test set.

        Args:
            train_data: dataset to be used for training
            test_data: dataset to be used for testing

        Returns:
            np.array: scaled training data
            np.array: scaled test data
        """
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
        train_data: Optional[npt.NDArray[np.float64]] = None,
        test_data: Optional[npt.NDArray[np.float64]] = None,
    ) -> Tuple[Optional[npt.NDArray[np.float64]],
            Optional[npt.NDArray[np.float64]]]:
        """Scales previously computed principal components.

        Args:
            train_data: dataset to be used for training, containing principal
                components.
            test_data: dataset to be used for testing, containing principal
                components.

        Returns:
            np.array: scaled training data containing scaled principal
                components.
            np.array: scaled test data containing scaled principal
                components.
        """
        if train_data is not None:
            self.compressed_scaler = StandardScaler()
            self.compressed_scaler.fit(train_data)
            train_data = self.compressed_scaler.transform(train_data)

        if test_data is not None:
            if hasattr(self, "compressed_scaler"):
                test_data = self.compressed_scaler.transform(test_data)
            else:
                logger.error(
                    "Train data principle components has not been \
                     have to be scaled before scaling test PC."
                )
                raise AttributeError

        return train_data, test_data

    def get_principle_components(
        self,
        train_data: Optional[npt.NDArray[np.float64]] = None,
        test_data: Optional[npt.NDArray[np.float64]] = None,
    ) -> Tuple[Optional[npt.NDArray[np.float64]],
            Optional[npt.NDArray[np.float64]]]:
        """Computes principal components.

        Args:
            train_data: dataset to be used for training
            test_data: dataset to be used for testing

        Returns:
            np.array: scaled training data containing principal components.
            np.array: scaled test data containing principal components.
        """
        if train_data is not None:
            self.pca = PCA(self.retained_variance)
            self.pca.fit(train_data)
            train_data = self.pca.transform(train_data)

        if test_data is not None:
            if hasattr(self, "pca"):
                test_data = self.pca.transform(test_data)
            else:
                logger.error(
                    "Compress train data before compressing test data.")
                raise AttributeError

        return train_data, test_data

    def score(self,
        test_data: npt.NDArray[np.float64],
        test_labels: npt.NDArray[np.float64],
    ) -> float:
        """Emulates the binary classifiers `score` method and returns the mean
        accuracy for the given test set.

        Args:
            test_data: prepped test data
            test_labels: labels of `test_data`.

        Returns:
            float: mean accuracy.
        """
        self.clf_score = self.clf.score(test_data, test_labels)
        return self.clf_score

    def predict(
        self,
        dataid: int,
        db_name: str,
        db_folder: Optional[str] = None,
        readout_method_to_use: str = 'transport',
    ) -> List[Any]:
        """Classifies the trace of a QCoDeS dataset. Ideally, this data has
        been measured with nanotune and/or normalization constants saved
        to metadata under the `nt.meta_tag` key. Otherwise correct
        classification can not be guaranteed.

        Args:
            dataid: QCoDeS run ID.
            db_name: name of database
            db_folder: path to folder where database is located.

        Returns:
            array: containing the result as integers.
        """
        if db_folder is None:
            db_folder = nt.config["db_folder"]

        DATA_TYPE_MAPPING = dict(nt.config["core"]["data_types"])

        df = Dataset(dataid, db_name)

        condensed_data_all = prep_data(
            df, self.category, readout_method_to_use=readout_method_to_use)

        predictions = []

        for condensed_data in condensed_data_all:

            relevant_data = np.empty((1, 0))
            for data_type in self.data_types:
                to_append = condensed_data[DATA_TYPE_MAPPING[data_type]]

                if data_type == "features":
                    to_append[to_append == nt.config["core"]["fill_value"]] = np.nan
                    to_append = to_append[:, np.isfinite(to_append).any(axis=0)]  # type: ignore

                    try:
                        to_append = to_append[:, self.feature_indexes]  # type: ignore
                    except IndexError:
                        logger.warning(
                            f"Some data in {dataid} does not have the requested \
                            features. Make sure all data has been fitted with \
                            appropriate fit classes."
                        )

                relevant_data = np.append(relevant_data, to_append, axis=1)

            _, relevant_data = self.prep_data(test_data=relevant_data)  # type: ignore

            predictions.append(self.clf.predict(relevant_data))

        return predictions

    def compute_metrics(
        self,
        n_iter: Optional[int] = None,
        save_to_file: bool = True,
        filename: str = "",
        supp_train_data: Optional[List[str]] = None,
        n_supp_train: Optional[int] = None,
        perform_pca: bool = False,
        scale_pc: bool = False,
    ) -> Tuple[Dict[str, Dict[str, Any]], npt.NDArray[np.float64]]:
        """Computes different metrics of a classifier, averaging over
        a number of iterations.
        Beside metrics, the training time is tracked too. All information
        extracted is saved to a dict. Metrics computed are `accuracy_score`,
        `brier_score_loss`, `auc` (area under curve), `average_precision_recall`
        and the confusion matrix - all implemented in sklearn.metrics.

        Args:
            n_iter: number of train and test iterations over which the metrics
                statistic should be computed.
            save_to_file: whether to save metrics info to file.
            filename: name of json file if metrics are saved.
            supp_train_data: list of paths to files with additional training
                data.
            n_supp_train: number of datasets which should be added to
                the train set from additional data.
            perform_pca: whether the metrics should be computed using
                principal components for training and testing.
            scale_pc: whether principal components should be scaled.

        Returns:
            dict: summary od results, mapping the a string indicating the
                quantity onto the quantity itself. Containt mean and std
                each metric.
            np.array: metric results of all iterations. Shape is
                (len(METRIC_NAMES), n_iter),
                where the metrics appear in the order defined by
                METRIC_NAMES.
        """
        if n_iter is None:
            n_iter = DEFAULT_N_ITER[self.category]

        metrics = np.empty([len(METRIC_NAMES), n_iter])

        conf_matrix = []

        start_time = time.time()
        train_times = []

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
            mask_np = np.array(mask, dtype=bool)
            np.random.shuffle(mask_np)

            train_data_addon = train_data_addon[mask_np]
            train_labels_addon = train_labels_addon[mask_np]
        else:
            train_data_addon = None
            train_labels_addon = None

        for curr_iter in range(n_iter):
            start_time_train = time.time()

            (data_to_use, labels_to_use) = self.select_equal_populations(
                self.original_data, self.labels
            )
            (train_data,
             test_data,
             train_labels,
             test_labels) = self.split_data(data_to_use, labels_to_use)

            if train_data_addon is not None:
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

            train_times.append(time.time() - start_time_train)

            pred_labels = self.clf.predict(X_test)

            m_in = METRIC_NAMES.index("accuracy_score")
            metrics[m_in, curr_iter] = accuracy_score(test_labels, pred_labels)

            m_in = METRIC_NAMES.index("brier_score_loss")
            metrics[m_in, curr_iter] = brier_score_loss(
                test_labels, pred_labels)

            fpr, tpr, thresholds = roc_curve(test_labels, probas[:, 1])
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
        conf_matrix_np: npt.NDArray[np.float64] = np.array(conf_matrix)

        info_dict: Dict[str, Any] = {
            "n_iter": n_iter,
            "classifier": self.classifier_type,
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
            "std": np.std(conf_matrix_np, axis=0).tolist(),
            "mean": np.mean(conf_matrix_np, axis=0).tolist(),
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


    def _list_paths(self, filenames: List[str]) -> Tuple[List[str], str]:
        """Adds path to file names."""
        file_paths = []
        name = ""

        for filename in filenames:
            p = os.path.join(self.folder_path, filename)
            file_paths.append(p)
            name = name + os.path.splitext(filename)[0] + "_"
        return file_paths, name

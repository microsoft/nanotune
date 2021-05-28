import os
import glob
import logging
import json
import itertools
import numpy as np
import pandas as pd

import matplotlib
import matplotlib.pyplot as plt

from typing import Optional, List, Dict, Union, Tuple, Any

import nanotune as nt
from nanotune.classification.classifier import (
    DEFAULT_CLF_PARAMETERS,
    METRIC_NAMES,
    DEFAULT_DATA_FILES,
    Classifier,
)

from tensorflow import keras

logger = logging.getLogger(__name__)

metric_mapping = {
    "accuracy_score": "accuracy",
    "auc": "AUC",
    "average_precision_recall": "precision recall",
    "brier_score_loss": "Brier loss",
}


def qf_model(
    input_shape: Tuple[int, int, int, int],
    learning_rate: float = 0.001,
) -> keras.Sequential:

    model = keras.Sequential()
    model.add(
        keras.layers.Conv2D(
            32,
            kernel_size=(3, 3),
            activation="relu",
            input_shape=input_shape,
            data_format="channels_last",
            padding="same",
        )
    )

    model.add(keras.layers.MaxPooling2D(pool_size=(2, 2), strides=2))
    model.add(keras.layers.Flatten())

    model.add(keras.layers.Dense(1024, activation="relu"))
    model.add(keras.layers.Dropout(0.4))

    model.add(keras.layers.Dense(512, activation="relu"))
    model.add(keras.layers.Dropout(0.4))

    model.add(keras.layers.Dense(128, activation="relu"))
    model.add(keras.layers.Dropout(0.4))

    model.add(keras.layers.Dense(2, activation="softmax"))

    model.compile(
        loss=keras.losses.mean_squared_error,  # categorical_crossentropy,
        optimizer=keras.optimizers.Adam(lr=learning_rate),
        metrics=["accuracy"],
    )

    return model


def my_model(
    input_shape: Tuple[int, int, int, int],
    learning_rate: float = 0.001,
) -> keras.Sequential:
    """ """
    model = keras.Sequential()
    model.add(
        keras.layers.Conv2D(
            32,
            kernel_size=(3, 3),
            activation="relu",
            input_shape=input_shape,
            data_format="channels_last",
            padding="same",
        )
    )

    model.add(
        keras.layers.Conv2D(
            32,
            kernel_size=(3, 3),
            activation="relu",
            input_shape=input_shape,
            data_format="channels_last",
            padding="same",
        )
    )

    model.add(
        keras.layers.Conv2D(
            64,
            kernel_size=(3, 3),
            activation="relu",
            input_shape=input_shape,
            data_format="channels_last",
            padding="same",
        )
    )

    model.add(keras.layers.MaxPooling2D(pool_size=(2, 2), strides=2))
    model.add(keras.layers.Flatten())

    model.add(keras.layers.Dense(1024, activation="relu"))
    model.add(keras.layers.Dropout(0.5))

    model.add(keras.layers.Dense(512, activation="relu"))
    model.add(keras.layers.Dropout(0.5))

    model.add(keras.layers.Dense(128, activation="relu"))
    model.add(keras.layers.Dropout(0.5))

    model.add(keras.layers.Dense(2, activation="softmax"))

    model.compile(
        loss=keras.losses.mean_squared_error,  # categorical_crossentropy,
        optimizer=keras.optimizers.Adam(lr=learning_rate),
        metrics=["accuracy"],
    )

    return model


def load_syn_data(
    data_files: Optional[Dict[str, List[str]]] = None,
    data_types: Optional[List[str]] = None,
    for_CNN: bool = True,
) -> Tuple[np.ndarray, np.ndarray]:
    """"""
    if data_files is None:
        # data_files = {
        #     'qflow': ['qflow_data_large.npy'],
        #     'capa': ['noiseless_data.npy'],
        # }
        data_files = {
            "qflow": [
                "augmented_qf_data1.npy",
                "augmented_qf_data2.npy",
                "augmented_qf_data3.npy",
            ],
            "capa": [
                "augmented_cm_data1.npy",
                "augmented_cm_data2.npy",
                "augmented_cm_data3.npy",
            ],
        }
    else:
        if not all(elem in data_files.keys() for elem in ["qflow", "capa"]):
            print('data_files must contain following keys: "qflow", "capa".')
            raise ValueError
    if data_types is None:
        data_types = ["signal"]

    qf_data, qf_labels = _load_data(
        data_files["qflow"],
        data_types=data_types,
    )
    qf_data = qf_data * 2

    cm_data, cm_labels = _load_data(
        data_files["capa"],
        data_types=data_types,
    )
    cm_data = cm_data * 0.6

    syn_data = np.concatenate((qf_data, cm_data), axis=0)
    syn_labels = np.concatenate((qf_labels, cm_labels), axis=0)

    p = np.random.permutation(len(syn_labels))
    syn_data = syn_data[p]
    syn_labels = syn_labels[p]

    if not for_CNN and len(data_types) == 2:
        syn_labels = np.argmax(syn_labels, axis=1)

        m = syn_labels.shape[0]
        syn_curr = syn_data[:, :, :, 0].reshape(m, -1)
        syn_freq = syn_data[:, :, :, 1].reshape(m, -1)

        syn_data = np.concatenate((syn_curr, syn_freq), axis=1)
    else:
        logger.warning(
            "No data reshaping for parametric binary classifiers" + " was performed."
        )

    return syn_data, syn_labels


def load_exp_data(
    which: List[str],
    data_files: Optional[Dict[str, List[str]]] = None,
    data_types: Optional[List[str]] = None,
    for_CNN: bool = True,
) -> List[Tuple[np.ndarray, np.ndarray]]:
    """"""
    if data_files is None:
        # data_files = {
        #     'clean': ['clean_exp_dots.npy'],
        #     'good': ['exp_dots_corrected.npy'],
        #     'bad': ['exp_dots_corrected.npy'],
        #     'good_minus_clean': ['exp_dots_minus_clean.npy'],
        #     'good_and_bad': ['exp_dots_corrected.npy'],
        #     'good_and_bad_minus_clean': None,

        # }
        data_files = {
            "clean": [
                "augmented_clean_exp_dots1.npy",
                "augmented_clean_exp_dots2.npy",
            ],
            "good": [
                "augmented_exp_dots_corrected1.npy",
                "augmented_exp_dots_corrected2.npy",
                "augmented_exp_dots_corrected3.npy",
            ],
            "bad": [
                "augmented_exp_dots_corrected1.npy",
                "augmented_exp_dots_corrected2.npy",
                "augmented_exp_dots_corrected3.npy",
            ],
            "good_minus_clean": [
                "augmented_exp_dots_minus_clean1.npy",
                "augmented_exp_dots_minus_clean2.npy",
                "augmented_exp_dots_minus_clean3.npy",
            ],
            "good_and_bad": [
                "augmented_exp_dots_corrected1.npy",
                "augmented_exp_dots_corrected2.npy",
                "augmented_exp_dots_corrected3.npy",
            ],
            "good_and_bad_minus_clean": [],
        }
    if data_types is None:
        data_types = ["signal"]

    exp_data_all = []

    for dtype in which:

        if dtype == "good_and_bad":
            all_data, all_labels = _load_good_and_poor(
                data_files[dtype], data_types=data_types
            )

            exp_data_all.append((all_data, all_labels))

        elif dtype == "bad":
            all_data, all_labels = _load_data(
                data_files[dtype], data_types=data_types, relevant_labels=[0, 2]
            )

            exp_data_all.append((all_data, all_labels))

        elif dtype == "good_and_bad_minus_clean":
            if data_files["good_and_bad_minus_clean"] is None:
                f_name = data_files["good_minus_clean"]
            else:
                f_name = data_files["good_and_bad_minus_clean"]
            all_data, all_labels = _load_good_and_poor(f_name, data_types=data_types)

            exp_data_all.append((all_data, all_labels))

        elif dtype in ["clean", "good", "good_minus_clean"]:
            # not in ['good_and_bad', 'good_and_bad_minus_clean', 'bad']:
            data, labels = _load_data(
                data_files[dtype],
                data_types=data_types,
            )
            exp_data_all.append((data, labels))
        else:
            logger.error("Trying to load unknown data.")

    if not for_CNN and len(data_types) == 2:
        for idd, sub_data in enumerate(exp_data_all):

            data = sub_data[0]
            labels = sub_data[1]

            labels = np.argmax(labels, axis=1)

            m = labels.shape[0]
            curr = data[:, :, :, 0].reshape(m, -1)
            freq = data[:, :, :, 1].reshape(m, -1)

            data = np.concatenate((curr, freq), axis=1)

            exp_data_all[idd] = (data, labels)
    else:
        logger.warning(
            "No data reshaping for parametric binary classifiers" + " was performed."
        )

    return exp_data_all


def _load_good_and_poor(
    filenames: List[str],
    data_types: Optional[List[str]] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """"""
    if data_types is None:
        data_types = ["signal"]

    if isinstance(filenames, str):
        filenames = [filenames]
    singledots, single_labels = _load_data(
        filenames,
        data_types=data_types,
        regime="singledot",
    )

    doubledots, double_labels = _load_data(
        filenames,
        data_types=data_types,
        regime="doubledot",
    )

    single_labels = np.argmax(single_labels, axis=1)
    double_labels = np.argmax(double_labels, axis=1)

    n_each = int(np.min([len(single_labels), len(double_labels)]))

    sd_ids = np.random.choice(n_each, n_each, replace=False).astype(int)
    dd_ids = np.random.choice(n_each, n_each, replace=False).astype(int)

    singledot = singledots[sd_ids]
    sd_labels = np.zeros(n_each, dtype=int)

    doubledot = doubledots[dd_ids]
    dd_labels = np.ones(n_each, dtype=int)

    all_data = np.concatenate((singledot, doubledot), axis=0)
    all_labels = np.concatenate((sd_labels, dd_labels), axis=0)

    p = np.random.permutation(len(all_labels))
    all_data = all_data[p]
    all_labels = all_labels[p]

    all_labels = keras.utils.to_categorical(all_labels)

    return all_data, all_labels


def _load_data(
    files: List[str],
    regime: str = "dotregime",
    data_types: Optional[List[str]] = None,
    shuffle: bool = True,
    relevant_labels: Optional[List[int]] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load data from multiple data files but do it seperaterly to ensure
    'select_equal_populations' won't accidentially
    select data mainly from one file
    """
    if data_types is None:
        data_types = ["signal", "frequencies"]

    data = np.empty([0, 50, 50, len(data_types)])
    labels = np.empty([0])

    for dfile in files:

        data_loader = Classifier(
            [dfile],
            regime,
            data_types=data_types,
            relevant_labels=relevant_labels,
        )

        (sub_data, sub_labels) = data_loader.select_equal_populations(
            data_loader.original_data, data_loader.labels
        )

        m = sub_data.shape[0]

        if len(data_types) > 2:
            raise NotImplementedError

        if len(data_types) == 2:
            data_sig = sub_data[:, :2500].reshape(m, 50, 50, 1)
            data_frq = sub_data[:, 2500:].reshape(m, 50, 50, 1)

            sub_data = np.concatenate((data_sig, data_frq), axis=3)
        #         print(sub_data.shape)
        #         print(data.shape)
        if len(data_types) == 1:
            sub_data = sub_data.reshape(m, 50, 50, 1)

        data = np.concatenate((data, sub_data), axis=0)
        labels = np.concatenate((labels, sub_labels), axis=0)

    if shuffle:
        p = np.random.permutation(len(labels))
        data = data[p]
        labels = labels[p]

    labels = keras.utils.to_categorical(labels)

    return data, labels


def select_equal_populations(
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


def print_data_stats(data, labels) -> None:

    print("number of samples: {}".format(data.shape[0]))
    print(
        "populations (number and count): {}, {}".format(
            *np.unique(labels, return_counts=True)
        )
    )

    print("\n")

    print("max value: {}".format(np.max(data)))
    print("min value: {}".format(np.min(data)))
    print("std: {}".format(np.std(data)))
    print("median: {}".format(np.median(data)))

    a = np.hstack(data[:500].flatten())
    #     _ = plt.hist(a, bins=100,  range=[np.min(data), np.min(data) + 2*np.std(data)])  # arguments are passed to np.histogram
    _ = plt.hist(a, bins=100, range=[np.min(data), 1])
    plt.title("Histogram qf_data")
    plt.show()


def feature_combination_metrics(
    classifier: str,
    data_filenames: List[str],
    category: str,
    metric: str = "accuracy_score",
    filename: Optional[str] = None,
    classifier_parameters: Dict[str, Union[str, float, int]] = {},
    feature_indexes: List[int] = [0],
    n_iter: int = 75,
) -> Dict[str, Any]:
    """"""
    # TODO: Fix this method. It is broken. Has not been used for a while

    # n_feat = len(feature_indexes)
    # scores: List[str] = []

    # for k in range(1, n_feat+1):
    #     f_indx = itertools.combinations(range(1, n_feat+1), k)
    #     for f_combo in feature_indexes:
    #         qclf = Classifier(data_filenames,
    #                               category,
    #                               classifier=classifier,
    #                               hyper_parameters=classifier_parameters,
    #                               data_types=['features'],
    #                               feature_indexes=list(f_combo),
    #                               )
    #     infos = qclf.compute_metrics(n_iter=n_iter)
    #     features_str = ''
    #     sub_feat = [features[f] for f in f_combo]
    #     scores.append([', '.join(sub_feat), infos[metric]['mean'],
    #                    infos[metric]['std']])

    # info_dict = {
    #     'stage': category,
    #     'classifier': qclf.clf_type,
    #     'classifier_parameters': qclf.clf.get_params(),
    #     'n_iter': n_iter,
    #     'data_files': qclf.file_paths,
    #     'scores': scores,
    #     'metric': metric,
    # }

    # if filename is None:
    #     filename = qclf.clf_type + '_' + metric + '.json'

    # path = os.path.join(nt.config['db_folder'], category + '_features_metrics')
    # if not os.path.exists(path):
    #     os.makedirs(path)
    # path = os.path.join(path, filename)
    # with open(path, 'w') as f:
    #     json.dump(info_dict, f)

    # return info_dict
    logger.warning("feature_combination_metrics under construction")
    return {}


# def feature_metric_to_latex(directory: str,
#                             filenames: List[str],
#                             tables_folder: str) -> None:
#     """
#     """
#     metric = 'accuracy_score'
#     classifiers = ['SVC_rbf', 'SVC_linear', 'MLPClassifier']
#     stage = 'po'
#     directory = '/Users/jana/Documents/code/nanotune/measurements/databases/' + stage + '_features_metrics'

#     for classifier in classifiers:
#         filename = classifier + '_' + metric + '.json'

#         path = os.path.join(directory, filename)
#         with open(path) as f:
#             feat_data = json.load(f)

#         header =  ['features', 'mean ' + metric, 'std']
#         scores = sorted(feat_data['scores'], key=itemgetter(1), reverse=True)[0:40]

#         df = pd.DataFrame(scores)
#         filepath = os.path.join(tables_folder,  stage + '_' + classifier + '_' + metric +'.tex')
#         with open(filepath, 'w') as tf:
#             with pd.option_context("max_colwidth", 1000):
#                 tf.write(df.to_latex(index=False,
#                                      formatters=[dont_format, format_float,
#                                                  format_float],
#                                      header=header,
#                                      column_format='lcc').replace('\\toprule', '\\hline').replace('\\midrule', '\\hline').replace('\\bottomrule','\\hline'))


def performance_metrics_to_latex(
    tables_directory: str,
    metric: str = "accuracy_score",
    file_directory: Optional[str] = None,
) -> None:
    """"""
    categories: Dict[str, Tuple[str, List[List[str]]]] = {
        "pinchoff": (
            "pinchoff",
            [["signal"], ["frequencies"], ["frequencies", "signal"], ["features"]],
        ),
        "singledot": ("dots", [["signal"], ["frequencies"], ["signal", "frequencies"]]),
        "doubledot": ("dots", [["signal"], ["frequencies"], ["frequencies", "signal"]]),
        "dotregime": ("dots", [["signal"], ["frequencies"], ["frequencies", "signal"]]),
    }
    classifiers = [
        "DecisionTreeClassifier",
        "GaussianProcessClassifier",
        "KNeighborsClassifier",
        "LogisticRegression",
        "MLPClassifier",
        "QuadraticDiscriminantAnalysis",
        "RandomForestClassifier",
        "SVC",
    ]

    if file_directory is None:
        file_directory = os.path.join(nt.config["db_folder"], "classifier_metrics")

    header2 = [
        "classifier ",
        metric_mapping[metric],
        "evaluation time [s]",
        metric_mapping[metric],
        "evaluation time [s]",
    ]

    for category, settings in categories.items():
        data_file = settings[0]
        data_types = settings[1]

        for data_type in data_types:

            scores = []
            base_pattern = data_file + "_" + category + "*"
            all_files = glob.glob(os.path.join(file_directory, base_pattern))
            pattern = "_".join(data_type)
            rel_files = [f for f in all_files if pattern in f]
            for d_type in nt.config["core"]["data_types"]:
                if d_type not in data_type:
                    print(d_type)
                    rel_files = [f for f in rel_files if d_type not in f]

            for classifier in classifiers:
                clf_files = [f for f in rel_files if classifier in f]

                sub_score = [classifier]
                for pca_setting in ["no_PCA", "PCA."]:
                    if pca_setting == "PCA.":
                        files = [f for f in clf_files if pca_setting in f]
                    else:
                        files = [f for f in clf_files if "PCA" not in f]

                    if len(files) > 1:
                        print("error")
                        print(files)
                    with open(files[0]) as json_file:
                        data = json.load(json_file)

                    sub_score.extend(
                        [
                            "{0:.3f}".format(float(data[metric]["mean"]))
                            + " $\pm$ "
                            + "{0:.3f}".format(float(data[metric]["std"])),
                            format_time(float(data["mean_test_time"]))
                            + " $\pm$ "
                            + format_time(float(data["std_test_time"])),
                        ]
                    )
                scores.append(sub_score)

            df = pd.DataFrame(scores)
            filepath = tables_directory + category + "_" + pattern + ".tex"
            with open(filepath, "w") as tf:
                output = df.to_latex(
                    index=False,
                    formatters=[
                        dont_format,
                        dont_format,
                        dont_format,
                        dont_format,
                        dont_format,
                    ],
                    header=header2,
                    column_format="@{\extracolsep{6pt}}lcccc",
                    escape=False,
                )
                output = output.replace(
                    "\\toprule",
                    "\\hline \\hline & \multicolumn{2}{c}{PCA} & \multicolumn{2}{c}{no PCA} \\\\ \cline{2-3} \cline{4-5} ",
                )
                output = output.replace("\\midrule", "\\hline")
                output = output.replace("\\bottomrule", "\\hline \\hline")
                tf.write(output)


def performance_metrics_to_figure(
    data_file: str,
    category: str,
    data_types: List[str],
    metric: str,
    figure_directory: Optional[str] = None,
    file_directory: Optional[str] = None,
    rcparams: Optional[Dict[str, object]] = None,
) -> None:
    """"""
    if rcparams is not None:
        matplotlib.rcParams.update(rcparams)
    if file_directory is None:
        file_directory = os.path.join(nt.config["db_folder"], "classifier_metrics")

    base_pattern = data_file + "_" + category + "*"
    all_files = glob.glob(os.path.join(file_directory, base_pattern))

    for data_type in data_types:
        pattern = "_".join(data_type)
        rel_files = [f for f in all_files if pattern in f]
        for d_type in nt.config["core"]["data_types"]:
            if d_type not in data_type:
                rel_files = [f for f in rel_files if d_type not in f]

        file_dict = {
            "PCA": [f for f in rel_files if "PCA." in f],
            "PCA_scaled_PC": [f for f in rel_files if "PCA_scaled" in f],
            "no_PCA": [f for f in rel_files if "PCA" not in f],
        }

        for pca_setting, files in file_dict.items():
            names = []
            means = []
            stds = []

            for file in files:
                with open(file) as json_file:
                    data = json.load(json_file)
                    names.append(data["classifier"])
                    means.append(float(data[metric]["mean"]))
                    stds.append(float(data[metric]["std"]))

            means = [x for _, x in sorted(zip(names, means), key=lambda pair: pair[0])]
            stds = [x for _, x in sorted(zip(names, stds), key=lambda pair: pair[0])]
            names = sorted(names)

            fig = plt.figure()
            plt.plot(names, means, "o")
            plt.xticks(rotation=45, ha="right")
            plt.ylim([0, 1])

            fig.tight_layout()
            if figure_directory is None:
                figure_directory = file_directory

            filepath = figure_directory + category + "_" + pattern + "_"
            filepath = filepath + metric + "_" + pca_setting + "_all_clf.eps"

            plt.savefig(
                filepath,
                format="eps",
                bbox_inches="tight",
            )
            plt.show()


def plot_metric_fluctuations(
    category: str,
    data_types: List[List[str]],
    data_file: Optional[str] = None,
    metrics: Optional[List[str]] = None,
    classifier: Optional[str] = None,
    figure_directory: Optional[str] = None,
    file_directory: Optional[str] = None,
    rcparams: Optional[Dict[str, object]] = None,
) -> None:
    """"""
    if rcparams is not None:
        matplotlib.rcParams.update(rcparams)
    if data_file is None:
        data_file = DEFAULT_DATA_FILES[category]
    if metrics is None:
        metrics = METRIC_NAMES
    if file_directory is None:
        file_directory = os.path.join(nt.config["db_folder"], "classifier_stats")

    base_pattern = "metric_fluctuations_" + os.path.splitext(data_file)[0]
    base_pattern = base_pattern + "_" + category

    if classifier is None:
        base_pattern = base_pattern + "*"
    else:
        assert classifier in DEFAULT_CLF_PARAMETERS.keys()
        base_pattern = base_pattern + "_" + classifier + "*"

    all_files = glob.glob(os.path.join(file_directory, base_pattern))

    for data_type in data_types:
        pattern = "_".join(data_type)
        rel_files = [f for f in all_files if pattern in f]
        for d_type in nt.config["core"]["data_types"]:
            if d_type not in data_type:
                rel_files = [f for f in rel_files if d_type not in f]
                rel_files = [f for f in rel_files if ".json" in f]
        # print(rel_files)
        for file in rel_files:
            with open(file) as json_file:
                info_dict = json.load(json_file)

            means = info_dict["mean_metric_variations"]
            stds = info_dict["std_metric_variations"]

            colors = ["r", "g", "b", "y", "m"]

            fig, ax = plt.subplots(1, 1)
            for m_id, metric in enumerate(metrics):
                # if metric is not 'confusion_matrix':
                ax.plot(
                    means[m_id], c=colors[m_id], label="mean " + metric_mapping[metric]
                )
                ax.plot(
                    stds[m_id],
                    c=colors[m_id],
                    linestyle="dotted",
                    label="std " + metric_mapping[metric],
                )

            title = "Metric Fluctuations " + info_dict["category"]
            title = title + " (" + " ".join(data_type) + ")"
            ax.set_title(title)
            ax.set_xlabel("\# re-draws")
            ax.set_ylabel("score")
            ax.set_ylim((0, 1))
            ax.set_xticks(np.round(np.linspace(0, len(means[0]), 5), 2))

            ax.legend(loc="upper left", bbox_to_anchor=(0.8, 1))
            fig.tight_layout()
            print(figure_directory)
            if figure_directory is None:
                figure_directory = file_directory

            filename = os.path.splitext(os.path.basename(file))[0]
            print(filename)

            plt.savefig(
                os.path.join(figure_directory, filename + ".eps"),
                format="eps",
                bbox_inches="tight",
            )
            plt.show()


def summarize_hyper_parameter_optimization(
    directory: Optional[str] = None,
    filename: Optional[str] = None,
    to_latex: bool = True,
    table_folder: Optional[str] = None,
) -> Dict[str, Dict[str, Any]]:
    """"""
    if directory is None:
        directory = os.path.join(nt.config["db_folder"], "classifier_hyperparams")
    if filename is None:
        all_files = glob.glob(directory + "/*.json")
        filename = max(all_files, key=os.path.getctime)
    if table_folder is None:
        table_folder = directory

    f_path = os.path.join(directory, filename)
    with open(f_path, "r") as f:
        hparams = json.load(f)

    besties: Dict[str, Dict[str, Any]] = {}
    for clf in hparams.keys():
        for category in hparams[clf].keys():
            try:
                besties[category][clf] = {}
            except Exception:
                besties[category] = {}
                besties[category][clf] = {}
            best_score = 0.0
            best_option = {}
            best_dtype = None
            for dtype in hparams[clf][category].keys():
                for hparams_comb in hparams[clf][category][dtype]:
                    if float(hparams_comb[1]) > best_score:

                        best_option = hparams_comb[0]
                        best_score = hparams_comb[1]
                        best_dtype = dtype

                        #             besties[clf][category][best_dtype] = [best_option, best_score]
                        besties[category][clf] = [best_dtype, best_option, best_score]
    besties2 = {}
    table_besties: Dict[str, Any] = {}
    for cat in besties.keys():
        best_score = 0.0
        for clf in besties[cat].keys():
            if besties[cat][clf][2] > best_score:
                besties2[cat] = [clf] + besties[cat][clf]
                table_besties[cat] = []
                table_besties[cat].append(["category", cat])
                table_besties[cat].append(["classifier", clf])
                table_besties[cat].append(["data_type", besties[cat][clf][0]])
                table_besties[cat].append(["accuracy_score", besties[cat][clf][2]])
                for pname, param in besties[cat][clf][1].items():
                    table_besties[cat].append([pname, param])

                best_score = besties[cat][clf][2]
                # table_besties[cat] = [clf] + besties[cat][clf][0:1]
                # table_besties[cat] += [besties[cat][clf][-1]]

    if to_latex:
        header = ["parameter", "value"]
        for cat, param_table in table_besties.items():
            df = pd.DataFrame.from_dict(param_table)
            filename = "best_clf_params_" + cat + ".tex"
            filepath = os.path.join(table_folder, filename)
            with open(filepath, "w") as tf:
                output = df.to_latex(
                    index=False,
                    #  multirow=True,
                    #  col_space=10000,
                    formatters=[
                        dont_format,
                        dont_format,
                        #  format_dict,
                        #  format_float,
                    ],
                    header=header,
                    column_format="cc",
                )
                output = output.replace("\\toprule", "\\hline \\hline ")
                output = output.replace("\\midrule", "\\hline")
                output = output.replace("\\bottomrule", "\\hline \\hline")
                tf.write(output)
    return besties2


def merge_optimization_files(
    directory: Optional[str] = None,
) -> Tuple[Dict[str, Dict[str, Dict[str, Any]]], Dict[str, Dict[str, Dict[str, Any]]]]:
    """"""
    if directory is None:
        base, db = os.path.split(os.path.dirname(nt.config["db_folder"]))
        directory = os.path.join(base, "classifier_opimization", "results")

    all_files = glob.glob(directory + "/*.json")

    d_types = []
    categories = []
    clf_names = []

    best_params_all: Dict[str, Dict[str, Dict[str, Any]]] = {}

    for file in all_files:
        # print('\n')
        # print(file)
        with open(file, "r") as f:
            hparams = json.load(f)
        # print(hparams)
        for clf, sub_dict in hparams.items():
            # print('\n')
            # print(clf)
            clf_names.append(clf)
            if clf not in best_params_all.keys():
                best_params_all[clf] = {}

            for category, sub_sub_dict in sub_dict.items():
                if category not in best_params_all[clf].keys():
                    best_params_all[clf][category] = {}
                # print(category)
                categories.append(category)

                best_core = 0
                current_best_params = {}
                for data_type, params_list in sub_sub_dict.items():
                    if data_type not in best_params_all[clf][category].keys():
                        best_params_all[clf][category][data_type] = [{}, 0]

                    d_types.append(data_type)
                    # print(data_type)

                    for param in params_list:

                        if param[1] > best_core:
                            current_best_params = param.copy()

                    if (
                        current_best_params[1]
                        > best_params_all[clf][category][data_type][1]
                    ):
                        param = current_best_params
                        best_params_all[clf][category][data_type] = param.copy()

    categories_set = set(categories)
    d_types_set = set(d_types)
    clf_names_set = set(clf_names)
    clf_names_sorted = sorted(clf_names_set)

    params_by_category: Dict[str, Dict[str, Dict[str, Any]]] = {}
    for category in categories_set:
        if category not in params_by_category.keys():
            params_by_category[category] = {}
        for d_type in d_types_set:
            if d_type not in params_by_category[category].keys():
                params_by_category[category][d_type] = {}

            for clf in clf_names_sorted:

                if clf in best_params_all.keys():
                    if category in best_params_all[clf].keys():
                        if d_type in best_params_all[clf][category].keys():
                            param = best_params_all[clf][category][d_type].copy()
                            params_by_category[category][d_type][clf] = param

    return best_params_all, params_by_category


def dont_format(x):
    return x


def unpack(x):
    return np.array(x)


def format_float(x):
    return "{0:.3f}".format(x)


def format_time(x):
    return "{0:.4f}".format(x)

import os
import glob
import json
import copy

import pprint

import numpy as np
from typing import Optional, List, Dict, Union, Tuple, Any

pp = pprint.PrettyPrinter(indent=4)
train_score = "training"
empty = "- (-)"

# TODO: FIx PEP8

name_mapping: Dict[str, str] = {
    "training": "training data",
    "clean": "clean experimental",
    "good": "good experimental",
    "good_and_bad": "all experimental",
    "white": "white",
    "rnt": "random telegraph",
    "random_blobs": "random current modulation",
    "current_drop": "pinchoff modulation",
    "one_over_f": "1/f",
    "charge_jumps": "charge jumps",
    "training_score": train_score,
    "train_score": train_score,
    "synthetic_good": "synthetic and good experimental",
    "synthetic_good_and_bad": "synthetic and all experimental",
    "noisy_synthetic_good": "noisy synthetic and good experimental",
    "noisy_synthetic_good_and_bad": "noisy synthetic and all experimental",
    "pretrained_acc": "pre-train",
    "transfer_acc": "transfer",
    "diff_acc": "diff",
    "pretrained_conf_mat": "pre-trained",
    "transfer_conf_mat": "transfer",
    "noisy_synthetic": "synthetic with noise",
}


def merge_lists(old_list: np.ndarray, new_list: np.ndarray):
    """"""
    # TODO: rename to reflect what it actually does.
    if old_list.shape[0] == 0:
        old_list = new_list
    else:
        for conf_mat in new_list:
            #             print(conf_mat.shape)
            #             print(old_list.shape)
            #             print('\n')
            if (
                conf_mat[0, 0] not in old_list[:, 0, 0]
                and conf_mat[0, 1] not in old_list[:, 0, 1]
                and conf_mat[1, 0] not in old_list[:, 1, 0]
                and conf_mat[1, 1] not in old_list[:, 1, 1]
            ):
                old_list = np.append(old_list, conf_mat.reshape(1, 2, 2), axis=0)
    return old_list


def format_score(mean, std):
    return "{:.2f} ({:.2f})".format(mean, std)


def format_mat(mat1, mat2):

    #     mat_str =   '{:.2f} ({:.2f}) & {:.2f} ({:.2f}) \\\\ {:.2f} ({:.2f}) & {:.2f} ({:.2f})'
    #     mat_str =   '{}({})&{}({})\\\\{}({})&{}({})'
    #     format_num = []
    #     mat2 = mat2.flatten()
    # #     mat_str = ' '.join(['$ \\begin{bmatrix} ', str(round(mat1[0. 0])), str(round(mat2[0. 0])), '&', str(round(mat1[0. 1])), str(round(mat2[0. 1])), '\\\\',
    # #                        str(round(mat1[1. 0])), str(round(mat2[1. 0])), '&', str(round(mat1[1. 1])), str(round(mat2[1. 1])), ' \\end{bmatrix} $' ])
    #     for ie, elm in enumerate(mat1.flatten()):
    #         format_num.append(str(round(float(elm),2)))
    #         format_num.append(str(round(float(mat2[ie]),2)))
    #     print(mat_str)
    #     print(format_num)
    #     print(len(format_num))
    #     print(mat1)
    if len(mat1.flatten()) == 4:
        #     if len(format_num) == 8:
        #         mat_str = '$ \\begin{bmatrix} ' + str(round(float(mat1[0, 0]), 2)) + ' ' + str(round(float(mat2[0, 0]), 2))+ ' ' + '&'+ ' ' + str(round(float(mat1[0, 1]), 2))+ ' ' + str(round(float(mat2[0, 1]), 2))+ ' ' + '\\\\' + ' '
        #         mat_str = mat_str + str(round(float(mat1[1, 0]), 2))+ ' ' + str(round(float(mat2[1, 0]), 2))+ ' ' +'&'+ ' ' +str(round(float(mat1[1, 1]), 2))+ ' ' + str(round(float(mat2[1, 1]), 2))+ ' ' +' \\end{bmatrix} $'

        mat_str = (
            "$\\begin{bmatrix}"
            + str(int(mat1[0, 0]))
            + "&"
            + str(int(mat1[0, 1]))
            + "\\\\"
        )  # str(int(mat1[0, 1])) #+ '  ('  + str(int(mat1[0, 1])) + ') \\\\'
        #         mat_str = mat_str +  str(int(mat1[1, 0])) + '  (' + str(int(mat2[1, 0])) + ') & ' + str(int(mat1[1, 1])) + '  ('  + str(int(mat2[1, 1]))
        mat_str += (
            str(int(mat1[1, 0])) + "&" + str(int(mat1[1, 1])) + "\\end{bmatrix}$"
        )  #' '.join(['$ \\begin{bmatrix} ', '10 & 20 \\\\ 30 & 40  ', ' \\end{bmatrix} $'])

    #         mat_str = mat_str.format(*format_num)
    #         print(mat_str)
    #         mat_str = '$\\begin{bmatrix}'+ mat_str+'\\end{bmatrix}$'
    #         print(mat_str)
    else:
        mat_str = "$ \\begin{bmatrix}  & \\\\ & \\end{bmatrix} $"

    return mat_str


#     return '$ \\begin{bmatrix} ' + str(round(float(mat1[0, 0]), 2)) + ' & \\\\ & \\end{bmatrix} $' #' '.join(['$ \\begin{bmatrix} ', '10 & 20 \\\\ 30 & 40  ', ' \\end{bmatrix} $'])


metrics = {
    "test_score": format_score,
    "conf_mat": format_mat,
}


def summarize_results(
    model: str,
    metric: str,
    results_f_name: List[str],
    folders: Optional[List[str]] = None,
    add_missing_keys: Optional[bool] = True,
) -> Dict[str, Dict[str, Dict[str, Any]]]:
    """ """
    if folders is None:
        folders = ["performance_stats"]
    #     results_f_name = "performances_results"
    f_name = "_".join([*results_f_name, model])

    results: Dict[str, Dict[str, Dict[str, Any]]] = {}
    results[model] = {}

    for folder in folders:
        path = os.path.join(folder, f_name)
        list_of_files = glob.glob(path + "*")
        for path in list_of_files:
            with open(path, "r") as file:
                res_l = json.load(file)

            for model in res_l.keys():
                for training_data, res in res_l[model].items():

                    try:
                        old_vals = results[model][training_data]["train_score"]
                    except KeyError:
                        results[model][training_data] = {}
                        results[model][training_data]["train_score"] = []
                        old_vals = []

                    new_vals = np.unique(
                        old_vals + res_l[model][training_data]["train_score"]
                    )
                    results[model][training_data]["train_score"] = new_vals.tolist()

                    for test_data in res.keys():
                        if test_data != "train_score":
                            try:
                                old_vals = results[model][training_data][test_data][
                                    "test_score"
                                ]
                            except KeyError:
                                results[model][training_data][test_data] = {}
                                results[model][training_data][test_data][
                                    "test_score"
                                ] = np.empty((0))
                                results[model][training_data][test_data][
                                    "conf_mat"
                                ] = np.empty((0, 2, 2))

                            old_vals = results[model][training_data][test_data][
                                "test_score"
                            ]
                            new_vals = np.unique(
                                np.append(
                                    old_vals,
                                    res_l[model][training_data][test_data][
                                        "test_score"
                                    ],
                                )
                            )
                            results[model][training_data][test_data][
                                "test_score"
                            ] = new_vals

                            old_vals = results[model][training_data][test_data][
                                "conf_mat"
                            ]
                            new_vals = np.asarray(
                                res_l[model][training_data][test_data]["conf_mat"]
                            )
                            new_vals = merge_lists(old_vals, new_vals)
                            #                     new_vals[0][0] = np.unique(new_vals[0][0]).tolist()
                            #                     new_vals[0][0] = np.unique(new_vals[0][0]).tolist()
                            results[model][training_data][test_data][
                                "conf_mat"
                            ] = new_vals

    if add_missing_keys:
        possible_missing_keys = [
            "goodWbad_exp_predict_clean_exp",
            "goodWbad_exp_predict_good_exp",
            "goodExp_predict_goodWBad_exp",
            "syn_w_good_exp_pred_goodBad",
            "syn_w_good_exp_pred_clean",
            "syn_w_good_bad_exp_pred_good",
            "syn_w_good_bad_exp_pred_clean",
            "syn_w_good_bad_exp_pred_good_and_bad",
        ]
        for missing_key in possible_missing_keys:
            if missing_key not in list(results[model].keys()):
                results[model][missing_key] = {
                    "train_score": [],
                    "clean": {"test_score": [], "conf_mat": []},
                    "good": {"test_score": [], "conf_mat": []},
                    "good_and_bad": {"test_score": [], "conf_mat": []},
                }

    stats: Dict[str, Dict[str, Dict[str, Any]]] = {}
    for model in results.keys():
        stats[model] = {}
        for training_data in results[model].keys():
            training_scores = results[model][training_data]["train_score"]
            try:
                stats[model][training_data]["training_score"] = [
                    np.mean(training_scores) * 100,
                    np.std(training_scores) * 100,
                ]
            except KeyError:
                stats[model][training_data] = {}
                stats[model][training_data]["training_score"] = [
                    np.mean(training_scores) * 100,
                    np.std(training_scores) * 100,
                ]

            for test_data in results[model][training_data].keys():
                if test_data != "train_score":
                    if test_data not in stats[model][training_data].keys():
                        stats[model][training_data][test_data] = {}
                    test_score = results[model][training_data][test_data]["test_score"]
                    stats[model][training_data][test_data]["test_score"] = [
                        np.mean(test_score) * 100,
                        np.std(test_score) * 100,
                    ]

                    conf_mats = results[model][training_data][test_data]["conf_mat"]
                    stats[model][training_data][test_data]["conf_mat"] = [
                        np.mean(conf_mats, axis=0),
                        np.std(conf_mats, axis=0),
                    ]

    return stats


def reformat_results(
    stats: Dict[str, Dict[str, Dict[str, Any]]],
    model: str,
    metric: str,
) -> Dict[str, Dict[str, Dict[str, Any]]]:
    mdl = model
    format_fct = metrics[metric]

    sub_res: Dict[str, Dict[str, Dict[str, Any]]] = {}
    sub_res[mdl] = {}

    # --------------------------------- #
    sub_res[mdl]["synthetic"] = {}
    sub_res[mdl]["synthetic"][train_score] = format_score(
        stats[mdl]["syn_pred_syn"]["training_score"][0],
        stats[mdl]["syn_pred_syn"]["training_score"][1],
    )

    sub_res[mdl]["synthetic"]["synthetic"] = format_fct(
        stats[mdl]["syn_pred_syn"]["partial_synthetic"][metric][0],
        stats[mdl]["syn_pred_syn"]["partial_synthetic"][metric][1],
    )

    sub_res[mdl]["synthetic"][name_mapping["clean"]] = format_fct(
        stats[mdl]["syn_pred_exp"]["clean"][metric][0],
        stats[mdl]["syn_pred_exp"]["clean"][metric][1],
    )

    sub_res[mdl]["synthetic"][name_mapping["good"]] = format_fct(
        stats[mdl]["syn_pred_exp"]["good"][metric][0],
        stats[mdl]["syn_pred_exp"]["good"][metric][1],
    )

    sub_res[mdl]["synthetic"][name_mapping["good_and_bad"]] = format_fct(
        stats[mdl]["syn_pred_exp"]["good_and_bad"][metric][0],
        stats[mdl]["syn_pred_exp"]["good_and_bad"][metric][1],
    )

    # --------------------------------- #
    sub_res[mdl]["synthetic with noise"] = {}
    sub_res[mdl]["synthetic with noise"][train_score] = format_score(
        stats[mdl]["syn_wNoise_pred_exp"]["training_score"][0],
        stats[mdl]["syn_wNoise_pred_exp"]["training_score"][1],
    )

    sub_res[mdl]["synthetic with noise"][name_mapping["clean"]] = format_fct(
        stats[mdl]["syn_wNoise_pred_exp"]["clean"][metric][0],
        stats[mdl]["syn_wNoise_pred_exp"]["clean"][metric][1],
    )

    sub_res[mdl]["synthetic with noise"][name_mapping["good"]] = format_fct(
        stats[mdl]["syn_wNoise_pred_exp"]["good"][metric][0],
        stats[mdl]["syn_wNoise_pred_exp"]["good"][metric][1],
    )

    sub_res[mdl]["synthetic with noise"][name_mapping["good_and_bad"]] = format_fct(
        stats[mdl]["syn_wNoise_pred_exp"]["good_and_bad"][metric][0],
        stats[mdl]["syn_wNoise_pred_exp"]["good_and_bad"][metric][1],
    )

    sub_res[mdl]["synthetic with noise"]["synthetic"] = empty

    # --------------------------------- #
    sub_res[mdl]["good experimental"] = {}
    sub_res[mdl]["good experimental"][train_score] = format_score(
        stats[mdl]["good_exp_pred_good_exp"]["training_score"][0],
        stats[mdl]["good_exp_pred_good_exp"]["training_score"][1],
    )

    sub_res[mdl]["good experimental"][name_mapping["clean"]] = format_fct(
        stats[mdl]["expGood_pred_expClean"]["clean"][metric][0],
        stats[mdl]["expGood_pred_expClean"]["clean"][metric][1],
    )

    sub_res[mdl]["good experimental"][name_mapping["good"]] = format_fct(
        stats[mdl]["good_exp_pred_good_exp"]["partial_good_exp"][metric][0],
        stats[mdl]["good_exp_pred_good_exp"]["partial_good_exp"][metric][1],
    )

    sub_res[mdl]["good experimental"][name_mapping["good_and_bad"]] = format_fct(
        stats[mdl]["goodExp_predict_goodWBad_exp"]["good_and_bad"][metric][0],
        stats[mdl]["goodExp_predict_goodWBad_exp"]["good_and_bad"][metric][1],
    )

    sub_res[mdl]["good experimental"]["synthetic"] = empty

    # --------------------------------- #
    sub_res[mdl]["all experimental"] = {}
    sub_res[mdl]["all experimental"][train_score] = format_score(
        stats[mdl]["goodWbad_exp_pred_goodWbad_exp"]["training_score"][0],
        stats[mdl]["goodWbad_exp_pred_goodWbad_exp"]["training_score"][1],
    )

    sub_res[mdl]["all experimental"][name_mapping["good_and_bad"]] = format_fct(
        stats[mdl]["goodWbad_exp_pred_goodWbad_exp"]["partial_good_and_bad"][metric][0],
        stats[mdl]["goodWbad_exp_pred_goodWbad_exp"]["partial_good_and_bad"][metric][1],
    )

    sub_res[mdl]["all experimental"][name_mapping["clean"]] = format_fct(
        stats[mdl]["goodWbad_exp_predict_clean_exp"]["clean"][metric][0],
        stats[mdl]["goodWbad_exp_predict_clean_exp"]["clean"][metric][1],
    )

    sub_res[mdl]["all experimental"][name_mapping["good"]] = format_fct(
        stats[mdl]["goodWbad_exp_predict_good_exp"]["good"][metric][0],
        stats[mdl]["goodWbad_exp_predict_good_exp"]["good"][metric][1],
    )

    sub_res[mdl]["all experimental"]["synthetic"] = empty

    # --------------------------------- #
    sub_res[mdl]["synthetic and good experimental"] = {}
    sub_res[mdl]["synthetic and good experimental"][train_score] = format_score(
        stats[mdl]["syn_w_good_exp_pred_goodBad"]["training_score"][0],
        stats[mdl]["syn_w_good_exp_pred_goodBad"]["training_score"][1],
    )

    sub_res[mdl]["synthetic and good experimental"][
        name_mapping["good_and_bad"]
    ] = format_fct(
        stats[mdl]["syn_w_good_exp_pred_goodBad"]["good_and_bad"][metric][0],
        stats[mdl]["syn_w_good_exp_pred_goodBad"]["good_and_bad"][metric][1],
    )

    sub_res[mdl]["synthetic and good experimental"][name_mapping["clean"]] = format_fct(
        stats[mdl]["syn_w_good_exp_pred_clean"]["clean"][metric][0],
        stats[mdl]["syn_w_good_exp_pred_clean"]["clean"][metric][1],
    )

    sub_res[mdl]["synthetic and good experimental"][name_mapping["good"]] = format_fct(
        stats[mdl]["syn_w_good_exp_pred_goodBad"]["good"][metric][0],
        stats[mdl]["syn_w_good_exp_pred_goodBad"]["good"][metric][1],
    )

    sub_res[mdl]["synthetic and good experimental"]["synthetic"] = empty

    # --------------------------------- #
    sub_res[mdl]["synthetic and all experimental"] = {}
    sub_res[mdl]["synthetic and all experimental"][train_score] = format_score(
        stats[mdl]["syn_w_good_bad_exp_pred_good_and_bad"]["training_score"][0],
        stats[mdl]["syn_w_good_bad_exp_pred_good_and_bad"]["training_score"][1],
    )

    sub_res[mdl]["synthetic and all experimental"][
        name_mapping["good_and_bad"]
    ] = format_fct(
        stats[mdl]["syn_w_good_bad_exp_pred_good_and_bad"]["good_and_bad"][metric][0],
        stats[mdl]["syn_w_good_bad_exp_pred_good_and_bad"]["good_and_bad"][metric][1],
    )

    sub_res[mdl]["synthetic and all experimental"][name_mapping["clean"]] = format_fct(
        stats[mdl]["syn_w_good_bad_exp_pred_clean"]["clean"][metric][0],
        stats[mdl]["syn_w_good_bad_exp_pred_clean"]["clean"][metric][1],
    )

    sub_res[mdl]["synthetic and all experimental"][name_mapping["good"]] = format_fct(
        stats[mdl]["syn_w_good_bad_exp_pred_good"]["good"][metric][0],
        stats[mdl]["syn_w_good_bad_exp_pred_good"]["good"][metric][1],
    )

    sub_res[mdl]["synthetic and all experimental"]["synthetic"] = empty

    return sub_res


def summarize_noise_results(
    model: str,
    metric: str,
    results_f_name: Optional[List[str]] = None,
    #    folder: Optional[str] = None,
    folders: Optional[List[str]] = None,
) -> Dict[str, Dict[str, Dict[str, Dict[str, Dict[str, Any]]]]]:
    if results_f_name is None:
        results_f_name = ["noise", "strengths", "results"]
    if folders is None:
        folders = ["performance_stats"]

    f_name = "_".join([*results_f_name, model])
    # path = os.path.join(folder, f_name)
    results: Dict[str, Dict[str, Dict[str, Dict[str, Any]]]] = {}
    results[model] = {}

    for folder in folders:
        path = os.path.join(folder, f_name)
        list_of_files = glob.glob(path + "*")
        # list_of_files = glob.glob(path + '*')
        # results = {}
        # results[model] = {}
        for path in list_of_files:
            with open(path, "r") as file:
                res_l = json.load(file)
            for model in res_l.keys():
                #             pp.pprint(res_l[model])
                for noise_type, res in res_l[model].items():
                    if noise_type not in results[model].keys():
                        results[model][noise_type] = {}
                    for strength in res_l[model][noise_type].keys():
                        try:
                            old_vals = results[model][noise_type][strength][
                                "train_score"
                            ]
                        except KeyError:
                            results[model][noise_type][strength] = {}
                            results[model][noise_type][strength]["train_score"] = []
                            old_vals = []

                        new_vals = np.unique(
                            old_vals + res_l[model][noise_type][strength]["train_score"]
                        )
                        results[model][noise_type][strength][
                            "train_score"
                        ] = new_vals.tolist()

                        for test_data in res_l[model][noise_type][strength].keys():
                            if test_data != "train_score":
                                try:
                                    old_vals = results[model][noise_type][strength][
                                        test_data
                                    ]["test_score"]
                                except KeyError:
                                    results[model][noise_type][strength][test_data] = {}
                                    results[model][noise_type][strength][test_data][
                                        "test_score"
                                    ] = np.empty((0))
                                    results[model][noise_type][strength][test_data][
                                        "conf_mat"
                                    ] = np.empty((0, 2, 2))

                                old_vals = results[model][noise_type][strength][
                                    test_data
                                ]["test_score"]
                                new_vals = np.unique(
                                    np.append(
                                        old_vals,
                                        res_l[model][noise_type][strength][test_data][
                                            "test_score"
                                        ],
                                    )
                                )
                                results[model][noise_type][strength][test_data][
                                    "test_score"
                                ] = new_vals

                                old_vals = results[model][noise_type][strength][
                                    test_data
                                ]["conf_mat"]
                                new_vals = np.asarray(
                                    res_l[model][noise_type][strength][test_data][
                                        "conf_mat"
                                    ]
                                )
                                new_vals = merge_lists(old_vals, new_vals)
                                #                     new_vals[0][0] = np.unique(new_vals[0][0]).tolist()
                                #                     new_vals[0][0] = np.unique(new_vals[0][0]).tolist()
                                results[model][noise_type][strength][test_data][
                                    "conf_mat"
                                ] = new_vals

    # pp.pprint(results)
    stats: Dict[str, Dict[str, Dict[str, Dict[str, Any]]]] = {}
    for model in results.keys():
        stats[model] = {}
        for noise_type in results[model].keys():
            stats[model][noise_type] = {}
            for strength in results[model][noise_type].keys():
                stats[model][noise_type][strength] = {}

                training_scores = results[model][noise_type][strength]["train_score"]
                stats[model][noise_type][strength]["training_score"] = [
                    np.mean(training_scores) * 100,
                    np.std(training_scores) * 100,
                ]
                #                 try:
                #                     stats[model][noise_type][strength]['training_score'] = [np.mean(training_scores)*100, np.std(training_scores)*100]
                #                 except KeyError:
                #                     stats[model][noise_type][strength] = {}
                #                     stats[model][noise_type][strength]['training_score'] = [np.mean(training_scores)*100, np.std(training_scores)*100]

                for test_data in results[model][noise_type][strength].keys():
                    if test_data != "train_score":
                        if test_data not in stats[model][noise_type][strength].keys():
                            stats[model][noise_type][strength][test_data] = {}
                        test_score = results[model][noise_type][strength][test_data][
                            "test_score"
                        ]
                        stats[model][noise_type][strength][test_data]["test_score"] = [
                            np.mean(test_score) * 100,
                            np.std(test_score) * 100,
                        ]

                        conf_mats = results[model][noise_type][strength][test_data][
                            "conf_mat"
                        ]
                        stats[model][noise_type][strength][test_data]["conf_mat"] = [
                            np.mean(conf_mats, axis=0),
                            np.std(conf_mats, axis=0),
                        ]

    return stats


def reformat_noise_stats(
    stats: Dict[str, Dict[str, Dict[str, Dict[str, Any]]]],
    model: str,
    metric: str,
    folders: Optional[List[str]] = None,
) -> Dict[str, Dict[str, Dict[str, Dict[str, Any]]]]:
    if folders is None:
        folders = ["performance_stats"]

    mdl = model
    format_fct = metrics[metric]

    sub_res: Dict[str, Dict[str, Dict[str, Dict[str, Any]]]] = {}
    sub_res[mdl] = {}

    for noise_type in stats[mdl].keys():
        sub_res[mdl][name_mapping[noise_type]] = {}
        for strength in stats[mdl][noise_type].keys():
            sub_res[mdl][name_mapping[noise_type]][strength] = {}
            sub_res[mdl][name_mapping[noise_type]][strength][train_score] = format_fct(
                stats[mdl][noise_type][strength]["training_score"][0],
                stats[mdl][noise_type][strength]["training_score"][1],
            )
            for test_data in stats[mdl][noise_type][strength].keys():
                if test_data != "training_score":
                    res = format_fct(
                        stats[mdl][noise_type][strength][test_data][metric][0],
                        stats[mdl][noise_type][strength][test_data][metric][1],
                    )
                    sub_res[mdl][name_mapping[noise_type]][strength][
                        name_mapping[test_data]
                    ] = res

    ## Get no noise acc
    stats_nonoise = summarize_results(
        model,
        metric,
        ["performances", "results"],
        folders=folders,
    )
    sub_res_nonoise = reformat_results(stats_nonoise, model, metric)
    sub_res[mdl]["No noise"] = {}
    sub_res[mdl]["No noise"]["0"] = {}

    for test_data in sub_res_nonoise[mdl]["synthetic"].keys():
        if test_data != "synthetic":
            sub_res[mdl]["No noise"]["0"][test_data] = sub_res_nonoise[mdl][
                "synthetic"
            ][test_data]
    return sub_res


def summarize_transfer_learning_results(
    model: str,
    metric: str,
    results_f_name: Optional[List[str]] = None,
    folders: Optional[List[str]] = None,
) -> Dict[str, Dict[str, Dict[str, Any]]]:
    if folders is None:
        folders = ["performance_stats"]
    if results_f_name is None:
        results_f_name = ["transfer", "learning", "results"]

    f_name = "_".join([*results_f_name, model])

    results: Dict[str, Dict[str, Dict[str, Any]]] = {}
    results[model] = {}
    for folder in folders:
        path = os.path.join(folder, f_name)
        list_of_files = glob.glob(path + "*")

        for path in list_of_files:
            with open(path, "r") as file:
                res_l = json.load(file)

            for model in res_l.keys():
                for data_combo, res in res_l[model].items():
                    if data_combo not in results[model].keys():
                        results[model][data_combo] = {}

                    try:
                        old_vals = results[model][data_combo]["train_score"]
                    except KeyError:
                        results[model][data_combo] = {}
                        results[model][data_combo]["train_score"] = []
                        old_vals = []

                    new_vals = np.unique(
                        old_vals + res_l[model][data_combo]["train_score"]
                    )
                    results[model][data_combo]["train_score"] = new_vals.tolist()

                    for test_data in res.keys():
                        #                     results[model][data_combo][test_data] = {}
                        if test_data != "train_score":
                            try:
                                old_vals = results[model][data_combo][test_data][
                                    "pretrained_acc"
                                ]
                            except KeyError:
                                results[model][data_combo][test_data] = {}
                                results[model][data_combo][test_data][
                                    "diff_acc"
                                ] = np.empty((0))
                                results[model][data_combo][test_data][
                                    "pretrained_acc"
                                ] = np.empty((0))
                                results[model][data_combo][test_data][
                                    "pretrained_conf_mat"
                                ] = np.empty((0, 2, 2))
                                results[model][data_combo][test_data][
                                    "transfer_acc"
                                ] = np.empty((0))
                                results[model][data_combo][test_data][
                                    "transfer_conf_mat"
                                ] = np.empty((0, 2, 2))

                            for score in ["pretrained_acc", "transfer_acc", "diff_acc"]:
                                old_vals = results[model][data_combo][test_data][score]
                                new_vals = np.unique(
                                    np.append(
                                        old_vals,
                                        res_l[model][data_combo][test_data][score],
                                    )
                                )
                                results[model][data_combo][test_data][score] = new_vals
                            for mat in ["pretrained_conf_mat", "transfer_conf_mat"]:
                                old_vals = results[model][data_combo][test_data][mat]
                                new_vals = np.asarray(
                                    res_l[model][data_combo][test_data][mat]
                                )
                                new_vals = merge_lists(old_vals, new_vals)
                                #                     new_vals[0][0] = np.unique(new_vals[0][0]).tolist()
                                #                     new_vals[0][0] = np.unique(new_vals[0][0]).tolist()
                                results[model][data_combo][test_data][mat] = new_vals
    #     print(results)
    stats: Dict[str, Dict[str, Dict[str, Any]]] = {}
    for model in results.keys():
        stats[model] = {}
        for training_data in results[model].keys():
            training_scores = results[model][training_data]["train_score"]
            try:
                stats[model][training_data]["training_score"] = [
                    np.mean(training_scores) * 100,
                    np.std(training_scores) * 100,
                ]
            except KeyError:
                stats[model][training_data] = {}
                stats[model][training_data]["training_score"] = [
                    np.mean(training_scores) * 100,
                    np.std(training_scores) * 100,
                ]

            for test_data in results[model][training_data].keys():
                if test_data != "train_score":
                    #                     data_combo = '_'.join([train_data, trans_data])
                    if test_data not in stats[model][training_data].keys():
                        stats[model][training_data][test_data] = {}
                    for score in ["pretrained_acc", "transfer_acc", "diff_acc"]:
                        if score == "diff_acc":
                            test_score = -1 * np.array(
                                results[model][training_data][test_data][score]
                            )
                        #                             test_score = test_score - np.array(results[model][training_data][test_data]['pretrained_acc'])
                        else:
                            test_score = results[model][training_data][test_data][score]

                        stats[model][training_data][test_data][score] = [
                            np.mean(test_score) * 100,
                            np.std(test_score) * 100,
                        ]

                    for mat in ["pretrained_conf_mat", "transfer_conf_mat"]:
                        conf_mats = results[model][training_data][test_data][mat]
                        stats[model][training_data][test_data][mat] = [
                            np.mean(conf_mats, axis=0),
                            np.std(conf_mats, axis=0),
                        ]

    return stats


def reformat_transfer_learning(
    stats: Dict[str, Dict[str, Dict[str, Any]]],
    model: str,
    metric: str,
) -> Dict[str, Dict[str, Dict[str, Dict[str, Any]]]]:
    mdl = model
    format_fct = metrics[metric]

    if metric == "test_score":
        scores = ["pretrained_acc", "transfer_acc"]
    elif metric == "conf_mat":
        scores = ["transfer_conf_mat"]

    sub_res: Dict[str, Dict[str, Dict[str, Dict[str, Any]]]] = {}
    sub_res[mdl] = {}

    for ti, training_data in enumerate(stats[mdl].keys()):
        sub_res[mdl][name_mapping[training_data]] = {}
        sub_res[mdl][name_mapping[training_data]][train_score] = format_fct(
            stats[mdl][training_data]["training_score"][0],
            stats[mdl][training_data]["training_score"][1],
        )

        for it, test_data in enumerate(["clean", "good", "good_and_bad"]):
            for score in scores:
                new_str = " ".join([name_mapping[score], "r" + str(it)])
                try:
                    res = format_fct(
                        stats[mdl][training_data][test_data][score][0],
                        stats[mdl][training_data][test_data][score][1],
                    )
                except KeyError:
                    res = "nan (nan)"
                sub_res[mdl][name_mapping[training_data]][new_str] = res

    return sub_res

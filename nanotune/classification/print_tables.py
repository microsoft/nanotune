import os

import numpy as np
import matplotlib.pyplot as plt

import pandas as pd

from nanotune.classification.format_results import *

import pprint
import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)
pp = pprint.PrettyPrinter(indent=4)

index_list = [
    "synthetic",
    "good experimental",
    "all experimental",
    "synthetic and good experimental",
    "synthetic and all experimental",
    "synthetic with noise",
]

transfer_index_list = [
    "synthetic and good experimental",
    "synthetic and all experimental",
    "noisy synthetic and good experimental",
    "noisy synthetic and all experimental",
]

# TODO: Fix PEP8


def print_performances(
    models: List[str],
    metrics: List[str],
    tables_directory: str,
    res_folders: Optional[List[str]] = None,
) -> None:
    """"""
    for model in models:
        for metric in metrics:
            stats = summarize_results(
                model,
                metric,
                ["performances", "results"],
                folders=res_folders,
            )
            sub_res = reformat_results(stats, model, metric)

            df = pd.DataFrame.from_dict(sub_res[model], orient="index")

            df = df.loc[index_list, :]

            table_name = model + "_" + metric + ".tex"

            filepath = os.path.join(tables_directory, table_name)

            with open(filepath, "w") as tf:
                output = df.to_latex(
                    index=True, column_format="@{\extracolsep{8pt}}lccccc", escape=False
                )
                if metric == "conf_mat":
                    output = output.replace(
                        "\\toprule",
                        "\\hline \\hline \n "
                        + " training data & \multicolumn{5}{c}{average accuracy}  \\\\ \cline{2-6} \n",
                    )
                else:
                    output = output.replace(
                        "\\toprule",
                        "\\hline \\hline \n "
                        + " training data & \multicolumn{5}{c}{average accuracy (std)}  \\\\ \cline{2-6} \n",
                    )
                output = output.replace("\\midrule", "\\hline")
                output = output.replace("\\bottomrule", "\\hline \\hline")

                output = output.replace("\end{bmatr...", "\end{bmatrix}$")
                tf.write(output)


def print_transfer_learning(
    models: List[str],
    metrics: List[str],
    tables_directory: str,
    res_folders: Optional[List[str]] = None,
) -> None:
    """"""
    for model in models:
        for metric in metrics:
            stats = summarize_transfer_learning_results(
                model, metric, ["transfer", "learning", "results"], folders=res_folders
            )
            sub_res = reformat_transfer_learning(stats, model, metric)

            df = pd.DataFrame.from_dict(sub_res[model], orient="index")
            df = df.loc[transfer_index_list, :]
            table_name = "_".join(["transfer", "stats", model, metric]) + ".tex"

            filepath = os.path.join(tables_directory, table_name)

            with open(filepath, "w") as tf:
                output = df.to_latex(
                    index=True,
                    column_format="@{\extracolsep{5pt}}lccccccc",
                    escape=False,
                )
                #             output = output.replace('\\toprule', '\\hline \\hline \n ')
                output = output.replace(
                    "\\toprule",
                    "\hline \\hline \n "
                    + "  training \&  transfer data &   \multicolumn{7}{c}{average accuracy (std)} \\\\  \cline{2-8} \n "
                    + " & transfer training   & \multicolumn{2}{c}{clean experimental} & \multicolumn{2}{c}{good experimental} & \multicolumn{2}{c}{all experimental} \\\\ "
                    + " \cline{2-2} \cline{3-4} \cline{5-6} \cline{7-8} \n"
                    +
                    #                             '  &  training ACC &   synthetic &     clean exp &      good exp &  all exp \\\\\n ',
                    "",
                )  # & \multicolumn{2}{c}{PCA} & \multicolumn{2}{c}{no PCA} \\\\ \cline{2-3} \cline{4-5} ')
                output = output.replace("\\midrule", "\\hline")
                output = output.replace("\\bottomrule", "\\hline \\hline")
                output = output.replace("r0", "")
                output = output.replace("r1", "")
                output = output.replace("r2", "")
                output = output.replace("training &", " &")
                #             output = output.replace('r10', '')
                #             output = output.replace('r11', '')
                #             output = output.replace('r12', '')
                #             output = output.replace('r20', '')
                #             output = output.replace('r21', '')
                #             output = output.replace('r22', '')
                tf.write(output)


def print_noise_strenghts(
    models: List[str],
    metrics: List[str],
    tables_directory: str,
    res_folders: Optional[List[str]] = None,
) -> None:
    """"""
    for model in models:
        for metric in metrics:
            table_name = "_".join(["noise", "stats", model, metric]) + ".tex"

            filepath = os.path.join(tables_directory, table_name)

            stats = summarize_noise_results(
                model,
                metric,
                ["noise", "strengths", "results"],
                folders=res_folders,
            )
            # print(stats)
            sub_res = reformat_noise_stats(stats, model, metric, folders=res_folders)

            output = "\\begin{tabular}{@{\\extracolsep{8pt}}lcccc} \n \\hline \\hline \n & \\multicolumn{4}{c}{average accuracy (std)}  \\\\ \\cline{2-5} \n"
            output += "max noise amplitude  & training & clean experimental & good experimental & all experimental \\\\\n"
            output += "\\hline "

            noise_types = list(sub_res[model].keys())

            noise_types.remove("No noise")
            noise_types.insert(0, "No noise")

            for noise_type in noise_types:
                output += "\multicolumn{5}{c}{" + noise_type + "} \\\\ \cline{1-5} "
                df = pd.DataFrame.from_dict(
                    sub_res[model][noise_type],
                    orient="index",
                    # dtype="string",
                )
                append_out = df.to_latex(
                    index=True,
                    #                          formatters=[dont_format, dont_format,
                    #                                      dont_format, dont_format,
                    #                                      dont_format],
                    #                          header=header2,
                    column_format="",
                    escape=False,
                )
                #             sub_header = noise_type + '& strength'
                append_out = append_out.replace(
                    "\\toprule", ""
                )  # & \multicolumn{2}{c}{PCA} & \multicolumn{2}{c}{no PCA} \\\\ \cline{2-3} \cline{4-5} ')
                append_out = append_out.replace("\\midrule", "")
                append_out = append_out.replace("\\bottomrule", "")
                append_out = append_out.replace("\\begin{tabular}", "")
                append_out = append_out.replace("\\end{tabular}", "\\hline  \\hline ")
                append_out = append_out.replace(
                    "{} &   training & clean experimental & good experimental & all experimental \\\\\n",
                    " ",
                )
                append_out = append_out.replace(
                    "{} &      training & clean experimental & good experimental & all experimental \\\\\n",
                    " ",
                )

                output += append_out

            output += "\\hline \\hline\n \\end{tabular}\n"

            with open(filepath, "w") as tf:
                tf.write(output)


def print_binary_stats(
    models: List[str],
    metrics: List[str],
    tables_directory: str,
    res_folders: Optional[List[str]] = None,
) -> None:
    """"""
    if res_folders is None:
        res_folders = ["performance_stats"]
    for metric in metrics:
        output = "\\begin{tabular}{@{\\extracolsep{8pt}}lccccc}\n\\hline \\hline \n"
        output += " training data & \multicolumn{5}{c}{average accuracy (std)}  \\\\ \\hline \n"  # \cline{2-6}
        for model in models:
            output += "\multicolumn{6}{c}{" + model + "}  \\\\ \cline{1-6} \n"
            stats = summarize_results(
                model,
                metric,
                ["binary", "performances", "results"],
                folders=res_folders,
            )
            sub_res = reformat_results(stats, model, metric)

            df = pd.DataFrame.from_dict(
                sub_res[model],
                orient="index",
            )
            df = df.loc[index_list, :]
            suboutput = df.to_latex(
                index=True, column_format="@{\extracolsep{8pt}}lccccc", escape=False
            )
            suboutput = suboutput.replace("\\toprule", "\\hline \\hline \n ")
            suboutput = suboutput.replace("\\midrule", "\\hline")
            suboutput = suboutput.replace("\\bottomrule", "\\hline \\hline")
            suboutput = suboutput.replace(
                "\\begin{tabular}{@{\\extracolsep{8pt}}lccccc}\n\\hline \\hline \n", ""
            )
            suboutput = suboutput.replace("\n\\end{tabular}\n", "")
            output += suboutput

        output += "\n\\hline \\hline\n\\end{tabular}\n"

        table_name = "binary_clf_" + metric + ".tex"
        filepath = os.path.join(tables_directory, table_name)

        with open(filepath, "w") as tf:
            tf.write(output)

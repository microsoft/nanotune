import logging
from typing import Dict, Optional
from prettytable import PrettyTable
import numpy as np
from nanotune.classification.classifier import METRIC_NAMES

logger = logging.getLogger(__name__)

metric_mapping = {
    "accuracy_score": "accuracy",
    "auc": "AUC",
    "average_precision_recall": "precision recall",
    "brier_score_loss": "Brier loss",
}

def display_metrics(
        info_dict: Dict[str, Dict[str, float]],
        all_of_it: Optional[bool] = False,
    ) -> None:
        """Displays the metrics calculated by `Classifier.compute_metrics`."""
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

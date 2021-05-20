#pylint: disable=line-too-long, too-many-arguments, too-many-locals

import logging
import qcodes as qc


class QcodesDbConfig:
    """Context Manager for temporarily switching the qcodes database to another
    """

    def __init__(self, qcodes_db_path: str) -> None:
        """Initializes a QCoDeS database configurator.

        Args:
            qcodes_db_path: Path to database.
        """

        self._db_path = qcodes_db_path
        self._orig_db_path = qc.config["core"]["db_location"]

    def __enter__(self) -> None:
        self._orig_db_path = qc.config["core"]["db_location"]
        qc.config["core"]["db_location"] = self._db_path
        logging.info("Changed qcodes db to %s", self._db_path)

    def __exit__(self, *exc) -> None:
        qc.config["core"]["db_location"] = self._orig_db_path
        logging.info("Restoring qcodes db to %s", self._orig_db_path)


def dump_db(db_path: str) -> None:
    """ Utility method to dump the experiments, datasets, and parameters from a
    qcodes database
    """

    # Some test code to get started
    with QcodesDbConfig(db_path):

        print("Path to DB      : {0}".format(db_path))

        for exp in qc.experiments():
            print("Experiment name : {0}".format(exp.name))
            print("dataset count   : {0}".format(len(exp.data_sets())))
            print("first dataset   : {0}".format(exp.data_sets()[0].run_id))
            last_id = exp.last_data_set().run_id
            print("last dataset    : {0}".format(last_id))
            print("")

            for dataset in exp.data_sets():
                msg = [f"{p.label}({p.unit})" for p in dataset.get_parameters()]
                params = ", ".join(msg)
                print(f"{dataset.captured_run_id}: {dataset.exp_name} {params}")

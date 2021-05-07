#pylint: disable=line-too-long, too-many-arguments, too-many-locals

''' Contains utility classes related to QCoDeS '''

import logging
import qcodes as qc

class QcodesDbConfig:

    ''' Context Manager for temporarily switching the qcodes database to another '''

    def __init__(self, qcodes_db_path):
        self._db_path = qcodes_db_path
        self._orig_db_path = qc.config["core"]["db_location"]

    def __enter__(self):
        self._orig_db_path = qc.config["core"]["db_location"]
        qc.config["core"]["db_location"] = self._db_path
        logging.info("Changed qcodes db to %s", self._db_path)

    def __exit__(self, exc_type, exc_value, exc_tb):
        qc.config["core"]["db_location"] = self._orig_db_path
        logging.info("Restoring qcodes db to %s", self._orig_db_path)


def dump_db(db_path):

    ''' Utility method to dump the experiments, datasets, and parameters from a qcodes database '''

    # Some test code to get started
    with QcodesDbConfig(db_path):

        print("Path to DB      : {0}".format(db_path))

        for exp in qc.experiments():
            print("Experiment name : {0}".format(exp.name))
            print("dataset count   : {0}".format(len(exp.data_sets())))
            print("first dataset   : {0}".format(exp.data_sets()[0].run_id))
            print("last dataset    : {0}".format(exp.last_data_data_set().run_id))
            print("")

            for dataset in exp.data_sets():
                params = ", ".join([f"{p.label}({p.unit})" for p in dataset.get_parameters()])
                print(f"{dataset.captured_run_id}  : {dataset.exp_name}     {params}")

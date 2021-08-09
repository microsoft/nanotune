import logging
from sqlite3 import OperationalError
from typing import List, Optional, Tuple

import matplotlib
import PyQt5.QtCore as qtc
import PyQt5.QtGui as qtg
import PyQt5.QtWidgets as qtw
import qcodes as qc
from matplotlib import rcParams
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
from matplotlib.figure import Figure
from qcodes.dataset.experiment_container import (experiments,
                                                 load_experiment,
                                                 load_experiment_by_name)
from qcodes.dataset.plotting import plot_by_id
from qcodes.dataset.sqlite.queries import get_runs

import nanotune as nt

# from PyQt5.QtCore import *
# from PyQt5.QtGui import *
# from PyQt5.QtWidgets import *

logger = logging.getLogger(__name__)

label_bad = "Not Good"


class LabellingTool(qtw.QMainWindow):
    """"""

    def __init__(
        self,
        experiment_id: Optional[int] = None,
        db_folder: Optional[str] = None,
        db_name: Optional[str] = None,
        start_over: bool = False,
        figure_fontsize: int = 8,
    ) -> None:
        """"""
        if db_folder is None:
            db_folder = nt.config["db_folder"]

        LABELS = list(dict(nt.config["core"]["labels"]).keys())

        if db_name is None:
            logger.warning("Labelling default main database.")
            db_name = nt.config["db_name"]
        nt.set_database(db_name, db_folder)
        self.db_name = db_name
        self.db_folder = db_folder

        matplotlib.rc("font", size=figure_fontsize)
        super(LabellingTool, self).__init__()

        self.current_label = dict.fromkeys(LABELS, 0)
        self.experiment_id = experiment_id

        (self._iterator_list,
         self.labelled_ids,
         self.n_total) = self.get_data_ids(start_over)

        self._id_iterator = iter(self._iterator_list)
        try:
            self.current_id = self._id_iterator.__next__()
        except StopIteration:
            logger.warning(
                "All data of this experiment is already labelled."
            )
            print("All data already labelled.")
            return

        self._main_widget = qtw.QWidget(self)
        self.setCentralWidget(self._main_widget)
        self.initUI()
        self.show()

    def get_data_ids(
        self,
        start_over: bool = False,
    ) -> Tuple[List[int], List[int], int]:
        """"""

        if self.experiment_id is None:
            all_ids = []
            for experiment in experiments():
                runs = get_runs(experiment.conn, experiment.exp_id)
                all_ids += [run['captured_run_id'] for run in runs]
        else:
            try:
                experiment = load_experiment(self.experiment_id)
                runs = get_runs(experiment.conn, experiment.exp_id)
                all_ids = [run['captured_run_id'] for run in runs]
            except ValueError as v:
                msg = "Unable to load experiment."
                print(v)
                qtw.QMessageBox.warning(
                    self, "Error instantiating LabellingTool.", msg,
                    qtw.QMessageBox.Ok
                )

        if start_over:
            unlabelled_ids = all_ids
            labelled_ids = []
        else:
            unlabelled_ids_all = nt.get_unlabelled_ids(
                self.db_name,
                db_folder=self.db_folder,
                get_run_id=False,
            )
            unlabelled_ids = []
            labelled_ids = []
            for run_id in all_ids:
                if run_id in unlabelled_ids_all:
                    unlabelled_ids.append(run_id)
                else:
                    labelled_ids.append(run_id)

        return unlabelled_ids, labelled_ids, len(all_ids)

    def initUI(self) -> None:
        """"""

        self.progressbar = qtw.QProgressBar()
        self.progressbar.setMinimum(1)
        self.progressbar.setMaximum(100)

        self.statusBar().addPermanentWidget(self.progressbar)

        pp = len(self.labelled_ids) / self.n_total * 100
        self.progressbar.setValue(pp)

        self.statusBar().showMessage("Are we there yet?")

        self.setGeometry(300, 250, 700, 1100)
        # integers are:
        # X coordinate
        # Y coordinate
        # Width of the frame
        # Height of the frame
        self.setWindowTitle("nanotune labelling tool")

        self._main_layout = self.initMainLayout()
        self._main_widget.setLayout(self._main_layout)

    def initMainLayout(self) -> qtw.QVBoxLayout:
        """"""
        # -----------  Main Layout  ----------- #
        layout = qtw.QVBoxLayout(self._main_widget)
        # -----------  Figure row  ----------- #
        figure_row = qtw.QVBoxLayout()

        self.l1 = qtw.QLabel()
        self.l1.setText("Plot ID: {}".format(self.current_id))
        self.l1.setAlignment(qtc.Qt.AlignCenter)
        figure_row.addWidget(self.l1)

        rcParams.update({"figure.autolayout": True})

        self._figure = Figure(tight_layout=True)
        self._axes = self._figure.add_subplot(111)
        self._cb = [None]

        while True:
            try:
                _, self._cb = plot_by_id(self.current_id, axes=self._axes)
                break
            except (RuntimeError, IndexError) as r:
                logger.warning("Skipping current dataset" + str(r))
                self.labelled_ids.append(self.current_id)
                try:
                    self.current_id = self._id_iterator.__next__()
                except StopIteration:
                    logger.error("All datasets labelled.")
                    break

        self._canvas = FigureCanvasQTAgg(self._figure)
        self._canvas.setParent(self._main_widget)

        self._canvas.draw()
        self._canvas.draw()

        self._canvas.update()
        self._canvas.flush_events()

        figure_row.addWidget(self._canvas)
        figure_row.addStretch(10)

        # -----------  Buttons row  ----------- #
        self._buttons = []
        self._quality_group = qtw.QButtonGroup(self)
        button_row = qtw.QHBoxLayout()

        # left part of button row
        quality_column = qtw.QVBoxLayout()

        btn_good = qtw.QPushButton("Good")
        btn_good.setObjectName("good")
        btn_good.setCheckable(True)
        self._quality_group.addButton(btn_good)

        self._buttons.append(btn_good)

        btn_bad = qtw.QPushButton(label_bad)
        btn_bad.setObjectName(label_bad)
        btn_bad.setCheckable(True)
        self._quality_group.addButton(btn_bad)
        self._buttons.append(btn_bad)

        quality_column.addWidget(btn_good)
        quality_column.addWidget(btn_bad)

        button_row.addLayout(quality_column)

        # right part of button row
        # list of instances of class label
        labels_column = qtw.QVBoxLayout()
        LABELS = list(dict(nt.config["core"]["labels"]).keys())
        LABELS_MAP = dict(nt.config["core"]["labels"])
        for label in LABELS:
            if label not in ["good"]:

                btn = qtw.QPushButton(LABELS_MAP[label])
                btn.setObjectName(label)
                btn.setCheckable(True)
                labels_column.addWidget(btn)
                self._buttons.append(btn)

        button_row.addLayout(labels_column)

        # -----------   Finalize row   ----------- #
        finalize_row = qtw.QHBoxLayout()

        clear_btn = qtw.QPushButton("Clear")
        finalize_row.addWidget(clear_btn)
        clear_btn.clicked.connect(self.clear)

        save_btn = qtw.QPushButton("Save")
        finalize_row.addWidget(save_btn)
        save_btn.clicked.connect(self.save_labels)

        # -----------   Exit row   ----------- #
        exit_row = qtw.QHBoxLayout()

        exit_btn = qtw.QPushButton("Exit")
        exit_row.addWidget(exit_btn)
        exit_btn.clicked.connect(self.exit)

        empty_space = qtw.QHBoxLayout()
        empty_space.addStretch(1)
        exit_row.addLayout(empty_space)

        # -----------   Add all rows to main vertial box   ----------- #
        layout.addLayout(figure_row)
        layout.addLayout(button_row)
        layout.addLayout(finalize_row)
        layout.addLayout(exit_row)

        return layout

    def next(self) -> None:
        """"""
        self._axes.clear()
        self._axes.relim()

        if self._cb[0] is not None:
            for cbar in self._cb:
                cbar.ax.clear()
                cbar.ax.relim()
                cbar.remove()
        self._figure.tight_layout()

        while True:
            try:
                self.labelled_ids.append(self.current_id)
                self.current_id = self._id_iterator.__next__()

                # Update GUI
                self.l1.setText("Plot ID: {}".format(self.current_id))
                pp = len(self.labelled_ids) / self.n_total * 100
                self.progressbar.setValue(pp)

                _, self._cb = plot_by_id(self.current_id, axes=self._axes)

                self._figure.tight_layout()
                self._canvas.draw()
                self._canvas.update()
                self._canvas.flush_events()
                break

            except StopIteration:
                msg1 = "You are done!"
                msg2 = "All datasets of " + self.experiment.name
                msg2 += " are labelled."
                qtw.QMessageBox.information(self, msg1, msg2, qtw.QMessageBox.Ok)
                return
            except (RuntimeError, IndexError) as r:
                logger.warning("Skipping this dataset " + str(r))
                self.labelled_ids.append(self.current_id)
                try:
                    self.current_id = self._id_iterator.__next__()
                except StopIteration:
                    msg1 = "You are done!"
                    msg2 = "All datasets of " + self.experiment.name
                    msg2 += " are labelled."
                    qtw.QMessageBox.information(self, msg1, msg2, qtw.QMessageBox.Ok)
                    return

    def clear(self) -> None:
        """"""
        self._quality_group.setExclusive(False)
        for btn in self._buttons:
            btn.setChecked(False)
        self._quality_group.setExclusive(True)

        LABELS = list(dict(nt.config["core"]["labels"]).keys())
        self.current_label = dict.fromkeys(LABELS, 0)

    def save_labels(self) -> None:
        """"""

        for button in self._buttons:
            if button.objectName() == label_bad:
                continue
            checked = button.isChecked()
            self.current_label[button.objectName()] = int(checked)

        if self._quality_group.checkedId() == -1:
            msg = 'Please choose quality. \n \n Either "Good" or '
            msg += '"' + label_bad + '"' + " has"
            msg += " to be selected."
            qtw.QMessageBox.warning(
                self, "Cannot save label.", msg, qtw.QMessageBox.Ok)
        else:
            ds = qc.load_by_run_spec(captured_run_id=self.current_id)

            for label, value in self.current_label.items():
                ds.add_metadata(label, value)

            self.clear()
            self.next()

    def exit(self) -> None:
        """"""
        self.n_total - len(self.labelled_ids)
        quit_msg1 = "Please don't go"
        quit_msg2 = "Would you like to give it another try?"
        reply = qtw.QMessageBox.question(
            self, quit_msg1, quit_msg2, qtw.QMessageBox.Yes, qtw.QMessageBox.No
        )

        if reply == qtw.QMessageBox.No:
            self.close()

import logging

from typing import List, Optional, Tuple

# from PyQt5.QtCore import *
# from PyQt5.QtGui import *
# from PyQt5.QtWidgets import *

import PyQt5.QtCore as qtc
import PyQt5.QtGui as qtg
import PyQt5.QtWidgets as qtw

import matplotlib
from matplotlib import rcParams
from sqlite3 import OperationalError

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
from matplotlib.figure import Figure

import qcodes as qc
from qcodes.dataset.plotting import plot_by_id
from qcodes.dataset.experiment_container import (
    load_by_id,
    experiments,
    load_experiment_by_name,
    load_experiment,
)
import nanotune as nt

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
            db_name = nt.config["main_db"]
        nt.set_database(db_name)
        self.db_name = db_name

        self.db_folder = db_folder

        # print(qc.config['core']['db_location'])
        matplotlib.rc("font", size=figure_fontsize)
        super(LabellingTool, self).__init__()

        self.current_label = dict.fromkeys(LABELS, 0)
        self.experiment_id = experiment_id

        if self.experiment_id is None:
            logger.error(
                "Please select an experiment. Labelling entire "
                + " database is not supported yet."
            )
            raise NotImplementedError
            # all_experiments = experiments()
            # for e in all_experiments:
            # self.experiment = e
            # (self._iterator_list,
            #  self.labelled_ids,
            #  self.n_total) = self.get_data_ids(start_over)
        else:
            try:
                self.experiment = load_experiment(self.experiment_id)

                (
                    self._iterator_list,
                    self.labelled_ids,
                    self.n_total,
                ) = self.get_data_ids(start_over)

                self._id_iterator = iter(self._iterator_list)
                try:
                    self.current_id = self._id_iterator.__next__()
                except StopIteration:
                    logger.warning(
                        "All data of this experiment is already " + "labelled"
                    )
                    raise

            except ValueError:
                msg = "Unable to load experiment."
                # ee = experiments()
                # for e in ee:
                #     msg += e.name + '\n'
                qtw.QMessageBox.warning(
                    self, "Error instantiating LabellingTool.", msg, qtw.QMessageBox.Ok
                )
            except IndexError:
                msg = "Did not find any unlabelled data in experiment "
                msg += self.experiment.name + "."
                qtw.QMessageBox.warning(
                    self, "Error instantiating LabellingTool.", msg, qtw.QMessageBox.Ok
                )

        self._main_widget = qtw.QWidget(self)
        self.setCentralWidget(self._main_widget)

        self.initUI()
        self.show()

    def get_data_ids(
        self,
        start_over: bool = False,
    ) -> Tuple[List[int], List[int], int]:
        """"""
        unlabelled_ids: List[int] = []
        labelled_ids: List[int] = []
        print("getting datasets")

        last_id = nt.get_last_dataid(self.db_name, db_folder=self.db_folder)
        all_ids = list(range(1, last_id))

        # dds = self.experiment.data_sets()
        # if len(dds) == 0:
        #     logger.error('Experiment has no data. Nothing to label.')
        #     raise ValueError

        # start_id = dds[0].run_id
        # stop_id = dds[-1].run_id
        # all_ids = list(range(start_id, stop_id+1))
        print("len(all_ids): " + str(len(all_ids)))

        # check if database has label columns as column
        if not start_over:
            # Make sure database has nanotune label columns. Just a check.
            try:
                ds = load_by_id(1)
                ds.get_metadata("good")
            except OperationalError:
                logger.warning(
                    """No nanotune_label column found in current
                                database. Probably because no data has been
                                labelled yet. Hence starting over. """
                )
                start_over = True
            # except RuntimeError:
            #     logger.error('Probably data in experiment.')
            #     raise
        print("start_over: " + str(start_over))
        if start_over:
            unlabelled_ids = all_ids
            labelled_ids = []
        else:
            unlabelled_ids = nt.get_unlabelled_ids(self.db_name)
            labelled_ids = [x for x in all_ids if x not in unlabelled_ids]

        return unlabelled_ids, labelled_ids, len(all_ids)

    def initUI(self) -> None:
        """"""
        # msg = str(len(self.labelled_ids)) + ' labelled, '
        # msg += str(self.n_total - len(self.labelled_ids)) + ' to go.'
        self.progressbar = qtw.QProgressBar()
        self.progressbar.setMinimum(1)
        self.progressbar.setMaximum(100)
        # self.progressbar.setTextVisible(True)

        self.statusBar().addPermanentWidget(self.progressbar)
        # self.progressbar.setGeometry(30, 40, 200, 25)

        pp = len(self.labelled_ids) / self.n_total * 100
        self.progressbar.setValue(pp)

        self.statusBar().showMessage("Are we there yet?")

        self.setGeometry(300, 250, 700, 1100)
        # integers are:
        # X coordinate
        # Y coordinate
        # Width of the frame
        # Height of the frame
        self.setWindowTitle("nanotune Labelling Tool")

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
                # if self._cb[0] is not None:
                #     _, self._cb = plot_by_id(self.current_id, axes=self._axes,
                #                              colorbars=self._cb[0])
                # else:
                _, self._cb = plot_by_id(self.current_id, axes=self._axes)
                # title = 'Run #{}  Experiment #{}'.format(self.current_id,
                #                                          self.experiment_id)
                # for ax in self._axes:
                #     ax.set_title(title)
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
                # bl = QHBoxLayout()
                btn = qtw.QPushButton(LABELS_MAP[label])
                btn.setObjectName(label)
                btn.setCheckable(True)
                # bl.addWidget(btn)
                # bl.addStretch(1)
                # labels_column.addLayout(bl)
                labels_column.addWidget(btn)
                self._buttons.append(btn)

        button_row.addLayout(labels_column)

        # -----------   Finalize row   ----------- #
        finalize_row = qtw.QHBoxLayout()

        # go_back_btn = QPushButton('Go Back')
        # finalize_row.addWidget(go_back_btn)
        # go_back_btn.clicked.connect(self.go_back)

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
        # TO DO: Loop to the next unlabelled dataset ...
        self._axes.clear()
        self._axes.relim()
        # if len(self._figure.axes) > 1:
        #     self._figure.delaxes(self._figure.axes[1])
        if self._cb[0] is not None:
            for cbar in self._cb:
                cbar.ax.clear()
                cbar.ax.relim()
                cbar.remove()
                # cbar = None
        self._figure.tight_layout()

        while True:
            try:
                self.labelled_ids.append(self.current_id)
                self.current_id = self._id_iterator.__next__()

                # Update GUI
                self.l1.setText("Plot ID: {}".format(self.current_id))
                pp = len(self.labelled_ids) / self.n_total * 100
                self.progressbar.setValue(pp)
                # _, self._cb = plot_by_id(self.current_id, axes=self._axes,
                #                              colorbars=self._cb[0])
                # # if self._cb[0] is not None:
                # _, self._cb = plot_by_id(self.current_id, axes=self._axes,
                #                      colorbars=self._cb[0])
                # else:
                _, self._cb = plot_by_id(self.current_id, axes=self._axes)
                # for cbar in self._cb:
                #     cbar.ax.clear()
                #     cbar.ax.relim()
                #     cbar.remove()
                # title = 'Run #{}  Experiment #{}'.format(self.current_id,
                #                                          self.experiment_id)
                # for ax in self._axes:
                #     ax.set_title(title)

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
        # logger.error('Need to update label saving! -> One column per label.')
        # raise NotImplementedError
        for button in self._buttons:
            if button.objectName() == label_bad:
                continue
            checked = button.isChecked()
            self.current_label[button.objectName()] = int(checked)

        if self._quality_group.checkedId() == -1:
            msg = 'Please choose quality. \n \n Either "Good" or '
            msg += '"' + label_bad + '"' + " has"
            msg += " to be selected."
            qtw.QMessageBox.warning(self, "Cannot save label.", msg, qtw.QMessageBox.Ok)
        else:
            ds = load_by_id(self.current_id)

            for label, value in self.current_label.items():
                ds.add_metadata(label, value)

            self.clear()
            self.next()

    # def go_back(self):
    #     """
    #     """
    #     self._axes.clear()
    #     if self._cb[0] is not None:
    #         self._cb[0].ax.clear()

    #     while True:
    #         try:
    #             self.labelled_ids = self.labelled_ids[:-1]
    #             self.current_id -= 1

    #             # Update GUI
    #             self.l1.setText('Plot ID: {}'.format(self.current_id))
    #             pp = (len(self.labelled_ids)/self.n_total*100)
    #             self.progressbar.setValue(pp)
    #             if self._cb[0] is not None:
    #                 _, self._cb = plot_by_id(self.current_id, axes=self._axes,
    #                                          colorbars=self._cb[0])
    #             else:
    #                 _, self._cb = plot_by_id(self.current_id, axes=self._axes)
    #             self._figure.tight_layout()

    #             self._canvas.draw()
    #             self._canvas.update()
    #             self._canvas.flush_events()
    #             break

    #         except (RuntimeError, IndexError) as r:
    #             logger.warning('Skipping a dataset ' +
    #                             str(r))
    #             # self.labelled_ids.append(self.current_id)
    #             # self.current_id = self._id_iterator.__next__()
    #             self.current_id -= 1
    #             self.go_back()
    #             pass

    def exit(self) -> None:
        """"""
        # logger.warning('Saving labels.')
        self.n_total - len(self.labelled_ids)
        quit_msg1 = "Please don't go"  # , nanotune needs you! \n"
        # quit_msg1 += " " + str(n_missing) + ' datasets are calling for labels.'
        quit_msg2 = "Would you like to give it another try?"
        reply = qtw.QMessageBox.question(
            self, quit_msg1, quit_msg2, qtw.QMessageBox.Yes, qtw.QMessageBox.No
        )

        if reply == qtw.QMessageBox.No:
            self.close()

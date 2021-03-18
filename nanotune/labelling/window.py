import sys
from functools import partial
from typing import Optional, Dict

import PyQt5.QtWidgets as qtw

import nanotune as nt

import logging

logger = logging.getLogger(__name__)


class Window(qtw.QMainWindow):
    def __init__(
        self, title: str = "Labelling Tool", labels: Optional[Dict[str, str]] = None
    ):
        if labels is None:
            labels = dict(nt.config["core"]["labels"])

        super(Window, self).__init__()
        self.labels = labels

        self.initUI()

    def initUI(self) -> None:
        """"""
        self.statusBar().showMessage("Ready")

        self.setGeometry(300, 300, 300, 500)
        self.setWindowTitle("Statusbar")

        self.form_widget = Widgets(self)
        self.setCentralWidget(self.form_widget)

        self.show()

    def update_window(self, plot_id: int) -> None:
        pass

    # def close_application(self):

    #     choice = QMessageBox.question(self, 'Message',
    #                                  "Are you sure to quit?", QMessageBox.Yes |
    #                                  QMessageBox.No, QMessageBox.No)
    #     # Make sure to save labels

    #     if choice == QMessageBox.Yes:
    #         print('quit application')
    #         sys.exit()
    #     else:
    #         pass


class Widgets(qtw.QWidget):
    def __init__(self, window: Window) -> None:
        super(Widgets, self).__init__(window)
        layout = qtw.QVBoxLayout(self)

        n_labels = len(window.labels.items())

        figure_row = qtw.QVBoxLayout()

        l1 = qtw.QLabel()
        l1.setText("Plot ID: {}".format(window.labels["plot_id"][0]))
        l1.setAlignment(qtw.AlignCenter)
        figure_row.addWidget(l1)
        # display plot

        # -----------  Buttons row  ----------- #
        button_row = qtw.QHBoxLayout()

        # left part of button row
        goodness_column = qtw.QVBoxLayout()
        goodness_group = qtw.QButtonGroup(self)

        btn_good = qtw.QPushButton("Good")
        btn_good.setCheckable(True)
        btn_good.clicked.connect(partial(self.retain_label, "good"))
        goodness_group.addButton(btn_good)

        btn_bad = qtw.QPushButton("Bad")
        btn_bad.setCheckable(True)
        btn_bad.setChecked(True)
        # No action on click needed as 'bad' is labelled as absence of good.
        goodness_group.addButton(btn_bad)

        btn_goodish = qtw.QPushButton("Good-ish")
        btn_goodish.setCheckable(True)
        btn_goodish.clicked.connect(partial(self.retain_label, "good-ish"))
        goodness_group.addButton(btn_goodish)

        goodness_column.addWidget(btn_good)
        goodness_column.addWidget(btn_bad)
        goodness_column.addWidget(btn_goodish)

        button_row.addLayout(goodness_column)

        # right part of button row
        # list of instances of class label
        labels_column = qtw.QVBoxLayout()
        for short_name, value in window.labels.items():
            if short_name not in ["plot_id", "good", "good-ish"]:
                btn = qtw.QPushButton(value[1])
                btn.setCheckable(True)
                labels_column.addWidget(btn)

        button_row.addLayout(labels_column)

        # -----------   Finalize row   ----------- #
        finalize_row = qtw.QHBoxLayout()
        finalize_row.addWidget(qtw.QPushButton("Clear"))
        finalize_row.addWidget(qtw.QPushButton("Save"))

        # -----------   Exit row   ----------- #
        exit_row = qtw.QHBoxLayout()
        empty_space = qtw.QHBoxLayout()
        empty_space.addStretch(1)

        exit_button = qtw.QVBoxLayout()
        exit_button.addWidget(qtw.QPushButton("Exit"))

        exit_row.addLayout(exit_button)
        exit_row.addLayout(empty_space)

        # -----------   Add all rows to main vertial box   ----------- #
        layout.addLayout(figure_row)
        layout.addLayout(button_row)
        layout.addLayout(finalize_row)
        layout.addLayout(exit_row)

    def exit(self) -> None:
        """"""
        # TODO: Save labels
        logging.warning("Saving labels.")

    def retain_label(self, label: str):
        print(label)

    # def file_open(self):
    #     # need to make name an tupple otherwise i had an error and app crashed
    #     name, _ = QFileDialog.getOpenFileName(self, 'Open File', options=QFileDialog.DontUseNativeDialog)
    #     print('tot na dialog gelukt')  # for debugging
    #     file = open(name, 'r')
    #     print('na het inlezen gelukt') # for debugging
    #     self.editor()

    #     with file:
    #         text = file.read()
    #         self.textEdit.setText(text)

    # def file_save(self):
    #     name, _ = QFileDialog.getSaveFileName(self,'Save File', options=QFileDialog.DontUseNativeDialog)
    #     file = open(name, 'w')
    #     text = self.textEdit.toPlainText()
    #     file.write(text)
    #     file.close()

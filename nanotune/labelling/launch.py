import sys
import qcodes as qc
from PyQt5.QtWidgets import QApplication
from nanotune.labelling.labellingtool import LabellingTool


def launch() -> None:
    app = QApplication(sys.argv)
    Gui = LabellingTool(sys.argv[1])  # type: ignore
    sys.exit(app.exec_())


launch()

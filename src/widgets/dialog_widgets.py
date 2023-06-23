import sys
import matplotlib
matplotlib.use('QtAgg')

from PyQt6.QtWidgets import (
    QLabel, QVBoxLayout, QDialog, QDialogButtonBox
)

class BasicDialog(QDialog):
    def __init__(self, window_title:str, message:str):
        super().__init__()

        self.setWindowTitle(window_title)

        QBtn = QDialogButtonBox.StandardButton.Ok # | QDialogButtonBox.StandardButton.Cancel

        self.buttonBox = QDialogButtonBox(QBtn)
        self.buttonBox.accepted.connect(self.accept)
        self.buttonBox.rejected.connect(self.reject)

        self.layout = QVBoxLayout()
        message_widget = QLabel(message)
        self.layout.addWidget(message_widget)
        self.layout.addWidget(self.buttonBox)
        self.setLayout(self.layout)
import sys
import typing
from PyQt6 import QtCore, QtGui, QtWidgets
from PyQt6.QtCore import Qt
import pandas as pd
from PyQt6.QtWidgets import QApplication, QLabel, QMainWindow, QTextEdit, QVBoxLayout, QWidget

class TableModel(QtCore.QAbstractTableModel):

    def __init__(self, data):
        super(TableModel, self).__init__()
        self._data = data

    def data(self, index, role):
        if role == Qt.ItemDataRole.DisplayRole:
            value = self._data.iloc[index.row(), index.column()]
            return str(value)

    def rowCount(self, index):
        return self._data.shape[0]

    def columnCount(self, index):
        return self._data.shape[1]

    def headerData(self, section, orientation, role):
        # section is the index of the column/row.
        if role == Qt.ItemDataRole.DisplayRole:
            if orientation == Qt.Orientation.Horizontal:
                return str(self._data.columns[section])

            if orientation == Qt.Orientation.Vertical:
                return str(self._data.index[section])


class CustomQTableView(QtWidgets.QTableView):
    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)

    # def mouseMoveEvent(self, e):
    #     print("mouseMoveEvent")

    # def mousePressEvent(self, e):
    #     print("mousePressEvent")
    #     print(e.position())

    # def mouseReleaseEvent(self, e):
    #     print("mouseReleaseEvent")

    # def mouseDoubleClickEvent(self, e):
    #     print("mouseDoubleClickEvent")      


class MainWindow(QtWidgets.QMainWindow):

    def __init__(self):
        super().__init__()

        self.table = CustomQTableView()

        data = pd.DataFrame([
          [1, 9, 2],
          [1, 0, -1],
          [3, 5, 2],
          [3, 3, 2],
          [5, 8, 9],
        ], columns = ['A', 'B', 'C'], index=['Row 1', 'Row 2', 'Row 3', 'Row 4', 'Row 5'])

        self.model = TableModel(data)
        self.table.setModel(self.model)

        layout = QVBoxLayout()

        layout.addWidget(self.table)

        self.label = QLabel("Click in this window")
        layout.addWidget(self.label)
        widget = QWidget()
        widget.setLayout(layout)
        self.setCentralWidget(widget)

    def current_pos(self):
        index = self.table.indexAt(QtWidgets.QApplication.focusWidget().pos())
        if index.isValid():
            print(index.row(), index.column())

    def mouseMoveEvent(self, e):
        self.label.setText("mouseMoveEvent")

    def mousePressEvent(self, e):
        self.label.setText("mousePressEvent")
        self.current_pos()

    def mouseReleaseEvent(self, e):
        self.label.setText("mouseReleaseEvent")

    def mouseDoubleClickEvent(self, e):
        self.label.setText("mouseDoubleClickEvent")


app=QtWidgets.QApplication(sys.argv)
window=MainWindow()
window.show()
app.exec()
import sys
import typing
from PyQt6 import QtCore, QtGui, QtWidgets
from PyQt6.QtCore import Qt, QSize
import pandas as pd
from PyQt6.QtWidgets import QApplication, QLabel, QMainWindow, QTextEdit, QVBoxLayout, QWidget, QStyledItemDelegate, QGridLayout

import inspect
clsmembers = inspect.getmembers(sys.modules[__name__], inspect.isclass)

class ImageWindow(QWidget):
    """
    This "window" is a QWidget. If it has no parent, it
    will appear as a free-floating window as we want.
    """
    def __init__(self, path):
        super().__init__()
        layout = QVBoxLayout()
        # self.label = QLabel("Image Window")
        # layout.addWidget(self.label)
        pixmap = QtGui.QPixmap(path)
        layout = QGridLayout(self)
        for row in range(4):
            for column in range(4):
                label = QLabel(self)
                label.setPixmap(pixmap)
                layout.addWidget(label, row, column)
        self.setLayout(layout)
        self.setWindowTitle("My App")
        # self.setGeometry(500, 300, 300, 300)

class ImageDelegate(QStyledItemDelegate):

    def __init__(self, parent):
        QStyledItemDelegate.__init__(self, parent)

    def paint(self, painter, option, index):        

        painter.fillRect(option.rect, QtGui.QColor(191,222,185))

        # path = "path\to\my\image.jpg"
        path = "/Users/valentinospariza/Library/CloudStorage/OneDrive-UvA/Repositories/multimedia_analytics/mma_2023_uva_group_1/wordcloud.png"

        image = QtGui.QImage(str(path))
        pixmap = QtGui.QPixmap.fromImage(image)
        pixmap.scaled(50, 40, QtCore.Qt.AspectRatioMode.KeepAspectRatio)
        painter.drawPixmap(option.rect, pixmap) 

    def sizeHint(self, option, index) :
        return QSize(160, 90) # whatever your dimensions are

class TableModel(QtCore.QAbstractTableModel):

    def __init__(self, data):
        super(TableModel, self).__init__()
        self._data = data

    def data(self, index, role):
        if role == Qt.ItemDataRole.DisplayRole:
            value = self._data.iloc[index.row(), index.column()]
            return str(value)
        if role == Qt.ItemDataRole.DecorationRole:
            if index.column() == 1:
                label = QLabel()
                path = "/Users/valentinospariza/Library/CloudStorage/OneDrive-UvA/Repositories/multimedia_analytics/mma_2023_uva_group_1/wordcloud.png"
                image = QtGui.QImage(str(path)) 
                pixmap = QtGui.QPixmap.fromImage(image)
                label.setPixmap(pixmap)
                return label

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
        self.setItemDelegateForColumn(1, ImageDelegate(parent))

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

        self.table.selectionModel().selectionChanged.connect(
            self.on_selectionChanged
        )

        self.table
        self.table.clicked.connect(self.cell_was_clicked)

        layout = QVBoxLayout()

        layout.addWidget(self.table)

        self.label = QLabel("Click in this window")
        layout.addWidget(self.label)
        widget = QWidget()
        widget.setLayout(layout)
        self.setCentralWidget(widget)

    @QtCore.pyqtSlot(QtCore.QItemSelection, QtCore.QItemSelection)
    def on_selectionChanged(self, selected, deselected):
        print("=====Selected=====")
        for ix in selected.indexes():
            print(ix.row(), ix.column())
        print("=====Deselected=====")
        for ix in deselected.indexes():
            print(ix.row(), ix.column())

    def cell_was_clicked(self, item):
        print("Row %d and Column %d was clicked", (item.row(), item.column()))
        print(self.table.model().index(item.row(), item.column()).data())
        if item.column() == 1:
            self.show_new_window(checked=True)
        # self.ID = item.text()        

    def show_new_window(self, checked):
        if hasattr(self, "w") and self.w is not None:
            self.w.close()  # Close window.
            self.w = None  # Discard reference.

        self.w = ImageWindow("/Users/valentinospariza/Library/CloudStorage/OneDrive-UvA/Repositories/multimedia_analytics/mma_2023_uva_group_1/wordcloud.png")
        self.w.show()


    # def current_pos(self):
    #     index = self.table.indexAt(QtWidgets.QApplication.focusWidget().pos())
    #     if index.isValid():
    #         print(index.row(), index.column())

    # def mouseMoveEvent(self, e):
    #     self.label.setText("mouseMoveEvent")

    # def mousePressEvent(self, e):
    #     self.label.setText("mousePressEvent")
    #     self.current_pos()

    # def mouseReleaseEvent(self, e):
    #     self.label.setText("mouseReleaseEvent")

    # def mouseDoubleClickEvent(self, e):
    #     self.label.setText("mouseDoubleClickEvent")


app=QtWidgets.QApplication(sys.argv)
window=MainWindow()
window.show()
app.exec()
import sys
import typing
from PyQt6 import QtCore, QtWidgets
from PyQt6.QtWidgets import QVBoxLayout, QWidget, QAbstractItemView

from table_model import TableModel

from typing import List

class TableView(QtWidgets.QTableView):
    cellClicked = QtCore.pyqtSignal(object, QWidget)
    rowClicked = QtCore.pyqtSignal(object, QWidget)
    cellsSelected = QtCore.pyqtSignal(list, QWidget)

    def __init__(self, model: TableModel, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setModel(model)
        self.setVerticalScrollMode(QAbstractItemView.ScrollMode.ScrollPerPixel)
        self.w = None
        self.selectionModel().selectionChanged.connect(self.__cells_were_selected)
        self.clicked.connect(self.__cell_was_clicked)
        self._selected_entries = set()

    def update_model(self, model: TableModel):
        self.setModel(model)
        self.setVerticalScrollMode(QAbstractItemView.ScrollMode.ScrollPerPixel)
        self.update()

    @QtCore.pyqtSlot(QtCore.QModelIndex)
    def __cell_was_clicked(self, index):
        clicked_cell_value = self.model().get_data(index)
        self.cellClicked.emit(clicked_cell_value, self)
        clicked_row_data = self.model().get_row_data(index.row())
        self.rowClicked.emit(clicked_row_data, self)

    @QtCore.pyqtSlot(QtCore.QItemSelection, QtCore.QItemSelection)
    def __cells_were_selected(self, selected, deselected):
        # =====Selected=====
        for ix in selected.indexes():
            self._selected_entries.add((ix.row(), ix.column()))
        # =====Deselected=====
        for ix in deselected.indexes():
            self._selected_entries.discard((ix.row(), ix.column()))
        self.cellsSelected.emit(list(self._selected_entries),self)

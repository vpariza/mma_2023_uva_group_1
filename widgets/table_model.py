
from PyQt6 import QtCore
from PyQt6.QtCore import Qt
import pandas as pd
from typing import List
from enum import Enum

class TableModel(QtCore.QAbstractTableModel):
    def __init__(self, data:pd.DataFrame):
        super(TableModel, self).__init__()
        self._data = data

    def get_data(self, index):
        return self._data.iloc[index.row(), index.column()]
    
    def get_row_data(self, row:int):
        return self._data.iloc[row, :]
    
    def data(self, index, role):
        """
        Returnes the data specified in the position of the
        index if the role is to display the data otherwise
        if it is a decoration rule and the column is the
        image paths just return the the first of the images
        in each of the listings
        """
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
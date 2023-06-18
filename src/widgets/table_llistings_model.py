
from PyQt6 import QtCore
from PyQt6.QtCore import Qt
import pandas as pd
from typing import List
from enum import Enum

class TableListingsModel(QtCore.QAbstractTableModel):
    class ListingInfoKeys(Enum):
        IMGS_PATH = 'images_paths'
        LISTING_ID = 'funda_identifier'
    def __init__(self, data:pd.DataFrame, imgs_dir_path:str):
        super(TableListingsModel, self).__init__()
        self._data = data
        self._imgs_dir_path = imgs_dir_path

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

    def get_imgs_paths_column(self) -> List[str]:
        return self._data[self.ListingInfoKeys.IMGS_PATH.value].values.tolist()

    def get_imgs_paths(self, row):
        return self._data.iloc[row, self.get_imgs_column()]

    def get_entry_id(self, row):
        return self._data.iloc[row, self.get_entry_ids_column()]

    def get_imgs_column(self) -> int:
        return self._data.columns.get_loc(self.ListingInfoKeys.IMGS_PATH.value)

    def get_entry_ids_column(self) -> int:
        return self._data.columns.get_loc(self.ListingInfoKeys.LISTING_ID.value)

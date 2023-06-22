import matplotlib
matplotlib.use('QtAgg')

from PyQt6 import QtCore

import typing
import pandas as pd

from typing import List, Dict, Set

import numpy as np

class MultiLinePlotModel(QtCore.QObject):
    def __init__(self, data_x:pd.DataFrame|Dict[str, np.array], data_y:pd.DataFrame|Dict[str, np.array], parent: typing.Optional[QtCore.QObject] = None ) -> None:
        super(MultiLinePlotModel, self).__init__(parent)
        self._data_x = data_x if type(data_x) is pd.DataFrame else pd.DataFrame(data_x)
        self._data_y = data_x if type(data_y) is pd.DataFrame else pd.DataFrame(data_y)
    
    def get_headers(self):
        return list(self._data_x.keys())
    
    def get_column_x(self, column_name:str) -> pd.DataFrame:
        return self._data_x[column_name]
    
    def get_column_y(self, column_name:str) -> pd.DataFrame:
        return self._data_y[column_name]

    def get_data(self, cols_names:List[str] | Set[str]=None):
        """
        Get the column data from the column names in a list
        and their corresponding labels.
        """
        data_list_x = list()
        data_list_y = list()
        data_labels = list()
        if cols_names is None:
            cols_names = set(self.get_headers())
        else:
            cols_names = set(cols_names).intersection(set(self.get_headers()))
        for column_name in cols_names:
            data_labels.append(column_name)
            data_list_x.append(self.get_column_x(column_name).values)
            data_list_y.append(self.get_column_y(column_name).values)
        return data_list_x, data_list_y, data_labels

import matplotlib
matplotlib.use('QtAgg')

from PyQt6 import QtCore

import typing
import pandas as pd

from typing import List, Dict, Set

import numpy as np

class MultiBarPlotModel(QtCore.QObject):
    def __init__(self, data_x:pd.DataFrame|Dict[str, np.array], data_y:pd.DataFrame|Dict[str, np.array], feature_importances: dict = None, parent: typing.Optional[QtCore.QObject] = None ) -> None:
        super(MultiBarPlotModel, self).__init__(parent)
        self._data_x = data_x if type(data_x) is pd.DataFrame else pd.DataFrame(data_x)
        self._data_y = data_x if type(data_y) is pd.DataFrame else pd.DataFrame(data_y)
        self._data_feature_importances = feature_importances if type(feature_importances) is pd.DataFrame else pd.DataFrame(feature_importances)
        
        

    def get_headers(self):        
        return list(self._data_feature_importances)
    
    
    def get_column_features(self, column_name:str) -> pd.DataFrame:
        return self._data_feature_importances[column_name]


    def get_data(self, cols_names:List[str] | Set[str]=None):
        """
        Get the column data from the column names in a list
        and their corresponding labels.
        """
        data_labels = list()
        data_list_features = list()
        if cols_names is None:
            cols_names = set(self.get_headers())
        else:
            cols_names = set(cols_names).intersection(set(self.get_headers()))
        for column_name in cols_names:
            data_labels.append(column_name)
            data_list_features.append(self.get_column_features(column_name).values)
            
        return data_labels, self._data_feature_importances

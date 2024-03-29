import matplotlib
matplotlib.use('QtAgg')

from PyQt6 import QtCore

import typing
import pandas as pd
from enum import Enum

from copy import copy 

class MultiHistogramPlotModel(QtCore.QObject):
    class HouseInfoDistKeys(Enum):
        LONGITUDE = 'lon'
        LATITUDE = 'lat'
        # LISTING_ID = 'funda_identifier'
        PRICE = 'price'
        LIVING_AREA = 'living_area'
        BEDROOMS = 'bedrooms'
        ASK_PRICE_PER_SQ_M = 'asking_price_per_m²'
        STATUS = 'status'
        # ACCEPTANCE = 'acceptance'
        BUILDING_TYPE = 'building_type'
        NUM_BATH_ROOMS = 'number_of_bath_rooms'
        LABEL = 'label'
        # HEATING = 'heating'
        GARDEN = 'garden'
        FACILITIES_TYPE = 'type_of_facilities'
        PLOT_SIZE = 'plot_size'
        YEAR_OF_CONSTRUCTION = 'year_of_construction'
        POPULATION_CLOSEST_CITY = 'population_of_closest_city'
        DISTANCE_TO_CENTER = 'distance_to_center'

        @classmethod
        def list_values(cls):
            return list(map(lambda c: c.value, cls))

    def __init__(self, data:pd.DataFrame, parent: typing.Optional[QtCore.QObject] = None ) -> None:
        super(MultiHistogramPlotModel, self).__init__(parent)
        self._data = data
        self._headers = list(set(self.HouseInfoDistKeys.list_values()).union(set(self.get_numric_headers())))

    def get_numric_headers(self):
        numeric_headers = list()
        for column in self._data.columns:
            if pd.to_numeric(self._data[column], errors='coerce').notnull().all():
                numeric_headers.append(column)
        return numeric_headers


    def get_headers(self):
        return copy(self._headers)
    
    def get_column(self, column_name:str) -> pd.DataFrame:
        return self._data[column_name]

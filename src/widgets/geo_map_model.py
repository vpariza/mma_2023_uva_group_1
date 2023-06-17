

from PyQt6.QtCore import QObject

from enum import Enum
import typing
from typing import List
import pandas as pd

from shapely.geometry import Point
from shapely.geometry.polygon import Polygon

import numpy as np

class QGeoMapModel(QObject):
    class HouseInfoKeys(Enum):
        LONGITUDE = 'lon'
        LATITUDE = 'lat'
        POINT_COORDS = 'coords'
        LISTING_ID = 'funda_identifier'

    def __init__(self, data:pd.DataFrame, summary_keys:str=None, parent: typing.Optional['QObject'] = None ) -> None:
        super(QObject, self).__init__(parent)
        self._data = data
        self._summary_keys = summary_keys

    def __get_location_points(self):
        """
        Converts all the coordinates (latitude and longitude entries) to points so that it can process
        them easily.
        """
        coords = self.data[[self.HouseInfoKeys.LATITUDE.value, self.HouseInfoKeys.LONGITUDE.value]].to_numpy()
        return np.array([Point(coord) for coord in coords])

    def get_selected_entries_from_area(self, polygon_coords:List[List[float]]) -> list:
        """
        Given a set of coordinates defining a polygon return all the points in the data residing
        in the polygon.
        """
        polygon = Polygon(polygon_coords)
        idces = np.where(polygon.contains(self._data[self.HouseInfoKeys.POINT_COORDS.value].to_numpy()))[0]
        return self._data.iloc[idces][self.HouseInfoKeys.LISTING_ID.value].values.tolist()
    
    def get_selected_entry(self, coords:List[float]) -> object:
        """
        Given a set of coordinates defining a polygon return all the points in the data residing
        in the polygon.
        """
        index = np.where(np.logical_and(self._data[self.HouseInfoKeys.LATITUDE.value]==coords[0],self._data[self.HouseInfoKeys.LONGITUDE.value]==coords[1]))[0]
        return list(self._data.iloc[index][self.HouseInfoKeys.LISTING_ID.value])[0]
    
    def get_entries_summary(self):
        if self._summary_keys is None:
            summary_keys = set(self._data.keys())
        else:
            summary_keys = set(self._summary_keys)
        summary_keys = summary_keys.difference(set([self.HouseInfoKeys.LISTING_ID.value, self.HouseInfoKeys.LATITUDE.value, self.HouseInfoKeys.LONGITUDE.value]))
        info = dict()
        html_summaries = list()
        info['ids'] = self._data[self.HouseInfoKeys.LISTING_ID.value].to_numpy()
        info['coords'] = self._data[[self.HouseInfoKeys.LATITUDE.value, self.HouseInfoKeys.LONGITUDE.value]].to_numpy()
        for i, row_dict in enumerate(self._data.to_dict(orient="records")):
            html_summaries.append(self._get_html_entry(info['ids'][i], info['coords'][i], row_dict))
        info['html_summaries'] = html_summaries
        return info

    def _get_html_entry(self, id, coord: List[float] ,entry:dict):
        items = ["<li><b>{0}</b>: {1}</li>".format(key, value) for key, value in entry.items() if type(value) == str or type(value) == int or type(value) == float]
        items_str = '\n'.join(items)
        html=f"""
            <h3> 'Listing {id} at {coord[0], coord[1]}'</h3>
            <p>Listing Details:</p>
            <ul>
                {items_str}
            </ul>
            </p>
            """
        return html

    def get_centroid_point(self):
        return np.mean((self._data[self.HouseInfoKeys.LATITUDE.value], self._data[self.HouseInfoKeys.LONGITUDE.value]), axis=1)

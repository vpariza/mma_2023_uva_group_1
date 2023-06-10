import os, sys
import numpy as np
import pandas as pd

class FundaData():
    def __init__(self, **kwargs) -> None:
        super().__init__()
        self.base_dataset_path = kwargs.get('base_dataset_path', 'data/funda_images')
        self.data_path = kwargs.get('data_path', 'data/funda_data.pkl')
        self.images_path = kwargs.get('images_path', 'data/funda_image_features.h5')
        self.image_features = None
        self.data = pd.read_pickle(self.data_path)
        self.current_data = self.data


    def set_range_filter(self, field, min_value=None, max_value=None):
        if min_value is None:
            min_value = self.data[field].min()
        if max_value is None:
            max_value = self.data[field].max()
        self.current_data = self.current_data[min_value <= self.current_data[field] <= max_value]

    def set_value_filter(self, field, value):
        self.current_data = self.current_data[self.current_data[field] == value]

    def reset_filter(self):
        self.current_data = self.data
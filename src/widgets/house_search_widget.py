import sys
sys.path.append('/Users/valentinospariza/Library/CloudStorage/OneDrive-UvA/Repositories/multimedia_analytics/mma_2023_uva_group_1/')

import matplotlib
matplotlib.use('QtAgg')

from PyQt6 import QtCore, QtWidgets

from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QVBoxLayout, QPushButton
)
import typing
from typing import List, Dict


import pandas as pd
from src.widgets.filter_widget import FilterWidget
from src.widgets.query_widget import QueryWidget
from src.widgets.table_listings_model import TableListingsModel
from src.widgets.table_listings_view import TableListingsView
from src.widgets.dialog_widgets import BasicDialog
from src.utils.filtering_utils import apply_filters
from src.widgets.geo_map_model import QGeoMapModel
from src.widgets.geo_map_widget import GeoMapWidget

import numpy as np

class HouseSearchWidget(QWidget):
 
    txtQuerySubmitted = QtCore.pyqtSignal(str, QWidget)
    updatedShowedData = QtCore.pyqtSignal(pd.DataFrame, QWidget)


    def __init__(self, data:pd.DataFrame, config, widgets:Dict[str, QWidget]={}, parent: typing.Optional['QWidget']=None, *args, **kwargs):
        super(HouseSearchWidget, self).__init__(parent=parent, *args, **kwargs)
        self.widgets = widgets
        # Use pre-existing widgets if given
        self._query_widget = widgets.get('query_widget')
        self._filter_widget = widgets.get('filter_widget')
        self._table_listings_model = widgets.get('table_listings_model')
        self._table_listings_widget = widgets.get('table_listings_widget')
        # Save the given data
        # These are the data showed at each moment
        self._data_show = data.copy()
        # These are the original data
        self._data = data
        self._config = config
        self._images_dir_path = config['main']['images_dir_path']
        self.setLayout(self.create_layout())

    def get_dict_widgets(self):
        return {
            'query_widget': self._query_widget,
            'filter_widget': self._filter_widget,
            'table_listings_model': self._table_listings_model,
            'table_listings_widget': self._table_listings_widget
        }

    ###### Creating the general Layout ######
    def create_layout(self):
        main_layout = QHBoxLayout()
        main_layout.addLayout(self._left_layout())
        main_layout.addLayout(self._right_layout())
        return main_layout

    def _right_layout(self):
        right_layout = QVBoxLayout()
        ####### Add the Query Widget
        if self._query_widget is None:      
            self._query_widget = QueryWidget()
            self._query_widget.resize(600, 200) 
        # self._query_widget.querySubmitted.connect(self.on_query_submitted)
        right_layout.addWidget(self._query_widget)
        ####### Add the Filtering Widget
        # Define filters 
        combofilters_ = {'status': ['Available', 'Under option'],                                                  
                        'bedrooms': [str(i) for i in np.arange(0, 5 + 1)]}
        minmaxfilters_ = ['price', 'living_area']
        # Filter widgets tab 1 with layout option 1
        if self._filter_widget is None:
            self._filter_widget = FilterWidget(minmaxfilters = minmaxfilters_, combofilters = combofilters_, config = '1')
        self._filter_widget.searchbutton.filtersApplied.connect(self._on_filters_applied)
        right_layout.addWidget(self._filter_widget)
        ####### Add the Table Listings Widget
        if self._table_listings_widget is None:
            self._table_listings_model = TableListingsModel(self._data_show, self._images_dir_path)
            self._table_listings_widget = TableListingsView(self._table_listings_model)
        self._table_listings_widget.entryDoubleClicked.connect(self.on_table_entry_double_clicked)
        right_layout.addWidget(self._table_listings_widget)
        self._clear_all_button = QPushButton('Clear All', self)
        self._clear_all_button.clicked.connect(self.clear_all_button_clicked)
        right_layout.addWidget(self._clear_all_button)
        return right_layout

    def _left_layout(self):
        right_layout = QVBoxLayout()      
        ####### Add the Geo Map Model Widget   
        # Define the Geo Map Widget
        self._geo_map_model = QGeoMapModel(self._data_show )
        self._geo_map_widget = GeoMapWidget(self._geo_map_model)
        self._geo_map_widget.entryClicked.connect(self.on_map_entry_clicked)
        self._geo_map_widget.entriesSelected.connect(self.on_map_entries_selected)
        right_layout.addWidget(self._geo_map_widget)
        return right_layout

    ###### Update Methods ######
    def update_data_show(self, data:pd.DataFrame):
        self._data_show = data
        self.update()

    def update_original_data(self, data:pd.DataFrame):
        self._data = data.copy()
        self._data_show = data.copy()
        self.update()

    def update(self):
        self._table_listings_model = TableListingsModel(self._data_show, self._images_dir_path)
        self._table_listings_widget.update_model(self._table_listings_model)

        self._geo_map_model = QGeoMapModel(self._data_show )
        self._geo_map_widget.update_geo_map_model(self._geo_map_model)

    ###### Slot of Handling Signals ######
    @QtCore.pyqtSlot(object, QWidget)
    def _on_filters_applied(self, filters, source):
        filtered_df = apply_filters(self._data, filters) 
        if len(filtered_df.index) > 0:
            self._data_show = filtered_df
            self.updatedShowedData.emit(self._data_show, self)
            self.update()
        else:
            BasicDialog(window_title='No Results found!', message='There are no entries matching your filtering!').exec()
    
    @QtCore.pyqtSlot(str, QWidget)
    def _on_txt_query_submitted(self, txt_query, source):
        self.txtQuerySubmitted.emit(txt_query, self)

    @QtCore.pyqtSlot(list, QWidget)
    def on_map_entries_selected(self, entries, source):
        if len(entries) > 0:
            self._data_show = self._data_show[self._data_show['funda_identifier'].isin(entries)]
            self.updatedShowedData.emit(self._data_show, self)
            self.update()

    @QtCore.pyqtSlot(object, QWidget)
    def on_map_entry_clicked(self, entry, source):
        # TODO
        print('Map Entry Clicked', entry)
        pass

    @QtCore.pyqtSlot(object, QWidget)
    def on_table_entry_double_clicked(self, entry, source):
        self._geo_map_widget.focus_on_entry(entry)
        self.update()

    @QtCore.pyqtSlot()
    def clear_all_button_clicked(self):
        self._data_show = self._data.copy()
        self._geo_map_widget.focus_on_coord(None)
        self.updatedShowedData.emit(self._data_show, self)
        self.update()


import sys
import numpy as np
from PyQt6.QtWidgets import QApplication, QMainWindow, QVBoxLayout, QHBoxLayout, QWidget, QTabWidget, QSizePolicy, QLabel, QGridLayout

if __name__ == '__main__':
    class MainWindow(QMainWindow):
        def __init__(self):
            super().__init__()
            from src.utils.file_utils import load_from_pickle
            # data = load_from_pickle('data.pkl')
            data = pd.read_csv('./dataloading/data/dataset.csv')
            training_features = {
                'bedrooms': int,
                'building_type': "category",
                'living_area': int,
                'plot_size': int,
                'year_of_construction': int,
                'lat': float,
                'lon': float,
                'label': "category",
            }
            from src.utils.preprocessing import Preprocessing
            preprocessing = Preprocessing()
            config, tags, points, img_paths, df, images_dir_path = preprocessing.load_data()
            df['umap_x'] = points[:,0]
            df['umap_y'] = points[:,1] 
            # print(df[['umap_x','umap_y']])
            w = HouseSearchWidget(data=df, config=config, parent=self)
            self.setWindowState(QtCore.Qt.WindowState.WindowMaximized)
            
            self.setCentralWidget(w)
            # w.resize(800, 600)

    app = QApplication(sys.argv)
    main_window = MainWindow()
    main_window.show()
    sys.exit(app.exec())
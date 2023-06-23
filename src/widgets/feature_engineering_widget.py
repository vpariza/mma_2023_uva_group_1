import sys
sys.path.append('/Users/valentinospariza/Library/CloudStorage/OneDrive-UvA/Repositories/multimedia_analytics/mma_2023_uva_group_1/')

import matplotlib
matplotlib.use('QtAgg')

from PyQt6 import QtCore, QtWidgets

from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QVBoxLayout, QPushButton
)
import typing
from src.widgets.list_options_widget import ListOptionsWidget
from typing import List, Dict

from src.widgets.multi_hist_plot_model import MultiHistogramPlotModel
from src.widgets.multi_hist_plot_widget import MultiHistogramPlotWidget

import pandas as pd
from src.widgets.filter_widget import FilterWidget
from src.widgets.query_widget import QueryWidget
from src.widgets.plot_widget import ScatterPlotWidget, SelectClusterWidget

from src.widgets.multi_hist_plot_model import MultiHistogramPlotModel
from src.widgets.multi_hist_plot_widget import MultiHistogramPlotWidget
from src.widgets.table_listings_model import TableListingsModel
from src.widgets.table_listings_view import TableListingsView
from PyQt6.QtWidgets import QWidget, QListWidget
from src.widgets.dialog_widgets import BasicDialog
from src.utils.filtering_utils import apply_filters
from src.widgets.model_train_widget import ModelTrainWidget

from sklearn.preprocessing import minmax_scale
from sklearn.preprocessing import scale
from sklearn.preprocessing import robust_scale
from sklearn.preprocessing import maxabs_scale

class FeatureEngineeringWidget(QWidget):
 
    txtQuerySubmitted = QtCore.pyqtSignal(str, QWidget)
    modelToTrain = QtCore.pyqtSignal(str, pd.DataFrame, QWidget)
    modelDeleted = QtCore.pyqtSignal(str, QWidget)

    def __init__(self, data:pd.DataFrame, training_features:List[str], config, widgets:Dict[str, QWidget]={}, parent: typing.Optional['QWidget']=None, *args, **kwargs):
        super(FeatureEngineeringWidget, self).__init__(parent=parent, *args, **kwargs)
        self.widgets = widgets
        # Use pre-existing widgets if given
        self._query_widget = widgets.get('query_widget ')
        self._filter_widget = widgets.get('filter_widget')
        self._table_listings_model = widgets.get('table_listings_model')
        self._table_listings_widget = widgets.get('table_listings_widget')
        # Save the given data
        # These are the data showed at each moment
        self._data_show = data.copy()
        # These are the original data
        self._training_features = training_features
        self._selected_training_features = list()
        self._data = data
        self._config = config
        self._images_dir_path = config['main']['images_dir_path']
        self.setLayout(self.create_layout())

    def get_dict_widgets(self):
        return {
            'query_widget': self._query_widget,
            'filter_widget': self._filter_widget,
            'table_listings_model': self._table_listings_model,
            'table_listings_widget': self._table_listings_widget,
            **self._model_train_widget.get_dict_widgets()
        }

    ###### Creating the general Layout ######
    def create_layout(self):
        main_layout = QVBoxLayout()
        main_layout.addLayout(self._top_layout())
        main_layout.addLayout(self._bottom_layout())
        return main_layout

    def _top_layout(self):
        top_layout = QHBoxLayout()
        top_layout.addLayout(self._left_layout())
        top_layout.addLayout(self._right_layout())
        return top_layout

    def _left_layout(self):
        left_layout = QVBoxLayout()
        ####### Add the Query Widget
        if self._query_widget is None:      
            self._query_widget = QueryWidget()
            self._query_widget.resize(600, 200) 
        # self._query_widget.querySubmitted.connect(self.on_query_submitted)
        left_layout.addWidget(self._query_widget)
        ####### Add the Filtering Widget
        # Define filters 
        combofilters_ = {'status': ['Available', 'Under option'],                                                  
                        'bedrooms': [str(i) for i in np.arange(0, 5 + 1)]}
        minmaxfilters_ = ['price', 'living_area']
        # Filter widgets tab 1 with layout option 1
        if self._filter_widget is None:
            self._filter_widget = FilterWidget(minmaxfilters = minmaxfilters_, combofilters = combofilters_, config = '1')
        self._filter_widget.searchbutton.filtersApplied.connect(self._on_filters_applied)
        left_layout.addWidget(self._filter_widget)
        ####### Add the Table Listings Widget
        if self._table_listings_widget is None:
            self._table_listings_model = TableListingsModel(self._data_show, self._images_dir_path)
            self._table_listings_widget = TableListingsView(self._table_listings_model)
        # self.table_listings_widget.entryDoubleClicked.connect(self.on_table_entry_double_clicked)
        left_layout.addWidget(self._table_listings_widget)
        return left_layout

    def _right_layout(self):
        right_layout = QVBoxLayout()      
        ####### Add the Multi Histogram Widget   
        self._multi_hist_p_model = MultiHistogramPlotModel(self._data_show, self)
        options_fn = {
            'minmax_scale': minmax_scale,
            'standard_scale': scale,
            'robust_scale': robust_scale,
            'maxabs_scale': maxabs_scale
        }
        self._multi_hist_p_widget = MultiHistogramPlotWidget(self._multi_hist_p_model, options=list(options_fn.keys()), options_fn=options_fn, parent=self)
        right_layout.addWidget(self._multi_hist_p_widget)
        ####### Add the Clustering Widget 
        clustering_layout = QHBoxLayout()  
        self._select_scatter_plot = SelectClusterWidget()
        clustering_layout.addWidget(self._select_scatter_plot)
        self._scatter_plot_widget = ScatterPlotWidget(self._umap_points, self._config)
        clustering_layout.addWidget(self._scatter_plot_widget)
        right_layout.addLayout(clustering_layout)
        return right_layout

    def _bottom_layout(self):
        bottom_layout = QHBoxLayout()      
        self._model_train_widget = ModelTrainWidget(self._training_features, base_model_name='model_{}', widgets=self.widgets, parent=self)
        self._model_train_widget.modelToTrain.connect(self._on_model_to_train)
        self._model_train_widget.modelDeleted.connect(self._on_deleted_model)
        bottom_layout.addWidget(self._model_train_widget)
        return bottom_layout

    ###### Properties of the object ######
    @property
    def _umap_points(self):
        return self._data_show[['umap_x','umap_y']].values

    @property
    def _umap_points_x(self):
        return self._data_show['umap_x'].values

    @property
    def _umap_points_y(self):
        return self._data_show['umap_y'].values
    
    @property
    def _training_data(self):
        """
        Return the selected features we can train a model on.
        """
        return self._data_show[self._selected_training_features]

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

        self._multi_hist_p_model = MultiHistogramPlotModel(self._data_show, self)
        self._multi_hist_p_widget.update_model(self._multi_hist_p_model)
        
        umap_points = self._umap_points
        self._scatter_plot_widget.update_scatterplot(self._umap_points_x, self._umap_points_y)

    ###### Slot of Handling Signals ######
    @QtCore.pyqtSlot(object, QWidget)
    def _on_filters_applied(self, filters, source):
        filtered_df = apply_filters(self._data, filters) 
        if len(filtered_df.index) > 0:
            self._data_show = filtered_df
            self.update()
        else:
            BasicDialog(window_title='No Results found!', message='There are no entries matching your filtering!').exec()
    
    @QtCore.pyqtSlot(object, QWidget)
    def _on_txt_query_submitted(self, txt_query, source):
        self.txtQuerySubmitted.emit(txt_query, self)

    @QtCore.pyqtSlot(str, list, QWidget)
    def _on_model_to_train(self, model_name, features, source):
        self.modelToTrain.emit(model_name, self._data_show[features].copy(), self)

    @QtCore.pyqtSlot(str, QWidget)
    def _on_deleted_model(self, model_name, source):
        self.modelDeleted.emit(model_name,self)


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
            w = FeatureEngineeringWidget(data=df, training_features=list(training_features.keys()), config=config, parent=self)
            self.setWindowState(QtCore.Qt.WindowState.WindowMaximized)
            
            self.setCentralWidget(w)
            # w.resize(800, 600)

    app = QApplication(sys.argv)
    main_window = MainWindow()
    main_window.show()
    sys.exit(app.exec())
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
from src.widgets.elements.custom_blocks import TitleWidget, ButtonWidget, CheckBoxWidget


from sklearn.preprocessing import minmax_scale
from sklearn.preprocessing import scale
from sklearn.preprocessing import robust_scale
from sklearn.preprocessing import maxabs_scale

class FeatureEngineeringWidget(QWidget):
 
    txtQuerySubmitted = QtCore.pyqtSignal(str, QWidget)
    modelToTrain = QtCore.pyqtSignal(str, pd.DataFrame, QWidget)
    modelDeleted = QtCore.pyqtSignal(str, QWidget)
    updatedShowedData = QtCore.pyqtSignal(pd.DataFrame, QWidget)

    def __init__(self, data:pd.DataFrame, img_features, training_features:List[str], config, widgets:Dict[str, QWidget]={}, parent: typing.Optional['QWidget']=None, *args, **kwargs):
        super(FeatureEngineeringWidget, self).__init__(parent=parent, *args, **kwargs)
        self.widgets = widgets
        self.img_features = img_features
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
        self._data = data
        self._config = config
        self._images_dir_path = config['main']['images_dir_path']
        self.setLayout(self.create_layout())
        self._scatter_x = self._data_show['umap_x']
        self._scatter_y = self._data_show['umap_y']
        self.kmeans = False
        self.k = 1


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
        self.columnwidth = 600
        self.buttonwidth = 400
        self.buttonheight = 75
        self.titlewidth = 25
        self.main_layout = QGridLayout()
        v11, v21, v31 = self._col1_layout()
        v12, v22, v32 = self._col2_layout()
        v13, v23, v33 = self._col3_layout()

        self.main_layout.addWidget(v11, 0, 0)
        self.main_layout.addWidget(v21, 1, 0)
        self.main_layout.addWidget(v31, 2, 0, alignment = QtCore.Qt.AlignmentFlag.AlignCenter)
        
        self.main_layout.addWidget(v12, 0, 1)
        self.main_layout.addWidget(v22, 1, 1)
        self.main_layout.addWidget(v32, 2, 1, alignment = QtCore.Qt.AlignmentFlag.AlignCenter)
        self.main_layout.addWidget(v13, 0, 2)
        self.main_layout.addWidget(v23, 1, 2)
        
        return self.main_layout
    
    def _col3_layout(self):
        ####### Add the Table Listings Widget
        if self._table_listings_widget is None:
            self._table_listings_model = TableListingsModel(self._data_show, self._images_dir_path)
            self._table_listings_widget = TableListingsView(self._table_listings_model)
            self._table_listings_widget.setFixedWidth(575)
            self._table_listings_widget.setFixedHeight(400)
            
        v13 = self.add_block([self._table_listings_widget], QVBoxLayout(), size = [self.columnwidth], alignment_= QtCore.Qt.AlignmentFlag.AlignCenter)
        v13 = self.add_block([TitleWidget('Current datapoint selection:', size = [self.columnwidth, self.titlewidth]).title, v13], QVBoxLayout(), size = [self.columnwidth])
        
        self._model_train_widget = ModelTrainWidget(self._training_features, base_model_name='model_{}', widgets=self.widgets, parent=self)
        self._model_train_widget.modelToTrain.connect(self._on_model_to_train)
        self._model_train_widget.modelDeleted.connect(self._on_deleted_model)
        v23 = self.add_block([self._model_train_widget], QVBoxLayout(), size = [self.columnwidth])
        
        
        return v13, v23, 0

    def _col2_layout(self):
        ####### Add the Query Widget
        if self._query_widget is None:      
            self._query_widget = QueryWidget()
            self._query_widget.resize(600, 200) 
        self._query_widget.querySubmitted.connect(self._on_txt_query_submitted)

        self._select_scatter_plot = SelectClusterWidget()
        self._select_scatter_plot.searchbutton.filtersApplied.connect(self._on_scatterconfig_applied)
        
        
        v12 = self.add_block([TitleWidget('Query driven features:', size = [self.columnwidth, self.titlewidth]).title, self._query_widget, self._select_scatter_plot], QVBoxLayout(), size = [self.columnwidth])
        ####### Add the Clustering Widget 
        self._scatter_plot_widget = ScatterPlotWidget(self._umap_points, self._config)
        v22 = self.add_block([self._scatter_plot_widget], QHBoxLayout())
        
        self._store_qfeat_button = ButtonWidget('Store\nFeature', size = [self.buttonwidth, self.buttonheight]).button
        v32 = self._store_qfeat_button

        return v12, v22, v32


    def _col1_layout(self):
        # Filter widgets tab 1 with layout option 1
        # Define filters 
        combofilters_ = ['kind_of_house', 'building_type','number_of_rooms', 'bedrooms'] # combofilters -> {Element Title: [displayed textprompt]}
        minmaxfilters_ = ['price', 'living_area', 'year_of_construction']
        placeholdertext_ = ['Ex.: 100000', 'Ex.: 50', 'Ex.: 1990']

        if self._filter_widget is None:
            self._filter_widget = FilterWidget(self._data_show, minmaxfilters = minmaxfilters_, combofilters = combofilters_, placeholdertext = placeholdertext_, config = '1')
        self._filter_widget.searchbutton.filtersApplied.connect(self._on_filters_applied)
        
        v11 = self.add_block([TitleWidget('Data driven features:', size = [self.columnwidth, self.titlewidth]).title, self._filter_widget], QVBoxLayout(), size = [self.columnwidth])
        
        ####### Add the Multi Histogram Widget   
        self._multi_hist_p_model = MultiHistogramPlotModel(self._data_show, self)
        options_fn = {
            'minmax_scale': minmax_scale,
            'standard_scale': scale,
            'robust_scale': robust_scale,
            'maxabs_scale': maxabs_scale
        }
        self._multi_hist_p_widget = MultiHistogramPlotWidget(self._multi_hist_p_model, options=list(options_fn.keys()), options_fn=options_fn, parent=self)

        v21 = self.add_block([self._multi_hist_p_widget], QHBoxLayout())

        self.store_datfeat_button = ButtonWidget('Store\nFeature', size = [self.buttonwidth, self.buttonheight]).button
        v31 = self.store_datfeat_button
        
        return v11, v21, v31
    

    def add_block(self, widgetlist = [], block_type = QVBoxLayout(), alignment_ = QtCore.Qt.AlignmentFlag.AlignTop | QtCore.Qt.AlignmentFlag.AlignLeft, size = None):
        widget = QWidget()
        layout = block_type
        for wid in widgetlist:
            layout.addWidget(wid, alignment=alignment_)
        widget.setLayout(layout)
        if size is not None:
            widget.setFixedWidth(size[0])
        widget.setStyleSheet("background-color: #f5f5f5;")
        widget.setFixedWidth(self.columnwidth)
        return widget



    @property
    def model_names(self):
        return self._model_train_widget.model_names


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
    def _tsne_points(self):
        return self._data_show[['tsne_x','tsne_y']].values
    
    @property
    def _tsne_points_x(self):
        return self._data_show['tsne_x'].values
    
    @property
    def _tsne_points_y(self):
        return self._data_show['tsne_y'].values

    @property
    def _training_data(self):
        """
        Return the selected features we can train a model on.
        """
        return self._data_show[self._model_train_widget.selected_features]

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
        
        self._scatter_plot_widget.update_scatterplot(self._scatter_x, self._scatter_y, self.kmeans, k = self.k)

    def add_new_features(self, feature_names:list):
        self._model_train_widget.add_features(feature_names)

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
    
    @QtCore.pyqtSlot(object, QWidget)
    def _on_scatterconfig_applied(self, filters, source):      
        """  Update according to tsne or umap select
        """  
        if self._select_scatter_plot.dim_reduct_method.Filter.currentText() == 'umap':
            self._scatter_x = self._umap_points_x
            self._scatter_y = self._umap_points_y
        elif self._select_scatter_plot.dim_reduct_method.Filter.currentText() == 't-sne':
            self._scatter_x = self._tsne_points_x
            self._scatter_y = self._tsne_points_y
        
        if self._select_scatter_plot.clustering_method.Filter.currentText() == 'k-means':
            self.kmeans = True
            try:
                self.k = eval(self._select_scatter_plot.n_clusters_method.Filter.currentText())
            except (NameError, SyntaxError):
                # TODO: Setup request for k
                pass
        elif self._select_scatter_plot.clustering_method.Filter.currentText() == 'None':
            self.kmeans = False


            

        self.update()
        
        


    @QtCore.pyqtSlot(str, QWidget)
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
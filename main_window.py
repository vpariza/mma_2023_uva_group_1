import sys
sys.path.append('/Users/valentinospariza/Library/CloudStorage/OneDrive-UvA/Repositories/multimedia_analytics/mma_2023_uva_group_1/')

import sys
from PyQt6 import QtCore
from PyQt6.QtWidgets import QApplication, QMainWindow,QWidget, QTabWidget
from PyQt6.QtWidgets import QWidget
import pandas as pd
import numpy as np

from src.utils.preprocessing import Preprocessing 
from src.widgets.feature_engineering_widget import FeatureEngineeringWidget
from src.widgets.house_search_widget import HouseSearchWidget
from src.widgets.model_comparison_widget import ModelComparisonWidget

from src.models.price_predictor import PricePredictor

import pandas as pd

class MainWindow(QMainWindow):

    def __init__(self):
        super().__init__()

        #load data using the config file 'config.ini'
        preprocessing = Preprocessing()
        self.models_table_data = None
        self.p_data_x = {}
        self.p_data_y = {}
        # TODO: Return sirectory of Listings Images
        self.config, self.tags, self.points, self.img_paths, self.df, self.images_dir_path = preprocessing.load_data()

        ####### Load Data
        # Define the training features to use for the
        # allowing the user to train the model uppon
        self._training_features = {
            'bedrooms': int,
            'building_type': "category",
            'living_area': int,
            'plot_size': int,
            'year_of_construction': int,
            'lat': float,
            'lon': float,
            'label': "category",
        }
        for feature, dtype in self._training_features.items():
            self.df[feature] = self.df[feature].astype(dtype)
        self._preprocessing = Preprocessing()
        config, tags, points, img_paths, df, images_dir_path = self._preprocessing.load_data()
        # TODO: Preprocessing should include exactly the following columns for
        # accessing the umap features
        df['umap_x'] = points[:,0]
        df['umap_y'] = points[:,1] 

        self._data = self.df
        self._config = config
        self.setCentralWidget(self.create_central_widget())
        self.setWindowState(QtCore.Qt.WindowState.WindowMaximized)
        
    def create_central_widget(self):
        ####### Defining Tab 2
        # Define the Second Tab
        self._tab2_w = FeatureEngineeringWidget(data=self._data, 
                                                training_features=list(self._training_features.keys()), 
                                                config=self._config, widgets={}, parent=self)
        self._tab2_w.updatedShowedData.connect(self.on_updated_showed_data_tab_2)
        self._tab2_w.txtQuerySubmitted.connect(self.on_query_submitted)
        self._tab2_w.modelToTrain.connect(self.on_train_model)
        ####### Defining Tab 1
        self._tab1_w = HouseSearchWidget(data=self._data, config=self._config, widgets={}, parent=self)
        self._tab1_w.updatedShowedData.connect(self.on_updated_showed_data_tab_1)
        self._tab1_w.txtQuerySubmitted.connect(self.on_query_submitted)
        # ####### Defining Tab 3
        self._tab3_w = ModelComparisonWidget([],
                                              pd.DataFrame([]),
                                              p_data_x=pd.DataFrame([]),
                                              p_data_y=pd.DataFrame([]),
                                              widgets={})

        # Defien the Tabs widget 
        tabwidget = QTabWidget()
        ## Tab 1
        tabwidget.addTab(self._tab1_w, "House Search")
        ## Tab 2
        tabwidget.addTab(self._tab2_w, "Feature Engineering")
        # Tab 3
        tabwidget.addTab(self._tab3_w, "Compare Models")
        return tabwidget

    ###### HANDLING SINGALS FROM CHILD WIDGETS - SLOTS #######
    @QtCore.pyqtSlot(str, QWidget)
    def on_query_submitted(self, query, source):
        """
        query is a string and you must filter the 
        """
        # BEGIN: TODO: Insert your code for finding the similarity between the entries
        # (1) the self._data contains the dataset
        self._data
        print('Submitted Query',query)
        # (2) put your results in the self._show_data
        data = None
        # END: 
        self._tab1_w.update_original_data(data)
        self._tab2_w.update_original_data(data)
        
    @QtCore.pyqtSlot(pd.DataFrame, QWidget)
    def on_updated_showed_data_tab_1(self, show_data, source):
        self._tab2_w.update_data_show(show_data)

    @QtCore.pyqtSlot(pd.DataFrame, QWidget)
    def on_updated_showed_data_tab_2(self, show_data, source):
        self._tab1_w.update_data_show(show_data)

    @QtCore.pyqtSlot(str, pd.DataFrame, QWidget)
    def on_train_model(self, model_name, selected_data:pd.DataFrame, source):
        # BEGIN: TODO: Insert your code for training a model
        model = PricePredictor()

        # preprocess data
        model_df = self._data.loc[selected_data.index].copy()
        model_df = model.preprocess(model_df, selected_data.columns)
        train_X, test_X, val_X, train_y, test_y, val_y = model.split_data(model_df)
        # label which rows where in which split
        self._data.loc[train_X.index, "split"] = "train"
        self._data.loc[test_X.index, "split"] = "test"
        self._data.loc[val_X.index, "split"] = "val"

        # train model
        model.fit(train_X, train_y, val_X, val_y)

        # test and evaluate model
        pred_y, scores = model.predict(test_X, test_y)
        self._data.loc[test_X.index, f"{model_name}_prediction"] = pred_y
        val_scores = model.get_learning_curves()
        self.p_data_x[model_name] = np.arange(1, len(val_scores)+1).astype(float)
        self.p_data_y[model_name] = val_scores
        # make all values for all keys in dict the same length
        max_len = max([len(v) for v in self.p_data_y.values()])
        print(self.p_data_y)
        for k, v in self.p_data_y.items():
            self.p_data_x[k] = np.pad(self.p_data_x[k], (0, max_len-len(v)), mode='constant', constant_values=np.nan)
            self.p_data_y[k] = np.pad(v, (0, max_len-len(v)), mode='constant', constant_values=np.nan)
        print(self.p_data_y)

        #TODO update respective widget with feature importances
        feature_importances = model.get_feature_importances()

        # save model to disk
        model.save_model(model_name)

        # update widgets with results
        eval_function_name_mapping = {
            "mean_absolute_error": "Mean Absolute Error [%]",
            "mean_absolute_percentage_error": "Mean Absolute Percentage Error [%]",
            "r2_score": "R2 Score [%]",
        }
        scores = {eval_function_name_mapping.get(k, k): v for k, v in scores.items()}
        scores = {k: round(v, 2) for k, v in scores.items()}
        if self.models_table_data is None:
            self.models_table_data = pd.DataFrame(scores, index=[model_name])
        else:
            self.models_table_data.loc[model_name] = scores
        # Update the model names in the third tab
        self._tab3_w.update_model_names(self._tab2_w.model_names)
        # Update the table data in the third tab
        self._tab3_w.update_model_table_data(self.models_table_data)
        # Update the plot data in the third tab
        self._tab3_w.update_plot_data(self.p_data_x, self.p_data_y)
        #TODO update tabs because columns have changed


if __name__ == '__main__':
    app = QApplication(sys.argv)
    main_window = MainWindow()
    main_window.show()
    sys.exit(app.exec())
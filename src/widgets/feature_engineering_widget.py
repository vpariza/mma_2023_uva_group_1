

import matplotlib
matplotlib.use('QtAgg')

from PyQt6 import QtCore, QtWidgets

from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QVBoxLayout, QPushButton
)
import typing
from list_options_widget import ListOptionsWidget
from typing import List

from src.widgets.table_model import TableModel
from src.widgets.table_view import TableView
from src.widgets.multi_hist_plot_model import MultiHistogramPlotModel
from src.widgets.multi_hist_plot_widget import MultiHistogramPlotWidget

import pandas as pd

class FeatureEngineeringWidget(QWidget):
 
    def __init__(self, parent: typing.Optional['QWidget']=None, *args, **kwargs):
        super(FeatureEngineeringWidget, self).__init__(parent=parent, *args, **kwargs)
        main_layout = QHBoxLayout()
        # Left side layout

        ########## TOP - LEFT SIDE ########## 
        # Add a Title for Selection of models list
        left_layout = QVBoxLayout()
        self._list_options_w_title = QLabel()
        self._list_options_w_title.setText('Free-text driven features')
        left_layout.addWidget(self._list_options_w_title)
        # Add the Selection of models list
        self._list_options_w = ListOptionsWidget(model_names, parent=self)
        self._list_options_w.optionsSelected.connect(self.models_selected)
        left_layout.addWidget(self._list_options_w)
        self._m_line_p_m = MultiLinePlotModel(data_x = p_data_x if p_data_x is not None else pd.DataFrame([]), 
                                              data_y = p_data_y if p_data_y is not None else pd.DataFrame([]),
                                              parent=self)
        self._m_line_p_w = MultiLinePlotWidget(self._m_line_p_m, 
                                               plot_configs={'title': 'Model Performance Visualization', 'xlabel': 'Epochs', 'ylabel': 'Accuracy'},
                                               parent=self)
        left_layout.addWidget(self._m_line_p_w)
        # Right side layout
        right_layout = QVBoxLayout()
        # Add a Title for the Performance of Models Table
        self._models_table_model_title = QLabel()
        self._models_table_model_title.setText('Model Performance Measures')
        right_layout.addWidget(self._models_table_model_title)
        # Add the Table
        self._models_table_model = TableModel(models_table_data if models_table_data is not None else pd.DataFrame([]))
        self._models_table_model_full = self._models_table_model
        self._models_table_view = TableView(self._models_table_model, parent=self)
        right_layout.addWidget(self._models_table_view)
        # Merge all the Layouts
        main_layout.addLayout(left_layout)
        main_layout.addLayout(right_layout)
        self.setLayout(main_layout)

    def top_left_layout(self):
        left_layout = QVBoxLayout()
        self._list_options_w_title = QLabel()
        self._list_options_w_title.setText('Free-text driven features')
        left_layout.addWidget(self._list_options_w_title)

        

    @QtCore.pyqtSlot(list, QWidget)
    def models_selected(self, model_names, source):
        # Show plots only for the models selected
        self._m_line_p_w.show_lines(show_lines=model_names)
        # Show table rows for the models selected
        data = self._models_table_model_full.to_dataframe()
        self._models_table_model = TableModel(data[data['model'].isin(model_names)])
        self._models_table_view.update_model(self._models_table_model)



import numpy as np
from PyQt6.QtWidgets import QApplication, QMainWindow, QVBoxLayout, QHBoxLayout, QWidget, QTabWidget, QSizePolicy, QLabel, QGridLayout

if __name__ == '__main__':
    class MainWindow(QMainWindow):
        def __init__(self):
            super().__init__()
            model_names = ['model_1', 'model_2', 'model_3']
            models_table_data = np.array([
                model_names,
                np.arange(1,4),
                np.arange(1,4)*10,
                np.arange(1,4)*100
            ]).T

            p_data_y = {
                'model_1': np.arange(10)**2, 
                'model_2': np.arange(10)**3, 
                'model_3': np.arange(10)
            }
            p_data_x = {
                'model_1': np.arange(10), 
                'model_2': np.arange(10), 
                'model_3': np.arange(10)
            }
            models_table_data = pd.DataFrame(models_table_data, columns=['model', *[f'Conf {i}' for i in range(3)]])
            tabwidget = ModelComparisonWidget(model_names,
                                              models_table_data,
                                              p_data_x,
                                              p_data_y)
            self.setCentralWidget(tabwidget)

    app = QApplication(sys.argv)
    main_window = MainWindow()
    main_window.show()
    sys.exit(app.exec())
import ini
import sys
import matplotlib
matplotlib.use('QtAgg')

from PyQt6 import QtCore, QtGui, QtWidgets

from PyQt6.QtWidgets import (
    QMainWindow, QWidget,
    QLabel, QVBoxLayout, QHBoxLayout, QComboBox, QPushButton
)
import typing
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg, NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure
import pandas as pd
import matplotlib.pyplot as plt

from typing import List, Dict, Set

import numpy as np

class MplCanvas(FigureCanvasQTAgg):

    def __init__(self, parent=None, width=5, height=4, dpi=100):
        fig = Figure(figsize=(width, height), dpi=dpi)
        self.axes = fig.add_subplot(111)
        super(MplCanvas, self).__init__(fig)

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

class MultiLinePlotWidget(QWidget):
    """
    Define a custom widget for Plotting Histograms.
    """

    def __init__(self, line_p_model:MultiLinePlotModel, show_lines:List[str]=None, plot_configs:Dict[str, any]={}, parent: typing.Optional['QWidget']=None, *args, **kwargs):
        super(MultiLinePlotWidget, self).__init__(parent=parent, *args, **kwargs)
        # Save the model that holds the data
        self._line_p_model = line_p_model
        self._plot_configs = plot_configs
        self._show_lines = show_lines if show_lines is not None else self._line_p_model.get_headers()
        # The column to show initial data of
        self._selected_cols = [col for col in self._line_p_model.get_headers() if col in set(self._show_lines)]
        # Histogram Widget
        self._line_p_widget = QtWidgets.QWidget(self)
        lines_p_layout = QtWidgets.QVBoxLayout()
        # Create the histogram
        self._create_line_plot(self._selected_cols, self._line_p_widget)
        self._toolbar = NavigationToolbar(self._sc, self._line_p_widget)
        lines_p_layout.addWidget(self._toolbar)
        lines_p_layout.addWidget(self._sc)
        button = QPushButton("Hello World", self)
        button.clicked.connect(self.button_clicked)
        lines_p_layout.addWidget(button)
        self._line_p_widget.setLayout(lines_p_layout)
        layout = QVBoxLayout()
        layout.addWidget(self._line_p_widget)
        self.setLayout(layout)

    def update_model(self, line_p_model: MultiLinePlotModel):
        self._line_p_model = line_p_model
        self.show_lines()
        self.update()

    def show_lines(self, show_lines:List[str]=None):
        self._show_lines = show_lines if show_lines is not None else self._line_p_model.get_headers()
        self.update()

    def button_clicked(self):
        print('clicked')
        data_x = {
            'a': np.arange(10),
            'b': np.arange(10),
            'c': np.arange(10)
        }

        data_y = {
            'a': np.arange(10)*10,
            'b': np.arange(10)*5,
            'c': np.arange(10)
        }

        # df = pd.DataFrame()
        line_p_model = MultiLinePlotModel(data_x, data_y)
        self.update_model(line_p_model)

    def update(self):
        data_x, data_y, labels = self._line_p_model.get_data(self._show_lines)
        self._sc.axes.clear()
        self._line_plots = list()
        for x,y,label in zip(data_x, data_y, labels):
            self._line_plots.append(self._sc.axes.plot(x, y, label=label))
        self._sc.axes.legend()
        if self._plot_configs.get('ylabel') is not None:
            self._sc.axes.set_ylabel(self._plot_configs['ylabel'])
        if self._plot_configs.get('xlabel') is not None:
            self._sc.axes.set_xlabel(self._plot_configs['xlabel'])
        if self._plot_configs.get('title') is not None:
            self._sc.axes.set_title(self._plot_configs['title'])
        self._sc.draw()
        self._toolbar.update()
        self._line_p_widget.update()

    def _create_line_plot(self, col_names, parent):
        self._sc = MplCanvas(parent, width=5, height=4, dpi=100)
        data_x, data_y, labels = self._line_p_model.get_data(col_names)
        self._line_plots = list()
        for x,y,label in zip(data_x, data_y, labels):
            self._line_plots.append(self._sc.axes.plot(x, y, label=label))
        self._sc.axes.legend()
        if self._plot_configs.get('ylabel') is not None:
            self._sc.axes.set_ylabel(self._plot_configs['ylabel'])
        if self._plot_configs.get('xlabel') is not None:
            self._sc.axes.set_xlabel(self._plot_configs['xlabel'])
        if self._plot_configs.get('title') is not None:
            self._sc.axes.set_title(self._plot_configs['title'])

class MainWindow(QtWidgets.QMainWindow):
    
    def __init__(self, *args, **kwargs):
        super(MainWindow, self).__init__(*args, **kwargs)
        import ini
        
        data_x = {
            'a': np.arange(10),
            'b': np.arange(10)
        }

        data_y = {
            'a': np.arange(10)*10,
            'b': np.arange(10)*5
        }

        # df = pd.DataFrame()
        l_p_model = MultiLinePlotModel(data_x, data_y, self)

        h_p_widget = MultiLinePlotWidget(l_p_model,show_lines=None, plot_configs={'xlabel':'X', 'ylabel':'Y', 'title': 'title'})
        self.setCentralWidget(h_p_widget)
        self.show()


app = QtWidgets.QApplication(sys.argv)

w = MainWindow()
app.exec()
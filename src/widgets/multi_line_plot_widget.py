import sys
import matplotlib
matplotlib.use('QtAgg')

from PyQt6 import QtWidgets

from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout
)
import typing
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg, NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure
import pandas as pd
from src.widgets.multi_line_plot_model import MultiLinePlotModel
from typing import List, Dict, Set

import numpy as np

class MplCanvas(FigureCanvasQTAgg):

    def __init__(self, parent=None, width=5, height=4, dpi=100):
        fig = Figure(figsize=(width, height), dpi=dpi)
        self.axes = fig.add_subplot(111)
        super(MplCanvas, self).__init__(fig)

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
        self._line_p_widget.setLayout(lines_p_layout)
        layout = QVBoxLayout()
        layout.addWidget(self._line_p_widget)
        self.setLayout(layout)

    def update_model(self, line_p_model: MultiLinePlotModel):
        self._line_p_model = line_p_model
        self.show_lines()

    def show_lines(self, show_lines:List[str]=None):
        self._show_lines = show_lines if show_lines is not None else self._line_p_model.get_headers()
        self.update()

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
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
from src.widgets.multi_bar_plot_model import MultiBarPlotModel
from typing import List, Dict, Set

import numpy as np

class MplCanvas(FigureCanvasQTAgg):

    def __init__(self, parent=None, width=5, height=4, dpi=100):
        fig = Figure(figsize=(width, height), dpi=dpi)
        self.axes = fig.add_subplot(111)
        super(MplCanvas, self).__init__(fig)

class MultiBarPlotWidget(QWidget):
    """
    Define a custom widget for Plotting Histograms.
    """

    def __init__(self, bar_p_model:MultiBarPlotModel, show_bars:List[str]=None, plot_configs:Dict[str, any]={}, parent: typing.Optional['QWidget']=None, *args, **kwargs):
        super(MultiBarPlotWidget, self).__init__(parent=parent, *args, **kwargs)
        # Save the model that holds the data
        self._bar_p_model = bar_p_model
        self._plot_configs = plot_configs
        self._show_bars = show_bars if show_bars is not None else self._bar_p_model.get_headers()
        # The column to show initial data of
        self._selected_cols = [col for col in self._bar_p_model.get_headers() if col in set(self._show_bars)]
        # Histogram Widget
        self._bar_p_widget = QtWidgets.QWidget(self)
        bars_p_layout = QtWidgets.QVBoxLayout()
        # Create the histogram
        self._create_bar_plot(self._selected_cols, self._bar_p_widget)
        self._toolbar = NavigationToolbar(self._sc, self._bar_p_widget)
        bars_p_layout.addWidget(self._toolbar)
        bars_p_layout.addWidget(self._sc)
        self._bar_p_widget.setLayout(bars_p_layout)
        layout = QVBoxLayout()
        layout.addWidget(self._bar_p_widget)
        self.setLayout(layout)

    def update_model(self, bar_p_model: MultiBarPlotModel):
        self._bar_p_model = bar_p_model
        self.show_bars()

    def show_bars(self, show_bars:List[str]=None):
        self._show_bars = show_bars if show_bars is not None else self._bar_p_model.get_headers()
        self.update()

    def update(self):
        labels, features = self._bar_p_model.get_data(self._show_bars)
        self._sc.axes.cla()
        self._bar_plots = list()
        # For plotting
        categories = features.index.values
        values = features.values.T
        x_tick = np.arange(len(categories))
        num_values = len(values)
        width = 1.0 / (num_values + 1)

        for i, vals in enumerate(values):
            offset = (i - num_values / 2) * width
            self._bar_plots.append(self._sc.axes.bar(x_tick + offset, vals, width, label=features.keys().values[i].replace('_','\n')))
            
        self._sc.axes.set_xticks(x_tick)
        self._sc.axes.set_xticklabels(categories)
        self._sc.axes.legend()
        if self._plot_configs.get('ylabel') is not None:
            self._sc.axes.set_ylabel(self._plot_configs['ylabel'])
        if self._plot_configs.get('xlabel') is not None:
            self._sc.axes.set_xlabel(self._plot_configs['xlabel'])
        if self._plot_configs.get('title') is not None:
            self._sc.axes.set_title(self._plot_configs['title'])
        self._sc.draw()
        self._toolbar.update()
        self._bar_p_widget.update()

    def _create_bar_plot(self, col_names, parent):
        self._sc = MplCanvas(parent, width=5, height=4, dpi=100)
        labels, features = self._bar_p_model.get_data(col_names)
        self._bar_plots = list()
        for col in features:
            self._bar_plots.append(self._sc.axes.bar(features.index.values, features[col], label=col))
        self._sc.axes.legend()
        if self._plot_configs.get('ylabel') is not None:
            self._sc.axes.set_ylabel(self._plot_configs['ylabel'])
        if self._plot_configs.get('xlabel') is not None:
            self._sc.axes.set_xlabel(self._plot_configs['xlabel'])
        if self._plot_configs.get('title') is not None:
            self._sc.axes.set_title(self._plot_configs['title'])
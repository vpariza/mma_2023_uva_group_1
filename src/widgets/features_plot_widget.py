import sys
import matplotlib
matplotlib.use('QtAgg')

from PyQt6 import QtWidgets
from PyQt6 import QtCore
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QComboBox, QHBoxLayout, QLabel, 
)
import typing
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg, NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure
import pandas as pd
from typing import List, Dict, Set

class MplCanvas(FigureCanvasQTAgg):

    def __init__(self, parent=None, width=5, height=4, dpi=100):
        fig = Figure(figsize=(width, height), dpi=dpi, facecolor='#f5f5f5')
        fig.subplots_adjust(left=0.2, right=0.98, top=0.99, bottom=0.21)
        fig.tight_layout()
        self.axes = fig.add_subplot(111)
        super(MplCanvas, self).__init__(fig)

class FeaturesPlotWidget(QWidget):
    """
    Define a custom widget for Plotting Histograms.
    """

    def __init__(self, data:pd.DataFrame, parent: typing.Optional['QWidget']=None, *args, **kwargs):
        super(FeaturesPlotWidget, self).__init__(parent=parent, *args, **kwargs)
        # Save the model that holds the data
        self._data = data
        self._identify_numerical_headers()
        if len(self.numerical_headers) > 0:
            if "price" in self.numerical_headers:
                self._selected_label_x = "price"
            else:
                self._selected_label_x = self.numerical_headers[0] 
            if "living_area" in self.numerical_headers:
                self._selected_label_y = "living_area"
            else:
                self._selected_label_y = self.numerical_headers[1]
        else:
            self._selected_label_x  = self._selected_label_y = None
        # Plot Widget
        self._ftrs_p_widget = QtWidgets.QWidget(self)
        sc_p_layout = QVBoxLayout()
        # Create the Plot Widget
        self._create_sc_plot(self._ftrs_p_widget)
        # self._toolbar = NavigationToolbar(self._sc, self._ftrs_p_widget)
        # sc_p_layout.addWidget(self._toolbar)
        sc_p_layout.addWidget(self._sc)
        buttons_layout = QVBoxLayout()
        # Dropdown menu for labels of axis x
        self._label_x_widget = QLabel()
        self._label_x_widget.setText('Axis X Feature:')
        buttons_layout.addWidget(self._label_x_widget)
        self._dropdown_list_lx_widget = QComboBox(self)
        self._dropdown_list_lx_widget.addItems(self.numerical_headers)
        self._dropdown_list_lx_widget.setCurrentText(self._selected_label_x)
        # self._dropdown_list_lx_widget.currentTextChanged.connect(self.dropdown_label_x_changed)
        self._dropdown_list_lx_widget.view().pressed.connect(self.dropdown_label_x_changed_i)
        buttons_layout.addWidget(self._dropdown_list_lx_widget)
        # Dropdown menu for labels of axis y
        self._dropdown_list_ly_widget = QComboBox(self)
        self._label_y_widget = QLabel()
        self._label_y_widget.setText('Axis Y Feature:')
        buttons_layout.addWidget(self._label_y_widget)
        self._dropdown_list_ly_widget.addItems(self.numerical_headers)
        self._dropdown_list_ly_widget.setCurrentText(self._selected_label_y)
        # self._dropdown_list_ly_widget.currentTextChanged.connect(self.dropdown_label_y_changed)
        self._dropdown_list_ly_widget.view().pressed.connect(self.dropdown_label_y_changed_i)
        buttons_layout.addWidget(self._dropdown_list_ly_widget)
        self._ftrs_p_widget.setLayout(sc_p_layout)
        self._ftrs_p_widget
        layout = QHBoxLayout()
        layout.addWidget(self._ftrs_p_widget)
        layout.addLayout(buttons_layout)
        self.setLayout(layout)
    
    @property
    def numerical_headers(self):
        return self._numeric_headers

    def _identify_numerical_headers(self):
        self._numeric_headers = list()
        for column in self._data.columns:
            if pd.to_numeric(self._data[column], errors='coerce').notnull().all() and self._data[column].dtypes != bool:
                self._numeric_headers.append(column)
        return self._numeric_headers

    def upadte_model(self, data:pd.DataFrame):
        self._data = data
        self._identify_numerical_headers()
        self._dropdown_list_lx_widget.clear()
        self._dropdown_list_lx_widget.addItems(self.numerical_headers)
        self._dropdown_list_ly_widget.clear()
        self._dropdown_list_ly_widget.addItems(self.numerical_headers)
        self.update()

    @QtCore.pyqtSlot(str)
    def dropdown_label_x_changed(self, label):
        self._selected_label_x = label
        self.update()

    @QtCore.pyqtSlot(str)
    def dropdown_label_y_changed(self, label):
        # print('dropdown_label_y_changed', label)
        self._selected_label_y = label
        self.update()
    
    @QtCore.pyqtSlot(QtCore.QModelIndex)
    def dropdown_label_x_changed_i(self, index):
        self._selected_label_x = self._dropdown_list_lx_widget.model().itemFromIndex(index).text()
        self.update()

    @QtCore.pyqtSlot(QtCore.QModelIndex)
    def dropdown_label_y_changed_i(self, index):
        self._selected_label_y = self._dropdown_list_ly_widget.model().itemFromIndex(index).text()
        self.update()

    def update(self):
        self._sc.axes.clear()
        if self._selected_label_y is not None and self._selected_label_x is not None:
            data_x, data_y = self._data[self._selected_label_x], self._data[self._selected_label_y]
        else:
            data_x, data_y  = [], []
        self._sc_plot = self._sc.axes.scatter(data_x, data_y, alpha = 0.2) # s=self.points_size, c=self.points_color, alpha=self._alpha
        if self._selected_label_y is not None:
            self._sc.axes.set_ylabel(self._selected_label_y)
        if self._selected_label_x is not None:
            self._sc.axes.set_xlabel(self._selected_label_x)
        if self._selected_label_y is not None and self._selected_label_x is not None:
            self._sc.axes.set_title('{} vs {}'.format(self._selected_label_x, self._selected_label_y))
        self._sc.draw()
        self._ftrs_p_widget.setMinimumSize(170, 170)
        # self._toolbar.update()
        # self._ftrs_p_widget.update()

    def _create_sc_plot(self, parent):
        self._sc = MplCanvas(parent, width=5, height=4, dpi=100)
        if self._selected_label_y is not None and self._selected_label_x is not None:
            data_x, data_y = self._data[self._selected_label_x], self._data[self._selected_label_y]
        else:
            data_x, data_y  = [], []
        self._sc_plot = self._sc.axes.scatter(data_x, data_y) # s=self.points_size, c=self.points_color, alpha=self._alpha
        if self._selected_label_y is not None:
            self._sc.axes.set_ylabel(self._selected_label_y)
        if self._selected_label_x is not None:
            self._sc.axes.set_xlabel(self._selected_label_x)
        if self._selected_label_y is not None and self._selected_label_x is not None:
            self._sc.axes.set_title('{} vs {}'.format(self._selected_label_x, self._selected_label_y))
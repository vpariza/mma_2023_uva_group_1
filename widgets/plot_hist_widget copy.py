import sys
import matplotlib
matplotlib.use('QtAgg')

from PyQt6 import QtCore, QtGui, QtWidgets

from PyQt6.QtWidgets import (
    QMainWindow, QWidget,
    QLabel, QVBoxLayout, QPushButton, QComboBox
)
import typing
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg, NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure
import matplotlib.pyplot as plt

from enum import Enum

import pandas as pd

class HistogramPlotModel(QtCore.QObject):
    class HouseInfoDistKeys(Enum):
        LONGITUDE = 'lon'
        LATITUDE = 'lat'
        # LISTING_ID = 'funda_identifier'
        PRICE = 'price'
        LIVING_AREA = 'living_area'
        BEDROOMS = 'bedrooms'
        ASK_PRICE_PER_SQ_M = 'asking_price_per_m²'
        STATUS = 'status'
        ACCEPTANCE = 'acceptance'
        BUILDING_TYPE = 'building_type'
        NUM_BATH_ROOMS = 'number_of_bath_rooms'
        LABEL = 'label'
        # HEATING = 'heating'
        GARDEN = 'garden'
        FACILITIES_TYPE = 'type_of_facilities'
        # NEIGHBERHOOD = 'in_the_neighborhood'

        @classmethod
        def list_values(cls):
            return list(map(lambda c: c.value, cls))

    def __init__(self, data:pd.DataFrame, parent: typing.Optional[QtCore.QObject] = None ) -> None:
        super(HistogramPlotModel, self).__init__(parent)
        self._data = data
    
    def get_headers(self):
        return self.HouseInfoDistKeys.list_values()
    
    def get_column(self, column_name:str) -> pd.DataFrame:
        return self._data[column_name]

class HistogramPlotWidget(QWidget):
    """
    Define a custom widget for Plotting Histograms.
    """
    # Define the Signals that can be emitted
    # entriesSelected = QtCore.pyqtSignal(list, QWidget)
    # entryClicked = QtCore.pyqtSignal(object, QWidget)

    def __init__(self, hist_p_model:HistogramPlotModel, parent: typing.Optional['QWidget']=None, *args, **kwargs):
        super(HistogramPlotWidget, self).__init__(parent=parent, *args, **kwargs)
        # Save the model that holds the data
        self._hist_p_model = hist_p_model
        # The column to show initial data of
        self._selected_col = list(self._hist_p_model.get_headers())[0]
        # Histogram Widget
        self._hist = None
        self._hist_widget = QtWidgets.QWidget(self)
        hist_layout = QtWidgets.QVBoxLayout()
        self._create_hist(self._selected_col)
        self._toolbar = NavigationToolbar(self._canvas, self._hist_widget)
        hist_layout.addWidget(self._toolbar)
        hist_layout.addWidget(self._canvas)
        self._hist_widget.setLayout(hist_layout)
        layout = QVBoxLayout()
        self._dropdown_list_widget = QComboBox(self)
        self._dropdown_list_widget.addItems(self._hist_p_model.get_headers())
        self._dropdown_list_widget.currentTextChanged.connect(self.dropdown_label_changed)
        layout.addWidget(self._dropdown_list_widget)
        layout.addWidget(self._hist_widget)
        self.setLayout(layout)

    @QtCore.pyqtSlot(str)
    def dropdown_label_changed(self, label):
        print(label)
        self._selected_col = label
        self.update()

    def update(self):
        self._create_hist( self._selected_col)
        # self._create_hist(self._selected_col)
        # self._toolbar = NavigationToolbar(self._sc, self)
        # _hist_widget = QtWidgets.QWidget()
        # plot_layout = QtWidgets.QVBoxLayout()
        # plot_layout.addWidget(self._toolbar)
        # plot_layout.addWidget(self._sc)
        # _hist_widget.setLayout(plot_layout)
        # layout = QVBoxLayout()
        # _dropdown_list_widget = QComboBox()
        # _dropdown_list_widget.addItems(self._hist_p_model.get_headers())
        # _dropdown_list_widget.currentTextChanged.connect(self.dropdown_label_changed)
        # layout.removeWidget(self._dropdown_list_widget)
        # layout.removeWidget(self._hist_widget)
        # self._dropdown_list_widget = _dropdown_list_widget
        # self._hist_widget = _hist_widget
        # layout.addWidget(self._dropdown_list_widget)
        # layout.addWidget(self._hist_widget)

    def _create_hist(self, col_name):
        self._canvas = FigureCanvasQTAgg(plt.Figure(figsize=(15, 6)))
        data = self._hist_p_model.get_column(col_name)
        num_u_labels = len(set(data.values.tolist()))
        self._ax = self._canvas.figure.subplots()
        self._hist = self._ax.hist(data, density=False, bins=min(30, num_u_labels))  # density=False would make counts
        self._ax.set_ylabel('Counts')
        self._ax.set_xlabel(col_name)
        self._ax.tick_params(rotation=90)
        self._canvas.draw()

    def _update_hist(self, col_name):
        data = self._hist_p_model.get_column(col_name)
        num_u_labels = len(set(data.values.tolist()))
        if self._hist is not None:
            self._hist[-1].remove()
            # del self._hist
        self._hist = self._ax.hist(data, density=False, bins=min(30, num_u_labels))  # density=False would make counts
        self._canvas.draw()

class MainWindow(QtWidgets.QMainWindow):

    def __init__(self, *args, **kwargs):
        super(MainWindow, self).__init__(*args, **kwargs)
        import ini
        from utils.file_utils import load_from_pickle
        df = load_from_pickle('data.pkl')
        h_p_model = HistogramPlotModel(df)
        h_p_widget = HistogramPlotWidget(h_p_model)
        self.setCentralWidget(h_p_widget)
        self.show()


app = QtWidgets.QApplication(sys.argv)

w = MainWindow()
app.exec()
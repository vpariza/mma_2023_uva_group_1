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
import pandas as pd

class MplCanvas(FigureCanvasQTAgg):

    def __init__(self, parent=None, width=5, height=4, dpi=100):
        fig = Figure(figsize=(width, height), dpi=dpi)
        self.axes = fig.add_subplot(111)
        super(MplCanvas, self).__init__(fig)

class HistogramPlotModel(QObject):
    class HouseInfoDistKeys(Enum):
        LONGITUDE = 'lon'
        LATITUDE = 'lat'
        POINT_COORDS = 'coords'
        LISTING_ID = 'funda_identifier'

        @classmethod
        def list_values(cls):
            return list(map(lambda c: c.value, cls))

    def __init__(self, data:pd.DataFrame, parent: typing.Optional['QObject'] = None ) -> None:
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

    def __init__(self, histp_model:HistogramPlotModel, parent: typing.Optional['QWidget']=None, *args, **kwargs):
        super(HistogramPlotWidget, self).__init__(parent=parent, *args, **kwargs)
        self._histp_model = histp_model
        # the widget should have a layout
        layout = QVBoxLayout()
        dropdown_list_widget = QComboBox()
        dropdown_list_widget.addItems(["One", "Two", "Three"])

        self.map_view = QtWebEngineWidgets.QWebEngineView()
        page = WebEngineGeoMapPage(self.map_view)

        page.areaSelected.connect(self.area_was_selected)
        page.markerClicked.connect(self.marker_was_clicked)
        # self.setUpdatesEnabled()
        self.map_view.setPage(page)
        self.map_view.setHtml(self.__generate_html_map().getvalue().decode())
        self.map_view.resize(640, 480)
        layout.addWidget(self.map_view)
        self.setLayout(layout)
    

class MainWindow(QtWidgets.QMainWindow):

    def __init__(self, *args, **kwargs):
        super(MainWindow, self).__init__(*args, **kwargs)

        sc = MplCanvas(self, width=5, height=4, dpi=100)
        sc.axes.plot([0,1,2,3,4], [10,1,20,3,40])

        # Create toolbar, passing canvas as first parament, parent (self, the MainWindow) as second.
        toolbar = NavigationToolbar(sc, self)

        layout = QtWidgets.QVBoxLayout()
        layout.addWidget(toolbar)
        layout.addWidget(sc)

        # Create a placeholder widget to hold our toolbar and canvas.
        widget = QtWidgets.QWidget()
        widget.setLayout(layout)
        self.setCentralWidget(widget)

        self.show()


app = QtWidgets.QApplication(sys.argv)
w = MainWindow()
app.exec()
import matplotlib
matplotlib.use('QtAgg')

from PyQt6 import QtCore, QtWidgets

from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QComboBox
)
import typing
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg, NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure

from src.widgets.hist_plot_model import HistogramPlotModel
class MplCanvas(FigureCanvasQTAgg):

    def __init__(self, parent=None, width=5, height=4, dpi=100):
        self.fig = Figure(figsize=(width, height), dpi=dpi)
        self.axes = self.fig.add_subplot(111)
        super(MplCanvas, self).__init__(self.fig)

class HistogramPlotWidget(QWidget):
    """
    Define a custom widget for Plotting Histograms.
    """

    def __init__(self, hist_p_model:HistogramPlotModel, parent: typing.Optional['QWidget']=None, *args, **kwargs):
        super(HistogramPlotWidget, self).__init__(parent=parent, *args, **kwargs)
        # Save the model that holds the data
        self._hist_p_model = hist_p_model
        # The column to show initial data of
        self._selected_col = list(self._hist_p_model.get_headers())[0]
        # Histogram Widget
        self._hist_widget = QtWidgets.QWidget(self)
        hist_layout = QtWidgets.QVBoxLayout()
        self._create_hist(self._selected_col, self._hist_widget)
        self._toolbar = NavigationToolbar(self._sc, self._hist_widget)
        hist_layout.addWidget(self._toolbar)
        hist_layout.addWidget(self._sc)
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
        self._selected_col = label
        self.update()

    def update_model(self, hist_p_model: HistogramPlotModel):
        self._hist_p_model = hist_p_model
        self.update()

    def update(self):
        data = self._hist_p_model.get_column(self._selected_col)
        num_u_labels = len(set(data.values.tolist()))
        self._hist[-1].remove()
        self._sc.axes.clear()
        self._hist = self._sc.axes.hist(data, density=False, bins=min(30, num_u_labels))  # density=False would make counts
        self._sc.axes.set_ylabel('Counts')
        self._sc.axes.set_xlabel(self._selected_col)
        self._sc.axes.tick_params(rotation=90)
        self._sc.fig.tight_layout()
        self._sc.draw()
        self._toolbar.update()
        self._hist_widget.update()

    def _create_hist(self, col_name, parent):
        self._sc = MplCanvas(parent, width=5, height=4, dpi=100)
        data = self._hist_p_model.get_column(col_name)
        num_u_labels = len(set(data.values.tolist()))
        self._hist = self._sc.axes.hist(data, density=False, bins=min(30, num_u_labels))  # density=False would make counts
        self._sc.axes.set_ylabel('Counts')
        self._sc.axes.set_xlabel(col_name)
        self._sc.axes.tick_params(rotation=90)

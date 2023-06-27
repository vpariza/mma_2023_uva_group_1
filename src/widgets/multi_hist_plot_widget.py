import sys
sys.path.append('/Users/valentinospariza/Library/CloudStorage/OneDrive-UvA/Repositories/multimedia_analytics/mma_2023_uva_group_1/')

import sys
import matplotlib
matplotlib.use('QtAgg')

from PyQt6 import QtCore, QtGui, QtWidgets

from PyQt6.QtWidgets import (
    QMainWindow, QWidget,
    QLabel, QVBoxLayout, QHBoxLayout, QComboBox, QDialog, QDialogButtonBox
)
import typing
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg, NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure

from typing import List, Dict
from src.widgets.multi_hist_plot_model import MultiHistogramPlotModel
from src.widgets.checkbox_list_widget import CheckBoxListWidget

from src.widgets.dialog_widgets import BasicDialog
import numpy as np
import pandas as pd

class MplCanvas(FigureCanvasQTAgg):

    def __init__(self, parent=None, width=5, height=4, dpi=100):
        self.fig = Figure(figsize=(width, height), dpi=dpi, facecolor='#f5f5f5')
        self.axes = self.fig.add_subplot(111)
        super(MplCanvas, self).__init__(self.fig)

class MultiHistogramPlotWidget(QWidget):
    """
    Define a custom widget for Plotting Histograms.
    """

    def __init__(self, hist_p_model:MultiHistogramPlotModel, options:Dict[str,bool]|List[str], options_fn:Dict[str,object], parent: typing.Optional['QWidget']=None, *args, **kwargs):
        super(MultiHistogramPlotWidget, self).__init__(parent=parent, *args, **kwargs)
        # Save the model that holds the data
        self._hist_p_model = hist_p_model
        self._options_fn = options_fn
        # The column to show initial data of
        self._selected_col = list(self._hist_p_model.get_headers())[0]
        # Histogram Widget
        main_layout = QHBoxLayout()
        self._hist_widget = QtWidgets.QWidget(self)
        hist_layout = QtWidgets.QVBoxLayout()
        # Define the checkbox widget
        vbox =  QVBoxLayout()
        vbox.addWidget(QLabel('Feature\nTransformations'))
        self._checkbox = CheckBoxListWidget(options,self)
        self._checkbox.stateChanged.connect(self.options_state_changed)
        vbox.addWidget(self._checkbox)
        # Create the histogram
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
        main_layout.addLayout(layout)
        main_layout.addLayout(vbox)
        self.setLayout(main_layout)

    @QtCore.pyqtSlot(str)
    def dropdown_label_changed(self, label):
        self._selected_col = label
        self.update()

    @QtCore.pyqtSlot(dict, QWidget)
    def options_state_changed(self, options_state, source):
        self.update()

    def get_transformed_column_data(self, data:pd.DataFrame):
        column_data = dict()
        for option_name, fn in self._options_fn.items():
            if not self._checkbox.is_checked(option_name):
                continue
            new_data_column = '{}_{}'.format(self._dropdown_list_widget.currentText(),option_name)
            if new_data_column not in data.columns:
                column_data[new_data_column] = list(fn(data[self._dropdown_list_widget.currentText()].values))
        return pd.DataFrame(column_data)

    def update_model(self, hist_p_model: MultiHistogramPlotModel):
        self._hist_p_model = hist_p_model
        self.update()

    def apply_option_fns(self, data):
        option_names = list()
        options_data = list()
        for option_name, fn in self._options_fn.items():
            if not self._checkbox.is_checked(option_name):
                continue
            option_names.append(option_name)
            options_data.append(fn(data))
        return options_data, option_names

    def update_model(self, hist_p_model: MultiHistogramPlotModel):
        self._hist_p_model = hist_p_model
        self.update()

    def update(self):
        data = self._hist_p_model.get_column(self._selected_col)
        numeric_data = True
        try:
            options_data, labels = self.apply_option_fns(data.values)
        except ValueError as e:
            numeric_data = False
        numeric_data = numeric_data and pd.to_numeric(data, errors='coerce').notnull().all() == True
        if numeric_data == False:
            options_data = labels = []
            self._checkbox.uncheck_all()
        num_u_labels = len(set(data.values.tolist()))
        self._sc.axes.clear()
        if len(options_data) > 0 and numeric_data:
            self._hist = self._sc.axes.hist(options_data, density=False, bins=min(30, num_u_labels), label=labels)  # density=False would make counts
        else:
            self._hist = self._sc.axes.hist( data if numeric_data else data.astype(str), density=False, bins=min(30, num_u_labels), label='Original')  # density=False would make counts
        self._sc.axes.legend()
        self._sc.axes.set_ylabel('Counts')
        self._sc.axes.set_xlabel(self._selected_col)
        self._sc.axes.tick_params(rotation=90)
        self._sc.fig.tight_layout()
        self._sc.draw()
        self._toolbar.update()
        self._hist_widget.update()

    def _create_hist(self, col_name, parent):
        self._sc = MplCanvas(parent, width=5, height=4, dpi=100)
        data = self._hist_p_model.get_column(self._selected_col)
        options_data, labels = self.apply_option_fns(data.values)
        num_u_labels = len(set(data.values.tolist()))
        if len(options_data) > 0:
            self._hist = self._sc.axes.hist(options_data, density=False, bins=min(30, num_u_labels), label=labels)  # density=False would make counts
        else:
            self._hist = self._sc.axes.hist(data, density=False, bins=min(30, num_u_labels), label='Original')  # density=False would make counts
        self._sc.axes.legend()
        self._sc.axes.set_ylabel('Counts')
        self._sc.axes.set_xlabel(col_name)
        self._sc.axes.tick_params(rotation=90)
        self._sc.fig.tight_layout()
        self._sc.draw()


if __name__ == '__main__':

    from sklearn.preprocessing import minmax_scale
    from sklearn.preprocessing import scale
    from sklearn.preprocessing import robust_scale
    from sklearn.preprocessing import maxabs_scale
    class MainWindow(QtWidgets.QMainWindow):
        
        def __init__(self, *args, **kwargs):
            super(MainWindow, self).__init__(*args, **kwargs)
            
            from src.utils.file_utils import load_from_pickle
            from src.widgets.hist_plot_widget import HistogramPlotWidget
            df = load_from_pickle('data.pkl')
            h_p_model = MultiHistogramPlotModel(df)
            options_fn = {
                'minmax_scale': minmax_scale,
                'standard_scale': scale,
                'robust_scale': robust_scale,
                'maxabs_scale': maxabs_scale
            }
            h_p_widget = MultiHistogramPlotWidget(h_p_model, list(options_fn.keys()), options_fn)
            self.setCentralWidget(h_p_widget)
            self.show()


    app = QtWidgets.QApplication(sys.argv)

    w = MainWindow()
    app.exec() 
import sys
sys.path.append('/Users/valentinospariza/Library/CloudStorage/OneDrive-UvA/Repositories/multimedia_analytics/mma_2023_uva_group_1/')

import sys
from PyQt6 import QtCore
from PyQt6.QtCore import QSize
from PyQt6.QtWidgets import QApplication, QMainWindow, QVBoxLayout, QHBoxLayout, QWidget, QTabWidget, QSizePolicy
from src.widgets.query_widget import QueryWidget
from src.widgets.plot_widget import ScatterPlotWidget, SelectClusterWidget
from src.widgets.filter_widget import FilterWidget
from src.utils.preprocessing import Preprocessing
from src.widgets.geo_map_model import QGeoMapModel
from src.widgets.geo_map_widget import GeoMapWidget
from src.widgets.hist_plot_model import HistogramPlotModel
from src.widgets.hist_plot_widget import HistogramPlotWidget
import numpy as np 

from src.widgets.table_listings_view import TableListingsView
from src.widgets.table_llistings_model import TableListingsModel


from PyQt6.QtWidgets import QWidget, QHBoxLayout, QVBoxLayout, QPushButton

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        #load data using the config file 'config.ini'
        preprocessing = Preprocessing()
        # TODO: Return sirectory of Listings Images
        self.config, self.tags, self.points, self.img_paths, self.df = preprocessing.load_data()
        # BEGIN TEST: For Testing Purposes only
        from src.utils.file_utils import load_from_pickle
        self.df = load_from_pickle('data.pkl')
        self.df.iloc[:100]
        # END TEST: For Testing Purposes only
        self.df_show = self.df.copy()
        self.images_dir_path = '../Dataloading/Datasets/Funda/images'
        ## inialize widgets
        self.query_widgets = [QueryWidget() for i in range(2)]
        for query_widget in self.query_widgets:
            query_widget.querySubmitted.connect(self.on_query_submitted)

        # Define scatter plot widget
        self.select_scatter_plot = SelectClusterWidget()
        self.scatter_plot_widget = ScatterPlotWidget(self.points, self.config)
        
        # Select filters 
        # combolist -> {Element Title: [index values, displayed textprompt]}
        combofilters_ = {'Parking': ['no', 'yes'], 
                        'Bedroom': [str(i) for i in np.arange(0, 5 + 1)]}
        minmaxfilters_ = ['price', 'area']


        self.filter_widget = FilterWidget(minmaxfilters = minmaxfilters_, combofilters = combofilters_, config = '1')
        self.filter_widget.searchbutton.filtersApplied.connect(self.on_filters_applied)


        self.filter_widget_tab2 = FilterWidget(minmaxfilters = minmaxfilters_, combofilters = combofilters_, config = '2')
        self.filter_widget_tab2.searchbutton.filtersApplied.connect(self.on_filters_applied)

        # Define the Geo Map Widget
        geo_map_model = QGeoMapModel(self.df_show)
        self.geo_map_widget = GeoMapWidget(geo_map_model)
        self.geo_map_widget.entryClicked.connect(self.on_map_entry_clicked)
        self.geo_map_widget.entriesSelected.connect(self.on_map_entries_selected)

        # Define the Listings Table
        table_listings_model = TableListingsModel(self.df_show, self.images_dir_path)
        self.table_listings_widget = TableListingsView(table_listings_model)
        self.table_listings_widget.entryDoubleClicked.connect(self.on_table_entry_double_clicked)

        # Define the Histogram widget
        histmodel = HistogramPlotModel(self.df)
        self.hist_plot_widget = HistogramPlotWidget(histmodel)


        # Clear All Button Widget
        self.clear_all_button_widget = QPushButton('Clear All', self)
        self.clear_all_button_widget.clicked.connect(self.clear_all_button_clicked)

        ## set up main window 
        
        self.make_layout()
        
    def make_layout(self):   
        """Congifure layout for main window"""  

        # Configure main window apperance    
        self.setWindowTitle("READ: Real Estate Analytics Dashboard") # READ
        self.setMinimumSize(QSize(1250, 500))

        # Tab Widget
        tabwidget = QTabWidget()

        ## Combine widgets in right column
        filterwidgets = QWidget()
        filterwidgets_layout = QVBoxLayout(self, spacing=10)
        filterwidgets_layout.addWidget(self.query_widgets[0])
        filterwidgets_layout.addWidget(self.filter_widget)
        filterwidgets_layout.addWidget(self.table_listings_widget)
        filterwidgets_layout.addWidget(self.clear_all_button_widget)
        filterwidgets.setLayout(filterwidgets_layout)
   
        ## set the layout of tab 1
        tab1_widget = QWidget()
        tab1_layout = QHBoxLayout(self, spacing=10)
        tab1_layout.addWidget(self.geo_map_widget)
        tab1_layout.addWidget(filterwidgets)
        
        tab1_widget.setLayout(tab1_layout)
        
        
        ## set the layout of tab 2
        tab2_widget = QWidget()
        tab2_layout = QHBoxLayout(self, spacing=10)
        tab2_layout.addWidget(self.scatter_plot_widget)


        V2_widget = QWidget()
        V2_layout = QVBoxLayout(self, spacing=10)

        H2_widget = QWidget()
        H2_layout = QHBoxLayout(self, spacing=10)

        H2_layout.addWidget(self.select_scatter_plot)
        H2_layout.addWidget(self.hist_plot_widget)
        H2_widget.setLayout(H2_layout)
        V2_layout.addWidget(H2_widget)
        V2_layout.addWidget(self.filter_widget_tab2)

        V2_widget.setLayout(V2_layout)
        
        tab2_layout.addWidget(V2_widget)
        tab2_widget.setLayout(tab2_layout)
        
        
        # Build tabs
        tabwidget.addTab(tab1_widget, "House Search")
        tabwidget.addTab(tab2_widget, "House Feature Exploration")
        self.setCentralWidget(tabwidget)
        
    
    def update_table(self):
        geo_map_model = QGeoMapModel(self.df_show)
        self.geo_map_widget.update_geo_map_model(geo_map_model)

    def update_geo_map(self):
        table_listings_model = TableListingsModel(self.df_show, self.images_dir_path)
        self.table_listings_widget.update_model(table_listings_model)

    def update(self):
        """
        Update/Refresh all the widgets
        """
        self.update_geo_map()
        self.update_table()

    ###### HANDLING SINGALS FROM CHILD WIDGETS - SLOTS #######
    @QtCore.pyqtSlot(str, QWidget)
    def on_query_submitted(self, query, source):
        print('Submitted Query',query)

    @QtCore.pyqtSlot(object, QWidget)
    def on_filters_applied(self, filters, source):
        filtered_df = self.apply_filters(self.df, filters) 
        if len(filtered_df.index) > 0:
            self.df_show = filtered_df
            self.update()

    @QtCore.pyqtSlot(list, QWidget)
    def on_map_entries_selected(self, entries, source):
        if len(entries) > 0:
            self.df_show = self.df_show[self.df_show['funda_identifier'].isin(entries)]
            self.update()

    @QtCore.pyqtSlot(object, QWidget)
    def on_map_entry_clicked(self, entry, source):
        # TODO
        print('Map Entry Clicked', entry)
        pass

    @QtCore.pyqtSlot(object, QWidget)
    def on_table_entry_double_clicked(self, entry, source):
        self.geo_map_widget.focus_on_entry(entry)
        self.update()

    @QtCore.pyqtSlot()
    def clear_all_button_clicked(self):
        self.df_show = self.df.copy()
        self.geo_map_widget.focus_on_coord(None)
        self.update()

    ###### Other Utility Methods ######
    def apply_filters(self, df, filters): 
        new_df = self.df_show.copy()
        for tag, filter in filters.items():
            values = {'Max': filter.Max.QueryText.text(), 'Min': filter.Min.QueryText.text() }
            for bound, input in values.items():
                try: 
                    input = eval(input)
                    if bound == 'Min':
                        print(bound, tag, input)
                        new_df = new_df[new_df[tag] > input]
                    elif bound == 'Max':
                        print(bound, tag, input)
                        new_df = new_df[new_df[tag] < input]

                except SyntaxError:
                    print(bound, tag, 'not provided')
                except NameError:
                    print('invalid input for ', bound, tag)
        return new_df

if __name__ == '__main__':
    app = QApplication(sys.argv)
    main_window = MainWindow()
    main_window.show()
    sys.exit(app.exec())
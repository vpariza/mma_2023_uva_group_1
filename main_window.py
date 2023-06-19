import sys
sys.path.append('/Users/valentinospariza/Library/CloudStorage/OneDrive-UvA/Repositories/multimedia_analytics/mma_2023_uva_group_1/')

import sys
from PyQt6 import QtCore
from PyQt6.QtCore import QSize, Qt
from PyQt6.QtWidgets import QApplication, QMainWindow, QLabel, QVBoxLayout, QHBoxLayout, QWidget, QTabWidget
from src.widgets.query_widget import QueryWidget
from src.widgets.plot_widget import PlotWidget
from src.widgets.filter_widget import FilterWidget
from src.utils.preprocessing import Preprocessing
from src.widgets.geo_map_model import QGeoMapModel
from src.widgets.geo_map_widget import GeoMapWidget

from src.widgets.table_listings_view import TableListingsView
from src.widgets.table_llistings_model import TableListingsModel


from PyQt6.QtWidgets import QWidget, QLineEdit, QComboBox, QHBoxLayout, QVBoxLayout, QLabel, QPushButton

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
        # self.scatter_plot_widget = PlotWidget(self.points, self.config)
        self.filter_widget = FilterWidget()
        self.filter_widget.searchbutton.filtersApplied.connect(self.on_filters_applied)
        # Define the Geo Map Widget
        geo_map_model = QGeoMapModel(self.df_show)
        self.geo_map_widget = GeoMapWidget(geo_map_model)
        self.geo_map_widget.entryClicked.connect(self.on_map_entry_clicked)
        self.geo_map_widget.entriesSelected.connect(self.on_map_entries_selected)

        # Define the Listings Table
        table_listings_model = TableListingsModel(self.df_show, self.images_dir_path)
        self.table_listings_widget = TableListingsView(table_listings_model)
        self.table_listings_widget.entryDoubleClicked.connect(self.on_table_entry_double_clicked)

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
        
        # Build tabs
        tabwidget.addTab(tab1_widget, "House Search")
        label1 = QLabel("Sick Dashboard Visualizations.")
        tabwidget.addTab(label1, "House Feature Exploration")
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
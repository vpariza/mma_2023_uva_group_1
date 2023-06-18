

import sys
from PyQt6 import QtCore
from PyQt6.QtCore import QSize, Qt
from PyQt6.QtWidgets import QApplication, QMainWindow, QLabel, QVBoxLayout, QHBoxLayout, QWidget
from src.widgets.query_widget import QueryWidget
from src.widgets.plot_widget import PlotWidget
from src.widgets.filter_widget import FilterWidget
from src.utils.preprocessing import Preprocessing
from src.widgets.geo_map_model import QGeoMapModel
from src.widgets.geo_map_widget import GeoMapWidget

from src.widgets.table_listings_view import TableListingsView
from src.widgets.table_llistings_model import TableListingsModel

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        #load data using the config file 'config.ini'
        preprocessing = Preprocessing()
        # TODO: Return sirectory of Listings Images
        self.config, self.tags, self.points, self.img_paths, self.df = preprocessing.load_data()
        from src.utils.file_utils import load_from_pickle
        self.df = load_from_pickle('data.pkl')
        self.df_show = self.df.copy()
        self.images_dir_path = ''
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
        self.table_listings_widget.entryClicked.connect(self.on_table_entry_clicked)
        
        # Connect the widgets to slots for hanlding signals

        ## set up main window 
        self.make_layout()
        
        
    def make_layout(self):   
        """Congifure layout for main window"""  

        # Configure main window apperance    
        self.setWindowTitle("READ: Real Estate Analytics Dashboard") # READ
        self.setMinimumSize(QSize(1250, 500))

        ## Combine widgets in right column
        vbox = QWidget()
        vbox_layout = QVBoxLayout(self, spacing=0)

        vbox_layout.addWidget(self.query_widgets[0])
        #vbox_layout.addWidget(self.query_widgets[1])
        vbox_layout.addWidget(self.filter_widget)
        vbox_layout.addWidget(self.table_listings_widget)
        
        vbox.setLayout(vbox_layout)
   
        ## set the layout of the main window
        main_widget = QWidget()
        main_layout = QHBoxLayout(self, spacing=10)
        main_layout.addWidget(self.geo_map_widget)
        main_layout.addWidget(vbox)
        
        main_widget.setLayout(main_layout)
        self.setCentralWidget(main_widget)
    
    # Update/Refresh all the widgets
    def update(self):
        geo_map_model = QGeoMapModel(self.df_show)
        self.geo_map_widget.update_geo_map_model(geo_map_model)
        table_listings_model = TableListingsModel(self.df_show, self.images_dir_path)
        self.table_listings_widget.update_model(table_listings_model)

    ###### HANDLING SINGALS FROM CHILD WIDGETS - SLOTS #######
    @QtCore.pyqtSlot(str, QWidget)
    def on_query_submitted(self, query, source):
        print('Submitted Query',query)

    @QtCore.pyqtSlot(object, QWidget)
    def on_filters_applied(self, filters, source):
        self.apply_filters(filters) 
        self.update()

    @QtCore.pyqtSlot(list, QWidget)
    def on_map_entries_selected(self, entries, source):
        self.df_show = self.df_show[self.df_show['funda_identifier'].isin(entries)]
        self.update()

    @QtCore.pyqtSlot(object, QWidget)
    def on_map_entry_clicked(self, entry, source):
        pass

    @QtCore.pyqtSlot(object, QWidget)
    def on_table_entry_clicked(self, entry, source):
        pass

    ###### Other Utility Methods ######
    def apply_filters(self, filters): 
        new_df = self.df.copy()
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
        self.df_show = new_df

if __name__ == '__main__':
    app = QApplication(sys.argv)
    main_window = MainWindow()
    main_window.show()
    sys.exit(app.exec())
import sys
sys.path.append('/Users/valentinospariza/Library/CloudStorage/OneDrive-UvA/Repositories/multimedia_analytics/mma_2023_uva_group_1/')

import sys
from PyQt6 import QtCore
from PyQt6.QtCore import QSize
from PyQt6.QtWidgets import QApplication, QMainWindow, QVBoxLayout, QHBoxLayout, QWidget, QTabWidget, QSizePolicy, QLabel
from src.widgets.query_widget import QueryWidget
from src.widgets.plot_widget import ScatterPlotWidget, SelectClusterWidget
from src.widgets.filter_widget import FilterWidget
from src.utils.preprocessing import Preprocessing
from src.widgets.geo_map_model import QGeoMapModel
from src.widgets.geo_map_widget import GeoMapWidget
from src.widgets.hist_plot_model import HistogramPlotModel
from src.widgets.hist_plot_widget import HistogramPlotWidget
from src.widgets.elements.custom_blocks import TitleWidget, ButtonWidget, CheckBoxWidget
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
        self.config, self.tags, self.points, self.img_paths, self.df, self.images_dir_path = preprocessing.load_data()
        # BEGIN TEST: For Testing Purposes only
        from src.utils.file_utils import load_from_pickle
        #self.df = load_from_pickle('data.pkl')
        #self.df.iloc[:100]
        # END TEST: For Testing Purposes only
        self.df_show = self.df.copy()
        

        ####
        # Define all widgets 
        ####

        self.query_widgets = [QueryWidget() for i in range(2)]
        for query_widget in self.query_widgets:
            query_widget.querySubmitted.connect(self.on_query_submitted)

        # Define scatter plot widget
        self.select_scatter_plot = SelectClusterWidget()
        self.scatter_plot_widget = ScatterPlotWidget(self.points, self.config)

        self.feature_transform_widget = CheckBoxWidget(['MinMaxScaling', 'SQRT'],'Feature\nTransformations')

        
        # Define filters 
        combofilters_ = {'status': ['Available', 'Under option'],                                                  
                        'bedrooms': [str(i) for i in np.arange(0, 5 + 1)]}           # combofilters -> {Element Title: [displayed textprompt]}
        minmaxfilters_ = ['price', 'living_area']

        ## Filter widgets tab 1 with layoutoption 1
        self.filter_widget = FilterWidget(minmaxfilters = minmaxfilters_, combofilters = combofilters_, config = '1')
        self.filter_widget.searchbutton.filtersApplied.connect(self.on_filters_applied)

        ## Filter widgets tab 2 with layoutoption 2
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

        # Define the Listings Table tab2
        table_listings_model_2 = TableListingsModel(self.df_show, self.images_dir_path)
        self.table_listings_widget_2 = TableListingsView(table_listings_model_2)

        # Define the Histogram widget
        histmodel = HistogramPlotModel(self.df)
        self.hist_plot_widget = HistogramPlotWidget(histmodel)
        

        # Clear All Button Widget
        self.clear_all_button_widgets = [QPushButton('Clear All', self) for i in range(2)]
        for widget in range(len(self.clear_all_button_widgets)):
            self.clear_all_button_widgets[widget].clicked.connect(self.clear_all_button_clicked)

        # set up main window 
        self.make_main_layout()
        
    def make_main_layout(self):   
        """Congifure layout for main window"""  

        # Configure main window apperance    
        self.setWindowTitle("READ: Real Estate Analytics Dashboard") # READ
        self.showMaximized()  

        # Tab Widget
        tabwidget = QTabWidget()

        ## Tab 1
        tab1_widget = self.make_layout_tab_1()
        tabwidget.addTab(tab1_widget, "House Search")
        
        ## Tab 2
        tab2_widget = self.make_layout_tab_2()
        tabwidget.addTab(tab2_widget, "Feature Engineering")
        
        ## Tab 3
        tab3_widget = self.make_layout_tab_3()
        tabwidget.addTab(tab3_widget, "Compare Models")

        self.setCentralWidget(tabwidget)

    def make_layout_tab_1(self):
        widget = QWidget()
        layout = QHBoxLayout()

        filterwidgets = self.add_block([self.query_widgets[0], 
                                        self.filter_widget,
                                        self.table_listings_widget, 
                                        self.clear_all_button_widgets[0]],  
                                        QVBoxLayout())
        
        layout.addWidget(self.geo_map_widget)
        layout.addWidget(filterwidgets)
        widget.setLayout(layout)
        

        return widget
 
    def make_layout_tab_2(self):
        widget = QWidget()
        layout = QVBoxLayout()
        
        h1 = self.add_block([self.feature_transform_widget, ButtonWidget('Store\nFeature').button], QVBoxLayout())
        h1 = self.add_block([self.hist_plot_widget, h1], QHBoxLayout())
        h1 = self.add_block([TitleWidget('Data driven features:', size = [600, 25]).title, self.filter_widget_tab2, h1], QVBoxLayout(), QtCore.Qt.AlignmentFlag.AlignTop | QtCore.Qt.AlignmentFlag.AlignLeft, size = [600])
        
        

        h2 = self.add_block([self.select_scatter_plot, ButtonWidget('Store\nFeature').button], QVBoxLayout())
        h2 = self.add_block([self.scatter_plot_widget, h2], QHBoxLayout(), size = [600])
        h2 = self.add_block([TitleWidget('Query driven features:', size = [600, 25]).title, self.query_widgets[1], h2], QVBoxLayout(), QtCore.Qt.AlignmentFlag.AlignTop | QtCore.Qt.AlignmentFlag.AlignLeft, size = [600])
        

        v = self.add_block([h1, h2], QHBoxLayout())
        v = self.add_block([v, self.table_listings_widget_2], QVBoxLayout(), size = [1200])

        layout.addWidget(v)
        widget.setLayout(layout)
        return widget
    
    def make_layout_tab_3(self):
        widget = QWidget()
        layout = QHBoxLayout()
        
        widget.setLayout(layout)

    def add_block(self, widgetlist = [], block_type = QVBoxLayout(), alignment_flag = QtCore.Qt.AlignmentFlag.AlignTop, size = None):
        widget = QWidget()
        layout = block_type
        for wid in widgetlist:
            layout.addWidget(wid, alignment=alignment_flag)
        widget.setLayout(layout)
        if size is not None:
            widget.setFixedWidth(size[0])
        return widget
        


        return widget
    
    def update_table(self):
        geo_map_model = QGeoMapModel(self.df_show)
        self.geo_map_widget.update_geo_map_model(geo_map_model)

    def update_geo_map(self):
        table_listings_model = TableListingsModel(self.df_show, self.images_dir_path)
        self.table_listings_widget.update_model(table_listings_model)
    
    def update_scatterplot(self):
        self.scatter_plot_widget.update_scatterplot(self.df_show['umap_x'], self.df_show['umap_y'] )

    def update(self):
        """
        Update/Refresh all the widgets
        """
        self.update_geo_map()
        self.update_table()
        self.update_scatterplot()

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
        new_df = self.df.copy()
        for tag, filter in filters['range'].items():
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

        for tag, filter in filters['combo'].items():
            if filter.Filter.currentText() != '':
                new_df = new_df[new_df[tag] == filter.Filter.currentText()]
        
        
        return new_df

if __name__ == '__main__':
    app = QApplication(sys.argv)
    main_window = MainWindow()
    main_window.show()
    sys.exit(app.exec())
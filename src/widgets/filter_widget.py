from PyQt6.QtWidgets import QWidget, QLineEdit, QComboBox, QHBoxLayout, QVBoxLayout, QLabel, QPushButton
from PyQt6.QtCore import QSize
from PyQt6 import QtCore
import numpy as np

class FilterWidget(QWidget):
    """ Build main filtering widget for dashboard

    """
    def __init__(self, minmaxfilters = ['price', 'area'], combolist =  ['Parking', 'Bedroom']):
        """ Init filtering widget for dashboard

        Args:
            minmaxfilters: list of range filters to include 
            combofilters: list of roll-down menu filters to include 
        
        TODO: Automate filtering for roll-down menues once these are implemented into the dataset    
        TODO: Find better system to organize drop down menu elements
        TODO: Clean-up Syntax 
        """

        super().__init__()

        # Initialize main widget
        self.widget = QWidget(self)
        vbox_layout = QVBoxLayout(self, spacing=2)

        #################################
        # Block 1  - Main Box Title
        #################################

        querylabel = QLabel(self)
        querylabel.setText('Filtering of Houses appearing in the Map:')
        querylabel.setStyleSheet("border: 0px;")
        
        vbox_layout.addWidget(querylabel)

        #################################
        # Block 2  - Range Filters (MinMax format)
        #################################

        self.minmaxfilters_dic = {}

        for filter in minmaxfilters:
            minmaxfilter = Layout(RangeFilter, filter, 'MinMax')
            self.minmaxfilters_dic[filter] = minmaxfilter
            vbox_layout.addWidget(minmaxfilter.widget)

        #################################
        # Block 3  - Combo Filters Title
        #################################
        querylabel = QLabel(self)
        querylabel.setText('Other filtering options:')
        querylabel.setStyleSheet("border: 0px;")
        vbox_layout.addWidget(querylabel)

        #################################
        # Block 4  - Combo Filters (Roll-down format)
        #
        # Requires spesification of drop-down menu elements
        #################################

        self.combofilters = QWidget(self)
        combo_layout = QHBoxLayout(self, spacing=2)

        """ Parking filter"""
        ## Drop down menu elements
        binary = [1, 0]
        binary_tags = ['yes', 'no']    

        self.parkingfilter = ComboFilter('Parking Available:', binary, binary_tags)
        combo_layout.addWidget(self.parkingfilter.widget)

        """ Bedroom filter """
        ## Drop down menu elements
        bedroms, bedrom_tags = self.counter(1, 5)
        
        self.bedroomfilter = ComboFilter('Bedroom Available:', bedroms, bedrom_tags)
        combo_layout.addWidget(self.bedroomfilter.widget)

        # Configure widget style
        self.combofilters.setLayout(combo_layout)
        self.combofilters.setStyleSheet("border: 0px;") 
        vbox_layout.addWidget(self.combofilters)


        #################################
        # Block 5  - Configure filtering button
        #
        # Controller for updating the plot
        #################################
        self.searchbutton = SearchWidget(self.minmaxfilters_dic)
        vbox_layout.addWidget(self.searchbutton.widget)

        # Configure widget style
        self.widget.setLayout(vbox_layout)
        self.widget.setStyleSheet("border: 2px solid darkgray; background-color: #f5f5f5;")

    def counter(self, N_min, N_max):
        numerical = np.arange(N_min, N_max + 1)
        tags = [str(i) for i in numerical]    
        return numerical, tags



class Layout(QWidget):
    """ Class for organizing layout of subwidgets
    
    """
    def __init__(self, Filter, name, method = 'MinMax'):
        super().__init__()
        self.Filter = Filter
        self.name = name 

        if method == 'MinMax':
            self.minmax_layout()
        else:
            print(method, ' layout method not implemented')

    def minmax_layout(self):
        ## Initialize widgets
        self.widget = QWidget(self)
        minmax_layout = QHBoxLayout(self, spacing=2)
    
        self.Max = RangeFilter('Max ' + self.name)
        minmax_layout.addWidget(self.Max.widget)

        self.Min = RangeFilter('Min ' + self.name)
        minmax_layout.addWidget(self.Min.widget)

        # Configure widget style
        self.widget.setLayout(minmax_layout)
        self.widget.setStyleSheet("border: 0px;") 

class RangeFilter(QWidget):
    """ Class for building singular Query-input style widgets
    
    """
    def __init__(self, name):
        super().__init__()
        self.name = name
        self.widget =  QWidget(self)
        widget_layout = QHBoxLayout(self, spacing=5)

        # Set box lable
        self.QueryLabel = QLabel(self)
        self.QueryLabel.setText(self.name)
        self.QueryLabel.setStyleSheet("border: 0px;")
        self.QueryLabel.setFixedSize(QSize(125, 25))  
        widget_layout.addWidget(self.QueryLabel)
        
        # Set query input box
        self.QueryText = QLineEdit(self)
        self.QueryText.setStyleSheet("border: 1px solid darkgray;")   
        widget_layout.addWidget(self.QueryText)

        # Combine widgets
        self.widget.setLayout(widget_layout )
        self.widget.setStyleSheet("border: 0px;")
        self.widget.setFixedSize(QSize(250 , 50))
        

class ComboFilter(QWidget):
    """ Class for building singular drop-down menu style widgets
    
    """
    def __init__(self, name, filter_list, filter_tags):
        super().__init__()
        self.name = name
        self.filter_list = filter_list #[0, 100, 500, 1000]
        self.filter_tages = filter_tags #
        self.widget = QWidget(self)
        layout = QVBoxLayout(self)

        label = QLabel(self.name)
        label.setStyleSheet("border: 0px;")
        self.Filter = QComboBox(self)
        self.Filter.setFixedSize(QSize(50, 25))  
        self.Filter.addItems(filter_tags)

        # Connect signals to the methods.
        self.Filter.activated.connect(self.check_index)
        self.Filter.activated.connect(self.current_text)
        self.Filter.activated.connect(self.current_text_via_index)
        self.Filter.activated.connect(self.current_count)
        
        layout.addWidget(label)
        layout.addWidget(self.Filter)
        
        self.widget.setLayout(layout)
        self.widget.setStyleSheet("border: 0px;")
        
    
    def check_index(self, index):
        cindex = self.Filter.currentIndex()

    def current_text(self, _): 
        ctext = self.Filter.currentText()
        print(self.label, ctext)

    def current_text_via_index(self, index):
        ctext = self.Filter.itemText(index) 

    def current_count(self, index):
        count = self.Filter.count()

class SearchWidget(QWidget):
    """ Class for building main filtering button and update plots
    """

    # Signal for Emitting the filtered data
    filtersApplied = QtCore.pyqtSignal(object, QWidget)

    def __init__(self, minmax_filters):
        super().__init__()
        self.filters = minmax_filters

        self.widget = QWidget(self)
        layout = QVBoxLayout(self)
        
        # Set search button 
        self.QueryButton = QPushButton('Filter', self)
        self.QueryButton.clicked.connect(self.clickMethod)
        self.QueryButton.setFixedSize(QSize(150, 25))  
        self.QueryButton.setStyleSheet("border: 1px solid darkgray;")      
        layout.addWidget(self.QueryButton)
        
        # Combine widgets
        self.setLayout(layout)
        self.setStyleSheet("border: 0px;")  

    def clickMethod(self):
        self.filtersApplied.emit(self.filters, self)

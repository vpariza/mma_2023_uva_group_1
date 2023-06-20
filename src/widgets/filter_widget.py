from PyQt6.QtWidgets import QWidget, QLineEdit, QComboBox, QHBoxLayout, QVBoxLayout, QLabel, QPushButton
from PyQt6.QtCore import QSize
from PyQt6 import QtCore
import numpy as np

class FilterWidget(QWidget):
    """ Build main filtering widget for dashboard

    """
    def __init__(self, minmaxfilters = ['price', 'area'], combolist =  ['Parking', 'Bedroom'], config = '1'):
        """ Init filtering widget for dashboard

        Args:
            minmaxfilters: list of range filters to include 
            combofilters: list of roll-down menu filters to include 
        
        TODO: Automate filtering for roll-down menues once these are implemented into the dataset    
        TODO: Find better system to organize drop down menu elements
        TODO: Clean-up Syntax 
        """
        super().__init__()

        self.minmaxfilters = minmaxfilters


        #################################
        # Block 1  - Main Box Title
        #################################

        self.querylabel = QLabel(self)
        self.querylabel.setText('Filtering of Houses appearing in the Map:')

        #################################
        # Block 2  - Range Filters (MinMax format)
        #################################

        self.minmaxfilters_dic = {}
        for filter in self.minmaxfilters:
            minmaxfilter = Layout(RangeFilter, filter, 'MinMax')
            self.minmaxfilters_dic[filter] = minmaxfilter
            
        #################################
        # Block 3  - Combo Filters Title
        #################################

        self.querylabel2 = QLabel(self)
        self.querylabel2.setText('Other filtering options:')
        
        #################################
        # Block 4  - Combo Filters (Roll-down format)
        #
        # Requires spesification of drop-down menu elements
        #################################

        """ Parking filter"""
        ## Drop down menu elements
        binary = [1, 0]
        binary_tags = ['yes', 'no']    

        self.parkingfilter = ComboFilter('Parking Available:', binary, binary_tags)
        

        """ Bedroom filter """
        ## Drop down menu elements
        bedroms, bedrom_tags = self.counter(1, 5)
        
        self.bedroomfilter = ComboFilter('Bedroom Available:', bedroms, bedrom_tags)
        


        #################################
        # Block 5  - Configure filtering button
        #
        # Controller for updating the plot
        #################################
        self.searchbutton = SearchWidget(self.minmaxfilters_dic)
        

        # Configure widget style
        self.make_layout(config)
        
    def make_layout(self, config = '1'):
        # Initialize main widget
        hbox_layout = QHBoxLayout(self, spacing=2)
        vbox_layout = QVBoxLayout(self, spacing=2)
        if config == '1':
            vbox_layout.addWidget(self.querylabel)
            for filter in self.minmaxfilters:
                vbox_layout.addWidget(self.minmaxfilters_dic[filter])
            vbox_layout.addWidget(self.querylabel2)
            self.combofilters = QWidget(self)
            combo_layout = QHBoxLayout(self, spacing=2)
            # Configure widget style
            combo_layout.addWidget(self.parkingfilter)
            combo_layout.addWidget(self.bedroomfilter)
            self.combofilters.setLayout(combo_layout)
            vbox_layout.addWidget(self.combofilters)
            vbox_layout.addWidget(self.searchbutton)
            self.setLayout(vbox_layout)
            self.setStyleSheet(" background-color: #f5f5f5;")

        if config == '2':
            vbox_layout.addWidget(self.querylabel)
            for filter in self.minmaxfilters:
                vbox_layout.addWidget(self.minmaxfilters_dic[filter])
            
            vbox = QWidget(self)
            vbox.setLayout(vbox_layout)
            hbox = QWidget(self)
            hbox_layout.addWidget(vbox)
            
            combofilters = QWidget(self)
            combo_layout = QHBoxLayout(self, spacing=2)
            
            # Configure widget style
            combo_layout.addWidget(self.parkingfilter)
            combo_layout.addWidget(self.bedroomfilter)
            combofilters.setLayout(combo_layout)
            hbox_layout.addWidget(combofilters)
            
            hbox.setLayout(hbox_layout)
            #vbox_layout.addWidget(self.querylabel2)
            button_layout = QVBoxLayout(self, spacing=2)
            button_layout.addWidget(hbox)
            button_layout.addWidget(self.searchbutton, alignment=QtCore.Qt.AlignmentFlag.AlignCenter)
            
            self.setLayout(button_layout)
            self.setStyleSheet("background-color: #f5f5f5;")
            
            
            

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
        minmax_layout = QHBoxLayout(self, spacing=2)
    
        self.Max = RangeFilter('Max ' + self.name)
        minmax_layout.addWidget(self.Max)

        self.Min = RangeFilter('Min ' + self.name)
        minmax_layout.addWidget(self.Min)

        # Configure widget style
        self.setLayout(minmax_layout)
        self.setStyleSheet("border: 0px;") 

class RangeFilter(QWidget):
    """ Class for building singular Query-input style widgets
    
    """
    def __init__(self, name):
        super().__init__()
        self.name = name
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
        self.setLayout(widget_layout )
        self.setStyleSheet("border: 0px;")
        self.setFixedSize(QSize(250 , 50))
        

class ComboFilter(QWidget):
    """ Class for building singular drop-down menu style widgets
    
    """
    def __init__(self, name, filter_list = [], filter_tags = []):
        super().__init__()
        self.name = name
        self.filter_list = filter_list #[0, 100, 500, 1000]
        self.filter_tages = filter_tags #
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
        
        self.setLayout(layout)
        self.setStyleSheet("border: 0px;")
        
    
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

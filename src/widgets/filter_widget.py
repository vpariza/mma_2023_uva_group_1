from PyQt6.QtWidgets import QWidget, QLineEdit, QComboBox, QHBoxLayout, QVBoxLayout, QLabel, QPushButton
from PyQt6.QtCore import QSize
from PyQt6 import QtCore
import numpy as np

class FilterWidget(QWidget):
    """ Build main filtering widget for dashboard

    """
    def __init__(self, minmaxfilters = [], combofilters =  {}, config = '1'):
        """ Init filtering widget for dashboard

        Args:
            minmaxfilters: list of range filters to include 
            combofilters: list of roll-down menu filters to include 
        
        TODO: Automate filtering for roll-down menues once these are implemented into the dataset    
        """
        super().__init__()

        self.minmaxfilters = minmaxfilters
        self.combofilters = combofilters


        # Block 1  - Main Box Title
        self.boxtitle = QLabel()
        self.boxtitle.setText('Filtering of Houses appearing in the Map:')

        # Block 2  - Range Filters (MinMax format)
        self.minmaxfilters_widgets = {}
        for filter in self.minmaxfilters:
            minmaxfilter = MinMaxWidget(RangeFilter, filter)
            self.minmaxfilters_widgets[filter] = minmaxfilter
        
        # Block 3  - Combo Filters Title
        self.querylabel = QLabel()
        self.querylabel.setText('Other filtering options:')
        
        # Block 4  - Combo Filters (Roll-down format)
        self.combofilters_widgets = {}
        for filter in self.combofilters:
            combofilter = ComboFilter(filter, self.combofilters[filter])
            self.combofilters_widgets[filter] = combofilter
        
        # Block 5  - Configure filtering button
        self.filters = {}

        self.filters.update({'combo': self.combofilters_widgets})
        self.filters.update({'range': self.minmaxfilters_widgets})
        
        self.searchbutton = SearchWidget(self.filters)

        # Configure widget style
        self.make_main_layout(config)
        
    def make_main_layout(self, config = '1'):
        hblock = QWidget()
        hblock_layout = QHBoxLayout()
        vblock = QWidget()
        vblock_layout = QVBoxLayout()
        vblock2 = QWidget()
        vblock2_layout = QVBoxLayout()

        if config == '1':
            widget_layout = QVBoxLayout()
            widget_layout.addWidget(self.boxtitle)
            # Add minmax filters
            for filter in self.minmaxfilters_widgets:
                widget_layout.addWidget(self.minmaxfilters_widgets[filter])
            widget_layout.addWidget(self.querylabel)
            # Add drop-down filters
            for filter in self.combofilters_widgets:
                hblock_layout.addWidget(self.combofilters_widgets[filter])
            hblock.setLayout(hblock_layout)
            
        if config == '2':
            widget_layout = QVBoxLayout()
            # Add minmax filters
            vblock_layout.addWidget(self.querylabel)
            for filter in self.minmaxfilters_widgets:
                vblock_layout.addWidget(self.minmaxfilters_widgets[filter])
            vblock.setLayout(vblock_layout)
            hblock_layout.addWidget(vblock)
            # Add drop-down filters
            for filter in self.combofilters_widgets:
                vblock2_layout.addWidget(self.combofilters_widgets[filter])
            vblock2.setLayout(vblock2_layout)
            hblock_layout.addWidget(vblock2)
            hblock.setLayout(hblock_layout)
            
        widget_layout.addWidget(hblock)
        widget_layout.addWidget(self.searchbutton, alignment=QtCore.Qt.AlignmentFlag.AlignCenter)
        self.setLayout(widget_layout)
        self.setStyleSheet("background-color: #f5f5f5;")


class MinMaxWidget(QWidget):
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
        self.QueryLabel.setText(self.name.replace("_", " "))
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
    def __init__(self, name, filter_tags = []):
        super().__init__()
        self.name = name
        self.filter_tags = filter_tags #
        self.filter_list = np.arange(0, len(self.filter_tags))
        layout = QVBoxLayout(self)

        self.label = QLabel(self.name)
        self.label.setStyleSheet("border: 0px;")
        self.Filter = QComboBox(self)
        self.Filter.setFixedSize(QSize(50, 25))  
        self.Filter.addItems(filter_tags)

        # Connect signals to the methods.
        self.Filter.activated.connect(self.check_index)
        self.Filter.activated.connect(self.current_text)
        self.Filter.activated.connect(self.current_text_via_index)
        self.Filter.activated.connect(self.current_count)
        
        layout.addWidget(self.label)
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

    def __init__(self, filters):
        super().__init__()
        self.filters = filters
        
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

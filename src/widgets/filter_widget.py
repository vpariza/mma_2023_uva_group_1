from PyQt6.QtWidgets import QWidget, QLineEdit, QComboBox, QHBoxLayout, QVBoxLayout, QLabel, QPushButton
from PyQt6.QtCore import QSize
from PyQt6 import QtCore
import numpy as np

class FilterWidget(QWidget):
    """ Build main filtering widget for dashboard

    """
    
    def __init__(self, df, minmaxfilters = [], combofilters =  {}, placeholdertext = {}, config = '1'):    
        """ Init filtering widget for dashboard

        Args:
            minmaxfilters: list of range filters to include 
            combofilters: list of roll-down menu filters to include 
        
        TODO: Automate filtering for roll-down menues once these are implemented into the dataset    
        """
        super().__init__()
        self.df = df
        self.minmaxfilters = minmaxfilters
        self.combofilters = combofilters


        # Block 1  - Main Box Title
        self.boxtitle = QLabel()
        self.boxtitle.setText('Filtering of Houses appearing in the Map:')

        # Block 2  - Range Filters (MinMax format)
        self.minmaxfilters_widgets = {}
        for filter in range(len(self.minmaxfilters)):
            minmaxfilter = MinMaxWidget(RangeFilter, self.minmaxfilters[filter], placeholdertext, filter = self.minmaxfilters[filter])
            self.minmaxfilters_widgets[self.minmaxfilters[filter]] = minmaxfilter

        # Block 3  - Combo Filters Title
        self.querylabel = QLabel()
        self.querylabel.setText('Other filtering options:')
        
        # Block 4  - Combo Filters (Roll-down format)
        self.combofilters_widgets = {}
        for filter in self.combofilters:
            try:
                options = [str(item) for item in np.unique(self.df[filter].values)]
            except TypeError:
                options = [str(item) for item in set(self.df[filter].values)]
            combofilter = ComboFilter(filter, options)
            self.combofilters_widgets[filter] = combofilter
        
        # Block 5  - Configure filtering button
        self.filters = {}

        self.filters.update({'combo': self.combofilters_widgets})
        self.filters.update({'range': self.minmaxfilters_widgets})
        
        self.searchbutton = SearchWidget(self.filters)
        self._clear_all_button = ClearAllWidget(self.filters)
        
        # Configure widget style
        self.make_main_layout(config)

    def get_filters(self):
        return self.filters
    
    def clickmethod_clearall(self):
        self.buttonClearAll.emit(self)
    

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
        button_layout = QHBoxLayout()
        button_layout.addWidget(self.searchbutton, alignment=QtCore.Qt.AlignmentFlag.AlignCenter)
        button_layout.addWidget(self._clear_all_button, alignment=QtCore.Qt.AlignmentFlag.AlignCenter)
        widget_layout.addLayout(button_layout)
        self.setLayout(widget_layout)
        self.setStyleSheet("background-color: #f5f5f5;")


class MinMaxWidget(QWidget):
    """ Class for organizing layout of subwidgets
    
    """
    def __init__(self, Filter, name, placeholdertext = 'select', method = 'MinMax', filter = ''):
        super().__init__()
        self.Filter = Filter
        self.name = name 

        if placeholdertext != 'select':
            self.placeholdertext_min = 'min:  ' + str(placeholdertext['min_' + filter])
            self.placeholdertext_max = 'max:  ' + str(placeholdertext['max_' + filter])
        else:
            self.placeholdertext_min = placeholdertext
            self.placeholdertext_max = placeholdertext
        
        if method == 'MinMax':
            self.minmax_layout()
        else:
            # print(method, ' layout method not implemented')
            pass

    def minmax_layout(self):
        ## Initialize widgets
        minmax_layout = QHBoxLayout(self, spacing=2)

        self.Min = RangeFilter('Min ' + self.name, self.placeholdertext_min)
        minmax_layout.addWidget(self.Min)

        self.Max = RangeFilter('Max ' + self.name, self.placeholdertext_max)
        minmax_layout.addWidget(self.Max)

        # Configure widget style
        self.setLayout(minmax_layout)
        self.setStyleSheet("border: 0px;") 

class RangeFilter(QWidget):
    """ Class for building singular Query-input style widgets
    
    """
    def __init__(self, name, placeholdertext):
        super().__init__()
        self.name = name
        self.placeholdertext = placeholdertext
        self.layout = QVBoxLayout(self, spacing=5)

        # Set box lable
        self.QueryLabel = QLabel(self)
        self.QueryLabel.setText(self.name.replace("_", " "))
        self.QueryLabel.setStyleSheet("border: 0px;")
        #self.QueryLabel.setFixedSize(QSize(125, 25))  
        self.layout.addWidget(self.QueryLabel)
        
        # Set query input box
        self.QueryText = QLineEdit(self)
        self.QueryText.setStyleSheet("border: 1px solid darkgray;  background-color: white;")   
        self.layout.addWidget(self.QueryText)
        self.QueryText.setPlaceholderText(placeholdertext)

        # Combine widgets
        self.setLayout(self.layout)
        self.setStyleSheet("border: 0px;")
        #self.setFixedSize(QSize(250 , 50))
    
    def resetRangeFilter(self, new_placeholdertext):
        self.QueryText.clear()
        self.QueryText.setPlaceholderText(new_placeholdertext)



class ComboFilter(QWidget):
    """ Class for building singular drop-down menu style widgets
    
    """
    def __init__(self, name, filter_tags = []):
        super().__init__()
        self.name = name
        self.filter_tags = filter_tags #
        self.filter_list = np.arange(0, len(self.filter_tags))
        layout = QVBoxLayout(self)

        if self.name == 'bedrooms':
            self.label = QLabel('min # bedrooms')
        elif self.name == 'number_of_rooms':
            self.label = QLabel('min # rooms')
        else:
            self.label = QLabel(self.name.replace("_", " "))

        self.label.setStyleSheet("border: 0px;")
        self.Filter = QComboBox(self)
        self.Filter.setMaximumWidth(100)  
        self.Filter.addItems(filter_tags)
        self.Filter.setCurrentIndex(-1)
        self.Filter.setPlaceholderText('Select')
        self.Filter.setStyleSheet('background-color: white;')

        
        layout.addWidget(self.label)
        layout.addWidget(self.Filter)
        
        self.setLayout(layout)
        self.setStyleSheet("border: 0px;")

    def resetComboBoxes(self):
        for index in range(self.layout().count()):
            widget = self.layout().itemAt(index).widget()
            if isinstance(widget, QComboBox):
                widget.setCurrentIndex(-1)
                widget.setPlaceholderText('Select')



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
        #self.QueryButton.setFixedSize(QSize(150, 25))  
        layout.addWidget(self.QueryButton)
        
        # Combine widgets
        self.setLayout(layout)

    def clickMethod(self):
        self.filtersApplied.emit(self.filters, self)


class ClearAllWidget(QWidget):
    """ Class for building main filtering button and update plots
    """

    # Signal for Emitting the filtered data
    buttonClearAll = QtCore.pyqtSignal(object, QWidget)

    def __init__(self, filters):
        super().__init__()
        self.filters = filters
        
        layout = QVBoxLayout(self)
        
        # Set search button 
        self.QueryButton = QPushButton('Clear all', self)
        self.QueryButton.clicked.connect(self.clickMethod)
        #self.QueryButton.setFixedSize(QSize(150, 25))  
        layout.addWidget(self.QueryButton)
        
        # Combine widgets
        self.setLayout(layout)

    def clickMethod(self):
        self.buttonClearAll.emit(self.filters, self)

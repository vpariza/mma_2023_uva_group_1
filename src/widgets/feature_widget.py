from PyQt6.QtWidgets import QWidget, QLineEdit, QComboBox, QHBoxLayout, QVBoxLayout, QLabel, QPushButton, QGridLayout, QCheckBox
from PyQt6.QtCore import QSize
from PyQt6.QtGui import QFont
from PyQt6 import QtCore
from PyQt6.QtCore import Qt
import numpy as np



class FeatureBoxWidget(QWidget):
    """ Build custom title widget

    """
    def __init__(self, elements = ['option1', 'option2'], title_text = 'Title'):
        super().__init__()
        self.elements = elements
        box_list = []
        layout = QGridLayout()

        title = QLabel()
        title.setText(title_text)
        layout.addWidget(title)
        
        #layout.setContentsMargins(50, 80, 50, 80)
 
        for element in self.elements:
            check = QCheckBox(element)
            check.toggled.connect(self.showDetails)
            layout.addWidget(check)
            box_list.append(check)

        self.setLayout(layout)
 
      
 
    def showDetails(self):
        print("Selected: ", self.sender().isChecked(),
              "  Name: ", self.sender().text())


from PyQt6.QtWidgets import QWidget, QListWidget, QLineEdit, QComboBox, QHBoxLayout, QVBoxLayout, QLabel, QPushButton, QGridLayout, QCheckBox
from PyQt6.QtCore import QSize
from PyQt6.QtGui import QFont
from PyQt6 import QtCore
from PyQt6.QtCore import Qt
import numpy as np

class ModelBoxWidget(QWidget):
    """ Build custom title widget

    """
    def __init__(self, elements = ['option1', 'option2'], title_text = 'Title'):
        super().__init__()
        self.elements = elements
        box_list = []
        layout = QVBoxLayout()
        

        title = QLabel()
        title.setText(title_text)
        layout.addWidget(title)
        
        #layout.setContentsMargins(50, 80, 50, 80)
        list_widget = QListWidget()

        for element in self.elements:
            list_widget.addItem(element)
            
        # Set the selection mode to SingleSelection
        list_widget.setSelectionMode(QListWidget.SelectionMode.SingleSelection)

        # Connect the itemSelectionChanged signal to a slot
        list_widget.itemSelectionChanged.connect(self.itemSelectionChanged)

        layout.addWidget(list_widget) 
        self.setLayout(layout)


    def itemSelectionChanged(self):
        list_widget = self.list_widget.findChild(QListWidget)
        selected_items = list_widget.selectedItems()
        if selected_items:
            selected_item = selected_items[0]
            print("Selected:", selected_item.text())

            # Highlight the selected item
            list_widget.setCurrentItem(selected_item)

    
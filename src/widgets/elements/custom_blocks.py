from PyQt6.QtWidgets import QWidget, QLineEdit, QComboBox, QHBoxLayout, QVBoxLayout, QLabel, QPushButton, QGridLayout, QCheckBox
from PyQt6.QtCore import QSize
from PyQt6.QtGui import QFont
from PyQt6 import QtCore
from PyQt6.QtCore import Qt
import numpy as np

class ButtonWidget(QWidget):
    """ Build custom title widget

    """
    def __init__(self, button_title, size = None):
        super().__init__()
        self.button_title = button_title
        self.size = size
        

        self.button = QPushButton(self.button_title)
        #button1.clicked.connect(self.clickMethod)
        if self.size is not None:
            self.button.setFixedSize(QSize(self.size[0], self.size[1]))  
        self.button.setStyleSheet("border: 1px solid darkgray;")   

class TitleWidget(QWidget):
    """ Build custom title widget

    """
    def __init__(self, title_text, size = None):
        super().__init__()
        self.size = size

        self.title = QLabel()
        self.title.setText(title_text)
        font = QFont()
        
        font.setBold(True)
        font.setPointSize(16)
        self.title.setFont(font)
        self.title.setAlignment(Qt.AlignmentFlag.AlignTop | Qt.AlignmentFlag.AlignLeft)
        


        if self.size is not None:
            self.title.setFixedSize(QSize(self.size[0], self.size[1])) 

         

class CheckBoxWidget(QWidget):
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

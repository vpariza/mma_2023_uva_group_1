from PyQt6.QtWidgets import QWidget, QLineEdit, QComboBox, QHBoxLayout, QVBoxLayout, QLabel, QPushButton, QGridLayout, QCheckBox
from PyQt6.QtCore import QSize
from PyQt6.QtGui import QFont
from PyQt6 import QtCore
import numpy as np

class ButtonWidget(QWidget):
    """ Build custom title widget

    """
    def __init__(self, button_title, size = False, w = 150, h = 25):
        super().__init__()
        self.button_title = button_title
        self.size = size
        self.w, self.h = w, h

        self.button = QPushButton(self.button_title)
        #button1.clicked.connect(self.clickMethod)
        if self.size:
            self.button.setFixedSize(QSize(w, h))  
        self.button.setStyleSheet("border: 1px solid darkgray;")   

class TitleWidget(QWidget):
    """ Build custom title widget

    """
    def __init__(self, title_text, size = False):
        super().__init__()

        self.title = QLabel()
        self.title.setText(title_text)
        font = QFont()
        font.setBold(True)
        self.title.setFont(font)

        #if size:
            #button.setFixedSize(QSize(150, 25))  
         

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

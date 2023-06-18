from PyQt6.QtWidgets import QWidget, QLineEdit, QLabel, QPushButton, QVBoxLayout, QHBoxLayout
from PyQt6.QtCore import QSize
from PyQt6 import QtCore


class QueryWidget(QWidget):
    """ Build main query input widget for free-text search
    Can also be applied to build additional query widgets
    
    """
    # Signal for Emitting the requrested query text
    querySubmitted = QtCore.pyqtSignal(str, QWidget)

    def __init__(self):
        
        super().__init__()
        ## Initialize widgets
        self.vbox = QWidget(self)
        vbox_layout = QVBoxLayout(self, spacing=2)
        hbox =  QWidget(self)
        hbox_layout = QHBoxLayout(self, spacing=5)

        # Set box lable
        self.QueryLabel = QLabel(self)
        self.QueryLabel.setText('Free-text Query Search for House Descriptions:')
        self.QueryLabel.setStyleSheet("border: 0px;")
        vbox_layout.addWidget(self.QueryLabel)
        
        # Set query input box
        self.QueryText = QLineEdit(self)
        self.QueryText.setFixedSize(QSize(250, 25))
        self.QueryText.setStyleSheet("border: 1px solid darkgray;")   
        hbox_layout.addWidget(self.QueryText)
        
        # Set search button 
        self.QueryButton = QPushButton('Search Query', self)
        self.QueryButton.clicked.connect(self.clickMethod)
        self.QueryButton.setFixedSize(QSize(150, 25))  
        self.QueryButton.setStyleSheet("border: 1px solid darkgray;")      
        hbox_layout.addWidget(self.QueryButton)
        
        # Combine widgets
        hbox.setLayout(hbox_layout)
        hbox.setStyleSheet("border: 0px solid darkgray; background-color: #f5f5f5;")
        vbox_layout.addWidget(hbox)
        self.vbox.setLayout(vbox_layout)
        # set widget style
        self.vbox.resize(450, 100)
        self.vbox.setStyleSheet("border: 2px solid darkgray; background-color: #f5f5f5;")

    def clickMethod(self):
        self.querySubmitted.emit(self.QueryText.text(), self)
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
        vbox_layout = QVBoxLayout()
        hbox =  QWidget()
        hbox_layout = QHBoxLayout()

        # Set box lable
        self.QueryLabel = QLabel(self)
        self.QueryLabel.setText('Free-text Query Search for House Descriptions:')
        self.QueryLabel.setStyleSheet("border: 0px;")
        vbox_layout.addWidget(self.QueryLabel)
        
        # Set query input box
        self.QueryText = QLineEdit(self)
        self.QueryText.setPlaceholderText("Ex: 'red brick house'")
        self.QueryText.setFixedSize(QSize(250, 25))
        self.QueryText.setStyleSheet("border: 1px solid darkgray; background-color: white;")  
        hbox_layout.addWidget(self.QueryText)
        
        # Set search button 
        self.QueryButton = QPushButton('Search Query', self)
        self.QueryButton.clicked.connect(self.clickMethod)
        self.QueryButton.setFixedSize(QSize(150, 25))  
        hbox_layout.addWidget(self.QueryButton)
        
        # Combine widgets
        hbox.setLayout(hbox_layout)
        vbox_layout.addWidget(hbox)
        self.setLayout(vbox_layout)
        # set widget style
        self.resize(450, 100)
        
    def clickMethod(self):
        self.querySubmitted.emit(self.QueryText.text(), self)
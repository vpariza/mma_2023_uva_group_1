import sys
# from PyQt6.QtCore import *
# from PyQt6.QtGui import *

import matplotlib
matplotlib.use('QtAgg')
import numpy as np

from PyQt6 import QtCore, QtWidgets

from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QVBoxLayout, QPushButton, QFormLayout, QLineEdit, QInputDialog, QApplication
)



class InputTextDialog(QWidget):
   def __init__(self, parent = None):
      super(InputTextDialog, self).__init__(parent)
		
      layout = QFormLayout()
		
      self.btn1 = QPushButton("get name")
      self.btn1.clicked.connect(self.gettext)
		
      layout.addWidget(self.btn1)
      self.setLayout(layout)
      self.setWindowTitle("Input Dialog demo")
			
   def gettext(self):
        text, ok = QInputDialog.getText(self, 'Text Input Dialog', 'Enter your name:')
        if ok:
            print(text)
			
def main(): 
   app = QApplication(sys.argv)
   ex = InputTextDialog()
   ex.show()
   sys.exit(app.exec())
	
if __name__ == '__main__':
   main()
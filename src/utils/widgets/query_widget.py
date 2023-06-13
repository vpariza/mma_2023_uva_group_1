from PyQt6.QtWidgets import QWidget, QLineEdit, QLabel, QPushButton



class QueryWidget(QWidget):
    def __init__(self):
        super().__init__()
        
        self.QueryLabel = QLabel(self)
        self.QueryLabel.setText('Query:')
        self.Query = QLineEdit(self)

        self.Query.move(20, 40)
        self.Query.resize(200, 32)
        self.QueryLabel.move(20, 20)

        pybutton = QPushButton('Search', self)
        pybutton.clicked.connect(self.clickMethod)
        pybutton.resize(200,32)
        pybutton.move(20, 80)        

    def clickMethod(self):
        print('Query: ' + self.Query.text())
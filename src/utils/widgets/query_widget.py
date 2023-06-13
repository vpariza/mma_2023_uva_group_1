from PyQt6.QtWidgets import QWidget, QLineEdit, QLabel, QPushButton



class QueryWidget(QWidget):
    def __init__(self):
        super().__init__()
        
        self.nameLabel = QLabel(self)
        self.nameLabel.setText('Query:')
        self.query = QLineEdit(self)

        self.query.move(20, 40)
        self.query.resize(200, 32)
        self.nameLabel.move(20, 20)

        pybutton = QPushButton('Search', self)
        pybutton.clicked.connect(self.clickMethod)
        pybutton.resize(200,32)
        pybutton.move(20, 80)        

    def clickMethod(self):
        print('Query: ' + self.query.text())
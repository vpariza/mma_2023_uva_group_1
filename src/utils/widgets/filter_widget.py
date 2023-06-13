from PyQt6.QtWidgets import QWidget, QComboBox, QVBoxLayout, QLabel
from PyQt6.QtGui import QFont

class FilterWidget(QWidget):
    def __init__(self):
        super().__init__()

        boldfont = QFont()
        boldfont.setBold(True)
        
        widget = QWidget(self)
        layout = QVBoxLayout(self)
        title_label = QLabel("Filter Data")
        title_label.setFont(boldfont)
        
        layout.addWidget(title_label)
        
        self.pricefilter = PriceFilter()
        layout.addWidget(self.pricefilter.widget)
        
        widget.setLayout(layout)
        widget.move(20, 0)
        


class PriceFilter(QWidget):
    def __init__(self):
        super().__init__()

        self.widget = QWidget(self)
        layout = QVBoxLayout(self)

        price_label = QLabel("Select price range:")
        self.PriceFilter = QComboBox(self)

        self.PriceFilter.addItems(['0k-100k', '100k-500k', '500k-1000k'])
        self.PriceFilter.setPlaceholderText("Price Range")

        # Connect signals to the methods.
        self.PriceFilter.activated.connect(self.check_index)
        self.PriceFilter.activated.connect(self.current_text)
        self.PriceFilter.activated.connect(self.current_text_via_index)
        self.PriceFilter.activated.connect(self.current_count)
        self.PriceFilter.activated.connect(self.update_data)

        layout.addWidget(price_label)
        layout.addWidget(self.PriceFilter)
        
        self.widget.setLayout(layout)
        self.widget.move(20, 0)
    
    def check_index(self, index):
        cindex = self.PriceFilter.currentIndex()
        print(f"Index signal: {index}, currentIndex {cindex}")

    def current_text(self, _): 
        ctext = self.PriceFilter.currentText()
        print("Current Price Range", ctext)

    def current_text_via_index(self, index):
        ctext = self.PriceFilter.itemText(index) 

    def current_count(self, index):
        count = self.PriceFilter.count()
        print(f"Current Price Index {index+1}/{count}")

    def update_data(self, df):
        return 0
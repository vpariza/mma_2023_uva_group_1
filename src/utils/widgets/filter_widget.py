from PyQt6.QtWidgets import QWidget, QComboBox, QVBoxLayout, QLabel
from PyQt6.QtGui import QFont

class FilterWidget(QWidget):
    def __init__(self, df):
        super().__init__()
        self.df = df

        boldfont = QFont()
        boldfont.setBold(True)
        
        widget = QWidget(self)
        layout = QVBoxLayout(self)
        title_label = QLabel("Filter Data")
        title_label.setFont(boldfont)
        
        layout.addWidget(title_label)
        
        self.pricefilter = PriceFilter(self.df)
        layout.addWidget(self.pricefilter.widget)
        
        widget.setLayout(layout)
        widget.move(20, 0)
        


class PriceFilter(QWidget):
    def __init__(self, df):
        super().__init__()
        self.df = df
        self.price_list = [0, 100, 500, 1000]
        

        self.widget = QWidget(self)
        layout = QVBoxLayout(self)

        price_label = QLabel("Select price range:")
        self.MaxFilter = QComboBox(self)
        self.MaxFilter.addItems([str(price) + 'k' for price in self.price_list])
        self.MaxFilter.setPlaceholderText("Max Price")

        # Connect signals to the methods.
        self.MaxFilter.activated.connect(self.check_index)
        self.MaxFilter.activated.connect(self.current_text)
        self.MaxFilter.activated.connect(self.current_text_via_index)
        self.MaxFilter.activated.connect(self.current_count)
        self.MaxFilter.currentTextChanged.connect(self.on_combobox_changed)


        layout.addWidget(price_label)
        layout.addWidget(self.MaxFilter)
        
        self.widget.setLayout(layout)
        self.widget.move(20, 0)
    
    def check_index(self, index):
        cindex = self.MaxFilter.currentIndex()
        print(f"Index signal: {index}, currentIndex {cindex}")

    def current_text(self, _): 
        ctext = self.MaxFilter.currentText()
        print("Current Max Price", ctext)

    def current_text_via_index(self, index):
        ctext = self.MaxFilter.itemText(index) 

    def current_count(self, index):
        count = self.MaxFilter.count()
        print(f"Current Price Index {index+1}/{count}")

    def on_combobox_changed(self):
        new_df = self.df[self.df['price'] < 1e3*self.price_list[self.MaxFilter.currentIndex()]]
        
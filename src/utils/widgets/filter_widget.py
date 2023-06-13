from PyQt6.QtWidgets import QWidget, QComboBox, QVBoxLayout, QLabel
from PyQt6.QtGui import QFont

class FilterWidget(QWidget):
    def __init__(self, df, scatter_plot_widget):
        super().__init__()
        self.df = df

        boldfont = QFont()
        boldfont.setBold(True)

        # Set filterdict
        self.filters = {'Max Price': -1,
                        'Min Price': 0
                        } 

        # Configure price ranges
        price_list = [0, 100, 500, 1000]
        price_tags = [str(value) + 'k' for value in price_list]    

        # Construct filters
        self.pricefilter_max = ComboFilter('Max Price', self.df, scatter_plot_widget, price_list, price_tags, "Select max price:", self.filters)
        #self.pricefilter_min = ComboFilter('Min Price', self.df, scatter_plot_widget, price_list, price_tags, "Select min price:", filters)

        # Configure layout of widgets
        widget = QWidget(self)
        layout = QVBoxLayout(self)
        title_label = QLabel("Filter Data")
        title_label.setFont(boldfont)
        
        layout.addWidget(title_label)
        layout.addWidget(self.pricefilter_max.widget)
        widget.setLayout(layout)
        widget.move(20, 0)
        
    def make_layout():
        pass


class ComboFilter(QWidget):
    def __init__(self, name, df, scatter_plot_widget, filter_list, filter_tags, label, filters):
        super().__init__()
        self.name = name
        self.df = df
        self.label = label
        self.filters = filters
        self.scatterplot = scatter_plot_widget
        self.filter_list = filter_list #[0, 100, 500, 1000]
        self.filter_tages = filter_tags #
        self.widget = QWidget(self)
        layout = QVBoxLayout(self)

        label = QLabel(label)
        self.Filter = QComboBox(self)
        self.Filter.addItems(filter_tags)

        # Connect signals to the methods.
        self.Filter.activated.connect(self.check_index)
        self.Filter.activated.connect(self.current_text)
        self.Filter.activated.connect(self.current_text_via_index)
        self.Filter.activated.connect(self.current_count)
        self.Filter.currentTextChanged.connect(self.on_combobox_changed)

        layout.addWidget(label)
        layout.addWidget(self.Filter)
        
        self.widget.setLayout(layout)
        self.widget.move(20, 0)
    
    def check_index(self, index):
        cindex = self.Filter.currentIndex()
        #print(f"Index signal: {index}, currentIndex {cindex}")

    def current_text(self, _): 
        ctext = self.Filter.currentText()
        print(self.label, ctext)

    def current_text_via_index(self, index):
        ctext = self.Filter.itemText(index) 

    def current_count(self, index):
        count = self.Filter.count()
        #print(f"Current Price Index {index+1}/{count}")

    def on_combobox_changed(self):
        
        new_df = self.df[self.df['price'] < 1e3*self.filter_list[self.Filter.currentIndex()]]
        x = new_df['umap_x'].values
        y = new_df['umap_y'].values
        self.scatterplot.update_scatterplot(x, y)
import sys
from PyQt6.QtCore import QSize, Qt
from PyQt6.QtWidgets import QApplication, QMainWindow, QLabel, QVBoxLayout, QHBoxLayout, QWidget
from src.utils.widgets.query_widget import QueryWidget
from src.utils.widgets.plot_widget import PlotWidget
from src.utils.widgets.filter_widget import FilterWidget
from src.utils.preprocessing import Preprocessing


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        #load data using the config file 'config.ini'
        preprocessing = Preprocessing()
        self.config, self.tags, self.points, self.img_paths = preprocessing.load_data()

        ## inialize widgets
        self.query_widgets = [QueryWidget() for i in range(2)]
        self.scatter_plot_widget = PlotWidget(self.points, self.config)
        self.filter_widget = FilterWidget()
        
        ## set up main window 
        self.make_layout()
        
        

    
    def make_layout(self):   
        """Congifure layout for main window"""  

        # Configure main window apperance    
        self.setWindowTitle("Scatterplot Dashboard")
        self.setMinimumSize(QSize(1000, 500))

        ## Combine widgets in right column
        vbox = QWidget()
        vbox_layout = QVBoxLayout(self, spacing=0)

        vbox_layout.addWidget(self.query_widgets[0])
        #vbox_layout.addWidget(self.query_widgets[1])
        vbox_layout.addWidget(self.filter_widget)
        
        vbox.setLayout(vbox_layout)
   
        ## set the layout of the main window
        main_widget = QWidget()
        main_layout = QHBoxLayout(self, spacing=10)

        main_layout.addWidget(self.scatter_plot_widget)
        main_layout.addWidget(vbox)
        
        main_widget.setLayout(main_layout)
        self.setCentralWidget(main_widget)


if __name__ == '__main__':
    app = QApplication(sys.argv)
    main_window = MainWindow()
    main_window.show()
    sys.exit(app.exec())
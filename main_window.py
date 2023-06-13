import sys
from PyQt6.QtCore import QSize, Qt
from PyQt6.QtWidgets import QApplication, QMainWindow, QLabel, QVBoxLayout, QHBoxLayout, QWidget
from src.utils.widgets.query_widget import QueryWidget
from src.utils.widgets.plot_widget import PlotWidget
from src.utils.preprocessing import Preprocessing


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        #load data using the config file 'config.ini'
        preprocessing = Preprocessing()
        self.config, self.tags, self.points, self.img_paths = preprocessing.load_data()

        ## inialize widgets
        query_widget = QueryWidget()
        scatter_plot_widget = PlotWidget(self.points, self.config)
        
        ## set up main window 
        self.make_layout(scatter_plot_widget, query_widget)
        
        

    
    def make_layout(self, scatter_plot_widget, query_widget):   
        """Congifure layout for main window"""     
        self.setWindowTitle("Scatterplot Dashboard")
        self.setMinimumSize(QSize(500, 500))


if __name__ == '__main__':
    app = QApplication(sys.argv)
    main_window = MainWindow()
    main_window.show()
    sys.exit(app.exec())
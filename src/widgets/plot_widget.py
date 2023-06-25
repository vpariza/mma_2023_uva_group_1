import numpy as np
import matplotlib
matplotlib.use('QtAgg')

from PyQt6.QtWidgets import (
    QVBoxLayout, QWidget, QLabel, QComboBox, QSizePolicy
)
from PyQt6 import QtCore
from src.widgets.filter_widget import SearchWidget

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
from matplotlib.figure import Figure

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

class MplCanvas(FigureCanvasQTAgg):
    """Setup canvas for plotting"""

    def __init__(self, parent=None, width=5, height=4, dpi=100):
        self.fig = Figure(figsize=(width, height), dpi=dpi, facecolor='#f5f5f5')
        self.canvas = FigureCanvasQTAgg(self.fig)
        self.ax = self.fig.add_subplot(111)
        # set title of the plot
        self.fig.suptitle("Query Cosinesimilarity")
        super(MplCanvas, self).__init__(self.fig)


class ScatterPlotWidget(QWidget):
    """Widget that displays a scatterplot"""

    def __init__(self,points, config):
        super().__init__()
        # Setup plot configurations
        self.config = config
        self.load_configurations()
        
        # Load points
        self.points = points
        # self.mean_x = np.mean(self.points[:,0])
        # self.mean_y = np.mean(self.points[:,1])
        
        # create a matplotlib figure and add a subplot
        self.Figure = MplCanvas()

        # set up the layout
        layout = QVBoxLayout(self)
        layout.addWidget(self.Figure.canvas)

        self.draw_scatterplot()

    def load_configurations(self):
        """Load plot configurations"""
        self.points_size = float(self.config['scatterplot']['point_size'])
        self.points_color = self.config['scatterplot']['points_color']
        self.selection_color = self.config['scatterplot']['selection_color']
        self.selection_points_size = float(self.config['scatterplot']['selection_point_size'])
        self.rectangle_color = self.config['scatterplot']['rectangle_color']
        self.rectangle_opacity = float(self.config['scatterplot']['rectangle_opacity'])

    def draw_scatterplot(self, selected_points=None):
        """method that makes a scatterplot"""

        self.Figure.ax.scatter(self.points[:,0], self.points[:,1], s=self.points_size, c=self.points_color)
        self.Figure.canvas.draw()
        
    def update_scatterplot(self, x, y, kmeans = False, k = 2):
        """Update scatterplot with new dataframe"""
        if not kmeans:
            self.Figure.ax.cla() 
            self.Figure.ax.scatter(x, y, s=self.points_size, c=self.points_color)
            self.Figure.canvas.draw()
        if kmeans:
            #scaler = StandardScaler()
            #scaler.fit()
            kmeans = KMeans(n_clusters=k)
            # Fit the data to the k-means model
            data = np.array(list(zip(x, y)))
            kmeans.fit(data)
            labels = kmeans.labels_
            # Get the coordinates of the cluster centers
            cluster_centers = kmeans.cluster_centers_
            self.Figure.ax.cla() 
            self.Figure.ax.scatter(x, y, s=self.points_size, c=labels)
            self.Figure.ax.scatter(cluster_centers[:, 0], cluster_centers[:, 1], c='red', marker='x')
            self.Figure.canvas.draw()
            print('complete')

class SelectClusterWidget(QWidget):

    def __init__(self):
        super().__init__()
        # Histogram Widget
        self.select_widget = QWidget(self)
        select_layout = QVBoxLayout()
        Label = QLabel(self)
        Label.setText("Select Clustering Configurations")
        select_layout.addWidget(Label)

        filter = {}

        method = [0, 1]
        method_tags = ['umap', 't-sne']
        self.dim_reduct_method = ComboFilter('Dimensionality Reduction mehtod     ', method, method_tags)
        select_layout.addWidget(self.dim_reduct_method)
        filter['dimensionality_reduction_method'] = self.dim_reduct_method.Filter.currentText()

        method = [0]
        method_tags = ['k-means']
        self.clustering_method = ComboFilter('Clustering method                   ', method, method_tags)
        select_layout.addWidget(self.clustering_method)
        filter['clustering_method'] = self.clustering_method.Filter.currentText()

        method = [1, 2]
        method_tags = ['1', '2']
        self.n_clusters_method = ComboFilter('Numbers of clusters                 ', method, method_tags)
        select_layout.addWidget(self.n_clusters_method)
        filter['n_clusters_method'] = self.n_clusters_method.Filter.currentText()
        
        self.searchbutton = SearchWidget(filter)
        select_layout.addWidget(self.searchbutton)

        # Combine widgets
        self.setLayout(select_layout)
        self.setStyleSheet("border: 0px ; background-color: #f5f5f5;")
        self.adjustSize()


class ComboFilter(QWidget):
    """ Class for building singular drop-down menu style widgets
    
    """
    def __init__(self, name, filter_list = [], filter_tags = []):
        super().__init__()
        self.name = name
        self.filter_list = filter_list #[0, 100, 500, 1000]
        self.filter_tags = filter_tags #
        layout = QVBoxLayout(self)

        self.Filter = QComboBox(self)
        #self.Filter.setFixedSize(QtCore.QSize(100, 50))  
        self.Filter.addItems(filter_tags)
        self.Filter.setStyleSheet("background-color: white;")
        self.Filter.setCurrentIndex(-1)
        self.Filter.setPlaceholderText(self.name)
        self.Filter.resize(self.Filter.sizeHint())
        

        # Connect signals to the methods.
        self.Filter.activated.connect(self.check_index)
        self.Filter.activated.connect(self.current_text)
        self.Filter.activated.connect(self.current_text_via_index)
        self.Filter.activated.connect(self.current_count)
        
        layout.addWidget(self.Filter)
        
        self.setLayout(layout)
        #self.setMinimumSize(200, 200)
        #self.setMaximumSize(100, 200)
        self.adjustSize()
        self.setStyleSheet("border: 0px solid darkgray; background-color: #f5f5f5;")
        
    
    def check_index(self, index):
        cindex = self.Filter.currentIndex()

    def current_text(self, _): 
        ctext = self.Filter.currentText()
        print(  ctext)

    def current_text_via_index(self, index):
        ctext = self.Filter.itemText(index) 

    def current_count(self, index):
        count = self.Filter.count()

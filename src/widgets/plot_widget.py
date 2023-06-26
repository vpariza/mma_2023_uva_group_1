import numpy as np
import matplotlib
matplotlib.use('QtAgg')

from PyQt6.QtWidgets import (
    QPushButton, QVBoxLayout, QWidget, QLabel, QComboBox, QSizePolicy
)
from PyQt6 import QtCore
from src.widgets.filter_widget import SearchWidget
from src.widgets.checkbox_list_widget import CheckBoxListWidget
from src.utils.ZoomPan import ZoomPan
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
from matplotlib.figure import Figure
from matplotlib.patches import Rectangle

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

    # signal that emits the index of the selected points
    selected_idx = QtCore.pyqtSignal(list)
    #signal that emits the current mouse position
    label = QtCore.pyqtSignal(str)
    

    def __init__(self,points, config):
        super().__init__()
        # Setup plot configurations
        self.setMouseTracking(True)
        self.config = config
        self.load_configurations()

        # create a matplotlib figure and add a subplot
        self.Figure = MplCanvas()
        
        """"""
        # Load points
        self.points = points
        self.mean_x = np.mean(self.points[:,0])
        self.mean_y = np.mean(self.points[:,1])
        self.start_point = None
        self.end_point = None
        # create a button under the scatterplot that removes the rectangle and the selected points
        self.clear_button = QPushButton("Clear Selection")
        self.clear_button.clicked.connect(self.clear_selection)
        self.zp = ZoomPan()
        self.figZoom = self.zp.zoom_factory(self.Figure.ax, base_scale = 1.5)
        self.figPan = self.zp.pan_factory(self.Figure.ax)
        self.selected_points = []
        self.outside_points_visible = False
        """"""
        
        


        # set up the layout
        layout = QVBoxLayout(self)
        layout.addWidget(self.Figure.canvas)

        self.draw_scatterplot()
        """"""
        #connect the mouse press event to the selection method
        self.Figure.canvas.mpl_connect('button_press_event', self.on_canvas_click)
        self.Figure.canvas.mpl_connect('button_release_event', self.on_canvas_release)
        #connect the mouse move event to the label method
        self.Figure.canvas.mpl_connect('motion_notify_event', self.on_mouse_move)
        # Set initial widget size
        self.resize_widget()
        """"""


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
        # remove the old points from the plot
        if selected_points!=None:
            self.selected_points = selected_points
        for point in self.Figure.ax.collections:
            point.remove()
        

        self.Figure.ax.scatter(self.points[:,0], self.points[:,1], s=self.points_size, c=self.points_color)
        for i in self.selected_points:
            point = self.points[i]
            if self.is_point_in_rectangle(point) or self.outside_points_visible:
                self.Figure.ax.scatter(point[0], point[1], s=self.selection_points_size, c=self.selection_color)
        
        
        self.Figure.canvas.draw()
        
    def update_scatterplot(self, x, y, kmeans = False, k = 2, alpha_ = 1):
        """Update scatterplot with new dataframe"""
        if not kmeans:
            self.Figure.ax.cla() 
            self.Figure.ax.scatter(x, y, s=self.points_size, c=self.points_color, alpha=alpha_)
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
            self.Figure.ax.scatter(x, y, s=self.points_size, c=labels, alpha=alpha_)
            self.Figure.ax.scatter(cluster_centers[:, 0], cluster_centers[:, 1], c='red', marker='x')
            self.Figure.canvas.draw()

    
    def resizeEvent(self, event):
        # Resize widget while keeping it square
        size = min(self.width(), self.height())
        self.setFixedSize(size, size)
        self.resize_widget()

    def resize_widget(self):
        # Adjust scatter plot size to fit the widget
        size = min(self.Figure.canvas.width(), self.Figure.canvas.height())
        self.Figure.canvas.setFixedSize(size, size)

    def on_mouse_move(self, event):
        """Method that handles mouse movement on the canvas"""
        # Ignore mouse movement outside the plot area
        if event.inaxes is None:
            return
        x, y = event.xdata, event.ydata
        # emit a signal with the current mouse position
        self.label.emit(f"Current Mouse Position: {x:.2f}, {y:.2f}")

    def on_canvas_release(self, event):
        if event.inaxes is None:
            print("Releasing click outside axes. Ignoring. Please click inside the axes of the scatterplot")
            return
        pass

        if event.button == 1:
            if self.start_point:
                self.end_point = (event.xdata, event.ydata)
                if self.start_point == self.end_point:
                    return
                self.draw_selection_rectangle()
                # reset the drawing of the selected points
                self.selected_points = []
                self.draw_scatterplot()

    def on_canvas_click(self, event):
        """Method that handles mouse clicks on the canvas"""
        # Ignore clicks outside the plot area
        if event.inaxes is None:
            print("click outside axes. Ignoring. Please click inside the axes of the scatterplot")
            return
        # left click to select points
        if event.button == 1:
            self.start_point = (event.xdata, event.ydata)

    def draw_selection_rectangle(self):
        """Method that draws the selection rectangle on the plot"""
        # get coordinates of the selection rectangle
        x1, y1 = self.start_point[0], self.start_point[1]
        x2, y2 = self.end_point[0], self.end_point[1]
        
        #calculate the position and size of the rectangle
        x = min(x1, x2)
        y = min(y1, y2)
        w = abs(x1 - x2)
        h = abs(y1 - y2)
        
        #remove the old rectangle if it exists
        self.clear_selection()           
        # add the new rectangle to the plot
        self.rect = Rectangle((x, y), w, h, facecolor=self.rectangle_color, alpha=self.rectangle_opacity)
        self.Figure.ax.add_patch(self.rect)
        
        #update the plot
        self.Figure.canvas.draw()
        
        # emit a signal with the index of the selected points
        xmin, xmax = sorted([x1, x2])
        ymin, ymax = sorted([y1, y2])
        
        # get the indices of the selected points
        indices = [i for i, p in enumerate(self.points) if xmin <= p[0] <= xmax and ymin <= p[1] <= ymax]
        
        # update the selected points
        self.draw_scatterplot()
        
        # emit the signal
        self.selected_idx.emit(indices)

    def is_point_in_rectangle(self,point):
        """Method that checks if a point is in the selection rectangle"""
        # get the coordinates of the rectangle
        if self.start_point[0] < self.end_point[0]:
            x1, x2 = self.start_point[0], self.end_point[0]
        else:
            x1, x2 = self.end_point[0], self.start_point[0]
        if self.start_point[1] < self.end_point[1]:
            y1, y2 = self.start_point[1], self.end_point[1]
        else:   
            y1, y2 = self.end_point[1], self.start_point[1]
        
        # get the coordinates of the point
        x, y = point[0], point[1]

        # check if the point is in the rectangle
        if x1 <= x <= x2 and y1 <= y <= y2:
            return True
        else:
            return False
        
    def set_outside_points_visible(self, visible):
        self.outside_points_visible = visible
        self.draw_scatterplot()
    
    def clear_selection(self):
        # Remove the rectangle from the plot
        self.selected_idx.emit([])
        for patch in self.Figure.ax.patches:
            patch.remove()
        self.draw_scatterplot([])
    
    
            

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

        method = [0, 1]
        method_tags = ['k-means', 'None']
        self.clustering_method = ComboFilter('Clustering method                   ', method, method_tags)
        select_layout.addWidget(self.clustering_method)
        filter['clustering_method'] = self.clustering_method.Filter.currentText()

        method = np.arange(1,10 )
        method_tags = [str(i) for i in method]
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
        
        layout.addWidget(self.Filter)
        
        self.setLayout(layout)
        #self.setMinimumSize(200, 200)
        #self.setMaximumSize(100, 200)
        self.adjustSize()
        self.setStyleSheet("border: 0px solid darkgray; background-color: #f5f5f5;")
        
    


import numpy as np
import matplotlib
matplotlib.use('QtAgg')

from PyQt6.QtWidgets import QVBoxLayout, QWidget

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
from matplotlib.figure import Figure


class MplCanvas(FigureCanvasQTAgg):
    """Setup canvas for plotting"""

    def __init__(self, parent=None, width=5, height=4, dpi=100):
        self.fig = Figure(figsize=(width, height), dpi=dpi)
        self.canvas = FigureCanvasQTAgg(self.fig)
        self.ax = self.fig.add_subplot(111)
        # set title of the plot
        self.fig.suptitle("Plot")
        super(MplCanvas, self).__init__(self.fig)


class PlotWidget(QWidget):
    """Widget that displays a scatterplot"""

    def __init__(self,points, config):
        super().__init__()
        # Setup plot configurations
        self.config = config
        self.load_configurations()
        
        # Load points
        self.points = points
        self.mean_x = np.mean(self.points[:,0])
        self.mean_y = np.mean(self.points[:,1])
        
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
        
    def update_scatterplot(self, x, y):
        """Update scatterplot with new dataframe"""
        self.Figure.ax.cla() 
        self.Figure.ax.scatter(x, y, s=self.points_size, c=self.points_color)
        self.Figure.canvas.draw()
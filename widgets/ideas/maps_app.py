import io
import sys

import folium
from PyQt6 import QtWidgets, QtWebEngineWidgets

import sys
from PyQt6.QtWidgets import (
    QMainWindow, QApplication, QWidget,
    QLabel, QToolBar, QStatusBar, QVBoxLayout
)
from PyQt6.QtGui import QAction, QIcon
from PyQt6.QtCore import Qt

"""
Other useful resources
https://stackoverflow.com/questions/71831698/creating-a-folium-map-with-markers-with-different-colors
"""

class MainWindow(QMainWindow):

    def __init__(self):
        super(MainWindow, self).__init__()

        self.setWindowTitle("My App")

        layout = QVBoxLayout()
        m = folium.Map(
            location=[45.5236, -122.6750], tiles="Stamen Toner", zoom_start=13
        )
        data = io.BytesIO()
        m.save(data, close_file=False)

        w = QtWebEngineWidgets.QWebEngineView()
        w.setHtml(data.getvalue().decode())
        w.resize(640, 480)
        layout.addWidget(w)
        self.label = QLabel("Hello!")
        layout.addWidget(self.label)

        widget = QWidget()
        widget.setLayout(layout)
        self.setCentralWidget(widget)

    def mouseMoveEvent(self, e):
        self.label.setText("mouseMoveEvent")

    def mousePressEvent(self, e):
        self.label.setText("mousePressEvent")

    def mouseReleaseEvent(self, e):
        self.label.setText("mouseReleaseEvent")

    def mouseDoubleClickEvent(self, e):
        self.label.setText("mouseDoubleClickEvent")


if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    w = MainWindow()
    w.show()

    # m = folium.Map(
    #     location=[45.5236, -122.6750], tiles="Stamen Toner", zoom_start=13
    # )
    # data = io.BytesIO()
    # m.save(data, close_file=False)

    # w = QtWebEngineWidgets.QWebEngineView()
    # w.setHtml(data.getvalue().decode())
    # w.resize(640, 480)
    # w.show()

    sys.exit(app.exec())
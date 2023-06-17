# from PyQt5 import QtWebEngineWidgets, QtWidgets, QtCore, QtWebChannel

import io
import sys

import folium
from PyQt6 import QtWidgets, QtWebEngineWidgets, QtCore, QtWebChannel

import sys
from PyQt6.QtWidgets import (
    QMainWindow, QApplication, QWidget,
    QLabel, QToolBar, QStatusBar, QVBoxLayout
)

class Backend(QtCore.QObject):
    pointChanged = QtCore.pyqtSignal(float, float)

    @QtCore.pyqtSlot(float,float)
    def pointClicked(self, x, y):
        self.pointChanged.emit(x, y)


class MainWindow(QtWidgets.QMainWindow):
    def __init__(self, parent=None):
        super(MainWindow, self).__init__(parent)

        map_view = QtWebEngineWidgets.QWebEngineView()

        backend = Backend(self)
        backend.pointChanged.connect(self.onPointChanged)
        channel = QtWebChannel.QWebChannel(self)
        channel.registerObject('backend', backend)
        map_view.page().setWebChannel(channel)

        file = QtCore.QDir.current().absoluteFilePath("index.html")
        map_view.load(QtCore.QUrl.fromLocalFile(file))

        self.setCentralWidget(map_view)

    @QtCore.pyqtSlot(float,float)
    def onPointChanged(self, x, y):
        print("new points")
        print(x, y)


if __name__ == '__main__':
    import sys

    app = QtWidgets.QApplication(sys.argv)
    w = MainWindow()
    w.show()
    sys.exit(app.exec())
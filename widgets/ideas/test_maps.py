import folium
from jinja2 import Template
from folium.map import Marker

import io
import sys
from PyQt6 import QtWidgets, QtWebEngineWidgets

import sys
from PyQt6.QtWidgets import (
    QMainWindow, QApplication, QWidget,
    QLabel, QToolBar, QStatusBar, QVBoxLayout
)
from PyQt6.QtGui import QAction, QIcon
from PyQt6.QtCore import Qt

# Modify Marker template to include the onClick event
click_template = """{% macro script(this, kwargs) %}
    var {{ this.get_name() }} = L.marker(
        {{ this.location|tojson }},
        {{ this.options|tojson }}
    ).addTo({{ this._parent.get_name() }}).on('click', onClick);
{% endmacro %}"""

# Change template to custom template
Marker._template = Template(click_template)

location_center = [51.7678, -0.00675564]
m = folium.Map(location_center, zoom_start=13)

# Create the onClick listener function as a branca element and add to the map html
click_js = """function onClick(e) {
                 var point = e.latlng; alert(point)
                 }"""
                 
e = folium.Element(click_js)
html = m.get_root()
html.script.get_root().render()
html.script._children[e.get_name()] = e

#Add marker (click on map an alert will display with latlng values)
marker = folium.Marker([51.7678, -0.00675564]).add_to(m)


class MainWindow(QMainWindow):

    def __init__(self):
        super(MainWindow, self).__init__()

        self.setWindowTitle("My App")

        layout = QVBoxLayout()
        data = io.BytesIO()
        coords = [[51.7678, -0.00675564], [53, 0.2], [50, -0.1]]
        for lat, lon in coords:
        
            html=f"""
                <h1> {'Mystery Point'}</h1>
                <p>You can use any html here! Let's do a list:</p>
                <ul>
                    <li>Item 1</li>
                    <li>Item 2</li>
                </ul>
                </p>
                <p>And that's a <a href="https://www.python-graph-gallery.com">link</a></p>
                """
            iframe = folium.IFrame(html=html, width=200, height=200)
            popup = folium.Popup(iframe, max_width=2650)
            tooltip = "Click me!"
            # folium.Marker([lat, lon], popup='Mystery Point').add_to(m)
            folium.Marker([lat, lon], popup=popup, tooltip=tooltip).add_to(m)
        folium.Circle(
        # folium.CircleMarker(
            location=[51.7678, -0.00675564],
            radius=1000,
            popup="Laurelhurst Park",
            color="#3186cc",
            fill=True,
            fill_color="#3186cc",
        ).add_to(m)

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

    # def mouseMoveEvent(self, e):
    #     self.label.setText("mouseMoveEvent")

    # def mousePressEvent(self, e):
    #     self.label.setText("mousePressEvent")

    # def mouseReleaseEvent(self, e):
    #     self.label.setText("mouseReleaseEvent")

    # def mouseDoubleClickEvent(self, e):
    #     self.label.setText("mouseDoubleClickEvent")


if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    w = MainWindow()
    w.resize(640, 480)
    w.show()

    sys.exit(app.exec())
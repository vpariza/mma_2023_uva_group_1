
from elements.draw import Draw
import folium, io, sys, json
from PyQt6 import QtCore, QtWidgets, QtWebEngineWidgets, QtWebEngineCore

from PyQt6.QtWidgets import (
    QMainWindow, QWidget,
    QLabel, QVBoxLayout, QPushButton
)
from geo_map_model import QGeoMapModel

from jinja2 import Template
from folium.map import Marker

from enum import Enum
import typing

class WebEngineGeoMapPage(QtWebEngineCore.QWebEnginePage):
    """
    This represents a web page for the GeoMap. It can be use to
    interact with some web events from the interactions with the
    Folium Maps.
    """
    class InteractionTypes(Enum):
        MarkerClick = 'Point'
        AreaSelection = 'Polygon'
    # Define the Signals that can be emitted
    areaSelected = QtCore.pyqtSignal(list, QWidget)
    markerClicked = QtCore.pyqtSignal(list, QWidget)

    def __init__(self, parent: typing.Optional[QtCore.QObject] = None) -> None:
        super(WebEngineGeoMapPage, self).__init__(parent)

    def javaScriptConsoleMessage(self, level, msg, line, sourceID):
        """
        We want to capture the logging of the interactions in the console
        from the Folium Maps and then used them as we want in our widget.
        """
        if level == QtWebEngineCore.QWebEnginePage.JavaScriptConsoleMessageLevel.InfoMessageLevel:
            # We want to focus only on the information level loggings and not e.g., to the errors
            # Because these are the ones emitting the selection and click information
            coords_dict = json.loads(msg)
            if coords_dict.get(self.InteractionTypes.MarkerClick.value) is not None:
                coords = coords_dict[self.InteractionTypes.MarkerClick.value]
                self.markerClicked.emit([coords['lat'], coords['lng']], self)
            else:
                coords = coords_dict['geometry']['coordinates']
                interaction_type = coords_dict['geometry']['type']
                # Note that the coordinates returned are longitude, latitude so we
                # need to change their order as lat, lon to fit with the expectations
                # of the rest of the widget
                if interaction_type == self.InteractionTypes.MarkerClick.value:
                    self.markerClicked.emit([coords[1], coords[0]], self)
                elif interaction_type == self.InteractionTypes.AreaSelection.value:
                    self.areaSelected.emit([[coord[1], coord[0]] for coord in coords[0]], self)

class GeoMapWidget(QWidget):
    """
    Define a custom widget for the GeoMap.
    """
    # Define the Signals that can be emitted
    entriesSelected = QtCore.pyqtSignal(list, QWidget)
    entryClicked = QtCore.pyqtSignal(object, QWidget)

    def __init__(self, geo_map_model:QGeoMapModel, parent: typing.Optional['QWidget']=None, *args, **kwargs):
        super(GeoMapWidget, self).__init__(parent=parent, *args, **kwargs)
        self._geo_map_model = geo_map_model
        # the widget should have a layout
        layout = QVBoxLayout()
        # MAP LABEL
        # Maybe add a label for a title
        # self.label = QLabel("Hello!")
        # layout.addWidget(self.label)

        # UPDATE MAP BUTTON
        # button = QPushButton("Update Map", self)
        # button.clicked.connect(self.update_button_clicked)
        # layout.addWidget(button)
        # Create the Web Page for the GeoMap

        self.map_view = QtWebEngineWidgets.QWebEngineView()
        page = WebEngineGeoMapPage(self.map_view)

        page.areaSelected.connect(self.area_was_selected)
        page.markerClicked.connect(self.marker_was_clicked)
        # self.setUpdatesEnabled()
        self.map_view.setPage(page)
        self.map_view.setHtml(self.__generate_html_map().getvalue().decode())
        self.map_view.resize(640, 480)
        layout.addWidget(self.map_view)
        self.setLayout(layout)

    def update_geo_map_model(self, geo_map_model:QGeoMapModel):
        self._geo_map_model = geo_map_model
        self.update()

    def update(self):
        self.map_view.setHtml(self.__generate_html_map().getvalue().decode())
        self.map_view.resize(640, 480)
        self.map_view.update()

    def update_button_clicked(self):
        self.update_geo_map_model(geo_map_model)

    def __generate_html_map(self):
        m = self.__create_basic_map()
        self.__add_markers_on_map(m)
        self.__add_draw_features_on_map(m)
        data = io.BytesIO()
        m.save(data, close_file=False)
        return data

    def __create_basic_map(self):
        # Modify Marker template to include the onClick event
        # Change template to custom template
        Marker._template = Template(self.__click_marker_template)
        # Define the folium map
        m = folium.Map(
            location=self._geo_map_model.get_centroid_point(), tiles="OpenStreetMap", zoom_start=13
        )
        
        e = folium.Element(self.__click_marker_js)
        html = m.get_root()
        html.script.get_root().render()
        html.script._children[e.get_name()] = e
        return m

    def __add_markers_on_map(self, map):
        """
        Adding the markers on the map. Note that the more we
        increase the number of markers, the more time the rendering
        of the html takes.
        """
        info = self._geo_map_model.get_entries_summary()
        for id, coord, html   in zip(info['ids'], info['coords'], info['html_summaries']):
            # iframe = folium.IFrame(html=html, width=200, height=200)
            # popup = folium.Popup(iframe, max_width=2650)
            # tooltip = "Id: {}".format(id)
            # folium.Marker(coord, popup=popup, tooltip=tooltip).add_to(map)
            # folium.Marker(coord, popup='Listing Id: {}'.format(id)).add_to(map)
            folium.Marker(coord, popup='Listing Id: {} Location: {}'.format(id, coord)).add_to(map)

    def __add_draw_features_on_map(self, map):
        """     
        Define the Draw interaction element of the Folium Map
        This element allows to draw a selection on the map
        and at the same time send information about that
        selection to python
        """
        draw = Draw(
        show_geometry_on_click=True,
        draw_options={
            'polyline':False,
            'rectangle':True,
            'polygon':True,
            'circle':False,
            'marker':False,
            'circlemarker':False,
            },
        edit_options={'edit':True})
        map.add_child(draw)

    @property
    def __click_marker_template(self):
        return """{% macro script(this, kwargs) %}
            var {{ this.get_name() }} = L.marker(
                {{ this.location|tojson }},
                {{ this.options|tojson }}
            ).addTo({{ this._parent.get_name() }}).on('click', onClick);
        {% endmacro %}"""

    @property
    def __click_marker_js(self):
        return """function onClick(e) {
                 var dict = {
                 'Point': e.latlng
                 }; 
                 console.log(JSON.stringify(dict));
                 }"""

    @QtCore.pyqtSlot(list, QWidget)
    def area_was_selected(self, coords, source):
        entries = self._geo_map_model.get_selected_entries_from_area(coords) 
        print(entries)
        self.entriesSelected.emit(entries, self)

    @QtCore.pyqtSlot(list, QWidget)
    def marker_was_clicked(self, coords, source):
        entry = self._geo_map_model.get_selected_entry(coords) 
        print(entry)
        self.entryClicked.emit(entry, self)

if __name__ == '__main__': 
    #### TESTING CODE ####
    from utils.point_utils import add_points_as_coords
    app = QtWidgets.QApplication(sys.argv)
    w = QMainWindow()
    w.setWindowTitle("READ - Real Estate Analytics Dashboard - Demo")
    from utils import file_utils
    # Load the dataframe wit the Funda Data
    data = file_utils.load_from_pickle('./data.pkl')
    data = add_points_as_coords(data)
    geo_map_model = QGeoMapModel(data.iloc[:200])
    # geo_map_model.entriesSelected.connect()
    w.setCentralWidget(GeoMapWidget(geo_map_model))
    w.show()
    sys.exit(app.exec())
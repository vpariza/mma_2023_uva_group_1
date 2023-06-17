from PyQt6 import QtCore, QtWidgets, QtWebEngineWidgets, QtWebEngineCore
from draw import Draw
import folium, io, sys, json

if __name__ == '__main__': 
    app = QtWidgets.QApplication(sys.argv)
    
    m = folium.Map(location=[55.8527, 37.5689], zoom_start=13)
    
    draw = Draw(
       show_geometry_on_click=True,
        draw_options={
            'polyline':False,
            'rectangle':True,
            'polygon':True,
            'circle':True,
            'marker':False,
            'circlemarker':False,
            },
        edit_options={'edit':False})
    m.add_child(draw)

    data = io.BytesIO()
    m.save(data, close_file=False)

    class WebEnginePage(QtWebEngineCore.QWebEnginePage):
       def javaScriptConsoleMessage(self, level, msg, line, sourceID):
          print('level', level == QtWebEngineCore.QWebEnginePage.JavaScriptConsoleMessageLevel.InfoMessageLevel)
          coords_dict = json.loads(msg)
          coords = coords_dict['geometry']['coordinates'][0]
          print(coords_dict)
          print(coords)

    view = QtWebEngineWidgets.QWebEngineView()
    page = WebEnginePage(view)
    view.setPage(page)
    view.setHtml(data.getvalue().decode())
    view.show()
    sys.exit(app.exec())
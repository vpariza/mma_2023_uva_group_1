from PyQt6.QtWidgets import QWidget, QLineEdit, QComboBox, QHBoxLayout, QVBoxLayout, QLabel, QPushButton, QGridLayout, QCheckBox
from PyQt6 import QtCore
import sys
from PyQt6.QtWidgets import QWidget, QListWidget, QVBoxLayout, QLabel, QGridLayout, QCheckBox
from PyQt6 import QtCore
import copy

from typing import List

class ListOptionsWidget(QWidget):
    optionsSelected = QtCore.pyqtSignal(list, QWidget)

    def __init__(self, options:List[str], title_text:str=None, selection_mode=QListWidget.SelectionMode.MultiSelection, parent:QWidget=None):
        super().__init__(parent)
        # self._options = options
        layout = QVBoxLayout()
        if title_text is not None:
            # Add a Title only if it is defined
            self._title = QLabel()
            self._title.setText(title_text)
            layout.addWidget(self._title)
        # Define the widge of list of points
        self._list_widget = QListWidget()
        for option in options:
            self._list_widget.addItem(option)
        # Define the type of selection for the List Widget
        self._list_widget.setSelectionMode(selection_mode)
        self._list_widget.selectionModel().selectionChanged.connect(self.__options_were_selected)
        layout.addWidget(self._list_widget) 
        self.setLayout(layout)

    def update_options(self, options:List[str]):
        self._list_widget.clear()
        self.add_options(options)

    def add_options(self, options:List[str]):
        for option in options:
            self._list_widget.addItem(option)
        self.update()

    def set_selection(self, selected_options:List[str]):
        selected_options = set(selected_options)
        for i in range(self._list_widget.count()):
            item = self._list_widget.item(i)
            if item.text() in selected_options:
                item.setSelected(True)
            else:
                item.setSelected(False)

    @property
    def options(self):
        return [self._list_widget.item(i).text() for i in range(self._list_widget.count())]
    
    @property
    def selected_options(self):
        return [item.text() for item in self._list_widget.selectedItems()]

    @QtCore.pyqtSlot(QtCore.QItemSelection, QtCore.QItemSelection)
    def __options_were_selected(self, selected, deselected):
        self.optionsSelected.emit(self.selected_options, self)


                        
if __name__ == '__main__':
    from PyQt6.QtWidgets import QApplication, QWidget,  QGridLayout, QListWidget,  QPushButton
    from PyQt6.QtGui import QIcon

    class MainWindow(QWidget):

        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            layout = QGridLayout(self)
            self.w = ListOptionsWidget(options = ['option1', 'option2', 'option3', 'option4'])
            self.w.optionsSelected.connect(self.selected)
            layout.addWidget(self.w)
            self.setLayout(layout)
            self.show()
        
        def selected(self, selected, source):
            if len(selected) == 4:
                self.w.update_options(['O1', 'O2', 'O3', 'O4'])
                # self.w._list_widget.setCurrentRow(0)
                self.w._list_widget.item(0)
                self.w._list_widget.item(0).setSelected(True)
                self.w._list_widget.item(1).setSelected(True)
                self.w._list_widget.item(2).setSelected(False)
                self.w._list_widget.item(3).setSelected(True)
                # self.w._list_widget.setCurrentRow(2)
                # self.w._list_widget.setCurrentRow(3)
            print(selected)
    app = QApplication(sys.argv)
    window = MainWindow()
    app.exec()

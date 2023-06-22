
import sys

from PyQt6 import QtCore
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QApplication, QWidget, QCheckBox
)

import typing
from typing import List, Dict

import copy

class CheckBoxListWidget(QWidget):

    stateChanged = QtCore.pyqtSignal(dict, QWidget)
    checkboxClicked = QtCore.pyqtSignal(str, bool, QWidget)

    def __init__(self, options:List[str]| Dict[str, bool], parent: typing.Optional['QWidget']=None, *args, **kwargs):
        super(CheckBoxListWidget, self).__init__(parent=parent, *args, **kwargs)
        self._layout = self._create_checkboxes(options, default_state=False)
        self.setLayout(self._layout)

    def _create_checkboxes(self, options:List[str]| Dict[str, bool], default_state:bool=False):
        """"
        Method that creates a list of checkboxes
        """
        if isinstance(options, list):
            # If the options is a list convert it to a dictionary
            self.options_state = {option:default_state for option in options}
        if isinstance(options, dict):
            self.options_state = options
        layout = QVBoxLayout()
        self.checkboxes = dict()
        for option_name,checked in self.options_state.items():
            checkbox = QCheckBox(option_name, self)
            checkbox.toggled.connect(self.showDetails)
            checkbox.setChecked(checked)
            self.checkboxes[option_name] = checkbox
            # checkbox.move(100, 100)
            layout.addWidget(checkbox)
        return layout

    def showDetails(self):
        """
        Invoked when any of the checkboxes gets updated
        """
        self.options_state[self.sender().text()] = self.sender().isChecked()
        self.stateChanged.emit(copy.copy(self.options_state), self)
        self.checkboxClicked.emit(self.sender().text(), self.sender().isChecked(), self)

    def update_options_state(self, options_state:Dict[str, bool]):
        """
        Updates the state of the checkboxes.
        """
        self.options_state = copy.copy(options_state)
        for option_name, option_checked in self.options_state.items():
            self.checkboxes[option_name].setChecked(option_checked)
        self.update()

    def add_new_option(self, option_name, checked:bool):
        if option_name not in self.checkboxes:
            return 
        checkbox = QCheckBox(option_name, self)
        checkbox.toggled.connect(self.showDetails)
        checkbox.setChecked(checked)
        self.checkboxes[option_name] = checkbox
        self.options_state[option_name] = checked
        self._layout.addWidget(checkbox)
        self.update()

    def remove_option(self, option_name:str):
        if option_name not in self.checkboxes:
            return
        self._layout.removeWidget(self.checkboxes[option_name])
        del self.checkboxes[option_name]
        del self.options_state[option_name]
        self.update()

    def is_checked(self, option_name):
        return self.options_state[option_name]

    def uncheck_all(self):
        for option_name, _ in self.options_state.items():
            self.options_state[option_name] = False
            self.checkboxes[option_name].setChecked(False)
        self.update()

    @property
    def current_options_state(self):
        """
        Updates the state of the checkboxes.
        """
        return copy.copy(self.options_state)
    
    @property
    def option_names(self):
        """
        Updates the state of the checkboxes.
        """
        return list(self.options_state.keys())


if __name__ == '__main__':
    class Window(QWidget):
        change = False
        def __init__(self):
            super().__init__()
            self.resize(300, 250)
            self.setWindowTitle("CodersLegacy")
            layout = QVBoxLayout()
            options = {f'Option {i}':False for i in range(10)}
            w = CheckBoxListWidget(options)
            
            def temp(t,s,src):
                print(t,s,src)
            w.checkboxClicked.connect(temp)
            from PyQt6.QtWidgets import QApplication, QWidget, QPushButton
            button = QPushButton("Do not click me!", self)
            button.clicked.connect(lambda x: w.remove_option('Option 2'))
            layout.addWidget(w)
            layout.addWidget(button)
            self.setLayout(layout)
            

    
    app = QApplication(sys.argv)
    window = Window()
    window.show()
    sys.exit(app.exec())
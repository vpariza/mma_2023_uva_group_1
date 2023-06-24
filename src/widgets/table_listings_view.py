import sys
import typing
from PyQt6 import QtCore, QtGui, QtWidgets
from PyQt6.QtCore import Qt, QSize
import pandas as pd
from PyQt6.QtWidgets import QApplication, QLabel, QMainWindow,  QVBoxLayout, QWidget, QStyledItemDelegate, QGridLayout, QAbstractItemView

from enum import Enum
import os

from src.widgets.table_listings_model import TableListingsModel

from typing import List
class ImageWindow(QWidget):
    """
    This "window" is a QWidget. If it has no parent, it
    will appear as a free-floating window as we want.
    """
    def __init__(self, entry_id:str, dir_name:str, imgs_paths:List[str]):
        super().__init__()
        layout = QGridLayout(self)
        imgs_num = len(imgs_paths)
        # Caclulate the number of rows to plot
        COLS = 2
        rows = imgs_num // COLS
        rows += 0 if imgs_num % COLS == 0 else 1
        i = 0
        for row in range(rows):
            for column in range(COLS):
                if i >= imgs_num:
                    break
                path = os.path.join(dir_name, imgs_paths[i])
                label = QLabel(self)
                pixmap = QtGui.QPixmap(path)
                label.setPixmap(pixmap)
                label.setScaledContents(True)
                layout.addWidget(label, row, column)
                i+=1
        self.setLayout(layout)
        self.setWindowState(QtCore.Qt.WindowState.WindowMaximized)
        self.setWindowTitle("Images for {}".format(entry_id))

class ListingsImageDelegate(QStyledItemDelegate):

    def __init__(self, parent, column_imgs:int, imgs_dir_path:str, imgs_rel_paths:List[List[str]], show_blank_img=False):
        QStyledItemDelegate.__init__(self, parent)
        self._column_imgs = column_imgs
        self._imgs_rel_paths = imgs_rel_paths
        self._imgs_dir_path = imgs_dir_path
        self._show_blank_img = show_blank_img

    def paint(self, painter, option, index):
        if index.column() != self._column_imgs:
            return     
        painter.fillRect(option.rect, QtGui.QColor(255,255,255))
        imgs_rel_paths_local = self._imgs_rel_paths[index.row()]  
        if len(imgs_rel_paths_local) > 0 and not self._show_blank_img:
            path = os.path.join(self._imgs_dir_path, imgs_rel_paths_local[0])
            # In case loading of the image fails just 
            # return nothing
            image = QtGui.QImage(str(path))
            pixmap = QtGui.QPixmap.fromImage(image)
            pixmap.scaled(50, 40, QtCore.Qt.AspectRatioMode.KeepAspectRatio)
            painter.drawPixmap(option.rect, pixmap) 

    def sizeHint(self, option, index) :
        return QSize(160, 90) # whatever your dimensions are

class TableListingsView(QtWidgets.QTableView):
    entryClicked = QtCore.pyqtSignal(object, QWidget)
    entryDoubleClicked = QtCore.pyqtSignal(object, QWidget)
    entriesSelected = QtCore.pyqtSignal(list, QWidget)

    def __init__(self, model: TableListingsModel, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setItemDelegateForColumn(model.get_imgs_column(), ListingsImageDelegate(parent, model.get_imgs_column(), model._imgs_dir_path, model.get_imgs_paths_column()))
        self.setModel(model)
        self.setVerticalScrollMode(QAbstractItemView.ScrollMode.ScrollPerPixel)
        self.w = None
        self.selectionModel().selectionChanged.connect(self.__cells_were_selected)
        self.clicked.connect(self.__cell_was_clicked)
        self.doubleClicked.connect(self.__cell_was_double_clicked)
        self._selected_entries = set()
        self.setFixedWidth(600)

    def update_model(self, model: TableListingsModel):
        self.setModel(model)
        self.setVerticalScrollMode(QAbstractItemView.ScrollMode.ScrollPerPixel)
        self.update()

    @QtCore.pyqtSlot(QtCore.QModelIndex)
    def __cell_was_double_clicked(self, index):
        d_clicked_entry = self.model().get_entry_id(index.row())
        if index.column() == self.model().get_imgs_column():
            self.show_new_window(d_clicked_entry,self.model()._imgs_dir_path, self.model().get_imgs_paths(index.row()))
        self.entryDoubleClicked.emit(d_clicked_entry, self)

    @QtCore.pyqtSlot(QtCore.QModelIndex)
    def __cell_was_clicked(self, index):
        clicked_entry = self.model().get_entry_id(index.row())
        if index.column() == self.model().get_imgs_column():
            self.show_new_window(clicked_entry,self.model()._imgs_dir_path, self.model().get_imgs_paths(index.row()))
        self.entryClicked.emit(clicked_entry, self)

    @QtCore.pyqtSlot(QtCore.QItemSelection, QtCore.QItemSelection)
    def __cells_were_selected(self, selected, deselected):
        # =====Selected=====
        for ix in selected.indexes():
            self._selected_entries.add(self.model().get_entry_id(ix.row()))
        # =====Deselected=====
        for ix in deselected.indexes():
            self._selected_entries.discard(self.model().get_entry_id(ix.row()))
        self.entriesSelected.emit(list(self._selected_entries),self)

    def show_new_window(self, entry_id:str, dir_name:str, imgs_paths:List[str]):
        if self.w is not None:
            self.w.close()  # Close window.
            self.w = None
        self.w = ImageWindow(entry_id, dir_name, imgs_paths)
        self.w.show()

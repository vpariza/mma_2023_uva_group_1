from PyQt6 import QtCore, QtWidgets

from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QVBoxLayout, QPushButton, QListWidget
)
import typing
from src.widgets.list_options_widget import ListOptionsWidget

from typing import List, Dict
from src.widgets.dialog_widgets import BasicDialog

class ModelTrainWidget(QWidget):

    modelSelected = QtCore.pyqtSignal(str, QWidget)
    trainFeaturesSelected = QtCore.pyqtSignal(str, list, QWidget)
    modelToTrain = QtCore.pyqtSignal(str, list, QWidget)
    modelDeleted = QtCore.pyqtSignal(str, QWidget)

    def __init__(self, training_features:List[str], base_model_name='model_{}', widgets:Dict[str, QWidget]={}, parent: typing.Optional['QWidget']=None, *args, **kwargs):
        super(ModelTrainWidget, self).__init__(parent=parent, *args, **kwargs)
        self._list_models_w = widgets.get('list_models_w')
        self._list_features_w = widgets.get('list_features_w')
        self._models_confs = dict()
        self._base_model_name = base_model_name
        self._model_count = 1
        self._training_features = training_features
        self.setLayout(self._create_layout())

    def get_dict_widgets(self):
        return {
            'list_models_w': self._list_models_w,
            'list_features_w': self._list_features_w
        }

    def _create_layout(self):
        layout = QHBoxLayout()      
        ####### Add the Models List
        if self._list_models_w is None:
            self._list_models_w = ListOptionsWidget(list(self._models_confs.keys()), 
                                                    title_text='Choose Base Model', 
                                                    selection_mode=QListWidget.SelectionMode.SingleSelection, 
                                                    parent=self)
        self._list_models_w.optionsSelected.connect(self._on_model_selected)
        layout.addWidget(self._list_models_w)
        ####### Add the Features List
        if self._list_features_w is None:
            self._list_features_w = ListOptionsWidget(self._training_features, 
                                                    title_text='Choose features', 
                                                    selection_mode=QListWidget.SelectionMode.MultiSelection, 
                                                    parent=self)
        self._list_features_w.optionsSelected.connect(self._train_features_selected)
        # self._list_options_w.optionsSelected.connect(self.models_selected)
        layout.addWidget(self._list_features_w)
        ####### Add the Buttons Layout
        buttons_layout = QVBoxLayout()  
        self._create_empty_model_button = QPushButton('Create Empty Model', self)
        self._create_empty_model_button.clicked.connect(self._on_create_empty_train_model_button_clicked)
        buttons_layout.addWidget(self._create_empty_model_button)
        self._train_new_model_button = QPushButton('Train New Model', self)
        self._train_new_model_button.clicked.connect(self._on_train_model_button_clicked)
        buttons_layout.addWidget(self._train_new_model_button)
        self._delete_models_button = QPushButton('Delete Model', self)
        self._delete_models_button.clicked.connect(self._on_model_deleted_button_clicked)
        # self._delete_models_button.clicked.connect.connect(a)
        # self._delete_models_button.clicked.connect.connect(b)
        buttons_layout.addWidget(self._delete_models_button)
        layout.addLayout(buttons_layout)
        return layout

    @property
    def model_names(self):
        return self._list_models_w.options

    @QtCore.pyqtSlot(list, QWidget) 
    def _on_model_selected(self, model_names:List[str]):
        if len(model_names) > 0:
            mname = model_names[0]
            self._list_features_w.set_selection(self._models_confs[mname])
            self.modelSelected.emit(mname, self)
    
    @QtCore.pyqtSlot(list, QWidget) 
    def _train_features_selected(self, features:List[str]):
        mnames = self._list_models_w.selected_options
        if len(mnames) == 0:
            return
        mname = self._list_models_w.selected_options[0]
        self.trainFeaturesSelected.emit(mname, features, self)

    @QtCore.pyqtSlot() 
    def _on_model_deleted_button_clicked(self):
        mnames = self._list_models_w.selected_options
        if len(mnames) == 0:
            BasicDialog(window_title='No Model Selected for Deletion!', message='To delete a model you first need to select it from the list of models!').exec()
            return
        deleted_model_name = mnames[0]
        if deleted_model_name in self._models_confs:
            del self._models_confs[deleted_model_name]
        self._list_models_w.update_options(list(self._models_confs.keys()))
        self.modelDeleted.emit(deleted_model_name, self)

    @QtCore.pyqtSlot() 
    def _on_train_model_button_clicked(self):
        mnames = self._list_models_w.selected_options
        if len(mnames) == 0:
            BasicDialog(window_title='No Model Selected for Training!', message='To train a model you need to select a model from the list of models!').exec()
            return
        if len(self._list_features_w.selected_options) == 0:
            BasicDialog(window_title='No Features Selected for Training!', message='To train a model you need to select at least one feature!').exec()
            return
        mname = self._list_models_w.selected_options[0]
        self._models_confs[mname] = self._list_features_w.selected_options
        self.modelToTrain.emit(mname, self._models_confs[mname], self)
        
    @QtCore.pyqtSlot() 
    def _on_create_empty_train_model_button_clicked(self):
        mname = self._base_model_name.format(self._model_count)
        self._model_count += 1
        self._models_confs[mname] = list()
        self._list_models_w.update_options(list(self._models_confs.keys()))

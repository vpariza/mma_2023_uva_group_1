from PyQt6.QtWidgets import QWidget, QGridLayout, QLabel, QVBoxLayout
from PyQt6.QtWidgets import QGridLayout, QLabel
from PyQt6.QtGui import QPixmap
import os
import numpy as np
import random

class SentenceWidget(QWidget):
    def __init__(self, data_size=None, max_sentences=5):
        super().__init__()
        
        self.sentences = None
        self.selected_sentences = []
        self.selected_points = []
        self.max_sentences = max_sentences
        self.data_size = data_size
        
        self.layout = QVBoxLayout()
        
        header_label = QLabel("Sample Sentences")
        self.layout.addWidget(header_label)

        self.bullet_points = []
        for _ in range(self.max_sentences):
            bullet_point = QLabel("")
            bullet_point.setWordWrap(True)
            self.layout.addWidget(bullet_point)
            self.bullet_points.append(bullet_point)

        self.setLayout(self.layout)

    def update_sentences(self, sentences):
        self.sentences = sentences
        self.set_selected_points(list(range(len(self.sentences))))

    def update(self):
        """Update the widget with the current sentences"""
        sentence = "test sentence"
        for i, sentence in enumerate(self.selected_sentences):
            self.bullet_points[i].setText(f"• {sentence}")
        if len(self.selected_sentences) < self.max_sentences:
            for i in range(len(self.selected_sentences), self.max_sentences):
                self.bullet_points[i].setText("")

    def set_selected_points(self, selected_points):
        """Method that sets the selected sentences and updates the widget"""
        if self.sentences is None:
            return
        if selected_points==[]:
            #self.selected_sentences = random.sample(self.sentences, min(len(self.sentences), self.max_sentences))
            self.selected_points = []
            self.selected_sentences = []
        else:
            self.selected_points = selected_points
            self.selected_sentences = random.sample(self.sentences.iloc[self.selected_points].tolist(),min(len(self.sentences.iloc[self.selected_points]),self.max_sentences))
        
        self.update()
        
        
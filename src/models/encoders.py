import os, sys
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import h5py
import clip

device = "cuda" if torch.cuda.is_available() else "cpu"

class Model():
    def __init__(self, **kwargs):
        pass

    def get_similarity_ranking(self, query, top_k=None):
        pass
    
    def get_similarity_scores(self, query, normalize=False):
        raise NotImplementedError

    def cosine_similarity(self, a, b):
        return (np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))).squeeze(0)


class VisionModel(Model):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.model_name = kwargs.get('model_name', 'ViT-B/32')
        self.model, self.preprocess = clip.load(self.model_name, device=device)

        # Load precomputed clip embeddings for images
        image_features_path = kwargs.get('image_features_path', 'data/funda_image_features.h5')
        with h5py.File(image_features_path, "r") as hf:
            self.image_features = hf["image_features"][:]
        
    def get_similarity_scores(self, query, normalize=False):
        query = clip.tokenize(query).to(device)
        text_features = self.model.encode_text(query).detach().cpu().numpy()
        similarity_scores = self.cosine_similarity(text_features, self.image_features.T)
        if normalize:
            similarity_scores /= np.max(similarity_scores)
        return similarity_scores



class LanguageModel(Model):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        
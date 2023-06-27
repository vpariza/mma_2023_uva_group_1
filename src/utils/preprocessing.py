import configparser
import umap
import pickle
import os

import pandas as pd
from sklearn.manifold import TSNE

import numpy as np 

class Preprocessing():
    def __init__(self):
        super().__init__()

    def compute_umap(self, image_features):
        return umap.UMAP(n_neighbors=6, n_components=2, metric='cosine').fit_transform(image_features)
    
    def compute_tsne(self, image_features):
        return TSNE(n_components=2, perplexity=30, n_iter=1000, metric='cosine').fit_transform(image_features)
    
    def load_data(self):
        config = configparser.ConfigParser()
        config.read('config.ini')
        
        data_csv_path = config['main']['csv_path']
        image_dir_path = config['main']['images_dir_path']
        
        # load data
        df = pd.read_csv(data_csv_path)
        df = df.set_index("funda_identifier", drop=False)

        # restrict number of considered samples
        num_samples = int(config['main']['num_samples'])
        if num_samples > 0:
            df = df[:num_samples]
        
        # load image features and append to dataframe
        with open(config['main']['image_features_path'], "rb") as f:
            image_df = pickle.load(f)
        image_df = image_df.set_index(["funda_id"])
        image_df = image_df[image_df["image_id"] == "1"]
        image_df["image_path"] = image_df.index.astype(str) + f"/image1.jpeg"
        df = df.merge(image_df, left_index=True, right_index=True, how="left")

        return config, df, image_dir_path

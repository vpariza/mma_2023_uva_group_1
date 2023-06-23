import configparser
import h5py, umap

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
        
        data_path = config['main']['pkl_path']
        data_csv_path = config['main']['csv_path']
        # 16.865 image folders 
        images_path = config['main']['images_path']
        image_dir_path = config['main']['images_dir_path']
        
        # config - 1000
        num_samples = int(config['main']['num_samples'])
        
        sample_selection = str(config['main']['sample_selection'])
        ## dataloading 1000, 7
        #df = pd.read_pickle(data_path)
        
        #print('--------------------- Dataset ---------------------')
        df = pd.read_csv(data_csv_path)
        #print(df.head())
        

        with h5py.File(images_path, "r") as hf:
            # Shape: (1000, 512)
            image_features = hf["image_features"][:]
        
                        
        ## select num_samples samples between
        if sample_selection == 'new':
            image_features = image_features[:num_samples]
            points = self.compute_umap(image_features)
            tags = []
            image_features = image_features[:num_samples]
            img_paths = []
            df = df[:num_samples]
            df['umap_x'] = points[:,0]
            df['umap_y'] = points[:,1]

        if sample_selection == 'random':
            # Chooses 1000 random datapoints
            random_indices = np.random.choice(len(df), num_samples, replace=False)
            tags = df['tags'].iloc[random_indices].values
            points = df[['umap_x','umap_y']].iloc[random_indices].values
            image_features = image_features[random_indices]
            img_paths = df['filepaths'].iloc[random_indices].values

        if sample_selection == 'first':
            tags = df['tags'].iloc[:num_samples].values
            points = df[['umap_x','umap_y']].iloc[:num_samples].values
            image_features = image_features[:num_samples]
            img_paths = df['filepaths'].iloc[:num_samples].values
        ## recompute the embedding coordinates
        if bool(config['main']['recompute_embedding']):
            if str(config['main']['embedding'])=='umap':
                points = self.compute_umap(image_features)
            elif str(config['main']['embedding'])=='tsne':
                points = self.compute_tsne(image_features)
        
        return config, tags, points, img_paths, df, image_dir_path

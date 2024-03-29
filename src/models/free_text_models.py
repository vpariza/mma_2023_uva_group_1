import os, sys
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import clip
import pickle
from sentence_transformers import SentenceTransformer
from nltk.tokenize import sent_tokenize

device = "cuda" if torch.cuda.is_available() else "cpu"

class Model():
    """
    Base class for all models implementing basic functionality
    """
    def __init__(self, **kwargs):
        self.model = None
        self.modality = None
        self.features = None
        self.feature_name = None
        self.precomputed_features_path = kwargs.get("precomputed_features_path", None)
        assert self.precomputed_features_path is not None and os.path.exists(self.precomputed_features_path), "No precomputed features path provided"

    def load_precomputed_features(self):
        """
        Load precomputed features from pickle file
        """
        with open(self.precomputed_features_path, "rb") as f:
            self.df = pickle.load(f)
        self.df.set_index(["funda_id", f"{self.modality}_id"], inplace=True, drop=True)
        self.features = np.array(self.df[f"{self.modality}_features"].tolist())
    
    def encode_query(self, query):
        """
        Encode a query into a vector representation
        
        Args:
            query (str): Query string to be encoded
        Returns:
            (np.array): Vector representation of the query
        """
        raise NotImplementedError

    def calculate_similarity(self, query, df, feature_name=None):
        """
        Calculate the similarity between a query and a specific or multiple instance(s)
        
        Args:
            query (str): Query string for ranking
            df (pd.DataFrame): Dataframe to which to add the results
            feature_name (str): Name of the column in the dataframe to store the similarity scores
        Returns:
            (pd.DataFrame): Dataframe containing results
        """

        if self.features is None:
            self.load_precomputed_features()

        if feature_name is None:
            feature_name = query.lower().replace(" ", "_")
        self.feature_name = feature_name + f"_{self.modality}_similarity"

        query_features = self.encode_query(query)

        similarity_scores = self.cosine_similarity(query_features, self.features)
        print(similarity_scores)

        max_scores, max_scores_ids, max_scores_idxs = self.max_aggregation(similarity_scores)

        df = self.merge_to_df(df, max_scores, max_scores_ids, max_scores_idxs)

        return df

    def max_aggregation(self, scores):
        """
        Aggregate scores for each funda_id by taking the maximum score for each image / sentence
        
        Args:
            scores (np.array): Array of scores for each image / sentence
        Returns:
            max_scores (pd.DataFrame): Dataframe with columns funda_id, image_id, score
            max_scores_ids (list of tuples): List of tuples (funda_id, x_id) with x_id the index of the image / sentence with the highest score
        """
        self.df[self.feature_name] = scores
        max_scores_idxs = self.df.groupby("funda_id")[self.feature_name].idxmax()
        max_scores_ids = max_scores_idxs.apply(lambda x: x[1])
        max_scores_ids.name = self.feature_name + "-max_id"
        max_scores = self.df.loc[max_scores_idxs][self.feature_name]
        max_scores.name = self.feature_name + "-max_score"
        max_scores.index = max_scores.index.droplevel(1)
        
        return max_scores, max_scores_ids, max_scores_idxs
    
    def merge_to_df(self, df, max_scores, max_scores_ids, max_scores_idxs):
        """
        Merge the similarity scores and indices of items with maximum scores to the original dataframe
        
        Args:
            df (pd.DataFrame): Original dataframe
            max_scores (pd.Series): Series with maximum scores for each funda_id
            max_scores_ids (pd.Series): Series with indices of items with maximum scores for each funda_id
        Returns:
            (pd.DataFrame): Dataframe with containing additional columns
        """
        for col in df.columns:
            if col.startswith(self.feature_name):
                # If the column already exists, drop it
                df.drop(col, axis=1, inplace=True)
        df_merged = df.merge(max_scores, how="left", left_index=True, right_index=True)
        df_merged = df_merged.merge(max_scores_ids, how="left", left_index=True, right_index=True)
        df_merged = self.add_extra_info(df_merged)
        df_merged = self.add_2d_representation(df_merged, max_scores_idxs)
        return df_merged

    def add_extra_info(self, df):
        """
        Add extra information to the dataframe depending on type of model/modality
        
        Args:
            df (pd.DataFrame): Original dataframe
            add_2d_representation (bool): Whether to add a 2D representations (UMAP & t-SNE) of the image
        Returns:
            (pd.DataFrame): Dataframe potentially containing additional columns
        """
        return df
    
    def add_2d_representation(self, df, max_scores_idxs):
        """
        Add 2D representations (UMAP & t-SNE) of the image to the dataframe
        
        Args:
            df (pd.DataFrame): Original dataframe
            max_scores_idxs (pd.Series): Series with indices of items with maximum scores for each funda_id
        Returns:
            (pd.DataFrame): Dataframe containing additional columns
        """
        umap_x = self.df.loc[max_scores_idxs].umap_x
        umap_x.index = umap_x.index.droplevel(1)
        umap_x.name = self.feature_name + "-umap_x"
        df_merged = df.merge(umap_x, how="left", left_index=True, right_index=True)
        umap_y = self.df.loc[max_scores_idxs].umap_y
        umap_y.index = umap_y.index.droplevel(1)
        umap_y.name = self.feature_name + "-umap_y"
        df_merged = df_merged.merge(umap_y, how="left", left_index=True, right_index=True)
        tsne_x = self.df.loc[max_scores_idxs].tsne_x
        tsne_x.index = tsne_x.index.droplevel(1)
        tsne_x.name = self.feature_name + "-tsne_x"
        df_merged = df_merged.merge(tsne_x, how="left", left_index=True, right_index=True)
        tsne_y = self.df.loc[max_scores_idxs].tsne_y
        tsne_y.index = tsne_y.index.droplevel(1)
        tsne_y.name = self.feature_name + "-tsne_y"
        df_merged = df_merged.merge(tsne_y, how="left", left_index=True, right_index=True)
        return df_merged
    
    def cosine_similarity(self, a, b):
        """
        Compute the cosine similarity between two vectors
        
        Args:
            a (np.array): First vector
            b (np.array): Second array of vectors
        Returns:
            (np.array): Cosine similarity between the two vectors
        """
        return (np.dot(b, a.T).squeeze(-1) / (np.linalg.norm(a) * np.linalg.norm(b, axis=1)))
    

class VisionModel(Model):
    """
    CLIP-based image retrieval and ranking model
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.modality = "image"
        self.model_name = kwargs.get('model_name', 'ViT-B/32')
        self.model, self.preprocess = clip.load(self.model_name, device=device)

    def encode_query(self, query):
        """
        Encode a query into a vector representation using CLIP text encoder
        
        Args:
            query (str): Query string to be encoded
        Returns:
            (np.array): Vector representation of the query
        """
        query = clip.tokenize(query).to(device)
        query_features = self.model.encode_text(query).detach().cpu().numpy()
        return query_features



class LanguageModel(Model):
    """
    BERT-based text retrieval and ranking model
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.modality = "text"
        self.model_name = kwargs.get('model_name', 'krlng/sts-GBERT-bi-encoder')
        self.model = SentenceTransformer(self.model_name)

    def encode_query(self, query):
        """
        Encode a query into a vector representation using BERT text encoder
        
        Args:
            query (str): Query string to be encoded
        Returns:
            (np.array): Vector representation of the query
        """
        query_features = self.model.encode(query)
        query_features = np.expand_dims(query_features, axis=0)
        return query_features
    
    def add_extra_info(self, df):
        """
        Add extra information to the dataframe depending on type of model/modality
        
        Args:
            df (pd.DataFrame): Original dataframe
            max_scores_ids (pd.Series): Series with indices of items with maximum scores for each funda_id
        Returns:
            (pd.DataFrame): Dataframe potentially containing additional columns
        """
        def get_nth_sentence(data):
            n = data[self.feature_name + "-max_id"]
            description = data["description"]
            return sent_tokenize(description)[n]

        if df[self.feature_name + "-max_id"].max() > 1:
            df[self.feature_name + "-max_sentence"] = df.apply(get_nth_sentence, axis=1)
        return df
        
import pandas as pd
import numpy as np
import torch
import pickle
from sentence_transformers import SentenceTransformer
from nltk.tokenize import sent_tokenize
import clip

import matplotlib.pyplot as plt


device = "cuda" if torch.cuda.is_available() else "cpu"


def cosine_similarity(a, b):
    return (np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))).squeeze(0)
    

def aggregate_scores(embedding_df, scores):
    embedding_df["scores"] = scores
    indices = embedding_df.groupby("funda_id")["scores"].idxmax()
    return indices


def get_description_similarity(query):

    # loading precomputed text features
    precomputed_text_features_path = "dataloading/Funda/real_estate_sentence_features.pkl"
    with open(precomputed_text_features_path, "rb") as f:
        text_df = pickle.load(f)
        text_df.set_index(["funda_id", "text_id"], inplace=True, drop=True)
        text_features = np.array(text_df["text_features"].tolist())

    # query embedding
    model = SentenceTransformer('krlng/sts-GBERT-bi-encoder')
    q_embedding = model.encode(query)
    q_embedding = np.expand_dims(q_embedding, axis=0)

    # Calculate cosine similarity
    similarity_scores = cosine_similarity(q_embedding, text_features.T)

    # Max aggregation of scores
    max_scores_idxs = aggregate_scores(text_df, similarity_scores)
    max_scores_ids = max_scores_idxs.apply(lambda x: x[1])
    max_scores_ids.name = query + "-text_embedding-" + "max_id"
    max_scores = text_df.loc[max_scores_idxs].scores
    max_scores.name = query + "-text_embedding-" + "max_score"

    return max_scores, max_scores_ids


def get_image_similarity(query):

    # loading precomputed image features
    precomputed_image_features_path = "dataloading/Funda/real_estate_image_features.pkl"
    with open(precomputed_image_features_path, "rb") as f:
        image_df = pickle.load(f)
        image_df.set_index(["funda_id", "image_id"], inplace=True, drop=True)
        image_features = np.array(image_df["image_features"].tolist())

    # query embedding
    model, preprocess = clip.load("ViT-B/32", device=device)
    q_embedding = model.encode_text(clip.tokenize(query).to(device)).detach().cpu().numpy()

    # Calculate cosine similarity
    similarity_scores = cosine_similarity(q_embedding, image_features.T)

    # Max aggregation of scores
    max_scores_idxs = aggregate_scores(image_df, similarity_scores)
    max_scores = image_df.loc[max_scores_idxs].scores
    max_scores.name = query + "-image_embedding-" + "max_score"
    max_scores_ids = max_scores_idxs.apply(lambda x: x[1])
    max_scores_ids.name = query + "-image_embedding-" + "max_id"

    return max_scores, max_scores_ids


def get_nth_sentence(data, max_scores_ids_name):
    n = data[max_scores_ids_name]
    description = data["description"]
    return sent_tokenize(description)[n]


def merge_to_main_df(df, text_max_scores, image_max_scores, text_max_scores_ids=None, image_max_scores_ids=None):

    # Text features
    text_max_scores.index = text_max_scores.index.droplevel(1)
    df_merged = df.merge(text_max_scores, how="left", left_index=True, right_index=True)

    if text_max_scores_ids is not None:
        # Add the max id
        df_merged = df_merged.merge(text_max_scores_ids, how="left", left_index=True, right_index=True)
        # Add the max sentence
        df_merged[query + "-max_sentence"] = df_merged.apply(get_nth_sentence, axis=1, args=(text_max_scores_ids.name,))

    # Image features
    image_max_scores.index = image_max_scores.index.droplevel(1)
    df_merged = df_merged.merge(image_max_scores, how="left", left_index=True, right_index=True)

    if image_max_scores_ids is not None:
        # Add the max id
        df_merged = df_merged.merge(image_max_scores_ids, how="left", left_index=True, right_index=True)

    return df_merged


if __name__ == "__main__":

    # Load dataset
    print("Loading dataset...")
    csv_path = "dataloading/Funda/dataset.csv"
    df = pd.read_csv(csv_path, dtype=str, index_col=0)
    df = df.set_index('funda_identifier', drop=True)
    df.index = df.index.astype(int)

    # QUERY 1

    print("Embedding query and calculating similarity scores...")
    query = "typical dutch house"
    text_max_scores, text_max_scores_ids = get_description_similarity(query)
    image_max_scores, image_max_scores_ids = get_image_similarity(query)

    # Merge to main df
    print("Merging to main df...")
    df_merged = merge_to_main_df(
        df, text_max_scores, image_max_scores, text_max_scores_ids, image_max_scores_ids
        )
    
    # Print column names
    print("Columns:", df_merged.columns)
    
    # Print results for text embedding
    df_sorted = df_merged.sort_values(text_max_scores.name, ascending=False)
    print(df_sorted[query + "-max_sentence"].head(5))
    print(df_sorted[query + "-max_sentence"].tail(5))

    # Print results for image embedding
    df_sorted = df_merged.sort_values(image_max_scores.name, ascending=False)
    print(df_sorted[image_max_scores_ids.name].head(5))
    print(df_sorted[image_max_scores_ids.name].tail(5))


    # QUERY 2

    print("Embedding query and calculating similarity scores...")
    query = "a princess castle"
    text_max_scores, text_max_scores_ids = get_description_similarity(query)
    image_max_scores, image_max_scores_ids = get_image_similarity(query)

    # Merge to main df
    print("Merging to main df...")
    df_merged = merge_to_main_df(
        df_merged, text_max_scores, image_max_scores, text_max_scores_ids, image_max_scores_ids
        )
    
    # Print column names
    print("Columns:", df_merged.columns)
    
    # Print results for text embedding
    df_sorted = df_merged.sort_values(text_max_scores.name, ascending=False)
    print(df_sorted[query + "-max_sentence"].head(5))
    print(df_sorted[query + "-max_sentence"].tail(5))

    # Print results for image embedding
    df_sorted = df_merged.sort_values(image_max_scores.name, ascending=False)
    print(df_sorted[image_max_scores_ids.name].head(5))
    print(df_sorted[image_max_scores_ids.name].tail(5))

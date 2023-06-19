import os, sys
import json
import re
import traceback

import pandas
from PIL import Image
import clip
import pandas as pd
import umap
import argparse
import torch
from torchvision.transforms import functional as F
from tqdm import tqdm
import h5py
import numpy as np
from sklearn.manifold import TSNE
from nltk.tokenize import sent_tokenize

from sentence_transformers import SentenceTransformer


NUM_DATAPOINTS = 1024
BATCH_SIZE = 32


def get_clip_embeddings(model_name, device, df, image_path):
    # load the model
    model, preprocess = clip.load(model_name, device=device)
    funda_ids = []
    image_ids = []
    images = []
    image_features = []
    skipped_indices = []

    with torch.no_grad():
        print("Loading images...")
        for i, funda_id in tqdm(enumerate(df.index)):
            for image_name in df.loc[funda_id]['images_paths']:
                image_id = image_name.split('/')[-1].split('.')[0].replace("image", "")
                ## load the image, preprocess them and load clip embeddings
                try:
                    image = preprocess(Image.open(os.path.join(image_path + image_name))).unsqueeze(0).to(device)
                    #image_input = image.unsqueeze(0).to(device)
                    #image_features.append(model.encode_image(image_input).cpu().numpy())
                    funda_ids.append(funda_id)
                    image_ids.append(image_id)
                except Exception as ex:
                    traceback.print_exc()
                    skipped_indices.append(i)

        print("Computing image features...")
        for batch_idx in tqdm(range(NUM_DATAPOINTS // BATCH_SIZE)):
            try:
                batch_embeddings = model.encode_image(image[batch_idx*BATCH_SIZE : (batch_idx+1)*BATCH_SIZE]).cpu().numpy()
                image_features.append(batch_embeddings)
            except Exception as ex:
                traceback.print_exc()
                skipped_indices.append(batch_idx)
    # Print skipped indices
    print("Skipped Indices:", skipped_indices)

    df = pd.DataFrame({'funda_id': funda_ids, 'image_id': image_ids, 'image_features': image_features})
    # df.to_pickle(output_file)

    # # Save image features to an H5 file
    # image_features = np.concatenate(image_features, axis=0)

    # with h5py.File(output_file, "w") as hf:
    #     hf.create_dataset("image_features", data=image_features)
    return df


def get_bert_embeddings(df, device, split_documents=False):
    
    skipped_indices = []

    # preprocess the text
    funda_ids = []
    text_ids = []
    texts = []
    text_features = []
    for funda_id, doc in df[["funda_id", 'description']]:
        if split_documents:
            text_id = 0
            for text in sent_tokenize(doc):
                texts.append(text)
                funda_ids.append(funda_id)
                text_ids.append(text_id)
                text_id += 1
        else:
            funda_ids.append(funda_id)
            texts.append(doc)
            text_ids.append(0)
        
    texts = [re.sub(r'\s+', ' ', t) for t in texts]
    texts = [t.strip() for t in texts]

    # load the model
    model = SentenceTransformer('krlng/sts-GBERT-bi-encoder').to(device)

    total_batches = NUM_DATAPOINTS // BATCH_SIZE
    with torch.no_grad():
        for batch_idx in tqdm(range(total_batches)):
            try:
                batch_embeddings = model.encode(texts[batch_idx*BATCH_SIZE : min(len(texts) - 1, (batch_idx+1)*BATCH_SIZE)])
                text_features.append(batch_embeddings)
            except Exception as ex:
                traceback.print_exc()
                skipped_indices.append(batch_idx)
    # Print skipped indices
    print("Skipped Indices:", skipped_indices)

    text_features = np.concatenate(text_features, axis=0)

    df = pd.DataFrame({'funda_id': funda_ids, 'text_id': text_ids, 'text_features': text_features})

    return df


def make_umap(data, n_neighbors=6, n_components=2, metric='cosine'):
    return umap.UMAP(n_neighbors=n_neighbors, n_components=n_components, metric=metric).fit_transform(data)


def make_tsne(data, n_components=2, perplexity=30, n_iter=1000, metric='cosine'):
    return TSNE(n_components=n_components, perplexity=perplexity, n_iter=n_iter, metric=metric).fit_transform(data)


def process_data(df, output_path, image_path, model_name, device, **kwargs):
    """
    pre-compute image-specific features
    """
    image_feature_df = get_clip_embeddings(model_name, device, df, image_path=image_path)

    # add the UMAP coordinates to the dataframe
    image_features = image_feature_df['image_features'].tolist()
    umap_embeddings = make_umap(image_features)
    image_feature_df['umap_x'] = umap_embeddings[:, 0]
    image_feature_df['umap_y'] = umap_embeddings[:, 1]

    # add the TSNE coordinates to the dataframe
    tsne_embeddings = make_tsne(image_features)
    image_feature_df['tsne_x'] = tsne_embeddings[:, 0]
    image_feature_df['tsne_y'] = tsne_embeddings[:, 1]

    # save the dataframes
    image_feature_df.to_pickle(output_path.replace('.', '_image_features.'))

    """
    pre-compute text-specific features
    """
    text_features_df = get_bert_embeddings(df, device, split_documents=False)

    # add the UMAP coordinates to the dataframe
    text_features = text_features_df['text_features'].tolist()
    umap_embeddings = make_umap(text_features)
    text_features_df['umap_x'] = umap_embeddings[:, 0]
    text_features_df['umap_y'] = umap_embeddings[:, 1]

    # add the TSNE coordinates to the dataframe
    tsne_embeddings = make_tsne(text_features)
    text_features_df['tsne_x'] = tsne_embeddings[:, 0]
    text_features_df['tsne_y'] = tsne_embeddings[:, 1]

    # save the dataframes
    text_features_df.to_pickle(output_path.replace('.', '_text_features.'))

    """
    pre-compute text-specific features with sentence splitting
    """
    text_features_df = get_bert_embeddings(df, device, split_documents=True)

    # add the UMAP coordinates to the dataframe
    text_features = text_features_df['text_features'].tolist()
    umap_embeddings = make_umap(text_features)
    text_features_df['umap_x'] = umap_embeddings[:, 0]
    text_features_df['umap_y'] = umap_embeddings[:, 1]

    # add the TSNE coordinates to the dataframe
    tsne_embeddings = make_tsne(text_features)
    text_features_df['tsne_x'] = tsne_embeddings[:, 0]
    text_features_df['tsne_y'] = tsne_embeddings[:, 1]

    # save the dataframes
    text_features_df.to_pickle(output_path.replace('.', '_sentence_features.'))
    
    return image_feature_df, text_features_df

def old_parse_real_estate_json(json_path):
    with open(json_path, 'r') as ads_json:
        images_paths = []
        labels = []
        parsed = 0
        while parsed < NUM_DATAPOINTS:
            ad = json.loads(ads_json.readline())
            if 'construction' in ad['features'] and 'kind of house' in ad['features']['construction']:
                images_paths.append(ad['images_paths'][0])
                house_type = re.sub('\(.*\)', '', ad['features']['construction']['kind of house'])
                labels.append(house_type)
                parsed += 1
        return pandas.DataFrame({'labels': labels, 'filepaths': images_paths})
    
def parse_real_estate_json(json_path):
    df = pd.read_json(json_path, lines=True)
    df = df.set_index("funda_identifier")
    return df
    
def argparser():
    parser = argparse.ArgumentParser(description='Load the Funda ads jsonlines file containing image_paths to real estate images and save the table in pickle format with an added column: "clip_embeddings" which contains the CLIP embedding of the images and produce a column "umap_x" and "umap_y" which contains the UMAP coordinates of the images based on the clip embeddings')
    parser.add_argument('--json_path', type=str, help='path to the json file containing the image paths', default='./dataloading/Funda/ads_trans.jsonlines')
    parser.add_argument('--output_path', type=str, help='path to the output pickle file',default='./dataloading/Funda/real_estate.pkl')
    parser.add_argument('--image_path', type=str, help='path to the folder containing the images', default='./dataloading/Funda/images/')
    parser.add_argument('--model_name', type=str, default='ViT-B/32', help='name of the CLIP model that should be used')
    parser.add_argument('--device', type=str, default='cuda', help='device that should be used to run the CLIP model on')
    args = parser.parse_args()
    kwargs = vars(args)
    return kwargs

if __name__ == '__main__':
    kwargs = argparser()
    data = parse_real_estate_json(kwargs["json_path"])
    process_data(data, **kwargs)
import sys, os
from xgboost import XGBRegressor
import xgboost as xgb
import pandas as pd
import numpy as np
from src.utils.preprocessing import Preprocessing
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt


class Predictor():
    def __init__(self) -> None:
        self.model = XGBRegressor(n_estimators=1000, learning_rate=0.05, n_jobs=-1, early_stopping_rounds=5)

    def fit(self, train_X: pd.DataFrame, train_y: pd.Series, val_X: pd.DataFrame = None, val_y: pd.Series = None):
        if val_X is None or val_y is None:
            self.model.fit(train_X, train_y)
        else:
            self.model.fit(train_X, train_y, eval_set=[(val_X, val_y)], verbose=False)

    def predict(self, test_X: pd.DataFrame, test_y: pd.Series = None) -> np.ndarray:
        pred_y = self.model.predict(test_X)
        if test_y is not None:
            # TODO: Add evaluation metrics
            print("RMSE: ", np.sqrt(np.mean((pred_y - test_y) ** 2)))
        return pred_y
    
    def get_feature_importances(self):
        importances = self.model.get_booster().get_score(importance_type='total_gain')
        # normalize importances
        total_importance = sum(importances.values())
        for key in importances:
            importances[key] /= total_importance
        # sort importances
        importances = {k: v for k, v in sorted(importances.items(), key=lambda item: item[1], reverse=True)}
        return importances
    
    def plot_tree(self):
        fig, ax = plt.subplots(figsize=(30, 30))
        xgb.plot_tree(self.model, num_trees=10, ax=ax)
        plt.savefig("temp.pdf")
        
    

if __name__ == "__main__":
    # Load data

    # df = pd.read_json('dataloading/Funda/ads.jsonlines', lines=True, dtype=str)
    df = pd.read_csv('dataloading/Funda/dataset.csv', dtype=str)

    #preprocessing = Preprocessing()
    # TODO: Return sirectory of Listings Images
    #config, tags, points, img_paths, df = preprocessing.load_data()

    #print(df.columns)
    #sys.exit()

    # specify features to use for prediction
    df = df[:100]
    print(df.columns)

    features = {
        'bedrooms': int,
        #'building_type': "category",
        'living_area': int,
        'plot_size': int,
        'year_of_construction': int,
        'lat': float,
        'lon': float,
        #'price': float,
        # 'label': "category",
    }

    # convert features to correct type
    for feature, dtype in features.items():
        df[feature] = df[feature].astype(dtype)
    df['price'] = df['price'].astype(float)

    # print(pd.to_numeric(df["label"]))
    print(df[features.keys()].head())

    # precentage of data to use for training, validation and testing
    train_X, test_X, train_y, test_y = train_test_split(df[features.keys()], df['price'], test_size=0.2, random_state=42)
    test_X, val_X, test_y, val_y = train_test_split(test_X, test_y, test_size=0.5, random_state=42)

    """train_X = df[features.keys()].head((int(len(df)*(train_n/100))))
    train_y = df['price'].head((int(len(df)*(train_n/100))))

    val_X = df[features.keys()].head((int(len(df)*(val_n/100))))
    val_y = df['price'].head((int(len(df)*(val_n/100))))

    test_X = df[features.keys()].head((int(len(df)*(test_n/100))))
    test_y = df['price'].head((int(len(df)*(test_n/100))))"""

    # train model
    predictor = Predictor()
    print("Training model...")
    predictor.fit(train_X, train_y, val_X, val_y)
    print("Done!")

    # predict
    pred_y = predictor.predict(test_X, test_y)
    print([(p, t) for p, t in zip(pred_y, test_y)])

    print("Feature importances:")
    print(predictor.get_feature_importances())

    predictor.plot_tree()

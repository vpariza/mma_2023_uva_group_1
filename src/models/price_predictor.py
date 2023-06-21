import sys, os
from xgboost import XGBRegressor
import xgboost as xgb
import pandas as pd
import numpy as np
import json
from src.utils.preprocessing import Preprocessing
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

def convert_categorical_to_one_hot(df: pd.DataFrame, categorical_features: list) -> pd.DataFrame:
    for feature in categorical_features:
        df = pd.concat([df, pd.get_dummies(df[feature], prefix=feature).astype(float)], axis=1)
        df = df.drop(feature, axis=1)
    return df

"""
Metrics for evaluating price prediction.
"""

def root_mean_squared_error(y_true, y_pred):
    return np.sqrt(np.mean((y_pred - y_true) ** 2))

def mean_squared_error(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)

def mean_absolute_percentage_error(y_true, y_pred):
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

def mean_absolute_error(y_true, y_pred):
    return np.mean(np.abs(y_true - y_pred))

def mean_percentage_error(y_true, y_pred):
    return np.mean((y_true - y_pred) / y_true) * 100

def mean_error(y_true, y_pred):
    return np.mean(y_true - y_pred)

def median_absolute_error(y_true, y_pred):
    return np.median(np.abs(y_true - y_pred))

def median_percentage_error(y_true, y_pred):
    return np.median((y_true - y_pred) / y_true) * 100

def median_error(y_true, y_pred):
    return np.median(y_true - y_pred)

"""
Class for predicting the price of a listing.
"""

class Predictor():
    def __init__(self) -> None:
        self.model = XGBRegressor(n_estimators=1000, learning_rate=0.05, n_jobs=-1, early_stopping_rounds=5)
        self.used_features = []
        self.categorical_features = []
        self.target = None

    def preprocess(self, df: pd.DataFrame, features: list, target: str = "price", transforms: dict = {}) -> pd.DataFrame:
        """
        Preprocess data by convert categorical features to one-hot encoding and potentially applying transform functions.
        
        Args:
            df (pd.DataFrame): Dataframe containing the data.
            features (list): List of features to use for prediction.
            target (str, optional): Name of the target feature. Defaults to "price".
            transforms (dict, optional): Dictionary of transform functions to apply to the data. Defaults to {}.

        Returns:
            pd.DataFrame: Preprocessed dataframe.
        """
        df_processed = df.copy()
        for feature in features:
            if df[feature].dtype == "category":
                df_processed = convert_categorical_to_one_hot(df_processed, [feature])
                self.used_features.extend([c for c in df_processed.columns if c.startswith(feature + "_")])
                self.categorical_features.append(feature)
            else:
                self.used_features.append(feature)
            if feature in transforms:
                df_processed[feature] = transforms[feature](df_processed[feature])

        self.target = target
        assert target not in self.used_features, "Target feature cannot be used as input feature"

        return df_processed
    
    def split_data(self, df: pd.DataFrame, test_size: float = 0.1, val_size: float = 0.1) -> tuple:
        """
        Split data into train, test and validation sets.

        Args:
            df (pd.DataFrame): Dataframe containing the data.
            test_size (float, optional): Percentage of data to use for testing. Defaults to 0.1.
            val_size (float, optional): Percentage of data to use for validation. Defaults to 0.1.
        
        Returns:
            tuple: (train_X, test_X, val_X, train_y, test_y, val_y)
        """
        tmp_perc = test_size + val_size
        test_perc = test_size / tmp_perc
        assert tmp_perc < 1, "test_size + val_size must be less than 1"

        train_X, tmp_X, train_y, tmp_y = train_test_split(df[self.used_features], df[self.target], test_size=tmp_perc, random_state=42)
        test_X, val_X, test_y, val_y = train_test_split(tmp_X, tmp_y, test_size=test_perc, random_state=42)

        return train_X, test_X, val_X, train_y, test_y, val_y

    
    def fit(self, train_X: pd.DataFrame, train_y: pd.Series, val_X: pd.DataFrame = None, val_y: pd.Series = None):
        """
        Fit the model to the data.

        Args:
            train_X (pd.DataFrame): Training data.
            train_y (pd.Series): Training labels.
            val_X (pd.DataFrame, optional): Validation data. Defaults to None.
            val_y (pd.Series, optional): Validation labels. Defaults to None.
        """
        if val_X is None or val_y is None:
            self.model.fit(train_X, train_y)
        else:
            self.model.fit(train_X, train_y, eval_set=[(val_X, val_y)], verbose=False)

    def predict(self, test_X: pd.DataFrame, test_y: pd.Series = None, eval_metrics: list = []) -> tuple:
        """
        Predict the target feature for the test data.
        
        Args:
            test_X (pd.DataFrame): Test data.
            test_y (pd.Series, optional): Test labels. Defaults to None.
            eval_metrics (list, optional): List of evaluation metric function to use. Defaults to [].

        Returns:
            tuple: (pred_y, metric_scores) (Predicted labels and evaluation metric scores as np.array)
        """
        pred_y = self.model.predict(test_X)
        metric_scores = {}
        if test_y is not None:
            for metric in eval_metrics:
                metric_score = metric(test_y, pred_y)
                metric_scores[metric.__name__] = metric_score
                print(metric.__name__, metric_score)
        return pred_y, metric_scores
    
    def get_feature_importances(self):
        """
        Get feature importances from the model.
        
        Returns:
            dict: Dictionary of feature importances.
        """
        importances = self.model.get_booster().get_score(importance_type='total_gain')
        # subsume one-hot encoded features into single feature
        for feature in self.categorical_features:
            feature_importance = 0
            for key in importances.copy():
                if key.startswith(feature + "_"):
                    feature_importance += importances[key]
                    del importances[key]
            importances[feature] = feature_importance
        # normalize importances to sum to 1
        total_importance = sum(importances.values())
        for key in importances:
            importances[key] /= total_importance
        # sort importances
        importances = {k: v for k, v in sorted(importances.items(), key=lambda item: item[1], reverse=True)}
        return importances
    
    def plot_tree(self, num_trees: int = 10) -> tuple:
        """
        Plot a tree from the model.
        
        Args:
            num_trees (int, optional): Tree number to plot. Defaults to 10.
            
        Returns:
            tuple: (fig, ax) (Figure and axis objects.)
        """
        fig, ax = plt.subplots(figsize=(30, 30))
        xgb.plot_tree(self.model, num_trees=num_trees, ax=ax)
        # plt.savefig("temp.pdf")
        return fig, ax

    def save_model(self, path: str):
        """
        Save the model to a file.
        """
        self.model.save_model(path)
        extra_information = {
            "categorical_features": self.categorical_features,
            "used_features": self.used_features,
            "target": self.target,
        }
        with open(path.replace(".json", "_extra.json"), "w") as f:
            json.dump(extra_information, f)

    def load_model(self, path: str):
        """
        Load the model from a file.
        """
        with open(path.replace(".json", "_extra.json"), "r") as f:
            extra_information = json.load(f)
        self.categorical_features = extra_information.get("categorical_features", [])
        self.used_features = extra_information.get("used_features", [])
        self.target = extra_information.get("target", "price")
        
        self.model.load_model(path)
        
    

if __name__ == "__main__":
    """
    Demo of how to use the Predictor class.
    #TODO remove again after proper implementation
    """

    df = pd.read_csv('dataloading/Funda/dataset.csv', dtype=str)

    features = {
        'bedrooms': int,
        'building_type': "category",
        'living_area': int,
        'plot_size': int,
        'year_of_construction': int,
        'lat': float,
        'lon': float,
        #'price': float,
        'label': "category",
    }
    feature_list = list(features.keys())

    # convert features to correct type
    for feature, dtype in features.items():
        df[feature] = df[feature].astype(dtype)
    df["price"] = df["price"].astype(float)

    predictor = Predictor()

    print("Preprocessing data...")
    transofrmations = {
        "lat": lambda x: x - df["lat"].min() / df["lat"].max() - df["lat"].min(),
        "lon": lambda x: x - df["lon"].min() / df["lon"].max() - df["lon"].min(),
    }
    df = predictor.preprocess(df, feature_list, target="price", transforms=transofrmations)

    print("Training model...")
    train_X, test_X, val_X, train_y, test_y, val_y = predictor.split_data(df)
    predictor.fit(train_X, train_y, val_X, val_y)

    print("Predicting...")
    pred_y, scores = predictor.predict(test_X, test_y, eval_metrics=[mean_absolute_error, mean_absolute_percentage_error])
    # for p, t in zip(pred_y, test_y):
    #     print(p, t)
    print(scores)

    print("Feature importances:")
    print(predictor.get_feature_importances())

    print("Saving model...")
    predictor.save_model("test_model.json")

    print("Loading model...")
    predictor2 = Predictor()
    predictor2.load_model("test_model.json")

    print("Predicting with loaded model...")
    pred_y, scores = predictor2.predict(test_X, test_y, eval_metrics=[mean_absolute_error, mean_absolute_percentage_error])
    print(scores)

    # predictor.plot_tree()

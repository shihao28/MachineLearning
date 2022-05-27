import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin


class FeatureEng(BaseEstimator, TransformerMixin):
    def __init__(self, features_names):
        self.features_names = features_names

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        # Reproduce original copy of data frame
        X_copy = pd.DataFrame(
            X[:, :len(self.features_names)], columns=self.features_names)
        X_new = pd.DataFrame()

        # Feature Engineering
        X_new['rooms_per_household'] = X_copy['total_rooms'] / X_copy['households']
        X_new['population_per_household'] = X_copy['population'] / X_copy['households']
        X_new['bedrooms_per_room'] = X_copy['total_bedrooms'] / X_copy['total_rooms']
        X = np.concatenate([X, X_new.values], 1)

        return X

from sklearn.base import TransformerMixin, BaseEstimator
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

from sklearn.preprocessing import MinMaxScaler, StandardScaler, OneHotEncoder, OrdinalEncoder
from sklearn.impute import SimpleImputer
import pandas as pd
import numpy as np
import sklearn
from scipy import stats


class FrequencyEncoder(BaseEstimator, TransformerMixin):
    def __init__(self, handle_unknown='fill_zero'):
        self.fmap = {}
        self.handle_unknown = handle_unknown

    def fit(self, X, y=None):
        X = pd.DataFrame(X)
        for col in X.columns:
            self.fmap[col] = X[col].value_counts().to_dict()
        return self

    def transform(self, X, y=None):
        X = pd.DataFrame(X)
        for col in X.columns:
            X[col] = X[col].astype(object).map(self.fmap[col])
            if self.handle_unknown is None:
                if X[col].isna().sum() > 0:
                    raise Exception(f"There's some unknown values in {col}")
            elif self.handle_unknown == 'fill_mean':
                mean = np.mean(list(self.fmap[col].values()))
                X[col] = X[col].fillna(mean)
            elif self.handle_unknown == 'fill_mode':
                mode = stats.mode(list(self.fmap[col].values()))
                X[col] = X[col].fillna(mode)
            elif self.handle_unknown == 'fill_zero':
                X[col] = X[col].fillna(0)
        return X.values


class ColumnPreprocessor(TransformerMixin, BaseEstimator):
  def __init__(self, transformers, remainder='drop', sparse_threshold=0.3, 
               n_jobs=None, transformer_weights=None, verbose=False, to_dataframe=True):
    self.transformers = transformers
    self.remainder = remainder
    self.sparse_threshold = sparse_threshold
    self.n_jobs = n_jobs
    self.transformer_weights = transformer_weights
    self.verbose = verbose
    self.to_dataframe = to_dataframe

  def get_feature_names(self):
    column_names = []
    for name, func, columns in self.column_transformer.transformers_:
      if type(func) == sklearn.pipeline.Pipeline:
        for name_, func_ in func.steps:
          if hasattr(func_, 'get_feature_names'):
            columns = list(func_.get_feature_names(columns))
      elif func == 'drop':
        columns = []
      else:
        if hasattr(func, 'get_feature_names'):
          columns = list(func.get_feature_names(columns))
      column_names+=columns
    return column_names

  def column_transformer_to_dataframe(self, X):
    return pd.DataFrame(X, columns=self.get_feature_names())

  def fit(self, X, y=None):
    X = X.copy()
    self.column_transformer = ColumnTransformer(self.transformers, remainder=self.remainder, 
                                                sparse_threshold=self.sparse_threshold, 
                                                n_jobs=self.n_jobs, transformer_weights=self.transformer_weights, 
                                                verbose=self.verbose)
    self.column_transformer.fit(X)
    return self

  def transform(self, X, y=None):
    X = X.copy()
    X = self.column_transformer.transform(X)
    if self.to_dataframe:
      X = self.column_transformer_to_dataframe(X)
    return X 
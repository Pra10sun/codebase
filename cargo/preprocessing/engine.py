from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from cargo.modeling.encoders import OneHotEncoder
from collections import defaultdict
from sklearn.impute import SimpleImputer
from cargo.common import BaseHelpers

import numpy as np
import pandas as pd


class LabelEncoderCust(LabelEncoder):
    def __init__(self):
        super().__init__()

    @property
    def mapping(self):
        # adds a property to sklearn LabelEncoder to return values - encoding pair as a dictionary
        return dict(zip(self.classes_, self.transform(self.classes_)))


class CategoricalDataEngine(BaseHelpers):
    # Class that facilitates categorical data cleaning in preparation for modelling

    def __init__(self, X, **kwargs):
        super(CategoricalDataEngine, self).__init__(**kwargs)
        self.X = X
        self.d_label_encoder = None  # to be initiated from self.run() if label encoding is used
        self.one_hot_encoder = None
        self.d_entries = None
        self.si = None  # Placeholder for simple imputer
        self.oh = None  # Placeholder for One Hot encoder

    def fill_na(self, missing_values=None, strategy='constant', fill_value=None):
        """ Populate NULLs using a SimpleImputer """
        self.si = SimpleImputer(missing_values=missing_values, strategy=strategy, fill_value=fill_value)
        self.X = pd.DataFrame(self.si.fit_transform(self.X), columns=self.X.columns)
        self.log.info('NULLs have been filled')

    def one_hot(self):
        self.oh = OneHotEncoder()
        self.X = self.oh.fit_transform(self.X)


class NumericalDataEngine(BaseHelpers):
    # Class that facilitates numerical data cleaning in preparation for modelling

    def __init__(self, X, **kwargs):
        super(NumericalDataEngine, self).__init__(**kwargs)
        self.df = None
        self.df_processed = None
        self.si = None
        self.X = X
        self.scaler = MinMaxScaler()

    def fill_na(self, missing_values=np.nan, strategy='median', fill_value=None):
        """ Populate NULLs using a SimpleImputer """
        self.si = SimpleImputer(missing_values=missing_values, strategy=strategy, fill_value=fill_value)
        # print(self.X.dtypes)
        self.X = pd.DataFrame(self.si.fit_transform(self.X), columns=self.X.columns)
        self.log.info('NULLs have been filled')


class InputProcessingEngine(BaseHelpers):

    # Class to prepare data for modeling

    def __init__(self, X, **kwargs):
        super(InputProcessingEngine, self).__init__(**kwargs)
        self.X_cat = None
        self.X_cnt = None
        self.cat = None
        self.num = None
        self.df = None
        self.df_processed = None
        self.check_for_invalid_typed(X)
        self.cat, self.num = self.cat_cnt_split(X=X)

    def get(self):
        """ Return join output from Categorical and Numerical engines """
        if self.X_cnt is None:
            return self.cat.X
        elif self.X_cat is None:
            return self.num.X
        else:
            return self.cat.X.join(self.num.X)

    def check_for_invalid_typed(self, df):
        """ Check for entries that must be addressed before transformations """
        datetime_columns = df.select_dtypes(include=[np.datetime64]).columns
        if len(datetime_columns) > 0:
            raise Exception(f'Detected datetime columns. Please transform them first: {list(datetime_columns)}')

    def cat_cnt_split(self, X):
        """
            Split numerical and continuous variables into separate dataframes
            If not categorical/numerical data -> return None
        """
        self.log.info('Splitting data into categorical and numeric subsets...')
        cat_vars = [col for col in X if X[col].dtype == object]
        self.log.info(f'Categorical variables: {cat_vars}')
        if len(cat_vars) == 0:
            self.X_cat = None
        else:
            self.X_cat = X[cat_vars]
        cnt_vars = [col for col in X if X[col].dtype != object]
        self.log.info(f'Continuous variables: {cnt_vars}')
        if len(cnt_vars) > 0:
            self.X_cnt = X[cnt_vars]
        else:
            self.X_cnt = None
        self.log.info('Splitting complete')

        # Initiate encoders
        return CategoricalDataEngine(self.X_cat, log=self.log), NumericalDataEngine(self.X_cnt, log=self.log)

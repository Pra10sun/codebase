from sklearn.preprocessing import LabelEncoder, MinMaxScaler, OneHotEncoder
from collections import defaultdict
import numpy as np
import pandas as pd
from cargo.common import BaseHelpers
# from skutil.preprocessing import SafeLabelEncoder


class LabelEncoderCust(LabelEncoder):
    def __init__(self):
        super().__init__()

    @property
    def mapping(self):
        # adds a property to sklearn LabelEncoder to return values - encoding pair as a dictionary
        return dict(zip(self.classes_, self.transform(self.classes_)))


class CategoricalDataEngine(BaseHelpers):
    # Class that facilitates categorical data cleaning in preparation for modelling

    def __init__(self, fill_na_with='not_specified', one_hot=False, encode=False, **kwargs):
        super(CategoricalDataEngine, self).__init__(**kwargs)
        self.fillna_with = fill_na_with
        self.one_hot = one_hot
        self.encode = encode
        self.d_label_encoder = None  # to be initiated from self.run() if label encoding is used
        self.one_hot_encoder = None
        self.d_entries = None

    def replace_nans(self, df):
        # This function fills None and NaN values with values specified in self.fillna_with
        self.log.info('Replacing invalid values in categorical data')
        df_processed = df.replace({np.nan: self.fillna_with, None: self.fillna_with})
        for col in df_processed:
            df_processed[col] = df_processed[col].astype(str)
        self.log.info('Replacing invalid values in categorical data complete')
        self.log.info(f'Sample: \n{df_processed.head()}')
        return df_processed

    def run(self, df):
        # wrapping function that triggers data cleaning
        self.log.info(f'Executing categorical processes... | one_hot: {self.one_hot}, encode: {self.encode}')
        self.log.info(f'Input sample: {df.head()}')
        df_processed = self.replace_nans(df)  # replace nans with a mask value
        if self.one_hot:  # One hot required
            if not self.one_hot_encoder:  # Encoder does not exist -> fit -> transform
                self.log.info('Fitting one hot encoder')
                self.one_hot_encoder = OneHotEncoder(handle_unknown='ignore')
                self.one_hot_encoder.fit(df_processed)
                self.d_entries = {feature: list(np.unique(df_processed[feature])) for feature in df_processed}
            else:
                self.log.info('One hot encoder exists - fitting the data through existing encoder')
                if len(self.d_entries) == df_processed.shape[1]:
                    self.log.info(f'DF shape dimensions - OKAY')
            # Remove unknown entries from the space - needed for transformation on new data
            self.log.info('Removing unknown entries from feature space...')
            self.log.info(f'Feature space: {self.d_entries}')
            for feature in df_processed:
                df_processed[feature] = df_processed[feature].apply(lambda x: x if x in self.d_entries[feature] else self.fillna_with)
                # Feature names will expand if new features are provided - note features from the fit
            excpected_features = self.one_hot_encoder.get_feature_names()
            space = self.one_hot_encoder.transform(df_processed).toarray()
            df_processed = pd.DataFrame(space, columns=self.one_hot_encoder.get_feature_names(), index=df.index)
            self.log.info(f'DF shape prior to feature reduction to fitted space: {df_processed.shape}')
            df_processed = df_processed[excpected_features]
            self.log.info(f'Categorical sample shape: {df_processed.shape}')
            if df_processed.shape[0] == 1:
                df_processed = pd.DataFrame(df_processed)
            return df_processed
        # !! ONE HOT DOES NOT REQUIRE ENCODINGS ANYMORE
        if self.encode:  # if encode == False, just return data without NaNs
            if not self.d_label_encoder:  # if doesn't exist - create the encoder
                self.log.info('Label encoder is not present - fitting a new one')
                self.d_label_encoder = defaultdict(LabelEncoderCust)  # a dictionary of label encoder instances per column in df
                df_processed = self.encode_data(df_processed)
            else:  # encoder exists - use existing encoder to transform data - for testing model once it's been trained
                df_processed = self.inherit_categorical_data_encoding(df_processed)
        self.log.info(f'Categorical sample shape: {df_processed.shape}')
        return df_processed

    def encode_data(self, X_cat):
        # encodes categorical data with an encoder and stores the encoder mapped against a variable name for further decoding
        self.log.info('Encoding categorical variables...')
        X_cat_encoded = X_cat.apply(lambda x: self.d_label_encoder[x.name].fit_transform(x))
        self.log.info('Encoding categorical variables complete')
        return X_cat_encoded

    def inherit_categorical_data_encoding(self, df):
        # uses existing encoder mapped to a variable to transform encodings back into original terminology
        assert self.d_label_encoder is not None
        self.log.info('Transforming data using existing encodings...')
        self.log.info(f'Available encoders: {self.d_label_encoder}')
        for i, col in enumerate(df):
            self.log.info(f'Columns {i} / {len(df.columns)}')
            encoder = self.d_label_encoder[col]
            self.log.info(f'Unique values in columns: {set(df[col])}')
            try:
                df[col] = encoder.transform(df[col])
            except:
                # encoder.tranform() fails to work with never seen values - use dictionary directly
                self.log.info('Encoder transform failed - likely unknown entries')
                self.log.info('Iterating with dictionary. Go have a coffee...')
                self.log.info(f'Mapping for {col}')
                _map = encoder.mapping
                self.log.info(f'Map: {_map}')
                _l = list(_map)
                def trans(x, _map):
                    print(_map.get(x))
                    return x
                # df[col] = df[col].apply(lambda x: encoder.mapping[x] if x in encoder.mapping else None)
                df[col] = df[col].apply(lambda x: _map.get(x) if x in _l else None)

            # print(encoder.mapping)
        # df_encoded = df.apply(lambda x: self.d_label_encoder[x.name].transform(x))
        return df

    def extract_map(self):
        d_out = {}
        for key in self.d_label_encoder:
            pairs = self.d_label_encoder.get(key).mapping
            pairs_upd = {k: float(pairs[k]) for k in pairs}
            d_out.update({key: pairs_upd})
        return d_out


class NumericalDataEngine(BaseHelpers):
    # Class that facilitates numerical data cleaning in preparation for modelling

    def __init__(self, fillna_with=None, mark_missing=False, scaling=False, **kwargs):
        super(NumericalDataEngine, self).__init__(**kwargs)
        self.mark_missing = mark_missing
        self.fillna_with = fillna_with
        self.df = None
        self.df_processed = None
        self.scaling = scaling
        self.scaler = MinMaxScaler()

    def replace_nans(self, df):
        # Fills NaN and None values with a mask specified in self.fillna_with
        df_processed = df.copy()
        self.log.info('Replacing invalid values in numerical data')
        df_processed = df_processed.replace([-np.inf, np.inf], np.nan)
        df_processed = df_processed.replace({np.nan: self.fillna_with, None: self.fillna_with})
        self.log.info('Replacing invalid values in numerical data complete')
        return df_processed

    def scale(self, df):
        # Scaling numerical data
        df_out = df.copy()
        df_out = pd.DataFrame(self.scaler.fit_transform(df_out), columns=df_out.columns)
        return df_out

    def mark_missing_values(self, df):
        df_out = df.copy()
        original_features = list(df.columns)

        # Create identifiers for missing values
        if self.mark_missing:
            missing_features = [feature + '_missing' for feature in original_features]
            for missing_feature in missing_features:
                df_out[missing_feature] = None
            for feature in original_features:
                df_out[feature + '_missing'] = df_out[feature].apply(lambda x: 1 if ~np.isfinite(x) else 0)

        if self.fillna_with is not None:
            df_out = df_out.fillna(self.fillna_with)
        return df_out

    def run(self, df):
        if self.scaling:  # scale data
            self.log.info("Scaling data")
            df = self.scale(df)
        if self.mark_missing:
            self.log.info('Generating one extra column for each NULL')
            df_out = self.mark_missing_values(df)
        else:
            self.log.info(f'Imputing using substitution with {self.fillna_with}')
            df_out = self.replace_nans(df)
        self.log.info(f'Numerical sample shape: {df_out.shape}')
        return df_out


class InputProcessingEngine(BaseHelpers):

    # Class to prepare data for modeling

    def __init__(self, cat_fillna_with, num_fillna_with, **kwargs):
        super(InputProcessingEngine, self).__init__(**kwargs)
        self.X_cat = None
        self.X_cnt = None
        self.categoricalDataEngine = CategoricalDataEngine(fill_na_with=cat_fillna_with, log=self.log)
        self.numericalDataEngine = NumericalDataEngine(fillna_with=num_fillna_with, log=self.log)
        self.df = None
        self.df_processed = None

    def run(self, X, cat_encode=False, num_encode=False, one_hot=False, encode=True, scaling=False, return_joined=False):
        self.log.info('Input processing engine initiated')
        self.log.info(f'cat_encode = {cat_encode}')
        X_cat, X_cnt = self.cat_cnt_split(X)
        if (X_cat is not None) & cat_encode:
            X_cat = self.categoricalDataEngine.run(X_cat, encode=encode, one_hot=one_hot)
        if (X_cnt is not None) & num_encode:
            X_cnt = self.numericalDataEngine.run(X_cnt, scaling=scaling)
        if return_joined:
            if X_cnt is None:
                return X_cat
            elif X_cat is None:
                return X_cnt
            else:
                return X_cnt.join(X_cat)
        return X_cat, X_cnt

    def cat_cnt_split(self, X):
        # Split numerical and continuous variables into separate dataframes
        # if not categorical/numerical data -> return None
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
        return self.X_cat, self.X_cnt

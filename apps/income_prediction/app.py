from cargo.common import BaseHelpers
from cargo.modeling.classifiers import RandomForestClassifier, MultiOutputRandomForestClassifier
# from sklearn.ensemble import RandomForestClassifier
from cargo.modeling.cross_validation import GroupGridSearchCV
from cargo.evaluation.binary_classifier import BinaryClassifierCurves
import pandas as pd
from decouple import config
import os
import numpy as np


def increment_field(df, column, value):
    """ Increment column in a pd.DataFrame() by <val> """
    df_out = df.copy()
    df_out[column] = df_out[column] + value
    return df_out


class App(BaseHelpers):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.df = None  # Original Data
        self.df_hist = None  # Added historical data
        self.df_train = None  # Train model on this
        self.df_holdout = None  # Evaluate the model on this
        self.target_column = 'target_income_growth'
        self.y = None  # target variable
        self.X = None  # Features
        self.model_params = {
            'estimator__n_estimators': [100, 200],
            # 'criterion': ['gini'],
            # 'max_depth': [None],
            # 'min_samples_leaf': [1],
            # 'min_weight_fraction_leaf': [0],
            # 'max_features': ['auto'],
            # 'max_leaf_nodes': [None],
            # 'min_impurity_decrease': [0],
            # 'bootstrap': [True],
            # 'oob_score': [False],
            # 'n_jobs': [-1],
            # 'random_state': [99],
            # 'verbose': [1],
            # 'warm_start': [False],
            # 'class_weight': [None],
            # 'ccp_alpha': [0],
            # 'max_samples': [None],
        }
        # self.model_params = {key: self.model_params[key][0] for key in self.model_params}
        self.model = MultiOutputRandomForestClassifier(estimator=RandomForestClassifier(), n_jobs=-1)

    def load_data(self):
        """ Load data into RAM """
        path = (os.path.join(config('PYTHONPATH'), 'data/master.csv'))
        self.log.info(f'Loading data from {path}')
        self.df = pd.read_csv(path)
        self.log.info(f'Data has been loaded: {self.df.shape}')

    def concatenate(self):
        """ Add historical time series to each company_id, reported_as_of pairs """
        self.df_hist = self.df.copy()
        self.df_hist['fiscal_year'] = self.df_hist['reported_as_of'].apply(lambda x: int(x.split('-')[0]))
        self.df_hist.drop_duplicates(subset=['company_id', 'fiscal_year'], inplace=True)
        index = ['company_id', 'fiscal_year']
        self.df_hist.set_index(index)
        self.log.info(f'Removing duplicates, row counts: {len(self.df)} -> {len(self.df_hist)}')
        self.log.info(f'Shape before join: {self.df_hist.shape}')
        self.df_hist = self.df_hist.set_index(index).join(
            increment_field(
                df=self.df_hist,
                column='fiscal_year',
                value=1
            ).set_index(index), rsuffix='_n-1'
        ).join(
            increment_field(
                df=self.df_hist,
                column='fiscal_year',
                value=2
            ).set_index(index), rsuffix='_n-2'
        )
        self.log.info(f'Shape after join: {self.df_hist.shape}')
        return self.df_hist

    def bin_income_growth(self, df, target_column_name):
        self.log.info('Binning the targets')
        bins = {
            0: -np.inf,
            1: -1,
            2: -0.5,
            3: 0,
            4: 0.5,
            5: 1,
        }

        def get_bin(val):
            """ Assign the smallest bin for a value """
            for key in sorted(bins, reverse=True):
                if val >= bins[key]:
                    return key
            raise Exception(f'Binning missed a value: {val}')
        df['target'] = df[target_column_name].apply(get_bin)
        return df

    def generate_growth(self, df, save=False):
        """ Define the target variable """
        df_out = df.copy()
        df_out[self.target_column] = (df_out['is_net_income'] - df_out['is_net_income_n-1'])/df_out['is_net_income_n-1']
        if save:
            self.log.warning('Saving target to cache')
            df_out[self.target_column].to_csv(os.path.join(config('PYTHONPATH'), 'cache/target.csv'))
        df_out.dropna(subset=[self.target_column], inplace=True)
        self.log.info(f'Shape after removing NULLs: {len(df)} -> {len(df_out)}')
        return df_out

    def generate_multiclass_y(self, y, save=False):
        """ Create dummies for Y with overlap """
        df_dummies = pd.get_dummies(y)
        for i in range(1, df_dummies.shape[1] - 1):
            for j in range(i+1, df_dummies.shape[1]):
                df_dummies[i] += df_dummies[j]
        if save:
            df_dummies.to_csv(os.path.join(config('PYTHONPATH'), 'cache/y_multiclass.csv'))
        return df_dummies

    def replicate_shared_targets(self, df):
        """ Each entry can be multiple targets, expand the to represent all targets """
        df_out = df.copy().reset_index()
        df_dummies = pd.get_dummies(df_out['target'])
        for i in range(1, df_dummies.shape[1] - 1):
            for j in range(i+1, df_dummies.shape[1]):
                df_dummies[i] += df_dummies[j]

        df_out_dummies = df_out.join(df_dummies)
        assert np.max(df_dummies.values) == 1, 'Error occurred in the loop above'
        assert len(df_out_dummies) == len(df_out), 'Dummies do not match the dimensions of the original df'
        df_target_melt = df_out_dummies.melt(id_vars=list(df_out.columns), value_vars=df_dummies.columns)
        df_target_melt = df_target_melt[df_target_melt['value'] == 1]
        df_target_melt['target'] = df_target_melt['variable'].astype(int)
        df_target_melt.set_index(df.index.names, inplace=True)
        self.log.info(f'Shape after dummies: {df_out.shape} -> {df_target_melt.shape}')
        return df_target_melt

    def withdraw_holdout(self, df):
        """ Splitting the data into training training and holdout """
        self.log.info('Extracting holdout')
        train_idx = df.reset_index()['fiscal_year'].apply(lambda x: x < 2018)
        holdout_idx = df.reset_index()['fiscal_year'].apply(lambda x: x >= 2018)

        df_train = df.reset_index()[train_idx].set_index(df.index.names)
        df_holdout = df.reset_index()[holdout_idx].set_index(df.index.names)
        self.log.info(f'Training / Holdout: {len(df_train)} / {len(df_holdout)}')

        assert len(df_train) > 0 and len(df_holdout) > 0
        assert (len(set(df_train.index) - set(df_holdout.index))) == len(set(df_train.index)), 'Indexes intersect'
        return df_train, df_holdout

    def pre_process(self, df):
        y = df.pop('target')
        y_multi_class = self.generate_multiclass_y(y=y, save=True)
        X = df[[col for col in df.columns if df.dtypes[col] in ['float64', 'int64']]]
        X = self.select_features(X)
        X.fillna(0, inplace=True)

        return X, y_multi_class

    def select_features(self, X):
        """ Remove future values """
        feature = [col for col in X if col[-2::] in ['-1', -2]]
        return X[feature]

    def train(self, X, y):
        gs = GroupGridSearchCV(
            X=X,
            y=y,
            n_splits=3,
            groups=X.reset_index()['company_id'],
            estimator=self.model,
            param_grid=self.model_params
        ).fit(X, y)
        self.model = gs.best_estimator_
        print(self.model.feature_importance(X))

    def evaluate(self, X, y):
        """ Generate evaluation metrics """
        self.log.info('Evaluating performance')
        y_probs = self.model.predict_proba(X)
        for class_ in y:
            y_prob = y_probs[class_][:, 1]
            y_true = y[class_]
            bcc = BinaryClassifierCurves(y_true=y_true, y_prob=y_prob)
            bcc.make_all_plots()
            bcc.save_all_plots(key=os.path.join(config('PYTHONPATH'), 'cache'), prefix=f'class_{class_}')
            bcc.save_all_dfs(key=os.path.join(config('PYTHONPATH'), 'cache'), prefix=f'class_{class_}')

    def run(self):
        self.load_data()
        df = self.concatenate()
        df = self.generate_growth(df, save=False)
        df = self.bin_income_growth(df=df, target_column_name=self.target_column)
        self.df_train, self.df_holdout = self.withdraw_holdout(df)
        X, y = self.pre_process(self.df_train)
        X_test, y_test = self.pre_process(self.df_holdout)
        self.train(X, y)
        self.evaluate(X_test, y_test)


if __name__ == '__main__':
    App(log_level='INFO').run()
from cargo.common import BaseHelpers
from cargo.modeling.classifiers import RandomForestClassifier
from cargo.modeling.cross_validation import GroupGridSearchCV
from cargo.evaluation.binary_classifier import BinaryClassifierCurves
from sklearn.impute import SimpleImputer
from sqlalchemy import create_engine
from decouple import config
import pandas as pd
import numpy as np
import os

features = [
    'company_id',
	'stock_prevmonth_growth',
	'stock_prevsemiyear_growth',
    'stock_prevyear_growth',
    'stock_prevfiveyear_growth',
    'sector_prevmonth_growth',
    'sector_prevsemiyear_growth',
    'sector_prevyear_growth',
    'sector_prevfiveyear_growth',
    'stock_fifty_days_avg',
    'stock_hundred_days_avg',
    'stock_two_hundred_days_avg',
    'sector_fifty_days_avg',
    'sector_hundred_days_avg',
    'sector_two_hundred_days_avg'
]

class App(BaseHelpers):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.engine = create_engine(config('ENGINE_PATH'))
        self.df = None  # Original Data
        self.df_train = None  # Train model on this
        self.df_holdout = None  # Evaluate the model on this
        self.target_column = 'stock_nextmonth_growth'
        self.y = None  # target variable
        self.X = None  # Features
        self.imputer = SimpleImputer(strategy='mean')
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
        self.model = RandomForestClassifier()

    def load_data(self):
        self.log.info('loading data...')
        #self.df = pd.read_sql_table('prices', con=self.engine, schema='research')
        df = pd.read_sql(
            """
                select * from research.prices
                where prediction_year >= 2018;
            """, con=self.engine)
        self.df = df.dropna(subset=[self.target_column])
        self.log.info('data successfully loaded and cleaned')

    def bin_target(self, target_column_name):
        self.log.info('Binning the targets')
        bins = {
            0: -np.inf,
            1: 0
        }

        def get_bin(val):
            """ Assign the smallest bin for a value """
            for key in sorted(bins, reverse=True):
                if val >= bins[key]:
                    return key
            raise Exception(f'Binning missed a value: {val}')
        # df['target'] = df[target_column_name].apply(get_bin)

        self.df['target'] = 1 * (self.df[target_column_name] > 0)
        assert self.df['target'].isna().sum() == 0
        return self.df

    def withdraw_holdout(self):
        """ Splitting the data into training training and holdout """
        self.log.info('Extracting holdout')
        train_idx = self.df['prediction_year'].apply(lambda x: x < 2020)
        holdout_idx = self.df['prediction_year'].apply(lambda x: x >= 2020)

        df_train = self.df[train_idx]
        df_holdout = self.df[holdout_idx]
        self.log.info(f'Training / Holdout: {len(df_train)} / {len(df_holdout)}')

        assert len(df_train) > 0 and len(df_holdout) > 0
        assert (len(set(df_train.index) - set(df_holdout.index))) == len(set(df_train.index)), 'Indexes intersect'
        return df_train, df_holdout

    def pre_process(self, df, mode='train'):
        y = df.pop('target')
        X = df[features]

        if (mode == 'train'):
            imputed_X = self.imputer.fit_transform(X)
        elif (mode == 'test'):
            imputed_X = self.imputer.transform(X)
        else: 
            raise Exception(f'unknown mode for pre-process: {mode}')

        X = pd.DataFrame(imputed_X, columns = X.columns)
        assert X.isna().any().sum() == 0
        return X, y

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

        feature_importance = pd.DataFrame(self.model.feature_importance(X))
        self.log.info(feature_importance)
        feature_importance.to_csv(os.path.join(config('PYTHONPATH'), 'cache/feature_importance.csv'))
        self.log.info('Saved feature importance to csv')

    def evaluate(self, X, y):
        """ Generate evaluation metrics """
        self.log.info('Evaluating performance')
        y_probs = self.model.predict_proba(X)
        y_prob = y_probs[:, 1]
        bcc = BinaryClassifierCurves(y_true=y, y_prob=y_prob)
        bcc.make_all_plots()
        bcc.save_all_plots(key=os.path.join(config('PYTHONPATH'), 'cache/curves'), prefix='')
        bcc.save_all_dfs(key=os.path.join(config('PYTHONPATH'), 'cache/curves'), prefix='')


    def run(self):
        self.load_data()
        self.bin_target(target_column_name=self.target_column)
        self.df_train, self.df_holdout = self.withdraw_holdout()
        X, y = self.pre_process(self.df_train)
        X_test, y_test = self.pre_process(self.df_holdout, mode='test')
        self.train(X, y)
        self.evaluate(X_test, y_test)

if __name__ == '__main__':
    App(log_level='INFO').run()
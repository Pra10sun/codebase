from cargo.common import BaseHelpers
from sklearn.metrics import roc_auc_score
import numpy as np
from decouple import config
import json
import os


class IterativeFeatureSelector(BaseHelpers):
    def __init__(self, Model, model_params, X_train, X_valid, y_train, y_valid):
        super().__init__(self)
        self.Model = Model
        self.model_params = model_params

        self.X_train = X_train
        self.X_valid = X_valid
        self.y_train = y_train
        self.y_valid = y_valid
        
        self.current_run = 0
        self.results = {}

    def run(self, performance_threshold=1):
        """ Iterative fit the model, dropping features one by one """
        self.log.info('Running drop-feature selection pipeline...')

        features_to_remove = []
        drop_features_all = []

        model = self.Model(**self.model_params)
        self.log.info(f'Model has been assembled: {model}')
        model.fit(X=self.X_train, y=self.y_train)
        baseline_performance = self.get_score(model=model, X=self.X_valid, y=self.y_valid)
        current_performance = baseline_performance

        while (baseline_performance - current_performance) < performance_threshold:
            self.current_run += 1
            self.log.info(f'Iteration {self.current_run}, features remaining: {len(self.X_valid.columns)}')
            drop_features_all += features_to_remove

            model = self.Model(**self.model_params)
            model.fit(X=self.X_train, y=self.y_train)
            current_performance = self.get_score(model=model, X=self.X_valid, y=self.y_valid)
            self.log.info(f'Base: {baseline_performance}, Current: {current_performance}')

            run_info = {
                'features_to_remove': features_to_remove,
                'features_remaining': list(self.X_valid.columns),
                'performance': current_performance,
                'performance_increment': baseline_performance - current_performance
            }

            self.results[self.current_run] = run_info

            features_to_remove = self.get_drop_features(model=model)
            self.drop_features(features_to_remove)

            if len(self.X_valid.columns) == 0:
                break

        self.save()

    def get_score(self, model, X, y):
        """ Get performance score from the model based on X and y """
        y_pred = model.predict_proba(X)[:, 1]
        # score = log_loss(y_true=y, y_pred=y_pred)
        score = roc_auc_score(y_true=y, y_score=y_pred)
        self.log.debug(f'Score: {score}')
        return score
    
    def drop_features(self, features):
        """ Remove features from both training and validation data to prepare for the next run """
        self.X_train.drop(columns=features, inplace=True)
        self.X_valid.drop(columns=features, inplace=True)
        self.log.info(f'Features {features} have been dropped')
        
    def get_drop_features(self, model, fraction=0.05):
        df_fi = model.feature_importance(self.X_valid)
        df_fi.sort_values(by='importance', ascending=True, inplace=True)
        n = int(np.ceil(len(df_fi) * fraction))
        return list(df_fi.iloc[:n].index)

    def save(self):
        """ Save files to JSON """
        path = os.path.join(config('PYTHONPATH'), 'cache/feature_selection.json')
        with open(path, 'w') as f:
            json.dump(self.results, f)

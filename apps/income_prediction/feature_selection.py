from cargo.common import BaseHelpers
from cargo.modeling.classifiers import RandomForestClassifier
# from sklearn.ensemble import RandomForestClassifier
from cargo.modeling.feature_selection import IterativeFeatureSelector
from apps.income_prediction.app import App
import numpy as np


class FeatureSelection(BaseHelpers):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.model_params = {
            'n_estimators': 10,
            'criterion': 'gini',
            'max_depth': None,
            'min_samples_leaf': 1,
            'min_weight_fraction_leaf': 0,
            'max_features': 'auto',
            'max_leaf_nodes': None,
            'min_impurity_decrease': 0,
            'bootstrap': True,
            'oob_score': False,
            'n_jobs': -1,
            'random_state': 99,
            'verbose': 0,
            'warm_start': False,
            'class_weight': None,
            'ccp_alpha': 0,
            'max_samples': None,
        }

    def run(self):
        """ Starting point to run feature selection routine """
        app = App()

        app.load_data()
        df = app.concatenate()
        df = app.generate_growth(df, save=False)
        df = app.bin_income_growth(df=df, target_column_name='target_income_growth')
        df, df_test = app.withdraw_holdout(df)
        df_train, df_valid = self.split(df)

        X_train, y_train = app.pre_process(df_train)
        X_valid, y_valid = app.pre_process(df_valid)

        ifs = IterativeFeatureSelector(
            Model=RandomForestClassifier,
            model_params=self.model_params,
            X_train=X_train,
            X_valid=X_valid,
            y_train=y_train.loc[:, 5],
            y_valid=y_valid.loc[:, 5]
        )

        ifs.run()

    def split(self, df, validation_fraction=0.4):
        """ Split data into training and validation """
        self.log.info(f'Splitting data into training ({(1 - validation_fraction) * 100}%) and validation ({validation_fraction * 100}%)')
        n = int(np.ceil((1-validation_fraction) * len(df)))
        df_train = df.sample(n)
        df_valid = df.loc[[idx for idx in df.index if idx not in df_train.index], :]

        self.log.info(f'Training: {len(df_train)}. Validation: {len(df_valid)}')

        assert len(df_train) + len(df_valid) == len(df)
        return df_train, df_valid


if __name__ == '__main__':
    FeatureSelection().run()
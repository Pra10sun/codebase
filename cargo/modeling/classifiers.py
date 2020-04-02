from sklearn.ensemble import RandomForestClassifier as SklearnRandomForestClassifier
from sklearn.multioutput import MultiOutputClassifier as SklearnMultiOutputClassifier
import pandas as pd
from collections import defaultdict


class RandomForestClassifier(SklearnRandomForestClassifier):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def set_params(self, **params):
        """
            GridSearchCV does not seem to work with class inheritance here. Implementing highly simplified version here
        """
        if not params:
            return self

        for key, value in params.items():
            setattr(self, key, value)

        return self

    def feature_importance(self, X, column_name='importance'):
        """ Return feature importances as pandas dataframe """
        df_feature_importances = pd.DataFrame(
            self.feature_importances_,
            index=X.columns,
            columns=[column_name]
        ).sort_values(column_name, ascending=False)
        df_feature_importances = df_feature_importances.reset_index().rename(columns={'index': 'feature'})
        for feature_name in X.columns:
            df_feature_importances['feature'] = df_feature_importances['feature'].apply(
                lambda x: feature_name if feature_name in x else x)
        df_feature_importances_gb = df_feature_importances.groupby('feature').sum()
        df_fi = df_feature_importances_gb.sort_values(by=column_name, ascending=False)
        return df_fi


# class MultiOutputClassifier(SklearnMultiOutputClassifier):
#     def __init__(self, estimator, n_jobs=None):
#         super().__init__(estimator, n_jobs=n_jobs)


class MultiOutputRandomForestClassifier(SklearnMultiOutputClassifier):
    def __init__(self, estimator, n_jobs=None):
        super().__init__(estimator, n_jobs=n_jobs)

    def feature_importance(self, X):
        """ Return combined feature importances for all estimators """
        df_out = None
        print(vars(self))
        for i, estimator in enumerate(self.estimators_):
            if df_out is None:
                df_out = estimator.feature_importance(X, column_name=f'class_{i}')
            else:
                df_out = df_out.join(estimator.feature_importance(X, column_name=f'class_{i}'))
        return df_out
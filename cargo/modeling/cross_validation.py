from sklearn import metrics
from cargo.common import BaseHelpers
from sklearn.model_selection import GroupKFold, TimeSeriesSplit, GridSearchCV
import pandas as pd
import numpy as np


class GroupGridSearchCV(GridSearchCV):
    def __init__(self, X, y, n_splits, groups, estimator, param_grid):
        cv = self.set_group_splits(X=X, y=y, n_splits=n_splits, groups=groups)
        print(param_grid)
        super().__init__(
            estimator=estimator,
            cv=cv,
            param_grid=param_grid
        )

    def set_group_splits(self, X, y, n_splits, groups):
        """ Perform cross val with GroupKFold """

        assert (n_splits >= 2)
        assert len(groups) == len(y)
        assert len(set(groups)) >= n_splits, "The number of distinct groups must be > k_folds + 1"

        self.cv = list(GroupKFold(n_splits=n_splits).split(X, y, groups))
        return self.cv

    def fit(self, X, y=None, groups=None, **kwargs):
        result = super().fit(X, y, **kwargs)
        df_results = pd.DataFrame(result.cv_results_)
        df_results.sort_values(by=['mean_test_score'], ascending=False, inplace=True)
        df_results.reset_index(inplace=True)
        return self
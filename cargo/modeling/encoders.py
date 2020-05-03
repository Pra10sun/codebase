from sklearn.preprocessing import OneHotEncoder as SklearnOneHotEncoder, KBinsDiscretizer
import pandas as pd


class OneHotEncoder(SklearnOneHotEncoder):
    def __init__(self, **kwargs):
        super(OneHotEncoder, self).__init__(**kwargs)
        self.fit_flag = False

    def fit(self, X, **kwargs):
        out = super().fit(X)
        self.fit_flag = True
        return out

    def transform(self, X, **kwargs):
        sparse_matrix = super(OneHotEncoder, self).transform(X)
        new_columns = self.get_new_columns(X=X)
        d_out = pd.DataFrame(sparse_matrix.toarray(), columns=new_columns, index=X.index)
        return d_out

    def fit_transform(self, X, **kwargs):
        self.fit(X)
        return self.transform(X)

    def get_new_columns(self, X):
        new_columns = []
        for i, column in enumerate(X.columns):
            j = 0
            while j < len(self.categories_[i]):
                new_columns.append(f'{column}_<{self.categories_[i][j]}>')
                j += 1
        return new_columns


class CustomKBinsDiscretizer(KBinsDiscretizer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.features = None

    def fit(self, X, **kwargs):
        out = super().fit(X)
        self.features = list(X.columns)
        self.fit_flag = True
        return out

    def transform(self, X, **kwargs):
        if self.encode == 'ordinal':
            array = super().transform(X)
            df_out = pd.DataFrame(array, columns=X.columns, index=X.index)
            return df_out
        else:
            sparse_matrix = super().transform(X)
            new_columns = self.get_new_columns(X)
            df_out = pd.DataFrame(sparse_matrix.toarray(), columns=new_columns, index=X.index)
            return df_out

    def get_new_columns(self, X):
        new_columns = []
        for i, column in enumerate(X.columns):
            j = 0
            while j < (len(self.bin_edges_[i])-1):
                new_columns.append(f'{column}_<{self.bin_edges_[i][j]:.2f} - {self.bin_edges_[i][j+1]:.2f}>')
                j += 1
        return new_columns

    def describe(self):
        for i, feature in enumerate(self.features):
            j = 0
            while j < (len(self.bin_edges_[i])-1):
                print(f'{feature}: {j} <{self.bin_edges_[i][j]:.2f} - {self.bin_edges_[i][j+1]:.2f}>')
                j += 1





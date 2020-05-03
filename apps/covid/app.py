import pandas as pd

from decouple import config
from sqlalchemy import create_engine
from cargo.common import BaseHelpers
from sklearn.impute import SimpleImputer
from cargo.modeling.classifiers import DecisionTreeClassifier
from cargo.preprocessing.engine import InputProcessingEngine


class App(BaseHelpers):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.si = SimpleImputer(strategy='median')
        self.ips = None
        self.engine = create_engine(config('ENGINE_PATH'))

    def load_data(self):
        """ Load data from the database """
        return pd.read_sql_table('covid', schema='research', con=self.engine)

    def pre_process(self, df):
        """ Transformations raw data, preparing it for X and y """
        df = df.copy()
        df['target'] = (df['price_change'] < -0.55).astype(float)
        X = df
        y = X.pop('target')
        X.drop(columns=['price_change', 'price_march', 'price_february', 'price', 'reported_as_of', 'filing_date', 'company_name', 'ticker'], inplace=True)
        self.ips = InputProcessingEngine(X=X)
        # Categorical
        self.ips.cat.fill_na(strategy='constant', fill_value='Unknown')
        self.ips.cat.one_hot()
        self.ips.num.fill_na(strategy='median')
        X = self.ips.get()
        print(X)
        # X = pd.DataFrame(self.si.fit_transform(X), columns=X.columns)
        return X, y

    def run(self):
        """ Entry point """
        df = self.load_data()
        X, y = self.pre_process(df=df)
        model = DecisionTreeClassifier()
        model.fit(X, y)
        print(model.feature_importance(X=X))


if __name__ == '__main__':
    app = App()
    app.run()
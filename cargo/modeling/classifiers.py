from sklearn.ensemble import RandomForestClassifier as SklearnRandomForestClassifier
from sklearn.multioutput import MultiOutputClassifier as SklearnMultiOutputClassifier
from sklearn.tree import DecisionTreeClassifier as SklearnDecisionTreeClassifier, export_graphviz
import pandas as pd
import graphviz
import pydotplus
from collections import defaultdict
from IPython.display import Image


class TreeMethods:
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


class DecisionTreeClassifier(TreeMethods, SklearnDecisionTreeClassifier):
    def __init__(self, **kwargs):
        super(DecisionTreeClassifier, self).__init__(**kwargs)

    def visualize(self, feature_names, save_to=False, **kwargs):
        """ Visualize the tree """
        dot_data = export_graphviz(
            self,
            out_file=None,
            feature_names=feature_names,
            class_names=['Negative', 'Positive'],
            filled=True, rounded=True,
            special_characters=True
        )

        graph = pydotplus.graph_from_dot_data(dot_data)
        edges = defaultdict(list)

        for edge in graph.get_edge_list():
            edges[edge.get_source()].append(int(edge.get_destination()))

        def transparency_function(x, trans_ratio):
            """ Update RGB values with transparency alpha """
            return int((((1 - trans_ratio) * 1) + (trans_ratio * x / 255)) * 255)

        def rgba2rgb(rgb, a):
            """ Convert a list of rgb, alpha values into hex-decimal string """
            return f'#{hex(transparency_function(rgb[0], a))[2:]}{hex(transparency_function(rgb[1], a))[2:]}{hex(transparency_function(rgb[2], a))[2:]}'

        def update_color(node):
            """ Update color of the node in a tree graph """
            label = node.obj_dict['attributes']['label']
            start = label.find('[') + 1
            end = label.find(']')
            values = [float(val) for val in label[start:end].replace(' ', '').split(',')]
            ratio = abs(round((values[1] - values[0]) / (values[0] + values[1]), 1))
            if values[1] >= values[0]:
                color = rgba2rgb([230, 54, 54], ratio)
            else:
                color = rgba2rgb([40, 162, 183], ratio)
            color = f'{color}'
            node.set_fillcolor(color)

        # Update color at the root
        update_color(graph.get_node('0')[0])

        # Update other nodes
        for edge in edges:
            edges[edge].sort()
            for i in range(2):
                dest = graph.get_node(str(edges[edge][i]))[0]
                update_color(dest)

        if save_to:
            graph.write_png(save_to)
        return Image(graph.create_png())


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
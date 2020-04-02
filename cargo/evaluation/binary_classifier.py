from bokeh.plotting import figure
from bokeh.io import export_png
import pandas as pd
import os
import numpy as np
from cargo.common.base import BaseHelpers
from sklearn.metrics import precision_recall_curve, roc_curve, auc, average_precision_score
import warnings
warnings.filterwarnings('ignore')


class ROCCurve():
    def __init__(self, title='Receiver Operating Characteristic Curve'):
        self.title = title

    def plot(self, fpr, tpr, random_model):
        p = figure(
            title=self.title,
            x_axis_label='True Positive Rate (% of Positives Identified)',
            y_axis_label='Precision (% of Positive Predictions Correct)',
            plot_width=600,
            plot_height=500
        )
        p.toolbar.logo = None
        p.toolbar_location = None
        p.line(fpr, tpr, line_width=2, line_color=(230, 54, 54), legend='Our Model')
        p.line(fpr, random_model, line_width=2, line_color=(221, 221, 221), legend='Random Model')
        p.xaxis.axis_label_text_font_size = '16px'
        p.yaxis.axis_label_text_font_size = '16px'
        p.title.text_font_size = '20pt'
        p.legend.label_text_font_size = '11pt'
        p.legend.location = "bottom_right"
        return p


class LiftCurve:
    def __init__(self, title='Lift Curve'):
        self.title = title

    def plot(self, centile, model, perfect_model, random_model):
        p = figure(
            title=self.title,
            x_axis_label='Centile (% of All Rows)',
            y_axis_label='True Positive Rate (% of Positives Identified)',
            plot_width=600,
            plot_height=500
        )
        p.toolbar.logo = None
        p.toolbar_location = None
        p.line(centile, perfect_model, line_width=2, line_color=(221, 221, 221), legend='Perfect Model')
        p.line(centile, random_model, line_width=2, line_color=(69, 69, 69), legend='Random Model')
        p.line(centile, model, line_width=2, line_color=(230, 54, 54), legend='Our Model')
        p.xaxis.axis_label_text_font_size = '16px'
        p.yaxis.axis_label_text_font_size = '16px'
        p.title.text_font_size = '20pt'
        p.legend.label_text_font_size = '11pt'
        p.legend.location = "bottom_right"
        return p


class PRCurve:
    def __init__(self, title='Precision - Recall (PR) Curve'):
        self.title = title

    def plot(self, precision, tpr, random_model):
        p = figure(
            title=self.title,
            x_axis_label='True Positive Rate (% of Positives Identified)',
            y_axis_label='Precision (% of Positive Predictions Correct)',
            plot_width=600,
            plot_height=500
        )
        p.toolbar.logo = None
        p.toolbar_location = None
        p.line(tpr, precision, line_width=2, line_color=(230, 54, 54), legend='Precision')
        p.line(tpr, random_model, line_width=2, line_color=(221, 221, 221), legend='Random Model')
        p.xaxis.axis_label_text_font_size = '16px'
        p.yaxis.axis_label_text_font_size = '16px'
        p.title.text_font_size = '20pt'
        p.legend.label_text_font_size = '11pt'
        p.legend.location = "top_right"
        return p


class ThresholdRecallCurve:
    def __init__(self, title='Threshold - Recall Curve'):
        self.title = title

    def plot(self, threshold, recall):
        p = figure(
            title=self.title,
            x_axis_label='True Positive Rate (% of Positives Identified)',
            y_axis_label='Precision (% of Positive Predictions Correct)',
            plot_width=600,
            plot_height=500
        )
        p.toolbar.logo = None
        p.toolbar_location = None
        p.line(threshold, recall, line_width=2, line_color=(230, 54, 54), legend='Our Model')
        p.xaxis.axis_label_text_font_size = '16px'
        p.yaxis.axis_label_text_font_size = '16px'
        p.title.text_font_size = '20pt'
        p.legend.label_text_font_size = '11pt'
        p.legend.location = "top_right"
        return p


class BinaryClassifierCurves(BaseHelpers):

    def __init__(self,
                 y_prob,
                 y_true,
                 **kwargs):
        super(BinaryClassifierCurves, self).__init__(**kwargs)

        self.y_prob = np.array(y_prob)
        self.y_true = np.array(y_true)
        self.plotting_dfs = {}
        self.plots = {}
        self.prec = []
        self.recall = []
        self.thresholds = []

    def make_all_plots(self):
        self.make_roc_plot()
        self.make_lift_plot()
        self.make_prec_recall_plot()
        self.make_recall_thresh_plot()

    def save_all_plots(self, key, prefix):
        """ Save all types of plots"""
        for name in self.plots:
            plot = self.plots[name]
            filename = os.path.join(key, f"{prefix}_{name}.png")
            self.log.info(f'Saving to {filename}')
            export_png(plot, filename=filename)

    def save_all_dfs(self, key, prefix):
        """ Save all types of plotting dfs"""
        for name in self.plotting_dfs:
            self.plotting_dfs[name].to_csv(
                os.path.join(key, f"{prefix}_{name}.csv")
            )

    def make_roc_plot(self):
        df = self.plotting_dfs.get("roc") if 'roc' in self.plotting_dfs else self.make_roc_df()
        plot = ROCCurve().plot(
            fpr=df['False Positive Rate (% of Negatives That Are False Positives)'],
            tpr=df['True Positive Rate (% of Positives Identified)'],
            random_model=df['Random Guess']
        )
        self.plots["roc"] = plot
        return plot

    def make_lift_plot(self):
        df = self.plotting_dfs["lift"] if 'lift' in self.plotting_dfs else self.make_lift_df()
        plot = LiftCurve(title=f'Lift Curve ({self.roc_auc:.3f})').plot(
            centile=df['% of All Rows'],
            perfect_model=df['Perfect Rank Ordering'],
            random_model=df['Random Guess'],
            model=df['True Positive Rate (% of Positives Identified)'])
        self.plots['lift'] = plot
        return plot

    def make_prec_recall_plot(self):
        df = self.plotting_dfs["prec_recall"] if 'prec_recall' in self.plotting_dfs else self.make_prec_recall_df()
        plot = PRCurve(title=f'Precision - Recall Curve ({self.prauc:.3f})').plot(
            precision=df['Precision (% of Positive Predictions Correct)'],
            tpr=df['True Positive Rate (% of Positives Identified)'],
            random_model=df['Random Guess']
        )
        self.plots["prec_recall"] = plot
        return plot

    def make_recall_thresh_plot(self):
        df = self.plotting_dfs["prec_recall"] if "prec_recall" in self.plotting_dfs else self.make_prec_recall_df()
        plot = ThresholdRecallCurve().plot(
            threshold=df['Probability Threshold'],
            recall=df['True Positive Rate (% of Positives Identified)']
        )
        self.plots["thresh_recall"] = plot
        return plot

    def make_roc_df(self):
        fpr, tpr, thresh_holds = roc_curve(self.y_true, self.y_prob)
        self.roc_auc = auc(fpr, tpr)
        self.plotting_dfs["roc"] = pd.DataFrame({
            "False Positive Rate (% of Negatives That Are False Positives)": fpr,
            "Random Guess": fpr,
            "True Positive Rate (% of Positives Identified)": tpr,
            "Probability Threshold": thresh_holds,
        })
        return self.plotting_dfs["roc"]

    def make_lift_df(self):
        df = pd.DataFrame([])
        df['probs'] = self.y_prob
        df.sort_values(by='probs', inplace=True,
                       ascending=False)
        if "roc" not in self.plotting_dfs:
            self.make_roc_plot()
        df_roc = self.plotting_dfs["roc"].copy()
        df = df.merge(df_roc,
                      left_on='probs',
                      right_on="Probability Threshold",
                      how='left')
        df['row_num'] = np.arange(0, len(df)) + 1
        df['% of All Rows'] = df.row_num / len(df)
        df['Random Guess'] = df['% of All Rows'].copy()
        df['Perfect Rank Ordering'] = self.get_perfect_lift_ranking(df['% of All Rows'])
        dfg = (df.groupby(
            ["Probability Threshold",
             "True Positive Rate (% of Positives Identified)"])
               .max()
               .reset_index())
        output_cols = [
            '% of All Rows',
            "True Positive Rate (% of Positives Identified)",
            "Perfect Rank Ordering",
            "Random Guess"
        ]
        self.plotting_dfs["lift"] = (dfg[output_cols]
                                     .append(
            pd.DataFrame([[0.0, 0.0, 0.0, 0.0]], columns=output_cols))  # ensures we start at 0
                                     .dropna(axis=0))
        return self.plotting_dfs["lift"]

    def make_prec_recall_df(self):
        prec, recall, thresholds = precision_recall_curve(self.y_true, self.y_prob)
        self.prauc = average_precision_score(self.y_true, self.y_prob)
        self.plotting_dfs["prec_recall"] = pd.DataFrame({
            "Precision (% of Positive Predictions Correct)": prec,
            "True Positive Rate (% of Positives Identified)": recall,
            "Probability Threshold": [0.0] + thresholds.tolist(),
            "Random Guess": self.y_true.sum() / self.y_true.shape[0]})

        self.prec = prec
        self.recall = recall
        self.thresholds = thresholds.tolist()
        return self.plotting_dfs["prec_recall"]

    def get_perfect_lift_ranking(self, perc_of_rows):
        nrows = len(self.y_true)
        positive_fraction = sum(self.y_true) / nrows
        slope = 1 / positive_fraction  # rise over run
        oracle_curve = [i * slope if i * slope < 1 else 1 for i in perc_of_rows]
        return oracle_curve

    def get_threshold_at(self, prec=None, recall=None):
        """
            Get threshold to meet either precision or recall requirenment (individually)
            :param prec:float
            :param recall:float
            :return: threshold at which selected metric is above provided value
        """


        if len(self.thresholds) == 0:
            raise Exception('Thresholds are not computed, run "make_prec_recall_df()" first')

        if (type(prec) is float) and (recall is None):
            metric = self.prec
            val = prec
        elif (type(recall) is float) and (prec is None):
            metric = self.recall
            val = recall
        else:
            raise Exception('Please provide either precision or recall')

        for i, th in enumerate(self.thresholds):
            if metric[i] > val:
                print('Found threshold: ', th)
                return th
        # If nothing happened, must be all positive
        return 1.

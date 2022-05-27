import pandas as pd
import numpy as np
from sklearn.metrics import *
import seaborn as sns
from matplotlib import pyplot as plt


METRICS_MAPPING = dict(
    accuracy=[accuracy_score],
    balanced_accuracy=[balanced_accuracy_score],
    top_k_accuracy=[top_k_accuracy_score],
    neg_brier_score=brier_score_loss,
    neg_log_loss=[log_loss],

    average_precision=[average_precision_score],
    average_precision_micro=[average_precision_score, 'micro'],
    average_precision_macro=[average_precision_score, 'macro'],
    average_precision_samples=[average_precision_score, 'samples'],
    average_precision_weighted=[average_precision_score, 'weighted'],

    f1=[f1_score, 'binary'],
    f1_micro=[f1_score, 'micro'],
    f1_macro=[f1_score, 'macro'],
    f1_weighted=[f1_score, 'weighted'],
    f1_samples=[f1_score, 'samples'],

    precision=[precision_score, 'binary'],
    precision_micro=[precision_score, 'micro'],
    precision_macro=[precision_score, 'macro'],
    precision_weighted=[precision_score, 'weighted'],
    precision_samples=[precision_score, 'samples'],

    recall=[recall_score, 'binary'],
    recall_micro=[recall_score, 'micro'],
    recall_macro=[recall_score, 'macro'],
    recall_weighted=[recall_score, 'weighted'],
    recall_samples=[recall_score, 'samples'],

    jaccard=[jaccard_score, 'binary'],
    jaccard_micro=[jaccard_score, 'micro'],
    jaccard_macro=[jaccard_score, 'macro'],
    jaccard_weighted=[jaccard_score, 'weighted'],
    jaccard_samples=[jaccard_score, 'samples'],

    roc_auc=[roc_auc_score],
    roc_auc_ovr=[roc_auc_score, 'macro', 'ovr'],
    roc_auc_ovo=[roc_auc_score, 'macro', 'ovo'],
    roc_auc_ovr_weighted=[roc_auc_score, 'weighted', 'ovr'],
    roc_auc_ovo_weighted=[roc_auc_score, 'weighted', 'ovo'],
    )


class ClassificationEval:
    def __init__(self, train_pipeline, y_true, y_prob, metrics):
        self.train_pipeline = train_pipeline
        self.label = self.train_pipeline.classes_
        self.y_true = y_true
        self.y_prob = y_prob
        self.metrics = metrics
        self.num_classes = len(np.unique(y_true))

        self.is_binary_cls = False
        if self.num_classes == 2:
            self.is_binary_cls = True
            self.y_prob = y_prob[:, 1]

    def __get_metrics(self, y_true, y_pred, label):
        cls_report = classification_report(
                y_true, y_pred, target_names=label, output_dict=True)
        return cls_report

    def __plot_conf_matrix(self, y_true, y_pred, label, threshold=None):
        fig, ax = plt.subplots()
        conf_matrix = ConfusionMatrixDisplay.from_predictions(
            y_true, y_pred, normalize="all",
            display_labels=label, include_values=True,
            ax=ax)
        if threshold is not None:
            ax.set_title(f"Normalized Confusion Matrix at Best Threshold={threshold}")
        return fig

    def __plot_roc(self, y_true, y_prob, label, ax):
        roc_plot = RocCurveDisplay.from_predictions(
            y_true, y_prob, ax=ax)
        ax.set_title(f"Class: {label}")

        return roc_plot

    def __plot_precision_recall(self, y_true, y_prob, label, ax):
        precision_recall_plot = PrecisionRecallDisplay.from_predictions(
            y_true, y_prob, ax=ax)
        ax.set_title(f"Class: {label}")

        return precision_recall_plot

    def eval(self):
        plt.ioff()
        if self.is_binary_cls:
            best_cls_report = None
            best_score = 0
            best_threshold = 0
            best_y_pred = None
            fig_all = dict()
            for threshold in np.arange(1, 10) / 10:
                y_pred = np.where(self.y_prob > threshold, 1, 0)
                cls_report = self.__get_metrics(
                    self.y_true, y_pred, self.label)
                if len(METRICS_MAPPING[self.metrics]) == 1:
                    metrics = METRICS_MAPPING[self.metrics][0]
                    score = metrics(self.y_true, y_pred)
                else:
                    metrics, avg_method = METRICS_MAPPING[self.metrics]
                    if self.is_binary_cls:
                        score = metrics(
                            self.y_true, y_pred, pos_label=1, average="binary")
                    else:
                        score = metrics(
                            self.y_true, y_pred, pos_label=1, average=avg_method)
                if score > best_score:
                    best_score = score
                    best_cls_report = cls_report
                    best_threshold = threshold
                    best_y_pred = y_pred
            conf_matrix_fig = self.__plot_conf_matrix(
                self.y_true, best_y_pred, self.label,
                best_threshold)
            fig, ax = plt.subplots(1, 2)
            roc_plot = self.__plot_roc(
                self.y_true, self.y_prob, self.label[1], ax[0])
            precision_recall_plot = self.__plot_precision_recall(
                self.y_true, self. y_prob, self.label[1], ax[1])
            fig_all[self.label[1]] = fig
            evaluation_results = dict(
                cls_report=best_cls_report,
                conf_matrix_fig=conf_matrix_fig,
                fig=fig_all,
                score=best_score,
                best_threshold=best_threshold)
        else:
            y_pred = np.argmax(self.y_prob, 1)
            # Beware of bugs here during onehotencoding
            y_true_onehot = pd.get_dummies(self.y_true).values
            cls_report = self.__get_metrics(
                self.y_true, y_pred, label=self.label)
            conf_matrix_fig = self.__plot_conf_matrix(
                self.y_true, y_pred, self.label)
            score = f1_score(
                self.y_true, y_pred,
                pos_label=1, average="macro")
            fig_all = dict()
            for i, label in enumerate(self.label):
                fig, ax = plt.subplots(1, 2)
                roc_plot = self.__plot_roc(
                    y_true_onehot[:, i], self.y_prob[:, i], label, ax=ax[0])
                precision_recall_plot = self.__plot_precision_recall(
                    y_true_onehot[:, i], self. y_prob[:, i], label, ax[1])
                fig_all[label] = fig
            evaluation_results = dict(
                cls_report=cls_report,
                conf_matrix_fig=conf_matrix_fig,
                fig=fig_all,
                score=score)
        plt.ion()

        return evaluation_results

import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
from itertools import combinations


class EDA:
    def __init__(self, problem_type, data, label):
        self.problem_type = problem_type
        self.data = data
        self.label = label
        self.numeric_features = self.data.select_dtypes(
            include=["int", "float"])
        self.category_features = self.data.select_dtypes(
            exclude=["int", "float"])
        self.num_plot_per_fig = 4

    def __general(self):
        numeric_features_count = len(self.numeric_features)
        category_features_count = len(self.category_features)
        missing_value_count = self.data.isna().sum()

        return numeric_features_count, category_features_count, missing_value_count

    def __univariate(self):
        # plt.ioff()

        # numeric features
        numeric_features_stats = self.numeric_features.agg([
            "min", "max", "mean", "median", "std",
            "skew", "kurtosis"])
        numeric_features_quantile = self.numeric_features.quantile([0.25, 0.75])
        numeric_features_stats = pd.concat(
            [numeric_features_stats, numeric_features_quantile], 0)
        numeric_features_stats.sort_values(by="std", axis=1, ascending=True, inplace=True)

        # Kde plot
        kdeplot_all = []
        for i, (name, column) in enumerate(self.numeric_features.iteritems()):
            if i % self.num_plot_per_fig == 0:
                kdeplot_fig, kdeplot_ax = plt.subplots(
                    int(self.num_plot_per_fig**0.5),
                    int(self.num_plot_per_fig**0.5))
            row_idx, col_idx = divmod(
                i % self.num_plot_per_fig, int(self.num_plot_per_fig**0.5))
            kdeplot_ax_tmp = column.plot.kde(
                ax=kdeplot_ax[row_idx, col_idx], secondary_y=True, title=name)
            # ax_tmp.text(0.5, 0.5, "test", fontsize=22)
            if (i + 1) % self.num_plot_per_fig == 0 or i == len(self.numeric_features.columns)-1:
                kdeplot_all.append(kdeplot_fig)

        # Boxplot
        boxplot_all = []
        fig, ax = plt.subplots()
        for i, (name, column) in enumerate(self.numeric_features.iteritems()):
            if i % self.num_plot_per_fig == 0:
                boxplot_fig, boxplot_ax = plt.subplots(
                    int(self.num_plot_per_fig**0.5),
                    int(self.num_plot_per_fig**0.5))
            row_idx, col_idx = divmod(
                i % self.num_plot_per_fig, int(self.num_plot_per_fig**0.5))
            boxplot_ax[row_idx, col_idx].boxplot(column, 0, 'gD')
            boxplot_ax[row_idx, col_idx].set_title(f"{name}")
            if (i + 1) % self.num_plot_per_fig == 0 or i == len(self.numeric_features.columns)-1:
                boxplot_all.append(boxplot_fig)

        # Category features
        # Freq plot
        freqplot_all = []
        for i, (name, column) in enumerate(self.category_features.iteritems()):
            if i % self.num_plot_per_fig == 0:
                freqplot_fig, freqplot_ax = plt.subplots(
                    int(self.num_plot_per_fig**0.5),
                    int(self.num_plot_per_fig**0.5))
            row_idx, col_idx = divmod(
                i % self.num_plot_per_fig, int(self.num_plot_per_fig**0.5))
            bar = freqplot_ax[row_idx, col_idx].bar(
                column.value_counts().index, column.value_counts().values)
            freqplot_ax[row_idx, col_idx].bar_label(bar)
            freqplot_ax[row_idx, col_idx].set_title(f"{name}")
            if (i + 1) % self.num_plot_per_fig == 0 or i == len(self.category_features.columns)-1:
                freqplot_all.append(freqplot_fig)

        # plt.ion()

        return numeric_features_stats, kdeplot_all, boxplot_all, freqplot_all

    def __bivariate(self):
        # Categorical vs Categorical
        cat_vs_cat_plot_all = []
        if self.problem_type == "classification":
            columns_comb = list(combinations(self.category_features.columns, 2))
            for i, (column_a, column_b) in enumerate(columns_comb):
                category_features_crosstab = pd.crosstab(
                    self.category_features[column_a],
                    self.category_features[column_b],
                    margins=True, values=self.category_features[self.label],
                    aggfunc=pd.Series.count, normalize="all")
                category_features_crosstab.fillna(0, inplace=True)
                if i % self.num_plot_per_fig == 0:
                    cat_vs_cat_plot_fig, cat_vs_cat_plot_ax = plt.subplots(
                        int(self.num_plot_per_fig**0.5),
                        int(self.num_plot_per_fig**0.5))
                row_idx, col_idx = divmod(
                    i % self.num_plot_per_fig, int(self.num_plot_per_fig**0.5))
                cat_vs_cat_plot_ax_tmp = sns.heatmap(
                    category_features_crosstab,
                    ax=cat_vs_cat_plot_ax[row_idx, col_idx],
                    annot=True, cbar=True, fmt=".2%")
                cat_vs_cat_plot_ax_tmp.set(title=f"{column_a} vs {column_b} with {self.label} as count")
                if (i + 1) % self.num_plot_per_fig == 0 or i == len(columns_comb)-1:
                    cat_vs_cat_plot_all.append(cat_vs_cat_plot_fig)

        # Numerical vs Categorical
        num_vs_cat_plot_all = []
        if self.problem_type == "classification":
            numeric_features_and_label = pd.concat(
                [self.numeric_features, self.category_features[self.label]], 1)
            for i, (name, column) in enumerate(self.numeric_features.iteritems()):
                if i % self.num_plot_per_fig == 0:
                    num_vs_cat_plot_fig, num_vs_cat_plot_ax = plt.subplots(
                        int(self.num_plot_per_fig**0.5),
                        int(self.num_plot_per_fig**0.5))
                row_idx, col_idx = divmod(
                    i % self.num_plot_per_fig, int(self.num_plot_per_fig**0.5))
                num_vs_cat_plot_ax_tmp = sns.boxplot(
                    x=self.label, y=column, data=numeric_features_and_label,
                    ax=num_vs_cat_plot_ax[row_idx, col_idx])
                if (i + 1) % self.num_plot_per_fig == 0 or i == len(columns_comb)-1:
                        num_vs_cat_plot_all.append(num_vs_cat_plot_fig)

        # Numerical vs Numerical
        # correlation plot
        corr_matrix = self.numeric_features.corr(method="pearson")
        corr_fig, corr_ax = plt.subplots()
        corr_ax_tmp = sns.heatmap(corr_matrix, ax=corr_ax, annot=True)
        corr_ax_tmp.set(title="Correlation Matrix (Pearson)")

        num_vs_num_plot_all = []
        columns_comb = list(combinations(self.numeric_features.columns, 2))
        for i, (column_a, column_b) in enumerate(columns_comb):
            if i % self.num_plot_per_fig == 0:
                num_vs_num_plot_fig, num_vs_num_plot_ax = plt.subplots(
                    int(self.num_plot_per_fig**0.5),
                    int(self.num_plot_per_fig**0.5))
            row_idx, col_idx = divmod(
                i % self.num_plot_per_fig, int(self.num_plot_per_fig**0.5))
            num_vs_num_plot_ax[row_idx, col_idx].scatter(
                x=self.numeric_features[column_a],
                y=self.numeric_features[column_b],
                c=None if self.problem_type == "regression" else self.category_features[self.label])
            num_vs_num_plot_ax[row_idx, col_idx].set(
                xlabel=column_a, ylabel=column_b,
                title=f"Plot of {column_b} vs {column_a}")
            if (i + 1) % self.num_plot_per_fig == 0 or i == len(columns_comb)-1:
                num_vs_num_plot_all.append(num_vs_num_plot_fig)

        return cat_vs_cat_plot_all, num_vs_cat_plot_all, corr_fig, num_vs_num_plot_all

    def generate_report(self):
        numeric_features_count, category_features_count, missing_value_count = self.__general()
        numeric_features_stats, kdeplot_all, boxplot_all, freqplot_all = self.__univariate()
        cat_vs_cat_plot_all, num_vs_cat_plot_all, corr_fig, num_vs_num_plot_all = self.__bivariate()

        return None
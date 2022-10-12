import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from sklearn.metrics import mean_squared_error

# Suppressing sns.distplot warning
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

# Increasing display width for pandas
pd.set_option('max.columns', 20)
pd.set_option('display.width', 2000)


class RegressionEval:

    def __init__(self, y_true, y_pred, n_features, data_group=None, datetime=None):

        # ========== Data Check ==========
        assert len(y_true) == len(y_pred), \
            f"Lengths of y_true ({len(y_true)}) and y_pred ({len(y_pred)}) are not equal."

        self.y_true = y_true
        self.y_pred = y_pred
        self.n_features = n_features
        self.data_group = 'Train' if data_group is None else data_group
        self.datetime = datetime

    def evaluate(self, plot_scatter=True, plot_trend=True, plot_error_distribution=True):

        # ========== Data Pre-processing ==========
        results_df = pd.DataFrame({
            'DataGroup': self.data_group,
            'Actual': self.y_true,
            'Predicted': self.y_pred
        })
        if self.datetime is not None:
            results_df['DateTime'] = self.datetime

        train_n_rows = (results_df['DataGroup'] == 'Train').sum()
        group_sorted = ['Train', 'Validation', 'Test']

        results_df = results_df.reset_index(drop=True)
        results_df['DataGroup'] = pd.Categorical(results_df['DataGroup'], [x for x in group_sorted if x in results_df['DataGroup'].unique()])
        results_df = results_df.sort_values('DataGroup')

        # ========== Error Metrics Calculation ==========
        results_df['Error'] = results_df['Predicted'] - results_df['Actual']

        def calc_error_metrics(x):
            grouped_x = {'R2': self.calc_r2_score(x['Actual'], x['Predicted']),
                         'Adjusted_R2': self.calc_r2_score(x['Actual'], x['Predicted'], adjusted=True,
                                                           n_rows=train_n_rows, n_features=self.n_features),
                         'MAE': np.mean(np.abs(x['Error'])),
                         'MAPE': np.mean(np.abs(x['Error'] / x['Actual']) * 100),
                         'MSE': mean_squared_error(x['Actual'], x['Predicted']),
                         'RMSE': np.sqrt(mean_squared_error(x['Actual'], x['Predicted']))}
            return pd.Series(grouped_x, index=grouped_x.keys())

        grouped_results_df = results_df.groupby('DataGroup').apply(lambda x: calc_error_metrics(x))
        print("======================================== Model Results ========================================")
        print(grouped_results_df)
        print("===============================================================================================")
        return_dict = {'evaluation_results': grouped_results_df}

        # Prediction vs Actual scatterplot
        if plot_scatter:
            sns_grid = self.prediction_actual_scatterplot(results_df)
            return_dict['scatterplot_fig'] = sns_grid

        # Prediction vs Actual trending
        if self.datetime is not None:
            if plot_trend:
                trend_fig = self.prediction_actual_trending(results_df)
                return_dict['trend_fig'] = trend_fig

        # Density plot
        if plot_error_distribution:
            error_distribution_fig = self.error_distribution(results_df)
            return_dict['error_distribution_fig'] = error_distribution_fig

        return return_dict

    @staticmethod
    def calc_r2_score(y_true, y_pred, adjusted=False, n_rows=None, n_features=None):
        sse = np.sum((y_pred - y_true) ** 2)
        sst = np.sum((np.mean(y_true) - y_true) ** 2)

        if not adjusted:
            return 1 - (sse / sst)
        else:
            assert n_rows is not None and n_features is not None, \
                "n_rows and n_features must not be None if `adjusted` is True"

            return 1 - (sse / (n_rows - n_features)) / (sst / (n_rows - 1))

    @staticmethod
    def prediction_actual_scatterplot(results_df):

        lim_range = (results_df[['Actual', 'Predicted']].min().min(), results_df[['Actual', 'Predicted']].max().max())
        sns_grid = sns.lmplot(x='Actual', y='Predicted', hue='DataGroup', data=results_df,
                              palette=['#93BFEB', '#432C5F', '#00A19C'], legend_out=False, height=8, aspect=1.2)
        sns_grid.ax.grid('both')
        sns_grid.ax.legend(loc='upper left')
        sns_grid.ax.set_xlim(lim_range)
        sns_grid.ax.set_ylim(lim_range)
        sns_grid.ax.plot(lim_range, lim_range, linestyle='--', color='black')
        sns_grid.fig.suptitle("Actual vs Predicted")
        sns_grid.fig.tight_layout()

        return sns_grid

    def prediction_actual_trending(self, results_df):

        plot_df = results_df.sort_values('DateTime').set_index('DateTime')
        trend_fig, ax = plt.subplots(figsize=(12, 6))
        plot_df[['Actual', 'Predicted']].plot(ax=ax, color=['#666666', '#005955'])
        if self.data_group is not None:
            ax.fill_between(x=plot_df.index, y1=plot_df[['Actual', 'Predicted']].min().min(),
                            y2=plot_df[['Actual', 'Predicted']].max().max(),
                            where=plot_df['DataGroup'] == 'Validation', facecolor='#432C5F', alpha=0.5)
            ax.fill_between(x=plot_df.index, y1=plot_df[['Actual', 'Predicted']].min().min(),
                            y2=plot_df[['Actual', 'Predicted']].max().max(),
                            where=plot_df['DataGroup'] == 'Test', facecolor='#00A19C', alpha=0.5)
        ax.set_xlabel('')
        ax.set_ylabel('')
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%d-%b-%y'))
        trend_fig.suptitle("Actual vs Predicted")

        return trend_fig

    @staticmethod
    def error_distribution(results_df):

        color_palette = ['#93BFEB', '#432C5F', '#00A19C']
        error_distribution_fig, axes = \
            plt.subplots(nrows=len(results_df['DataGroup'].unique()), figsize=(12, 8), constrained_layout=True)
        for i, datagroup_type in enumerate(results_df['DataGroup'].unique()):
            sns.distplot(results_df.loc[results_df['DataGroup'] == datagroup_type, 'Error'], ax=axes[i],
                         label=datagroup_type, color=color_palette[i])
            axes[i].axvline(0, color='grey')
            axes[i].axvline(0, color='grey')
            axes[i].set_title(datagroup_type)
        error_distribution_fig.suptitle("Error distribution")

        return error_distribution_fig



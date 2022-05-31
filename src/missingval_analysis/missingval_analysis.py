import pandas as pd
import numpy as np
from sklearn.impute import MissingIndicator
from sklearn.preprocessing import StandardScaler
from scipy import stats
from matplotlib import pyplot as plt
import missingno as msno


# https://towardsdev.com/how-to-identify-missingness-types-with-missingno-61cfe0449ad9
# https://towardsdatascience.com/statistical-test-for-mcar-in-python-9fb617a76eac


class MissingValAnalysis:
    def __init__(
        self, data, numeric_features_names, category_features_names,
            scale=True):
        self.data = data
        self.numeric_features_names = numeric_features_names
        self.category_features_names = category_features_names
        self.scale = scale
        self.missing_indicator = MissingIndicator(
            missing_values=np.nan, features="missing-only"
        )

    def __hypothesis_testing(self):
        # Generate missing value indicator
        missing_features_indicator = self.missing_indicator.fit_transform(self.data)
        missing_features_names = self.data.columns[self.missing_indicator.features_]
        missing_features_indicator = pd.DataFrame(
            missing_features_indicator, columns=missing_features_names)

        # Check type of missingness
        feature_a = []
        feature_b = []
        p_value = []
        missing_type = []
        for missing_feature_name, missing_feature in missing_features_indicator.iteritems():
            for feature_name, feature in self.data.iteritems():
                if missing_feature_name == feature_name:
                    continue

                # Binning continuous features
                if feature_name in self.numeric_features_names:
                    feature = pd.qcut(x=feature, q=10, labels=False)

                # Chi2 test
                ct = pd.crosstab(
                    missing_feature, feature, margins=False)
                _, p_val, _, _ = stats.chi2_contingency(ct)
                if p_val < 0.05:
                    missing_type.append("MAR")
                else:
                    missing_type.append("MCAR/ MNAR")
                feature_a.append(missing_feature_name)
                feature_b.append(feature_name)
                p_value.append(p_val)
        missingness_report = pd.DataFrame({
            "Feature_A": feature_a, "Feature_B": feature_b,
            "p-value": p_value, "Missing_Type": missing_type
        })
        final_missingness_report = pd.crosstab(missingness_report["Feature_A"], missingness_report["Missing_Type"])
        total_count = final_missingness_report.iloc[0].sum()
        final_missingness_report = final_missingness_report / total_count * 100
        final_missingness_report = final_missingness_report.idxmax(1)

        return missingness_report, final_missingness_report

    def generate_report(self):
        plt.ioff()

        # Replace empty string with nan
        self.data.replace("", np.nan, inplace=True)

        if self.scale:
            self.data[self.numeric_features_names] =\
                StandardScaler().fit_transform(
                    self.data[self.numeric_features_names])

        missing_columns = self.data.columns[self.data.isna().any(0)]
        missingness = msno.matrix(self.data)
        missingness_corr = msno.heatmap(self.data, cmap='rainbow')
        nonna_count_by_column = msno.bar(self.data)

        # Check MCAR
        # https://impyute.readthedocs.io/en/master/user_guide/diagnostics.html#little-s-mcar-test-1
        # msno.matrix(self.data.sort_values('age'))

        # Check MAR
        first_lvl_index = np.array([["missing"]*8, ["complete"]*8]).reshape(-1)
        second_lvl_index = np.tile(self.data.describe().index.values, 2)
        index = [first_lvl_index, second_lvl_index]
        mar = dict()
        for missing_column in missing_columns:
            missing_portion = self.data[self.data[missing_column].isna()].drop(missing_column, axis=1)
            complete_portion = self.data[~self.data[missing_column].isna()]
            mar[missing_column] = pd.concat([missing_portion.describe(), complete_portion.describe()], 0)
            mar[missing_column].index = index

        # Check MNAR

        # Conduct hypothesis testing to check type of missingness
        missingness_report, final_missingness_report = self.__hypothesis_testing()

        plt.ion()

        return missingness_report, final_missingness_report

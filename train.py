import logging
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder, LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import SimpleImputer, KNNImputer, IterativeImputer
from sklearn.feature_selection import *
from sklearn.linear_model import *
from sklearn.ensemble import *
from sklearn.svm import *
from sklearn.metrics import *

from config import Config
from src.eda import EDA
from src.missingval_analysis import MissingValAnalysis
from src.feature_eng import FeatureEng
from src.evaluation import ClassificationEval
from src.mlflow_logging import MlflowLogging


# Set log level
logging.basicConfig(
    level=logging.INFO,
    format="(%(asctime)s) | %(levelname)-8s | %(module)s: %(message)s",
    datefmt='%Y-%m-%d %H:%M:%S')


class Train:
    def __init__(self, config):

        logging.info("Initializing...")
        self.config = config
        self.problem_type = config["problem_type"]
        self.data = pd.read_csv(config["data"]["data_path"])
        self.label = config["data"]["label"]
        self.numeric_features_names = config["data"]["numeric_features"]
        self.category_features_names = config["data"]["category_features"]
        self.datetime_features_names = config["data"]["datetime_features"]
        self.model_algs = config["model"]
        self.split_ratio = config["train_val_test_split"]["split_ratio"]
        self.tune = config["tuning"]["tune"]
        self.tune_alg = config["tuning"]["search_method"]
        self.param_grids = config["tuning"]
        self.metrics = config["evaluation"]["classification"]
        self.train_data, self.test_data = None, None
        # if self.problem_type == "classification":
        #     self.label_encoder = LabelEncoder()
        #     self.data[self.label] = self.label_encoder.fit_transform(
        #         self.data[self.label])

    def __preprocessing(self):
        logging.info("Preprocess data...")
        # To do label smoothing, label encoder

        # self.data["above_median_house_value"] = 0

        # Binary class
        # self.data["above_median_house_value"] =\
        #     self.data["median_house_value"].apply(
        #         lambda x: 1 if x > self.data["median_house_value"].median() else 0)

        # Multi class
        # self.data.loc[self.data["median_house_value"] > self.data["median_house_value"].quantile(0.25), "above_median_house_value"] = 1
        # self.data.loc[self.data["median_house_value"] > self.data["median_house_value"].quantile(0.75), "above_median_house_value"] = 2

        if self.problem_type in ["regression", "classification"] and len(self.datetime_features_names) > 0:
            self.data.drop(self.datetime_features_names, axis=1, inplace=True)

        return None

    def __train_test_split(self):
        logging.info("Train-test splitting...")
        train_data, test_data = train_test_split(
            self.data, test_size=self.split_ratio,
            stratify=self.data[self.label] if self.problem_type == "classification" else None,
            random_state=self.config["seed"]
            )

        return train_data, test_data

    def __eda(self, data):

        logging.info("Generating EDA report...")
        eda = EDA(
            self.problem_type, data, self.label,
            self.numeric_features_names, self.category_features_names,
            self.datetime_features_names).generate_report()

        return None

    def __imbalanced2balanced(self):
        raise NotImplementedError("WIP")

    def __remove_outlier(self):
        raise NotImplementedError("WIP")

    def __select_feat(self):
        raise NotImplementedError("WIP")

    def __missing_val_analysis(self, data):
        missingness_report, final_missingness_report = MissingValAnalysis(
            data, self.numeric_features_names,
            self.category_features_names).generate_report()

        return missingness_report, final_missingness_report

    def __feat_eng(self):
        raise NotImplementedError("WIP")

    def __create_train_pipeline(
        self, train_data, model_alg, final_missingness_report=None):

        # Numeric pipeline
        if self.label in self.numeric_features_names:
            self.numeric_features_names.remove(self.label)
        numeric_pipeline = Pipeline([
            ("scaler", StandardScaler())
            ])

        # Category pipeline
        if self.label in self.category_features_names:
            self.category_features_names.remove(self.label)
        category_pipeline = Pipeline([
            ("encoder", OneHotEncoder(drop="if_binary"))
            ])

        # Train pipeline
        col_transformer = ColumnTransformer([
            ("numeric_pipeline", numeric_pipeline, self.numeric_features_names),
            ("category_pipeline", category_pipeline, self.category_features_names)])
        train_pipeline = Pipeline([
            ("column_transformer", col_transformer),
            # ("outlier", CustomTransformer(IsolationForest(contamination=0.1, n_jobs=-1))),
            ("imputation", KNNImputer(
                n_neighbors=5,
                add_indicator=final_missingness_report.isin(["MCAR/ MNAR"]).any()
                )),
            ("feature_eng", FeatureEng(
                features_names=train_data.columns.drop(self.label))),
            # ("select_feat", SelectKBest(score_func=f_classif, k=5)),
            ("model", model_alg),
        ])

        return train_pipeline

    def __eval(self, train_pipeline, test_data, metrics):
        logging.info("Evaluating model...")
        y_prob = train_pipeline.predict_proba(test_data.drop(
            self.label, axis=1))
        evaluation_results = ClassificationEval(
            train_pipeline, test_data[self.label].values,
            y_prob, metrics=metrics).eval()

        return evaluation_results

    def __mlflow_logging(self, best_train_assets, train_data):
        logging.info("Logging to mlflow...")
        mlflow_logging = MlflowLogging(
            tracking_uri=self.config["mlflow"]["tracking_uri"],
            backend_uri=self.config["mlflow"]["backend_uri"],
            artifact_uri=self.config["mlflow"]["artifact_uri"],
            mlflow_port=self.config["mlflow"]["port"],
            experiment_name=self.config["mlflow"]["experiment_name"],
            run_name=self.config["mlflow"]["run_name"],
            registered_model_name=self.config["mlflow"]["registered_model_name"]
        )
        mlflow_logging.activate_mlflow_server()
        mlflow_logging.logging(
            best_train_assets, train_data, self.label,
            self.split_ratio, self.tune,
            self.config["evaluation"]["classification"])

        return None

    def train(self):
        # Preprocessing
        self.__preprocessing()

        # Train-test split
        train_data, test_data = self.__train_test_split()

        # EDA
        self.__eda(train_data.copy())

        # Missing Value Analysis
        _, final_missingness_report =\
            self.__missing_val_analysis(train_data.copy())

        # Training
        train_assets = dict()
        best_score = 0
        for model_alg_name, model_alg in self.model_algs.items():
            logging.info(f"Training {model_alg_name} with hyperparameter tuning={self.tune}")
            train_pipeline = self.__create_train_pipeline(
                train_data, model_alg,
                final_missingness_report=final_missingness_report
                )

            if self.tune:
                param_grid = self.param_grids.get(model_alg_name)
                if self.tune_alg.__name__ == "GridSearchCV":
                    train_pipeline = self.tune_alg(
                        estimator=train_pipeline, param_grid=param_grid,
                        scoring=self.config["evaluation"]["classification"],
                        n_jobs=-1, cv=5).fit(
                            train_data.drop(self.label, axis=1),
                            train_data[self.label].squeeze()).best_estimator_

                elif self.tune_alg.__name__ == "RandomizedSearchCV":
                    train_pipeline = self.tune_alg(
                        estimator=train_pipeline, param_distributions=param_grid,
                        cv=5, scoring=self.config["evaluation"]["classification"],
                        n_jobs=-1, random_state=self.seed, n_iter=10).fit(
                        train_data.drop(self.label, axis=1),
                        train_data[self.label].squeeze()).best_estimator_

                elif self.tune_alg.__name__ == "BayesSearchCV":
                    train_pipeline = self.tune_alg(
                        estimator=train_pipeline, search_spaces=param_grid,
                        optimizer_kwargs={"base_estimator": "GP"},
                        scoring=self.config["evaluation"]["classification"],
                        n_jobs=-1, cv=5, random_state=self.seed,
                        n_iter=10).fit(
                            train_data.drop(self.label, axis=1),
                            train_data[self.label]).best_estimator_

            else:
                train_pipeline.fit(
                    train_data.drop(self.label, axis=1),
                    train_data[self.label]
                )

            # Evaluation
            evaluation_results = self.__eval(
                train_pipeline, test_data,
                self.config["evaluation"]["classification"])
            train_assets[model_alg_name] = {
                "train_pipeline": train_pipeline,
                "evaluation_results": evaluation_results
            }
            score = evaluation_results["score"]
            if score > best_score:
                best_score = score
                best_model_alg_name = model_alg_name

        # Get best model
        best_train_assets = train_assets[best_model_alg_name]

        # Logging model and evaluation assets
        self.__mlflow_logging(best_train_assets, train_data)

        # Print best evaluation score on test data
        logging.info(f"Best {self.config['evaluation']['classification']}: {best_score}")

        logging.info("Training completed")


if __name__ == "__main__":
    Train(Config.train).train()

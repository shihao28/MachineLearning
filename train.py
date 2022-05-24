import os
import logging
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import SimpleImputer, KNNImputer, IterativeImputer
from sklearn.feature_selection import *
from sklearn.linear_model import *
from sklearn.ensemble import *
from sklearn.svm import *
from sklearn.model_selection import GridSearchCV
from skopt import BayesSearchCV
from sklearn.metrics import *
from subprocess import Popen, DEVNULL
import mlflow
from mlflow.models.signature import infer_signature
from pathlib import Path
import shutil

from config import Config
from eval import ClassificationEval


# Set log level
logging.basicConfig(
    level=logging.DEBUG,
    format="(%(asctime)s) | %(levelname)-8s | %(module)s: %(message)s",
    datefmt='%Y-%m-%d %H:%M:%S')


class Train:
    def __init__(self, config):

        logging.info("Initializing...")
        self.config = config
        self.data = pd.read_csv(Config.data.get("data_path"))
        self.label = config.data.get("label")
        self.model_algs = config.model
        self.split_ratio = config.train_val_test_split.get("split_ratio")
        self.tune = config.param_grid.get("tune")
        self.param_grids = config.param_grid
        self.metrics = config.evaluation.get("classification")
        self.train_data, self.test_data = None, None

        # Activating mlflow
        tracking_uri = config.mlflow.get("tracking_uri")
        backend_uri = config.mlflow.get("backend_uri")
        artifact_uri = config.mlflow.get("artifact_uri")
        mlflow_port = config.mlflow.get("port")
        env = {
            "MLFLOW_TRACKING_URI": f"{tracking_uri}:{mlflow_port}",
            "BACKEND_URI": backend_uri,
            "ARTIFACT_URI": artifact_uri,
            "MLFLOW_PORT": mlflow_port
            }
        os.environ.update(env)
        cmd_mlflow_server = (
            f"mlflow server --backend-store-uri {backend_uri} "
            f"--default-artifact-root {artifact_uri} "
            f"--host 0.0.0.0 -p {mlflow_port}")
        with open("stderr.txt", mode="wb") as out, open("stdout.txt", mode="wb") as err:
            Popen(cmd_mlflow_server, stdout=out, stderr=err, stdin=DEVNULL,
                  universal_newlines=True, encoding="utf-8",
                  env=os.environ, shell=True)

    def __preprocessing(self):
        logging.info("Preprocess data...")
        # To do label smoothing
        self.data["above_median_house_value"] = 0

        # Binary class
        self.data["above_median_house_value"] =\
            self.data["median_house_value"].apply(
                lambda x: 1 if x > self.data["median_house_value"].median() else 0)

        # Multi class
        # self.data.loc[self.data["median_house_value"] > self.data["median_house_value"].quantile(0.25), "above_median_house_value"] = 1
        # self.data.loc[self.data["median_house_value"] > self.data["median_house_value"].quantile(0.75), "above_median_house_value"] = 2

        return None

    def __train_test_split(self):
        logging.info("Train-test splitting...")
        train_data, test_data = train_test_split(
            self.data, test_size=self.split_ratio,
            stratify=self.data[self.label])

        return train_data, test_data

    def __eda(self):
        pass

    def __imbalanced2balanced(self):
        pass

    def __remove_outlier(self):
        pass

    def __select_feat(self):
        pass

    def __impute_missing_val(self):
        pass

    def __feat_eng(self):
        pass

    def __create_train_pipeline(self, train_data, model_alg):
        # Numeric pipeline
        numeric_features = train_data.select_dtypes(
            include=["int", "float"]).columns
        numeric_features = numeric_features.drop(
            labels=self.label, errors="ignore")
        numeric_pipeline = Pipeline([
            ("scaler", StandardScaler())
            ])

        # Category pipeline
        category_features = train_data.select_dtypes(
            exclude=["int", "float"]).columns
        category_features = category_features.drop(
            labels=self.label, errors="ignore")
        category_pipeline = Pipeline([
            ("encoder", OneHotEncoder(drop="if_binary"))
            ])

        # Train pipeline
        col_transformer = ColumnTransformer([
            ("numeric_pipeline", numeric_pipeline, numeric_features),
            ("category_pipeline", category_pipeline, category_features)])
        train_pipeline = Pipeline([
            ("column_transformer", col_transformer),
            # ("outlier", CustomTransformer(IsolationForest(contamination=0.1, n_jobs=-1))),
            ("imputation", KNNImputer()),
            # ("select_feat", SelectKBest(score_func=f_classif, k=5)),
            ("model", model_alg),
        ])

        return train_pipeline

    def __eval(self, train_pipeline, test_data, metrics):
        y_prob = train_pipeline.predict_proba(test_data.drop(
            self.label, axis=1))
        evaluation_results = ClassificationEval(
            train_pipeline, test_data[self.label].values, 
            y_prob, metrics=metrics).eval()

        return evaluation_results

    def __mlflow_logging(self, best_train_assets, train_data):
        # check f1 score with cls report
        logging.info("Logging to mlflow...")
        best_train_pipeline = best_train_assets.get("train_pipeline")
        best_evaluation_results = best_train_assets.get("evaluation_results")
        best_cls_report = self.__get_mlflow_cls_report(best_evaluation_results)
        best_threshold = best_train_assets.get("best_threshold")

        mlflow.set_experiment(self.config.mlflow.get("experiment_name"))
        with mlflow.start_run(run_name=self.config.mlflow.get("run_name")):
            mlflow.log_param('target_variable', self.label)
            mlflow.log_param('split_ratio', self.split_ratio)
            mlflow.log_param('tune', self.tune)
            mlflow.log_param('eval_metrics', self.config.evaluation.get("classification"))

            mlflow.log_metrics(best_cls_report)
            if best_threshold is not None:
                mlflow.log_metrics("best_threshold", best_threshold)

            signature = infer_signature(
                train_data,
                pd.DataFrame({self.label: best_train_pipeline.predict(
                    train_data.drop(self.label, axis=1))}))
            mlflow.sklearn.log_model(
                sk_model=best_train_pipeline, artifact_path="sk_models",
                signature=signature, input_example=train_data.sample(5),
                registered_model_name=self.config.mlflow.get("registered_model_name")
                )

            # Store plots as artifacts
            artifact_folder = Path("mlflow_tmp")
            artifact_folder.mkdir(parents=True, exist_ok=True)

            # Storing only figures, pd.DataFrames are excluded
            conf_matrix_fig = best_evaluation_results.get("conf_matrix_fig")
            conf_matrix_fig.savefig(Path(artifact_folder, "conf_matrix.png"))
            fig_all = best_evaluation_results.get("fig")
            for label, fig in fig_all.items():
               fig.savefig(Path(artifact_folder, f"fig_{label}.png"))
            mlflow.log_artifacts(
                artifact_folder, artifact_path="evaluation_artifacts")
            shutil.rmtree(artifact_folder)
        return None

    def __get_mlflow_cls_report(self, best_evaluation_results):
        best_cls_report = pd.DataFrame(best_evaluation_results.get("cls_report"))
        best_cls_report["metrics"] = best_cls_report.index
        best_cls_report.reset_index(drop=True, inplace=True)
        best_cls_report = pd.melt(
            best_cls_report, "metrics", best_cls_report.columns[:-1])
        best_cls_report.loc[best_cls_report["variable"] == "accuracy", "metrics"] = ""
        best_cls_report["metrics"] = best_cls_report["metrics"] + "_" + best_cls_report["variable"].astype(str)
        best_cls_report.drop("variable", 1, inplace=True)
        best_cls_report = pd.Series(
            best_cls_report["value"].values,
            index=best_cls_report["metrics"].values).to_dict()

        return best_cls_report

    def train(self):
        # Preprocessing
        self.__preprocessing()

        # Train-test split
        train_data, test_data = self.__train_test_split()

        # Training
        train_assets = dict()
        best_score = 0
        for model_alg_name, model_alg in self.model_algs.items():
            logging.info(f"Training {model_alg_name}...")
            train_pipeline = self.__create_train_pipeline(
                train_data, model_alg)

            if self.tune:
                param_grid = self.param_grids.get(model_alg_name)

                # To include hyperparameter tuning method in config
                # Grid Search SV
                train_pipeline = GridSearchCV(
                    estimator=train_pipeline, param_grid=param_grid,
                    scoring=self.config.evaluation.get("classification"),
                    n_jobs=-1, cv=5).fit(
                        train_data.drop(self.label, axis=1), 
                        train_data[self.label].squeeze()
                        ).best_estimator_

                # Bayes Search
                # train_pipeline = BayesSearchCV(
                #     estimator=train_pipeline, search_spaces=param_grid, 
                #     optimizer_kwargs={"base_estimator": "GP"},
                #     scoring=self.config.evaluation.get("classification"), n_jobs=-1, cv=5).fit(
                #         train_data.drop(self.label, axis=1), train_data[self.label]
                #         ).best_estimator_
            else:
                train_pipeline.fit(
                    train_data.drop(self.label, axis=1), train_data[self.label].squeeze()
                )

            # Evaluation
            evaluation_results = self.__eval(
                train_pipeline, test_data, 
                self.config.evaluation.get("classification"))
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

        logging.info("Training completed")


if __name__ == "__main__":
    Train(Config).train()

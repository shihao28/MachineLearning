import pandas as pd
import logging
import mlflow
from mlflow.tracking import MlflowClient

from config import Config
from src.mlflow_logging import MlflowLogging


# Set log level
logging.basicConfig(
    level=logging.DEBUG,
    format="(%(asctime)s) | %(levelname)-8s | %(module)s: %(message)s",
    datefmt='%Y-%m-%d %H:%M:%S')


class Predict:
    def __init__(self, config):
        logging.info("Initializing...")
        self.config = config
        mlflow_logging = MlflowLogging(
            tracking_uri=self.config["mlflow"]["tracking_uri"],
            backend_uri=self.config["mlflow"]["backend_uri"],
            artifact_uri=self.config["mlflow"]["artifact_uri"],
            mlflow_port=self.config["mlflow"]["port"],
        )
        mlflow_logging.activate_mlflow_server()

    def __load_model(self):
        logging.info("Loading model...")
        client = MlflowClient()
        loaded_model = client.get_latest_versions(
            name=self.config["mlflow"]["model_name"], stages=["None"])[0]
        model_uri = f"models:/{loaded_model.name}/{loaded_model.version}"
        model = mlflow.pyfunc.load_model(model_uri=model_uri)

        return model

    def __preprocessing(self, X):
        logging.info("Preprocess data...")
        # To do label smoothing
        X["above_median_house_value"] = 0

        # Binary class
        X["above_median_house_value"] =\
            X["median_house_value"].apply(
                lambda x: 1 if x > X["median_house_value"].median() else 0)

        # Multi class
        # X.loc[X["median_house_value"] > X["median_house_value"].quantile(0.25), "above_median_house_value"] = 1
        # X.loc[X["median_house_value"] > X["median_house_value"].quantile(0.75), "above_median_house_value"] = 2

        return X

    def predict(self, X=None):
        model = self.__load_model()

        label = f"Predicted_{model.metadata.get_output_schema().input_names()[0]}"
        if X is None:
            X = pd.read_csv(self.config["data_path"])
            X = self.__preprocessing(X)

        logging.info("Predicting...")
        y_pred = model.predict(X)
        y_pred = pd.DataFrame({label: y_pred})
        output_data = pd.concat([X, y_pred], 1)

        logging.info("Prediction completed")

        return output_data


if __name__ == "__main__":
    output_data = Predict(Config.predict).predict()

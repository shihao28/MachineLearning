import os
from subprocess import Popen, run, PIPE, DEVNULL

import numpy as np
import pytest
import mlflow

from conf import Config
from src.modelling.pipelines.input_handler import InputHandler
np.set_printoptions(suppress=True, precision=6)

@pytest.fixture
def mlflow_setup():
    """fixture mlflow_setup is used to ensure model registry is enabled by setting tracking uri using sqlite db.
    """
    env = {
        "TRACKING_SERVER": "local",
        "MLFLOW_TRACKING_URI": f'{Config.MLFLOW["MLFLOW_RUN"]["local"]["tracking_host"]}:{Config.MLFLOW["ENDPOINT_PORT"]}',
        "BACKEND_URI": Config.MLFLOW["MLFLOW_RUN"]["local"]["backend_uri"],
        "ARTIFACT_URI": Config.MLFLOW["MLFLOW_RUN"]["local"]["artifact_uri"],
        "MLFLOW_PORT": Config.MLFLOW["ENDPOINT_PORT"],
    }

    os.environ.update(env)

    # If run locally, tracking uri is set to backend_uri
    mlflow.set_tracking_uri(os.environ['BACKEND_URI'])

@pytest.fixture
def sample_dataset():
    data_df = InputHandler.get_data()
    return data_df


@pytest.fixture
def expected_metrics():
    expected_metrics = {
        'LR': {
            'R2': {
                'Train': 0.653134,
                'Validation': 0.654454,
                'Test': 0.662999,
            },
            'Adjusted_R2': {
                'Train': 0.652943,
                'Validation': 0.654264,
                'Test': 0.662814,
            },
            'MAE': {
                'Train': 49222.515924,
                'Validation': 48791.740763,
                'Test': 49226.080379,
            },
            'MAPE': {
                'Train': 28.393152,
                'Validation': 29.121888,
                'Test': 28.649173,
            }
        },
        'RFR': {
            'R2': {
                'Train': 0.933583,
                'Validation': 0.786155,
                'Test': 0.790046,
            },
            'Adjusted_R2': {
                'Train': 0.933547,
                'Validation': 0.786038,
                'Test': 0.789931,
            },
            'MAE': {
                'Train': 19211.397277,
                'Validation': 34467.707964,
                'Test': 35130.954446,
            },
            'MAPE': {
                'Train': 10.481070,
                'Validation': 19.779849,
                'Test': 19.171568,
            }
        },
        'RFC': {
            'Accuracy': 0.863453,
            'Precision_0': 0.909831,
            'Precision_1': 0.857493,
            'Recall_0': 0.831935,
            'Recall_1': 0.896863,
            'F1': 0.866059,
            'MCC': 0.747142,
            'ROC_AUC': 0.864399,
        },
        'AdaC': {
            'Accuracy': 0.540031,
            'Precision_0': 0.324477,
            'Precision_1': 0.31118,
            'Recall_0': 0.538307,
            'Recall_1': 0.541852,
            'F1': 0.387507,
            'MCC': 0.08013,
            'ROC_AUC': 0.540079,
        },
    }
    return expected_metrics


@pytest.fixture
def expected_vars_input_handler():
    expected_cols = np.array(['longitude', 'latitude', 'housing_median_age', 'total_rooms', 'total_bedrooms',
                              'population', 'households', 'median_income', 'median_house_value', 'ocean_proximity',
                              'above_median_house_value'])

    expected_mean_values = np.array([-119.5706886, 35.6332213, 28.6330935, 2636.5042333, 537.8705525, 1424.9469486,
                                     499.4334655, 3.8711616, 206864.4131552, 0.4999755])
    return expected_cols, expected_mean_values


@pytest.fixture
def ratio():
    ratio = 0.25
    return ratio


@pytest.fixture(params=['LR', 'RFR'])
def regressor(request):
    return request.param


@pytest.fixture(params=['RFC', 'AdaC'])
def classifier(request):
    return request.param


@pytest.fixture
def expected_mean_vals_split():
    expected_mean_values = np.array([-119.5691471, 35.6311479, 28.6205951, 2647.4257374, 539.5901853,
                                     1430.1078048, 500.7253981, 3.8732096, 207109.151005, 0.5018925])
    return expected_mean_values

@pytest.fixture
def expected_mean_test_predict():
    expected_mean_values = np.array([10316.1760877, -119.5706886, 35.6332213, 28.6330935,
                                     2636.5042333, 537.8705525, 1424.9469486, 499.4334655,
                                     3.8711616, 207549.0040754])

    return expected_mean_values

@pytest.fixture
def expected_hash():
    expected_hash = "c26c91"
    return expected_hash

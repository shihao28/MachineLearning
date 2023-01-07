import pytest
import numpy as np
import pandas as pd
from src.modelling.pipelines.ml_trainer import MLTrainer
from src.modelling.pipelines.input_handler import InputHandler

np.set_printoptions(suppress=True, precision=6)

# Loading environment variables (for local use only)
from dotenv import load_dotenv
load_dotenv()


def test_input_handler(sample_dataset, expected_vars_input_handler):
    data_df = sample_dataset
    expected_cols, expected_mean_values = expected_vars_input_handler

    np.testing.assert_almost_equal(data_df[expected_cols].mean().values, expected_mean_values,
                                   err_msg="Raw input data data_df downloaded from DB does not match expected values")
    assert (data_df.columns == expected_cols).all(),\
        f"The column names in data_df downloaded from DB does not match expected columns: {expected_cols}"


def test_train_test_split(sample_dataset, ratio, expected_mean_vals_split, regressor):
    mltrainer_model = MLTrainer(
        target_variable='median_house_value',
        numeric_features=['longitude', 'latitude', 'housing_median_age', 'total_rooms', 'total_bedrooms',
                          'population', 'households', 'median_income'],
        categorical_features=['ocean_proximity'], model_type='regression',
        model_algs=regressor, tune=True, experiment_name='Experiment-UT', run_name='Run-Split',
        register_model_name=None, split_ratio=ratio, split_type='random', scaler='standard'
    )
    train_df, test_df = mltrainer_model._split_train_test(sample_dataset, ratio=ratio)
    expected_train_df_rows = np.floor(sample_dataset.shape[0] * (1 - ratio))
    expected_test_df_rows = sample_dataset.shape[0] - expected_train_df_rows

    assert train_df.shape[0] == expected_train_df_rows, \
        f"Number of rows in train_df ({train_df.shape[0]}) does not match expected ({expected_train_df_rows})"
    assert test_df.shape[0] == expected_test_df_rows, \
        f"Number of rows in test_df ({test_df.shape[0]}) does not match expected ({expected_test_df_rows})"

    np.testing.assert_almost_equal(train_df.mean().values, expected_mean_vals_split,
                                   err_msg="Generated split-train-test result does not match expected values"
                                   )


def test_regression(mlflow_setup, sample_dataset, ratio, expected_metrics, regressor):

    mltrainer_model = MLTrainer(
        target_variable='median_house_value',
        numeric_features=['longitude', 'latitude', 'housing_median_age', 'total_rooms', 'total_bedrooms',
                          'population', 'households', 'median_income'],
        categorical_features=['ocean_proximity'], model_type='regression',
        model_algs=[regressor], tune='grid', experiment_name='Experiment-UT', run_name='Run-UT-Regression',
        register_model_name="ut_regression_median_house_value", split_ratio=ratio, split_type='random', scaler='standard',
    )
    mltrainer_model.train(sample_dataset)

    # Checking if model metrics are same as expected_metrics
    model_metrics = mltrainer_model.mlpipeline_model_list[regressor]['evaluation_dict']['evaluation_results']
    expected_metrics = pd.DataFrame(expected_metrics[regressor])

    for column in expected_metrics.columns:
        np.testing.assert_array_almost_equal(
            model_metrics[column], expected_metrics[column],
            err_msg=f"{column} metrics do not match. Got {model_metrics[column]}, expected {expected_metrics[column]}"
        )


def test_classification(mlflow_setup, sample_dataset, expected_metrics, classifier):

    mltrainer_model = MLTrainer(
        target_variable='above_median_house_value',
        numeric_features=['longitude', 'latitude', 'housing_median_age', 'total_rooms', 'total_bedrooms',
                          'population', 'households', 'median_income'],
        categorical_features=['ocean_proximity'], model_type='classification',
        model_algs=[classifier], tune='grid', experiment_name='Experiment-UT', run_name='Run-UT-Classification',
        register_model_name="ut_class_above_median_house_value", split_ratio=0.3, split_type='random',
    )
    mltrainer_model.train(sample_dataset)

    # Checking if model metrics are same as expected_metrics
    model_metrics = mltrainer_model.mlpipeline_model_list[classifier]['evaluation_dict']['evaluation_results']
    expected_metrics = pd.DataFrame(expected_metrics[classifier], index=[0])

    for column in expected_metrics.columns:
        np.testing.assert_array_almost_equal(
            model_metrics[column].mean(), expected_metrics[column],
            err_msg=f"{column} metrics do not match. Got {model_metrics[column]}, expected {expected_metrics[column]}"
        )

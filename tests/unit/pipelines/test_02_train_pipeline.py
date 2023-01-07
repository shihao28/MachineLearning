import pytest
from src.modelling.pipelines.train_pipeline import TrainPipeline

# Loading environment variables (for local use only)
from dotenv import load_dotenv
load_dotenv()


def test_train_pipeline_regression(mlflow_setup):

    TrainPipeline().train_models(
        model_type='regression', model_algs=['LR', 'RFR'],
        tune='grid', experiment_name='Experiment-1', run_name='Run-1',
        register_model_name="ut_regression_median_house_value",
        numeric_features=['longitude', 'latitude', 'housing_median_age', 'total_rooms', 'total_bedrooms',
                          'population', 'households', 'median_income'],
        categorical_features=['ocean_proximity']
    )


def test_train_pipeline_classification(mlflow_setup):

    TrainPipeline().train_models(
        model_type='classification', model_algs=['RFC'],
        tune='grid', experiment_name='Experiment-1', run_name='Run-1',
        register_model_name="ut_class_above_median_house_value",
        target_variable_list=['above_median_house_value'],
        numeric_features=['longitude', 'latitude', 'housing_median_age', 'total_rooms', 'total_bedrooms',
                          'population', 'households', 'median_income'],
        categorical_features=['ocean_proximity']
    )

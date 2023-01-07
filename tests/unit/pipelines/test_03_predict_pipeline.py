import os
import pandas as pd
import numpy as np
import pytest
from src.modelling.pipelines.predict_pipeline import PredictPipeline

# Loading environment variables (for local use only)
from dotenv import load_dotenv
load_dotenv()


np.set_printoptions(suppress=True)


def test_predict_pipeline(mlflow_setup, expected_mean_test_predict):

    PredictPipeline().run_pipeline(
        na_treatment='drop',
        import_from='csv',
        import_settings={
            "new_data_filepath":
                "https://raw.githubusercontent.com/ageron/handson-ml/master/datasets/housing/housing.csv"
        },
        export_to='csv',
        export_settings={
            "export_filepath": '.',
            "export_filename": 'test_predict.csv'
        },
        model_name="ut_regression_median_house_value",
        version="1"
    )

    assert os.path.exists("test_predict.csv"), "Prediction failed, test_predict.csv not found."
    np.testing.assert_almost_equal(pd.read_csv("test_predict.csv").mean().values,
                                   expected_mean_test_predict), "The predicted output does not match with the expected prediction output"
    os.remove("test_predict.csv")


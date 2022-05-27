from sklearn.linear_model import *
from sklearn.ensemble import *
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from skopt import BayesSearchCV
from sklearn.svm import *


class Config(object):

    # Training config
    train = dict(

        # regression, classification, time_series
        problem_type="classification",

        data=dict(
            data_path="data/01_raw/housing_binary_class.csv",
            label="above_median_house_value",
            numeric_features=[
                'longitude', 'latitude', 'housing_median_age', 'total_rooms',
                'total_bedrooms', 'population', 'households', 'median_income',
                'median_house_value'
            ],
            category_features=[
                'ocean_proximity', 'above_median_house_value'
            ],
            datetime_features=[

            ]
        ),

        train_val_test_split=dict(
            split_ratio=0.3,
        ),

        model={
            SVC.__name__: SVC(probability=True),
            LogisticRegression.__name__: LogisticRegression()
        },

        tuning={
            "tune": False,
            "search_method": GridSearchCV,  # RandomizedSearchCV, BayesSearchCV
            SVC.__name__: dict(
                model__C=[1, 5],
                model__kernel=["linear", "poly", "rbf"]
                ),
            LogisticRegression.__name__: dict(
                model__penalty=["none", "l2"]
            )
        },

        evaluation=dict(
            # Get list of metrics from below
            # https://scikit-learn.org/stable/modules/model_evaluation.html#scoring-parameter
            regression="r2",
            classification="f1_weighted"
        ),

        mlflow=dict(
            tracking_uri="http://127.0.0.1",
            backend_uri="sqlite:///mlflow.db",
            artifact_uri="./mlruns/",
            experiment_name="Best Pipeline",
            run_name="trial",
            registered_model_name="my_cls_model",
            port="5000",
        ),

        seed=0
    )

    # Prediction config
    predict = dict(

        data_path="data/01_raw/housing.csv",

        mlflow=dict(
            tracking_uri="http://127.0.0.1",
            backend_uri="sqlite:///mlflow.db",
            artifact_uri="./mlruns/",
            model_name="my_cls_model",
            port="5000",
            model_version="latest"
        ),
    )

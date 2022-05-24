from sklearn.linear_model import *
from sklearn.ensemble import *
from sklearn.svm import *


class Config(object):
    data = dict(
        data_path="data/01_raw/housing.csv",
        label="above_median_house_value",
    )

    train_val_test_split = dict(
        split_ratio=0.3,
    )

    model = {
        SVC.__name__: SVC(probability=True),
        LogisticRegression.__name__: LogisticRegression()
    }

    param_grid = {
        "tune": False,
        SVC.__name__: dict(
            model__C=[1, 5],
            model__kernel=["linear", "poly", "rbf"]
            ),
        LogisticRegression.__name__: dict(
            model__penalty=["none", "l2"]
        )
    }

    evaluation = dict(
        # Get list of metrics from below
        # https://scikit-learn.org/stable/modules/model_evaluation.html#scoring-parameter
        classification="recall_weighted"
    )

    mlflow = dict(
        tracking_uri="http://127.0.0.1",
        backend_uri="sqlite:///mlflow.db",
        artifact_uri="./mlruns/",
        experiment_name="Best Pipeline",
        run_name="trial",
        registered_model_name="my_cls_model",
        port="5000",
    )

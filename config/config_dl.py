import torch
from torch import nn, optim
from src.model_dl import MLP


class ConfigDL(object):

    # Training config
    train = dict(

        device=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu'),

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
            MLP.__name__: MLP(17, (20, 5), 2),
        },

        criterion=nn.BCEWithLogitsLoss(),

        epochs=50,

        batch_size=32,

        optimizer=dict(
            alg=optim.SGD,
            param=dict(
                lr=0.01, momentum=0.9, weight_decay=0.0005
            )
        ),

        lr_scheduler=dict(
            alg=optim.lr_scheduler.StepLR,
            param=dict(
                step_size=20, gamma=0.1
            )
        ),

        # WIP
        # tuning={

        # },

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

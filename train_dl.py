import os
import torch
import copy
import logging
import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import SimpleImputer, KNNImputer, IterativeImputer
from sklearn.svm import *
from sklearn.metrics import *

from config import ConfigDL
from train_ml import TrainML
from src.feature_eng import FeatureEng
from src.utils import AverageMeter, accuracy


# Set log level
logging.basicConfig(
    level=logging.DEBUG,
    format="(%(asctime)s) | %(levelname)-8s | %(module)s: %(message)s",
    datefmt='%Y-%m-%d %H:%M:%S')


def train_one_epoch(
    dataloader_train, model, criterion, optimizer_, device):

    model.train()
    train_epoch_loss = AverageMeter()
    train_epoch_accuracy = AverageMeter()
    for inputs, labels in dataloader_train:
        inputs = inputs.to(device)
        labels = labels.to(device)

        optimizer_.zero_grad()

        with torch.set_grad_enabled(True):
            logits = model(inputs)
            train_batch_loss = criterion(logits.squeeze(), labels.float())

            train_batch_loss.backward()
            optimizer_.step()

        train_epoch_loss.update(train_batch_loss, inputs.size(0))
        acc1 = accuracy(logits, labels.data)[0]
        train_epoch_accuracy.update(acc1.item(), inputs.size(0))

    return model, train_epoch_loss.avg, train_epoch_accuracy.avg


def validate(
    dataloader_eval, model, criterion, device,
        print_cls_report=False, target_names=None):

    model.eval()
    val_epoch_loss = AverageMeter()
    val_epoch_accuracy = AverageMeter()
    labels_all = []
    preds_all = []
    for inputs, labels in dataloader_eval:
        inputs = inputs.to(device)
        labels = labels.to(device)

        with torch.set_grad_enabled(False):
            logits = model(inputs)
            val_batch_loss = criterion(logits.squeeze(), labels.float())
            if logits.size(1) > 1:
                # Multiclass
                _, preds = torch.max(logits, 1)
            else:
                # Binary Class
                preds = torch.sigmoid(logits)
                preds = torch.where(preds > 0.5, 1, 0)

        labels_all.append(labels)
        preds_all.append(preds)

        val_epoch_loss.update(val_batch_loss, inputs.size(0))
        acc1 = accuracy(logits, labels.data)[0]
        val_epoch_accuracy.update(acc1.item(), inputs.size(0))

    labels_all = torch.cat(labels_all, 0).cpu().numpy()
    preds_all = torch.cat(preds_all, 0).cpu().numpy()
    if print_cls_report:
        cls_report = classification_report(
            y_true=labels_all, y_pred=preds_all,
            target_names=target_names,
            digits=6)
        logging.info(f"\n{cls_report}")

    return val_epoch_loss.avg, val_epoch_accuracy.avg


class TrainDL(TrainML):
    def __init__(self, config):
        self.config = config
        self.device = config["device"]
        self.problem_type = config["problem_type"]
        self.data = pd.read_csv(config["data"]["data_path"])
        self.label = config["data"]["label"]
        self.numeric_features_names = config["data"]["numeric_features"]
        self.category_features_names = config["data"]["category_features"]
        self.datetime_features_names = config["data"]["datetime_features"]
        self.model_algs = config["model"]
        self.split_ratio = config["train_val_test_split"]["split_ratio"]
        self.epochs = config["epochs"]
        self.batch_size = config["batch_size"]
        self.criterion = config["criterion"]
        self.optimizer = config["optimizer"]
        self.lr_scheduler = config["lr_scheduler"]

    def __create_preprocessing_pipeline(self, train_data):
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

        # Preprocessing pipeline
        col_transformer = ColumnTransformer([
            ("numeric_pipeline", numeric_pipeline, self.numeric_features_names),
            ("category_pipeline", category_pipeline, self.category_features_names),
            ])
        preprocessing_pipeline = Pipeline([
            ("column_transformer", col_transformer),
            # ("outlier", CustomTransformer(IsolationForest(contamination=0.1, n_jobs=-1))),
            ("imputation", KNNImputer(
                n_neighbors=5,
                # add_indicator=final_missingness_report.isin(["MCAR/ MNAR"]).any()
                )),
            ("feature_eng", FeatureEng(
                features_names=train_data.columns.drop(self.label))),
            # ("select_feat", SelectKBest(score_func=f_classif, k=5)),
            # ("model", SVC()),
        ])

        # Label pipeline
        label_pipeline = LabelEncoder()

        return preprocessing_pipeline, label_pipeline

    def __setting_pytorch_utils(self):
        # Train-test split
        train_data, test_data = self.train_test_split(
            self.data, self.split_ratio, self.problem_type,
            self.label, self.config["seed"]
        )

        # Create preprocessing and label encoder pipeline
        preprocessing_pipeline, label_pipeline =\
            self.__create_preprocessing_pipeline(train_data)
        preprocessing_pipeline.fit(
            train_data.drop(self.label, axis=1),
            train_data[self.label])
        label_pipeline.fit(train_data[self.label])

        # Transform dataset
        X_train = preprocessing_pipeline.transform(
            train_data.drop(self.label, axis=1))
        X_test = preprocessing_pipeline.transform(
            test_data.drop(self.label, axis=1))
        y_train = label_pipeline.transform(train_data[self.label])
        y_test = label_pipeline.transform(test_data[self.label])

        # Load dataset onto pytorch loader
        X_train = torch.from_numpy(X_train).float()
        X_test = torch.from_numpy(X_test).float()
        y_train = torch.from_numpy(y_train)
        y_test = torch.from_numpy(y_test)
        datasets, dataloaders = dict(), dict()
        for type_, X, y in zip(["train", "val"], [X_train, X_test], [y_train, y_test]):
            datasets[type_] = torch.utils.data.TensorDataset(X, y)
            dataloaders[type_] = torch.utils.data.DataLoader(
                datasets[type_], batch_size=self.batch_size, shuffle=True,
                num_workers=os.cpu_count(), drop_last=False)

        return preprocessing_pipeline, label_pipeline, dataloaders

    def train(self):
        preprocessing_pipeline, label_pipeline, dataloaders = self.__setting_pytorch_utils()

        train_assets = dict()
        best_score = 0
        for model_alg_name, model in self.model_algs.items():
            logging.info(f"Training {model_alg_name}...")

            model.to(self.device)

            # Initialize optimizer and lr scheduler
            optimizer_ = self.optimizer["alg"](
                params=model.parameters(), **self.optimizer["param"])
            lr_scheduler_ = self.lr_scheduler["alg"](
                optimizer=optimizer_, **self.lr_scheduler["param"])

            best_state_dict = copy.deepcopy(model.state_dict())
            best_accuracy = 0
            train_loss, train_accuracy, val_loss, val_accuracy, lr = [], [], [], [], []
            for epoch in range(self.epochs):

                # Train
                model, train_epoch_loss, train_epoch_accuracy =\
                    train_one_epoch(
                        dataloaders['train'], model,
                        self.criterion, optimizer_, self.device)
                train_loss.append(train_epoch_loss.item())
                train_accuracy.append(train_epoch_accuracy)
                logging.info(
                    f"Epoch {epoch:3d}/{self.epochs-1:3d} {'Train':5s}, "
                    f"Loss: {train_epoch_loss:.4f}, "
                    f"Acc: {train_epoch_accuracy:.4f}")

                # Eval
                val_epoch_loss, val_epoch_accuracy = validate(
                    dataloaders['val'], model, self.criterion,
                    self.device, False, label_pipeline.classes_)
                val_loss.append(val_epoch_loss.item())
                val_accuracy.append(val_epoch_accuracy)
                logging.info(
                    f"Epoch {epoch:3d}/{self.epochs-1:3d} {'Val':5s}, "
                    f"Loss: {val_epoch_loss:.4f}, "
                    f"Acc: {val_epoch_accuracy:.4f}")

                lr.append(lr_scheduler_.get_last_lr()[0])

                if val_epoch_accuracy > best_accuracy:
                    best_accuracy = val_epoch_accuracy
                    best_state_dict = copy.deepcopy(model.state_dict())

                lr_scheduler_.step()

            logging.info('Best Val Acc: {:4f}'.format(best_accuracy))

            # Load best model
            model.load_state_dict(best_state_dict)

            # Classification report
            val_epoch_loss, val_epoch_accuracy = validate(
                dataloaders['val'], model, self.criterion,
                self.device, True, label_pipeline.classes_)

            # Save best model
            torch.save(model.state_dict(), f"{model_alg_name}.pth")

            pd.DataFrame({
                'Epochs': range(self.epochs), 'Learning Rate': lr,
                'Training Loss': train_loss, 'Training Accuracy': train_accuracy,
                'Validation Loss': val_loss, 'Validation Accuracy': val_accuracy
                }).to_csv(f"{model_alg_name}.csv", index=False)

            logging.info("Training completed")

        return None


if __name__ == '__main__':
    TrainDL(ConfigDL.train).train()

    # # Input
    # DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    # EPOCHS = 50
    # BATCH_SIZE = 64
    # LEARNING_RATE = 0.01
    # DATA_DIR = "data/input/car_cls"
    # ANNOT = ['train70.txt', 'val70.txt']
    # CLS_MAPPING = dict(car=1, others=0)
    # MODEL = "ResNet18"
    # MODEL_NAME = "src/car_cls/exp/model0"

    # # create transform
    # data_transforms = {
    #     "train": transforms.Compose([
    #         # transforms.Resize(56),
    #         # transforms.RandomCrop(56),
    #         transforms.RandomResizedCrop(56, scale=(0.8, 1.0)),
    #         transforms.RandomHorizontalFlip(0.5),
    #         transforms.ToTensor(),
    #         transforms.Normalize(
    #             [0.485, 0.456, 0.406],
    #             [0.229, 0.224, 0.225])]),
    #     "val": transforms.Compose([
    #         transforms.Resize(56),
    #         transforms.CenterCrop(56),
    #         transforms.ToTensor(),
    #         transforms.Normalize(
    #             [0.485, 0.456, 0.406],
    #             [0.229, 0.224, 0.225])])}

    # # create dataset
    # image_datasets = {x: MyDataset(
    #     root=DATA_DIR,
    #     annot=annot,
    #     cls_mapping=CLS_MAPPING,
    #     transform=data_transforms[x])
    #     for x, annot in zip(["train", "val"], ANNOT)}
    # dataloaders = {x: torch.utils.data.DataLoader(
    #     image_datasets[x], batch_size=BATCH_SIZE, shuffle=True,
    #     num_workers=os.cpu_count(), drop_last=False) for x in ["train", "val"]}
    # dataset_sizes = {x: len(image_datasets[x]) for x in ["train", "val"]}

    # # create model
    # if MODEL == "ResNet18":
    #     model = ResNet18(num_class=len(CLS_MAPPING))
    # model = model.to(DEVICE)

    # # create loss
    # if len(CLS_MAPPING) > 2:
    #     criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    # else:
    #     criterion = nn.BCEWithLogitsLoss()

    # # create optimizer and lr scheduler
    # optimizer_ = optim.SGD(
    #     model.parameters(), lr=LEARNING_RATE,
    #     momentum=0.9, weight_decay=0.0005)
    # lr_scheduler_ = optim.lr_scheduler.StepLR(
    #     optimizer_, step_size=20, gamma=0.1)
    # # lr_scheduler_ = optim.lr.scheduler.MultiStepLR(
    # #     optimizer_, milestones=[15, 80], gamma=0.1)
    # # lr_scheduler_ = optim.lr_scheduler.CosineAnnealingLR(
    # #     optimizer_, T_max=EPOCHS*dataset_sizes['train']/BATCH_SIZE,
    # #     verbose=False)
    # # lr_scheduler_ = optim.lr_scheduler.CosineAnnealingWarmRestarts(
    # #     optimizer_, T_0=20, T_mult=1, verbose=False)

    # # Save training details
    # config = dict(
    #     root=DATA_DIR,
    #     train_test_annot=','.join(ANNOT),
    #     cls_mapping=CLS_MAPPING,
    #     model_name=f"{MODEL_NAME}.pth",
    #     model=MODEL,
    #     loss=str(criterion))
    # with open(f"{MODEL_NAME}.yml", "w") as outfile:
    #     yaml.dump(config, outfile, default_flow_style=False)

    # # Train
    # model = train(
    #     dataloaders, model, criterion, optimizer_, lr_scheduler_,
    #     EPOCHS, DEVICE, CLS_MAPPING, MODEL_NAME)

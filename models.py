import os
from datetime import datetime

import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# np.random.seed(123)

import pandas as pd

# import matplotlib.pyplot as plt
# get_ipython().run_line_magic('matplotlib', 'inline')

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.model_selection import KFold
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import log_loss

# from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score


import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras import regularizers

# from keras.layers.advanced_activations import LeakyReLU, PReLU, ELU
from keras.callbacks import ModelCheckpoint
from keras import backend as K

import math
import lightgbm as lgb
import xgboost as xgb


def plot_confusion_matrix(y_val_true, y_val_pred, name=""):
    from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

    y_val_pred = y_val_pred >= 0.5

    cm = confusion_matrix(y_val_true, y_val_pred)
    ConfusionMatrixDisplay(cm, display_labels=np.array(["No Pobre", "Pobre"])).plot(
        cmap="Blues"
    )
    plt.savefig(f"confusion_matrix_{name}")


def lgb_plot_metrics(evals_result):
    print("Plotting metrics recorded during training...")
    ax = lgb.plot_metric(evals_result, metric="f1")
    plt.savefig("views/lgb_metrics.png")


def lgb_plot_importance(model, X, num=20, fig_size=(40, 20)):
    feature_imp = pd.DataFrame(
        {"Value": model.feature_importance(), "Feature": X.columns}
    )
    feature_imp = feature_imp[feature_imp.Value > 0]
    plt.figure(figsize=fig_size)
    sns.set(font_scale=1)
    sns.barplot(
        x="Value",
        y="Feature",
        data=feature_imp.sort_values(by="Value", ascending=False)[0:num],
    )
    plt.title("LightGBM Features (avg over folds)")
    plt.tight_layout()
    plt.savefig("lgbm_importances-01.png")
    plt.show()
    return feature_imp


def linreg_model(data):
    x_tr, y_tr, x_val, y_val, x_test = (
        data["x_tr"],
        data["y_tr"],
        data["x_val"],
        data["y_val"],
        data["x_test"],
    )

    reg = LinearRegression().fit(x_tr, y_tr)
    y_pred_val = reg.predict(x_val)
    y_pred_test = reg.predict(x_test)

    return y_pred_val, y_pred_test


def ridge_model(paras, data):
    x_tr, y_tr, x_val, y_val, x_test = (
        data["x_tr"],
        data["y_tr"],
        data["x_val"],
        data["y_val"],
        data["x_test"],
    )

    model = Ridge(alpha=paras["alpha"])
    reg = model.fit(x_tr, y_tr)
    y_pred_val = reg.predict(x_val)
    y_pred_test = reg.predict(x_test)

    return y_pred_val, y_pred_test


def nn_model(paras, data):
    x_tr, y_tr, x_val, y_val = data["x_tr"], data["y_tr"], data["x_val"], data["y_val"]
    #     y_pred_vals = []
    #     y_pred_tests = []
    # Parameters
    input_nodes = x_tr.shape[1]
    layer_1_nodes = paras["nn_l1"]
    layer_2_nodes = paras["nn_l2"]
    layer_3_nodes = 300
    batch = paras["batch"]
    number_of_epochs = paras["epochs"]
    dropout_rate = paras["dp"]  # + np.random.rand(1)
    if paras["classification"] == True:
        activation = "sigmoid"
        loss = "binary_crossentropy"
        savename = "cla"
    else:
        # Regression
        activation = "linear"
        loss = "mean_squared_error"
        savename = "reg"

    # The input layer and the first hidden layer
    nn_model = Sequential()
    nn_model.add(
        Dense(
            activation="relu",
            input_dim=input_nodes,
            units=layer_1_nodes,
            kernel_initializer="lecun_normal",
            kernel_regularizer=regularizers.l2(0.01),
        )
    )

    # The second hidden layer
    nn_model.add(Dropout(dropout_rate))
    nn_model.add(
        Dense(
            activation="relu",
            input_dim=layer_1_nodes,
            units=layer_2_nodes,
            kernel_initializer="lecun_normal",
            kernel_regularizer=regularizers.l2(0.01),
        )
    )

    # The third hidden layer
    nn_model.add(Dropout(dropout_rate))
    nn_model.add(
        Dense(
            activation="relu",
            input_dim=layer_2_nodes,
            units=layer_3_nodes,
            kernel_initializer="lecun_normal",
            kernel_regularizer=regularizers.l2(0.01),
        )
    )
    nn_model.add(Dropout(dropout_rate))

    # Prediction
    nn_model.add(
        Dense(activation=activation, units=1, kernel_initializer="lecun_normal")
    )

    # Compile the NN
    nn_model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss=loss,
    )
    filepath = f"nn_weights_{savename}.hdf5"
    checkpoint = ModelCheckpoint(
        filepath, monitor="val_loss", verbose=1, save_best_only=True
    )
    callbacks_list = [checkpoint]
    history_ts = nn_model.fit(
        x_tr,
        y_tr,
        batch_size=batch,
        epochs=number_of_epochs,
        validation_data=(x_val, y_val),
        callbacks=callbacks_list,
        verbose=1,
    )

    y_pred_val = nn_model.predict(x_val).ravel()
    y_pred_test = nn_model.predict(data["x_test"].values).ravel()
    return y_pred_val, y_pred_test


def lgb_model(
    paras, data, save_metrics_plot=False, select_features=False, select_features_th=0
):
    x_tr, y_tr, x_val, y_val = data["x_tr"], data["y_tr"], data["x_val"], data["y_val"]

    if paras["classification"] == True:
        loss = "binary_logloss"
    else:
        # Regression
        loss = "l2"

    lgb_params = {
        "task": "train",
        "boosting_type": "gbdt",
        "objective": "binary",
        "metric": loss,
        # "max_depth": paras["max_depth"],
        "is_training_metric": False,
        "learning_rate": paras["lr"],
        "feature_fraction": paras["feature_fraction"],
        "verbose": paras["verbos_"],
    }
    lgb_train = lgb.Dataset(x_tr, y_tr, feature_name=paras["col_names"])
    lgb_val = lgb.Dataset(
        x_val, y_val, feature_name=paras["col_names"], reference=lgb_train
    )
    watchlist = lgb_val
    evals_result = {}

    lgb_model = lgb.train(
        lgb_params,
        lgb_train,
        num_boost_round=20000,
        valid_sets=watchlist,
        callbacks=[
            lgb.log_evaluation(500),
            lgb.record_evaluation(evals_result),
            lgb.early_stopping(
                stopping_rounds=100,
            ),
        ],
        # feval=lgb_f1_score,
    )

    if save_metrics_plot:
        lgb_plot_metrics(evals_result)

    if select_features:
        feature_imp = lgb_plot_importance(lgb_model, x_tr, num=-1, fig_size=(20, 40))
        feature_imp = feature_imp[feature_imp.Value > select_features_th]
        return feature_imp.Feature.to_list()

    y_pred_val = lgb_model.predict(
        x_val, num_iteration=lgb_model.best_iteration
    ).ravel()
    y_pred_test = lgb_model.predict(
        data["x_test"], num_iteration=lgb_model.best_iteration
    ).ravel()
    return y_pred_val, y_pred_test


def xgb_model(paras, data):
    x_tr, y_tr, x_val, y_val = data["x_tr"], data["y_tr"], data["x_val"], data["y_val"]

    xgb_params = {
        "eta": paras["eta"],
        "max_depth": paras["max_depth"],
        "subsample": paras["subsample"],
        "colsample_bytree": paras["colsample_by_tree"],
        "objective": "binary:logistic",
        "eval_metric": "logloss",
        #'base_socre': 0.2,
        "seed": 825,
        "silent": 1,
    }
    dtrain = xgb.DMatrix(x_tr, label=y_tr, feature_names=paras["col_names"])
    dval = xgb.DMatrix(x_val, label=y_val, feature_names=paras["col_names"])
    dtest = xgb.DMatrix(data["x_test"].values, feature_names=paras["col_names"])

    xgb_model = xgb.train(
        xgb_params,
        dtrain,
        num_boost_round=20000,
        evals=[(dtrain, "train"), (dval, "val")],
        early_stopping_rounds=1000,
        verbose_eval=paras["verbos_"],
    )
    y_pred_val = xgb_model.predict(dval)
    y_pred_test = xgb_model.predict(dtest)
    return y_pred_val, y_pred_test


def cart_model(paras, data):
    from sklearn import tree

    # FIXME: add parameters
    x_tr, y_tr, x_val, y_val = data["x_tr"], data["y_tr"], data["x_val"], data["y_val"]

    clf = tree.DecisionTreeClassifier(criterion="log_loss")
    clf = clf.fit(x_tr, y_tr)
    y_pred_val = clf.predict(x_val)
    y_pred_test = clf.predict(data["x_test"])

    return y_pred_val, y_pred_test


def select_relevant_features(X, y):
    """
    Selects relevant features using a LightGBM model on pre-processed data.

    Parameters:
    - data_A (pd.DataFrame): Input DataFrame containing the dataset.

    Returns:
    - list: List of relevant feature columns selected by the LightGBM model.

    The function preprocesses the input data using the pre_process_data function
    with the specified options (type="train", nan="dummies"). It then splits the data
    into training and validation sets using train_test_split.

    The LightGBM model is trained with the following parameters:
    - 'max_depth': None (unlimited depth)
    - 'lr': Learning rate set to 0.01
    - 'feature_fraction': Fraction of features to consider when training each tree set to 0.07
    - 'verbose': -1 (no training output)
    - 'col_names': List of column names from the training set

    The relevant feature columns are selected by the LightGBM model using the lgb_model function
    with select_features=True and a selection threshold of 0. The selected feature columns are then
    returned as a list.

    Example usage:
    >>> data = pd.read_csv('your_dataset.csv')
    >>> relevant_features = select_relevant_features(data)
    """
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=528
    )

    # Define parameters for the model
    params = {
        "max_depth": None,
        "lr": 0.01,
        "feature_fraction": 0.07,
        "verbose": -1,
        "col_names": X_train.columns.tolist(),
    }

    data = {
        "x_tr": X_train,
        "x_val": X_val,
        "y_tr": y_train,
        "y_val": y_val,
        "x_test": None,
    }

    # Train the LightGBM model
    relevant_cols = lgb_model(params, data, select_features=True, select_features_th=0)

    return relevant_cols


def compute_loss_cla(y_true, y_pred, name):
    y_pred = y_pred >= 0.5
    loss = f1_score(y_true, y_pred)  # Cross-entropy
    print("Share of poors: ", y_pred.sum() / y_pred.shape[0])
    print(f"********************************* {name} F1 is: {loss}\n")
    return loss


def compute_loss_reg(y_true, y_pred, pov_line, name):
    print(y_true)
    poors_pred = y_pred <= pov_line
    poors_true = y_true <= pov_line
    loss = f1_score(poors_true, poors_pred)  # Cross-entropy
    print("Share of poors: ", poors_pred.sum() / poors_pred.shape[0])
    print(f"********************************* {name} F1 is: {loss}\n")
    return loss


def train_cla_models(X, y, paras, test_=None):
    paras["lgb"]["col_names"] = X.columns.to_list()
    paras["xgb"]["col_names"] = X.columns.to_list()
    paras["lgb"]["classification"] = True
    paras["nn"]["classification"] = True

    losses = {"cart": [], "lgb": [], "xgb": [], "nn": [], "bag": []}
    preds = {"cart": [], "lgb": [], "xgb": [], "nn": [], "bag": []}

    skf = StratifiedKFold(n_splits=paras["splits"], random_state=123, shuffle=True)
    for i, (train_index, val_index) in enumerate(skf.split(X, y)):
        print("-" * 60)
        print("Iteration: ", i, " Current time: ", str(datetime.now()))
        print("-" * 60)
        x_tr, x_val = X.values[train_index], X.values[val_index]
        y_tr, y_val = y[train_index], y[val_index]
        data = {
            "x_tr": x_tr,
            "x_val": x_val,
            "y_tr": y_tr,
            "y_val": y_val,
            "x_test": test_,
        }

        # CART model
        y_pred_val_cart, y_pred_test_cart = cart_model(paras, data)
        loss_cart = compute_loss_cla(y_val, y_pred_val_cart, name="CART")
        losses["cart"] += [loss_cart]
        preds["cart"] += [y_pred_test_cart]

        # lgb model
        y_pred_val_lgb, y_pred_test_lgb = lgb_model(paras["lgb"], data)
        loss_lgb = compute_loss_cla(y_val, y_pred_val_lgb, name="lgb")
        losses["lgb"] += [loss_lgb]
        preds["lgb"] += [y_pred_test_lgb]
        plot_confusion_matrix(y_val, y_pred_val_lgb, name="lgb")

        # # xgb model
        y_pred_val_xgb, y_pred_test_xgb = xgb_model(paras["xgb"], data)
        loss_xgb = compute_loss_cla(y_val, y_pred_val_xgb, name="xgb")
        losses["xgb"] += [loss_xgb]
        preds["xgb"] += [y_pred_test_xgb]
        plot_confusion_matrix(y_val, y_pred_val_xgb, name="xgb")

        # # neural network model
        y_pred_val_nn, y_pred_test_nn = nn_model(paras["nn"], data)
        loss_nn = compute_loss_cla(y_val, y_pred_val_nn, name="nn")
        losses["nn"] += [loss_nn]
        preds["nn"] += [y_pred_test_nn]

        # # bagging models:
        y_pred_val_bag = (
            y_pred_val_lgb * paras["w_cla_lgb"]
            + y_pred_val_xgb * paras["w_cla_xgb"]
            + y_pred_val_nn * paras["w_cla_nn"]
        )
        y_pred_test_bag = (
            y_pred_test_lgb * paras["w_cla_lgb"]
            + y_pred_test_xgb * paras["w_cla_xgb"]
            + y_pred_test_nn * paras["w_cla_nn"]
        )
        loss_bag = compute_loss_cla(y_val, y_pred_val_bag, name="bag")
        losses["bag"] += [loss_bag]
        preds["bag"] += [y_pred_test_bag]

    m_loss = {k: np.mean(model_loss) for k, model_loss in losses.items()}
    print("mean F1 for models:", m_loss)
    m_predictions = {
        k: np.mean(model_preds, axis=0) >= 0.5 for k, model_preds in preds.items()
    }

    # p_a = np.mean(preds, axis=0)

    return m_loss, m_predictions, preds


def train_reg_models(X, y, paras, test_=None):
    paras["lgb"]["col_names"] = X.columns.to_list()
    paras["lgb"]["classification"] = False
    paras["nn"]["classification"] = False

    losses = {"linreg": [], "lgb": [], "ridge": [], "nn": [], "bag": []}
    preds = {"linreg": [], "lgb": [], "ridge": [], "nn": [], "bag": []}

    skf = KFold(n_splits=paras["splits"], random_state=123, shuffle=True)
    for i, (train_index, val_index) in enumerate(skf.split(X, y)):
        print("-" * 60)
        print("Iteration: ", i, " Current time: ", str(datetime.now()))
        print("-" * 60)

        x_tr, x_val = X.values[train_index], X.values[val_index]
        y_tr, y_val = y.values[train_index, 0], y.values[val_index, 0]
        povline_val = y.values[val_index, 1]

        data = {
            "x_tr": x_tr,
            "x_val": x_val,
            "y_tr": y_tr,
            "y_val": y_val,
            "x_test": test_,
        }

        # Linear regresion model
        y_pred_val_linreg, y_pred_test_linreg = linreg_model(data)
        loss_linreg = compute_loss_reg(
            y_val,
            y_pred_val_linreg,
            povline_val,
            name="linreg",
        )
        losses["linreg"] += [loss_linreg]
        preds["linreg"] += [y_pred_test_linreg]

        # Ridge model
        y_pred_val_ridge, y_pred_test_ridge = ridge_model(paras["ridge"], data)
        loss_ridge = compute_loss_reg(
            y_val, y_pred_val_ridge, povline_val, name="ridge"
        )
        losses["ridge"] += [loss_ridge]
        preds["ridge"] += [y_pred_test_ridge]

        # lgb model
        y_pred_val_lgb, y_pred_test_lgb = lgb_model(paras["lgb"], data)
        loss_lgb = compute_loss_reg(y_val, y_pred_val_lgb, povline_val, name="lgb")
        losses["lgb"] += [loss_lgb]
        preds["lgb"] += [y_pred_test_lgb]

        # neural network model
        y_pred_val_nn, y_pred_test_nn = nn_model(paras["nn"], data)
        loss_nn = compute_loss_reg(y_val, y_pred_val_nn, povline_val, name="nn")
        losses["nn"] += [loss_nn]
        preds["nn"] += [y_pred_test_nn]

        # bagging models:
        y_pred_val_bag = (
            y_pred_val_ridge * paras["w_reg_ridge"]
            + y_pred_val_lgb * paras["w_reg_lgb"]
            + y_pred_val_nn * paras["w_reg_nn"]
        )
        y_pred_test_bag = (
            y_pred_test_ridge * paras["w_reg_ridge"]
            + y_pred_test_lgb * paras["w_reg_lgb"]
            + y_pred_test_nn * paras["w_reg_nn"]
        )
        loss_bag = compute_loss_reg(y_val, y_pred_val_bag, povline_val, name="bag")
        losses["bag"] += [loss_bag]
        preds["bag"] += [y_pred_test_bag]

    m_loss = {k: np.mean(model_loss) for k, model_loss in losses.items()}
    print("mean F1 for models:", m_loss)
    m_predictions = {
        k: np.mean(model_preds, axis=0) >= 0.5 for k, model_preds in preds.items()
    }

    return m_loss, m_predictions, preds

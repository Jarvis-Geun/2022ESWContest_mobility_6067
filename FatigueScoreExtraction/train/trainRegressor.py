import sys
import pandas as pd
import numpy as np
import joblib
import time

import random
import warnings
import os
from IPython.display import display
warnings.filterwarnings('ignore')

from sklearn.model_selection import learning_curve
from sklearn.model_selection import KFold, cross_val_score
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import MinMaxScaler, PolynomialFeatures
from sklearn.linear_model import LinearRegression, ElasticNet, Lasso, Ridge
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor, VotingRegressor
from sklearn.metrics import mean_squared_error, r2_score, make_scorer
import sklearn_evaluation

def set_seed(seed):
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    return

def LearningCurve(x_train, y_train, model):
    train_size, train_score, test_score = learning_curve(model, x_train, y_train, cv = 10)
    sklearn_evaluation.plot.learning_curve(train_score, test_score, train_size)
    return

def RegressorTraining(x_train, y_train):

    '''
    Make Pipeline for preprocessing & Regression.
    GridSearchCV for Hyper parameter tuning.

    preprocessing : MinMaxScaling, PolynomialfeatureScailing.
    regressor : LinearRegression, Lasso, Ridge, ElasticNet, SVR, RandomForestRegression.

    best_param grid model is saved after data training(pkl format)

    :param x_train: numpy array of feature data.
    :param y_train: numpy array of label data.
    :return: None
    '''

    global BestGridCsv, BestGridModel
    mse = make_scorer(mean_squared_error, greater_is_better=False)
    pipe = Pipeline([('preprocessing', MinMaxScaler()),('regressor', LinearRegression())])
    params = [
        {
            'preprocessing': [None, MinMaxScaler(), PolynomialFeatures()],
            'regressor' : [LinearRegression()],
        },

        {
            'preprocessing': [None,MinMaxScaler()],
            'regressor' : [Lasso()],
            'regressor__alpha' : [0.001, 0.01, 0.1, 0.3, 0.7, 1.0]
        },

        {
            'preprocessing': [None,MinMaxScaler()],
            'regressor': [Ridge()],
            'regressor__alpha': [0.001, 0.01, 0.1, 0.3, 0.7, 1.0]
        },

        {
            'preprocessing': [None,MinMaxScaler(), PolynomialFeatures()],
            'regressor': [ElasticNet()],
            'regressor__alpha': [0.001, 0.01, 0.1, 0.3, 0.7, 1.0],
            'regressor__l1_ratio' : [0.001, 0.01, 0.1, 0.3, 0.7, 1.0]
        },

        {
            'preprocessing': [None, MinMaxScaler()],
            'regressor': [SVR()],
            'regressor__kernel' : ['linear', 'rbf', 'poly', 'sigmoid'],
            'regressor__degree' : [2, 3, 4, 5],
            'regressor__C' : [0.01, 0.1, 1.0]
        },

        {
            'preprocessing' : [MinMaxScaler()],
            'regressor' : [RandomForestRegressor()],
            'regressor__n_estimators' : [50, 100, 200, 300],
            'regressor__max_depth' : [5, 7, 9],
            'regressor__min_samples_leaf' : [1, 2, 4],
            'regressor__min_samples_split' : [2, 5, 10]
        }
    ]

    # pipe2 = Pipeline([('regressor', LinearRegression())])
    # params2 = [
    #     {
    #         'regressor': [LinearRegression()],
    #     },
    #
    #     {
    #         'regressor': [Lasso()],
    #         'regressor__alpha': [0.001, 0.01, 0.1, 0.3, 0.7, 1.0]
    #     },
    #
    #     {
    #         'regressor': [Ridge()],
    #         'regressor__alpha': [0.001, 0.01, 0.1, 0.3, 0.7, 1.0]
    #     },
    #
    #     {
    #         'regressor': [ElasticNet()],
    #         'regressor__alpha': [0.001, 0.01, 0.1, 0.3, 0.7, 1.0],
    #         'regressor__l1_ratio': [0.001, 0.01, 0.1, 0.3, 0.7, 1.0]
    #     },
    #
    #     {
    #         'regressor': [SVR()],
    #         'regressor__kernel': ['linear', 'rbf', 'poly', 'sigmoid'],
    #         'regressor__degree': [2, 3, 4, 5],
    #         'regressor__C': [0.01, 0.1, 1.0]
    #     }
    #
    #     {
    #         'regressor' : [RandomForestRegressor()],
    #         'regressor__n_estimators' : [50, 100, 200, 300],
    #         'regressor__max_depth' : [5, 7, 9],
    #         'regressor__min_samples_leaf' : [1, 2, 4],
    #         'regressor__min_samples_split' : [2, 5, 10]
    #     }
    # ]

    grid = GridSearchCV(estimator = pipe,
                        param_grid = params,
                        scoring = mse,
                        n_jobs = 2,
                        cv = 7,
                        verbose = 3)

    grid.fit(x_train, y_train)
    grid.set_params
    print("grid best parameter -> {}".format(grid.best_params_))
    print("grid best index     -> {}".format(grid.best_index_))
    print("grid best MSE     -> {}".format(-grid.best_score_))

    if not os.path.exists("../models"):
        os.mkdir("../models")

    joblib.dump(grid, BestGridModel)
    pd.DataFrame(grid.cv_results_).T.to_csv(BestGridCsv)
    return

if __name__ == "__main__":

    seed = 42
    set_seed(seed)
    real = True # Using real dataset or not
    BestGridCsv = "../models/Preprocessing_RandomForestGridSearchCvResult.csv"
    BestGridModel = "../models/Preprocessing_RandomForestBestGrid.pkl"

    # Read csv file & Concatenate DataFrame
    if real:
        try:
            # RppgFeatureCsv = pd.read_csv("./data/RppgFeature.csv")
            # FacialFeatureCsv = pd.read_csv("./data/FacialFeature.csv")
            x_train = pd.read_csv("./data/FacialFeature.csv").values
            y_train = pd.read_csv("./data/Label.csv").values.reshape(-1, )
        except FileNotFoundError:
            csv_list = [i for i in os.listdir("./data") if ".csv" in i]
            print(" === csv file list ===")
            for i in csv_list:
                print(i)
            sys.exit()
        # x_train = np.concatenate((RppgFeatureCsv.values, FacialFeatureCsv.values), axis = 1)

    else:
        from sklearn.datasets import load_boston
        Dataset = load_boston()
        x_train = Dataset['data']
        y_train = Dataset['target']

    start = time.time()
    RegressorTraining(x_train, y_train)
    print("Training Time: {:.4f}\n".format(time.time() - start))

    print(" ======= GridSearch Result ======= ")
    display(pd.read_csv(BestGridCsv))
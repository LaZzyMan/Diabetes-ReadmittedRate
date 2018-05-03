import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.cross_validation import train_test_split
from xgboost.sklearn import XGBClassifier
from sklearn import metrics
from sklearn.grid_search import GridSearchCV
import matplotlib.pylab as plt
import time
from pylab import mpl
import csv

mpl.rcParams['font.sans-serif'] = ['FangSong']
mpl.rcParams['axes.unicode_minus'] = False


def optimize_n_estimators(data):
    train_x = data.drop(['readmitted'], axis=1)
    train_y = data.pop('readmitted')
    xgb1 = XGBClassifier(
        learning_rate=0.1,
        n_estimators=1000,
        max_depth=5,
        min_child_weight=5,
        gamma=0.0,
        subsample=0.7,
        colsample_bytree=0.7,
        objective='binary:logistic',
        scale_pos_weight=1,
        seed=1000,
        max_delta_step=1
    )
    return model_fit(xgb1, train_x, train_y)


def optimize_wieght_depth(data):
    train_X = data.drop(['readmitted'], axis=1)
    train_Y = data.readmitted
    param_test = {
        'max_depth': list(range(3, 10, 2)),
        'min_child_weight': list(range(1, 6, 2))
    }
    gsearch = GridSearchCV(estimator=XGBClassifier(learning_rate=0.1, n_estimators=182, max_depth=5,
                                                   min_child_weight=3, gamma=0, subsample=0.9, colsample_bytree=0.7,
                                                   objective='binary:logistic', scale_pos_weight=9,
                                                   seed=27, silent=False, max_delta_step=1, reg_alpha=225, reg_lambda=1),
                           param_grid=param_test, scoring='roc_auc', n_jobs=4, iid=False, cv=5)
    gsearch.fit(train_X, train_Y)
    print(gsearch.grid_scores_)
    print(gsearch.best_params_)
    print(gsearch.best_score_)


def optimize_gamma(data):
    train_X = data.drop(['readmitted'], axis=1)
    train_Y = data.readmitted
    param_test = {
        'gamma': [i / 10.0 for i in range(0, 5)]
    }
    gsearch = GridSearchCV(estimator=XGBClassifier(learning_rate=0.1, n_estimators=182, max_depth=5,
                                                   min_child_weight=5, gamma=0, subsample=0.8, colsample_bytree=0.8,
                                                   objective='binary:logistic', scale_pos_weight=50,
                                                   seed=27, silent=False),
                           param_grid=param_test, scoring='roc_auc', n_jobs=4, iid=False, cv=5)
    gsearch.fit(train_X, train_Y)
    print(gsearch.grid_scores_)
    print(gsearch.best_params_)
    print(gsearch.best_score_)


def optimize_sample(data):
    train_X = data.drop(['readmitted'], axis=1)
    train_Y = data.readmitted
    param_test = {
        'subsample': [i / 10.0 for i in range(6, 10)],
        'colsample_bytree': [i / 10.0 for i in range(6, 10)]
    }
    gsearch = GridSearchCV(estimator=XGBClassifier(learning_rate=0.1, n_estimators=182, max_depth=5,
                                                   min_child_weight=5, gamma=0, subsample=0.8, colsample_bytree=0.8,
                                                   objective='binary:logistic', scale_pos_weight=50,
                                                   seed=27, silent=False),
                           param_grid=param_test, scoring='roc_auc', n_jobs=4, iid=False, cv=5)
    gsearch.fit(train_X, train_Y)
    print(gsearch.grid_scores_, gsearch.best_params_, gsearch.best_score_)


def optimize_alpha(data):
    train_X = data.drop(['readmitted'], axis=1)
    train_Y = data.next_step
    param_test = {
        'reg_alpha': [200, 220, 225, 240, 250]
    }
    gsearch = GridSearchCV(estimator=XGBClassifier(learning_rate=0.1, n_estimators=114, max_depth=3,
                                                   min_child_weight=3, gamma=0, subsample=0.9, colsample_bytree=0.7,
                                                   objective='binary:logistic', scale_pos_weight=50,
                                                   seed=27, silent=False),
                           param_grid=param_test, scoring='roc_auc', n_jobs=4, iid=False, cv=5)
    gsearch.fit(train_X, train_Y)
    print(gsearch.grid_scores_)
    print(gsearch.best_params_)
    print(gsearch.best_score_)


def optimize_lambda(data):
    train_X = data.drop(['readmitted'], axis=1)
    train_Y = data.next_step
    param_test = {
        'max_delta_step': [0, 1, 2, 3, 4, 5]
    }
    gsearch = GridSearchCV(estimator=XGBClassifier(learning_rate=0.1, n_estimators=114, max_depth=3,
                                                   min_child_weight=3, gamma=0, subsample=0.9, colsample_bytree=0.7,
                                                   objective='binary:logistic', scale_pos_weight=9,
                                                   seed=27, silent=False, reg_alpha=225, reg_lambda=1),
                           param_grid=param_test, scoring='roc_auc', n_jobs=4, iid=False, cv=5)
    gsearch.fit(train_X, train_Y)
    print(gsearch.grid_scores_)
    print(gsearch.best_params_)
    print(gsearch.best_score_)


def model_fit(alg, train_x, train_y, cv_folds=5, early_stopping_rounds=100):
    xgb_param = alg.get_xgb_params()
    xgb_train = xgb.DMatrix(train_x, label=train_y)
    cv_result = xgb.cv(xgb_param, xgb_train, num_boost_round=alg.get_params()['n_estimators'], nfold=cv_folds,
                      metrics='auc', early_stopping_rounds=early_stopping_rounds, verbose_eval=True)

    alg.set_params(n_estimators=cv_result.shape[0])
    alg.fit(train_x, train_y, eval_metric='auc')
    train_predictions = alg.predict(train_x)
    train_prob = alg.predict_proba(train_x)[:, 1]
    print("\nModel Report")
    print("Accuracy : %.4g" % metrics.accuracy_score(train_y, train_predictions))
    print("AUC Score (Train): %f" % metrics.roc_auc_score(train_y, train_prob))
    alg.get_booster().save_model('./data/XGBoost.model')
    return {'train_prediction': train_predictions, 'train_prob': train_prob}


def output(model_name, f_map):
    bst = xgb.Booster()
    bst.load_model(model_name)

    with open('./data/feature_map.txt', 'w', encoding='utf-8', newline='') as f:
        i = 0
        for feature in f_map:
            f.write('%d\t%s\ti\n' % (i, feature))
            i += 1
        f.close()

    score_weight = bst.get_fscore(fmap='./data/feature_map.txt')
    score_gain = bst.get_score(importance_type='gain', fmap='./data/feature_map.txt')
    with open('./data/feature_result.csv', 'w', encoding='gb2312', newline='') as w:
        writer = csv.writer(w)
        writer.writerow(['f_name', 'score_weight', 'score_gain'])
        for f_name, weight in score_weight.items():
            writer.writerow([f_name, weight, score_gain[f_name]])
        w.close()


if __name__ == '__main__':
    train_data = pd.read_csv('./data/re/com_data.csv', encoding='gb2312')
    f_map = train_data.drop('readmitted', axis=1).columns.tolist()
    optimize_n_estimators(train_data)
    # optimize_wieght_depth(train_data)
    # optimize_sample(train_data)
    # optimize_alpha(train_data)
    # optimize_lambda(train_data)
    # optimize_gamma(train_data)
    output('./data/XGBoost.model', f_map)

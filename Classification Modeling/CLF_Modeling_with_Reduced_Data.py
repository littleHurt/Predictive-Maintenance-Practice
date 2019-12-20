

# Loading necessary module

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
plt.style.use('ggplot')
%matplotlib inline 

from sklearn import metrics
from sklearn import model_selection

from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier



# -----------------------------------------------------------------------------
# load the previous data
df_train_lb = pd.read_csv(r'C:\Users\Desktop\df_train_lb.csv')
df_test_lb  = pd.read_csv(r'C:\Users\Desktop\df_test_lb.csv')


# a variable to hold the set of features to experiment with
#features_org = ['st1','st2','st3',\
#                's1','s2','s3','s4','s5','s6','s7','s8','s9','s10',\
#                's11','s12','s13','s14','s15','s16','s17','s18','s19','s20','s21']

features_rd = [ 'st1','st2',\
                            's3', 's4',           's7',       's9',\
                's11','s12',      's14','s15',     's17',           's20','s21']

#features_plus = [ 'st1', 'st2','st3',\
#                   's1',  's2', 's3', 's4', 's5', 's6', 's7', 's8', 's9','s10',\
#                  's11', 's12','s13','s14','s15','s16','s17','s18','s19','s20','s21',\
#                  
#                 'avt1','avt2','avt3',\
#                  'av1', 'av2', 'av3', 'av4', 'av5', 'av6', 'av7', 'av8', 'av9','av10',\
#                 'av11','av12','av13','av14','av15','av16','av17','av18','av19','av20','av21',\
#                 
#                 'sdt1','sdt2','sdt3',\
#                  'sd1', 'sd2', 'sd3', 'sd4', 'sd5', 'sd6', 'sd7', 'sd8', 'sd9','sd10',\
#                 'sd11','sd12','sd13','sd14','sd15','sd16','sd17','sd18','sd19','sd20','sd21']

features_used = features_rd # (or features_rd, features_plus) which is used in following ML model.


X_train = df_train_lb[features_used]
y_train = df_train_lb['label_bc']

X_test = df_test_lb[features_used]
y_test = df_test_lb['label_bc']




# ----------------------------------------------------------------------------- 
# Logistic Regression R

model_name = 'Logistic Regression(R)'
# set the fixed parameters and parameters for grid-searching
clf_lgrR = LogisticRegression(random_state = 51)
gs_params = {'C': [0.01, 0.1, 1.0, 10]} 

# tuning model for 'Logistic Regression R'
clf_lgrR, pred_lgrR = classification_tuning(model_name, clf_lgrR, features_used, params = gs_params)

# print the best parameters setting of 'Logistic Regression R'
print('\nBest Parameters:\n', clf_lgrR)


# show brief performance metrics of 'Logistic Regression R'
brief_clf_metrics(model_name, y_test, pred_lgrR.y_pred, pred_lgrR.y_score)

# reform performance metrics of 'Logistic Regression R'
ref_metrics_lgrR = reform_clf_metrics(model_name, y_test, pred_lgrR.y_pred, pred_lgrR.y_score)



# use best parameters which searched from "classification_tuning()" to plot metrics
clf_lgrR = LogisticRegression(C = 0.1, random_state = 51)
# plot the "ROC", "Precision-Recall" and "Calibration" Curves of 'Logistic Regression R'
plot_metrics(clf_lgrR, X_train, y_train, X_test, y_test)



# ----------------------------------------------------------------------------- 
# kNN Classifiction R

model_name = 'k-NN(R)'

# set the fixed parameters and parameters for grid-searching
clf_knnR = KNeighborsClassifier(n_jobs = -1)
gs_params = {'n_neighbors': [3, 4, 5, 6, 7, 8, 9, 10, 11, 12]}

# tuning model for 'k-NN O'
clf_knnR, pred_knnR = classification_tuning( model_name, clf_knnR, features_used, params = gs_params)

# print the best parameters setting of 'k-NN R'
print('\nBest Parameters:\n', clf_knnR)

# show brief performance metrics of 'k-NN R'
brief_clf_metrics(model_name, y_test, pred_knnR.y_pred, pred_knnR.y_score)

# reform performance metrics of 'k-NN R'
ref_metrics_knnR = reform_clf_metrics(model_name, y_test, pred_knnR.y_pred, pred_knnR.y_score)



# use best parameters which searched from "classification_tuning()" to plot metrics
clf_knnR = KNeighborsClassifier(n_neighbors = 11, n_jobs = -1)
# plot the "ROC", "Precision-Recall" and "Calibration" Curves of 'k-NN R'
plot_metrics(clf_knnR, X_train, y_train, X_test, y_test)



# ---------------------------------------------------------------------------- 
# SVM Classifiction R

model_name = 'SVM(R)'

# set the fixed parameters and parameters for grid-searching
clf_svcR = SVC(random_state = 51, probability = True)
gs_params = {'C': [0.01, 0.1, 1.0]} # best [1.0]

# tuning model for 'SVM Classifiction R'
clf_svcR, pred_svcR = classification_tuning(\
                    model_name, clf_svcR, features_used, params = gs_params\
                    )

# print the best parameters setting of 'SVM Classifiction R'
print('\nBest Parameters:\n',clf_svcR)

# show brief performance metrics of 'SVM Classifiction R'
brief_clf_metrics(model_name, y_test, pred_svcR.y_pred, pred_svcR.y_score)

# reform performance metrics of 'Logistic Regression R'
ref_metrics_svcR = reform_clf_metrics(model_name, y_test, pred_svcR.y_pred, pred_svcR.y_score)



# use best parameters which searched from "classification_tuning()" to plot metrics
clf_svcR = SVC(C = 1, random_state = 51, probability = True)
# plot the "ROC", "Precision-Recall" and "Calibration" Curves of 'SVM Classifiction R'
plot_metrics(clf_svcR, X_train, y_train, X_test, y_test)



# ----------------------------------------------------------------------------- 
# Gaussian Naive Bayesian R
model_name = 'Gaussian Naive Bayesian(R)'

# set the fixed parameters and parameters for grid-searching
clf_gnbR = GaussianNB()
gs_params = {} 

# tuning model for 'Gaussian Naive Bayesian R'
clf_gnbR, pred_gnbR = classification_tuning(model_name, clf_gnbR, features_used, params = gs_params)

# print the best parameters setting of 'Gaussian Naive Bayesian R'
print('\nBest Parameters:\n', clf_gnbR)

# show brief performance metrics of 'Gaussian Naive Bayesian R'
brief_clf_metrics(model_name, y_test, pred_gnbR.y_pred, pred_gnbR.y_score)

# reform performance metrics of 'Gaussian Naive Bayesian R'
ref_metrics_gnbR = reform_clf_metrics(model_name, y_test, pred_gnbR.y_pred, pred_gnbR.y_score)


# use best parameters which searched from "classification_tuning()" to plot metrics
# plot the "ROC", "Precision-Recall" and "Calibration" Curves of 'Gaussian Naive Bayesian R'
plot_metrics(clf_gnbR, X_train, y_train, X_test, y_test)



# ----------------------------------------------------------------------------- 
# Decision Tree R

model_name = 'Decision Tree(R)'

# set the fixed parameters and parameters for grid-searching
clf_dtrR = DecisionTreeClassifier(random_state = 51)
gs_params =  {'max_depth': [3, 4, 5, 6, 7, 8, 9],\
              'criterion': ['gini', 'entropy'],\
              'min_samples_split': [2, 3, 4, 5, 6],\
              'max_features': [3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]}

# tuning model for 'Decision Tree R'
clf_dtrR, pred_dtrR = classification_tuning(model_name, clf_dtrR, features_used, params = gs_params)

# print the best parameters setting of 'Decision Tree R'
print('\nBest Parameters:\n', clf_dtrR)

# show brief performance metrics of 'Decision Tree R'
brief_clf_metrics(model_name, y_test, pred_dtrR.y_pred, pred_dtrR.y_score)

# reform performance metrics of 'Decision Tree R'
ref_metrics_dtrR = reform_clf_metrics(model_name, y_test, pred_dtrR.y_pred, pred_dtrR.y_score)




# use best parameters which searched from "classification_tuning()" to plot metrics
clf_dtrR = DecisionTreeClassifier(criterion='entropy', max_depth=4, max_features=13, min_samples_split=2,random_state = 51)
# plot the "ROC", "Precision-Recall" and "Calibration" Curves of 'Decision Tree R'
plot_metrics(clf_dtrR, X_train, y_train, X_test, y_test)

# ----------------------------------------------------------------------------- 
# Random Forest R

model_name = 'Random Forest(R)'

# set the fixed parameters and parameters for grid-searching
clf_rfcR = RandomForestClassifier(random_state = 51)
gs_params = {'n_estimators': [36, 49, 64, 81, 99],\
             'max_depth': [3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13],\
             'criterion': ['gini', 'entropy']}

# tuning model for 'Random Forest R'
clf_rfcR, pred_rfcR = classification_tuning(model_name, clf_rfcR, features_used, params = gs_params)

# print the best parameters setting of 'Random Forest R'
print('\nBest Parameters:\n', clf_rfcR)

# show brief performance metrics of Random Forest R'
brief_clf_metrics(model_name, y_test, pred_rfcR.y_pred, pred_rfcR.y_score)

# reform performance metrics of 'Random Forest R'
ref_metrics_rfcR = reform_clf_metrics(model_name, y_test, pred_rfcR.y_pred, pred_rfcR.y_score)



# use best parameters which searched from "classification_tuning()" to plot metrics
clf_rfcR = RandomForestClassifier(criterion='gini', max_depth=11, n_estimators = 99, random_state = 51)
# plot the "ROC", "Precision-Recall" and "Calibration" Curves of 'Random Forest R'
plot_metrics(clf_rfcR, X_train, y_train, X_test, y_test)



# ----------------------------------------------------------------------------- 
# Neural Network Classifiction R

model_name = 'Neural Network(R)'

# set the fixed parameters and parameters for grid-searching
clf_mlpR = MLPClassifier(learning_rate= 'adaptive', max_iter = 365, random_state = 51)
gs_params = {'hidden_layer_sizes': [(64,), (72,), (81,), (99,)],\
             'alpha': [0.0001, 0.001, 0.01, 0.1],\
             'learning_rate_init': [0.0001, 0.001, 0.01, 0.1]
             } 

# tuning model for 'Neural Network Classifiction '
clf_mlpR, pred_mlpR = classification_tuning(\
                    model_name, clf_mlpR, features_used, params = gs_params)

# print the best parameters setting of 'Neural Network Classifiction R'
print('\nBest Parameters:\n', clf_mlpR)

# show brief performance metrics of 'Neural Network Classifiction R'
brief_clf_metrics(model_name, y_test, pred_mlpR.y_pred, pred_mlpR.y_score)

# reform performance metrics of 'Neural Network Classifiction R'
ref_metrics_mlpR = reform_clf_metrics(model_name, y_test, pred_mlpR.y_pred, pred_mlpR.y_score)



# use best parameters which searched from "classification_tuning()" to plot metrics
clf_mlpR = MLPClassifier(learning_rate= 'adaptive', max_iter = 365, random_state = 51, hidden_layer_sizes=(81,), alpha = 0.0001, learning_rate_init = 0.001)
# plot the "ROC", "Precision-Recall" and "Calibration" Curves of 'Neural Network Classifiction R'
plot_metrics(clf_mlpR, X_train, y_train, X_test, y_test)



#------------------------------------------------------------------------------
# XGBoost Classifiction R

model_name = 'XGBoost(R)'

# set the fixed parameters and parameters for grid-searching
clf_xgbR = XGBClassifier(random_state = 51)
gs_params = {'max_depth': [3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13],\
             'learning_rate': [0.0001, 0.001, 0.01, 0.1],\
             'colsample_bytree': [0.5, 1.0]\
             } 

# tuning model for 'XGBoost Classifiction R'
clf_xgbR, pred_xgbR = classification_tuning(model_name, clf_xgbR, features_used, params = gs_params)

# print the best parameters setting of 'XGBoost Classifiction R'
print('\nBest Parameters:\n', clf_xgbR)

# show brief performance metrics of 'XGBoost Classifiction R'
brief_clf_metrics(model_name, y_test, pred_xgbR.y_pred, pred_xgbR.y_score)

# reform performance metrics of 'XGBoost Classifiction R'
ref_metrics_xgbR = reform_clf_metrics(model_name, y_test, pred_xgbR.y_pred, pred_xgbR.y_score)



# use best parameters which searched from "classification_tuning()" to plot metrics
clf_xgbR = XGBClassifier(colsample_bytree = 0.5, max_depth = 3, learning_rate = 0.1, random_state = 51)
# plot the "ROC", "Precision-Recall" and "Calibration" Curves of 'XGBoost Classifiction R'
plot_metrics(clf_xgbR, X_train, y_train, X_test, y_test)



# ----------------------------------------------------------------------------- 
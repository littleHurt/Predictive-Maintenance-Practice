

# Loading necessary modules

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
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier



# -----------------------------------------------------------------------------
# load the previous data
df_train_lb = pd.read_csv(r'C:\Users\Desktop\df_train_lb.csv')
df_test_lb  = pd.read_csv(r'C:\Users\Desktop\df_test_lb.csv')


# a variable to hold the set of features to experiment with
features_org = ['st1','st2','st3',\
                's1','s2','s3','s4','s5','s6','s7','s8','s9','s10',\
                's11','s12','s13','s14','s15','s16','s17','s18','s19','s20','s21']

#features_rd = [ 'st1','st2',\
#                            's3', 's4',           's7',       's9',\
#                's11','s12',      's14','s15',     's17',           's20','s21']
#
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

features_used = features_org # (or features_rd, features_plus) which is used in following ML model.

X_train = df_train_lb[features_used]
y_train = df_train_lb['label_bc']

X_test = df_test_lb[features_used]
y_test = df_test_lb['label_bc']




# ----------------------------------------------------------------------------- 
# Logistic Regression O

model_name = 'Logistic Regression(O)'
# set the fixed parameters and parameters for grid-searching
clf_lgrO = LogisticRegression(random_state = 51)
gs_params = {'C': [0.01, 0.1, 1.0, 10]} # Best "C" = 0.01

# tuning model for 'Logistic Regression O'
clf_lgrO, pred_lgrO = classification_tuning(model_name, clf_lgrO, features_used, params = gs_params)

# print the best parameters setting of 'Logistic Regression O'
print('\nBest Parameters:\n', clf_lgrO)


# show brief performance metrics of 'Logistic Regression O'
brief_clf_metrics(model_name, y_test, pred_lgrO.y_pred, pred_lgrO.y_score)

# reform performance metrics of 'Logistic Regression O'
ref_metrics_lgrO = reform_clf_metrics(model_name, y_test, pred_lgrO.y_pred, pred_lgrO.y_score)



# use best parameters which searched from "classification_tuning()" to plot metrics
clf_lgrO = LogisticRegression(C = 0.01, random_state = 51)
# plot the "ROC", "Precision-Recall" and "Calibration" Curves of 'Logistic Regression O'
plot_metrics(clf_lgrO, X_train, y_train, X_test, y_test)



# ----------------------------------------------------------------------------- 
# kNN Classifiction O

model_name = 'k-NN(O)'

# set the fixed parameters and parameters for grid-searching
clf_knnO = KNeighborsClassifier(n_jobs = -1)
gs_params = {'n_neighbors': [3, 4, 5, 6, 7, 8, 9, 10, 11, 12]}

# tuning model for 'k-NN O'
clf_knnO, pred_knnO = classification_tuning( model_name, clf_knnO, features_used, params = gs_params)

# print the best parameters setting of 'k-NN O'
print('\nBest Parameters:\n', clf_knnO)

# show brief performance metrics of 'k-NN O'
brief_clf_metrics(model_name, y_test, pred_knnO.y_pred, pred_knnO.y_score)

# reform performance metrics of 'k-NN O'
ref_metrics_knnO = reform_clf_metrics(model_name, y_test, pred_knnO.y_pred, pred_knnO.y_score)



# use best parameters which searched from "classification_tuning()" to plot metrics
clf_knnO = KNeighborsClassifier(n_neighbors = 7, n_jobs = -1)
# plot the "ROC", "Precision-Recall" and "Calibration" Curves of 'k-NN O'
plot_metrics(clf_knnO, X_train, y_train, X_test, y_test)



# ---------------------------------------------------------------------------- 
# SVM Classifiction O

model_name = 'SVM(O)'

# set the fixed parameters and parameters for grid-searching
clf_svcO = SVC(random_state = 51, probability = True)
gs_params = {'C': [0.01, 0.1, 1.0]}

# tuning model for 'SVM Classifiction O'
clf_svcO, pred_svcO = classification_tuning(\
                    model_name, clf_svcO, features_used, params = gs_params\
                    )

# print the best parameters setting of 'SVM Classifiction O'
print('\nBest Parameters:\n',clf_svcO)

# show brief performance metrics of 'SVM Classifiction O'
brief_clf_metrics(model_name, y_test, pred_svcO.y_pred, pred_svcO.y_score)

# reform performance metrics of 'Logistic Regression O'
ref_metrics_svcO = reform_clf_metrics(model_name, y_test, pred_svcO.y_pred, pred_svcO.y_score)



# use best parameters which searched from "classification_tuning()" to plot metrics
clf_svcO = SVC(C = 1, random_state = 51, probability = True)
# plot the "ROC", "Precision-Recall" and "Calibration" Curves of 'SVM Classifiction O'
plot_metrics(clf_svcO, X_train, y_train, X_test, y_test)



# ----------------------------------------------------------------------------- 
# Gaussian Naive Bayesian
model_name = 'Gaussian Naive Bayesian(O)'

# set the fixed parameters and parameters for grid-searching
clf_gnbO = GaussianNB()
gs_params = {} 

# tuning model for 'Gaussian Naive Bayesian O'
clf_gnbO, pred_gnbO = classification_tuning(model_name, clf_gnbO, features_used, params = gs_params)

# print the best parameters setting of 'Gaussian Naive Bayesian O'
print('\nBest Parameters:\n', clf_gnbO)

# show brief performance metrics of 'Gaussian Naive Bayesian O'
brief_clf_metrics(model_name, y_test, pred_gnbO.y_pred, pred_gnbO.y_score)

# reform performance metrics of 'Gaussian Naive Bayesian O'
ref_metrics_gnbO = reform_clf_metrics(model_name, y_test, pred_gnbO.y_pred, pred_gnbO.y_score)



# plot the "ROC", "Precision-Recall" and "Calibration" Curves of 'Gaussian Naive Bayesian O'
plot_metrics(clf_gnbO, X_train, y_train, X_test, y_test)



# ----------------------------------------------------------------------------- 
# Decision Tree O

model_name = 'Decision Tree(O)'

# set the fixed parameters and parameters for grid-searching
clf_dtrO = DecisionTreeClassifier(random_state = 51)
gs_params =  {'max_depth': [3, 4, 5, 6, 7, 8, 9],\
              'criterion': ['gini', 'entropy'],\
              'min_samples_split': [2, 3, 4, 5, 6],\
              'max_features': [3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]}

# tuning model for 'Decision Tree O'
clf_dtrO, pred_dtrO = classification_tuning(model_name, clf_dtrO, features_used, params = gs_params)

# print the best parameters setting of 'Decision Tree O'
print('\nBest Parameters:\n', clf_dtrO)

# show brief performance metrics of 'Decision Tree O'
brief_clf_metrics(model_name, y_test, pred_dtrO.y_pred, pred_dtrO.y_score)

# reform performance metrics of 'Decision Tree O'
ref_metrics_dtrO = reform_clf_metrics(model_name, y_test, pred_dtrO.y_pred, pred_dtrO.y_score)



# use best parameters which searched from "classification_tuning()" to plot metrics
clf_dtrO = DecisionTreeClassifier(criterion='entropy', max_depth=4, max_features=9, min_samples_split=2,random_state = 51)
# plot the "ROC", "Precision-Recall" and "Calibration" Curves of 'Decision Tree O'
plot_metrics(clf_dtrO, X_train, y_train, X_test, y_test)




# ----------------------------------------------------------------------------- 
# Random Forest O

model_name = 'Random Forest(O)'

# set the fixed parameters and parameters for grid-searching
clf_rfcO = RandomForestClassifier(random_state = 51)
gs_params = {'n_estimators': [36, 49, 64, 81, 99],\
             'max_depth': [3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13],\
             'criterion': ['gini', 'entropy']}

# tuning model for 'Random Forest O'
clf_rfcO, pred_rfcO = classification_tuning(model_name, clf_rfcO, features_used, params = gs_params)

# print the best parameters setting of 'Random Forest O'
print('\nBest Parameters:\n', clf_rfcO)

# show brief performance metrics of Random Forest O'
brief_clf_metrics(model_name, y_test, pred_rfcO.y_pred, pred_rfcO.y_score)

# reform performance metrics of 'Random Forest O'
ref_metrics_rfcO = reform_clf_metrics(model_name, y_test, pred_rfcO.y_pred, pred_rfcO.y_score)



# use best parameters which searched from "classification_tuning()" to plot metrics
clf_rfcO = RandomForestClassifier(criterion = 'gini', max_depth = 12, n_estimators = 81, random_state = 51)
# plot the "ROC", "Precision-Recall" and "Calibration" Curves of 'Random Forest O'
plot_metrics(clf_rfcO, X_train, y_train, X_test, y_test)



# ----------------------------------------------------------------------------- 
# Neural Network Classifiction O

model_name = 'Neural Network(O)'

# set the fixed parameters and parameters for grid-searching
clf_mlpO = MLPClassifier(learning_rate= 'adaptive', max_iter = 365, random_state = 51)
gs_params = {'hidden_layer_sizes': [(64,), (72,), (81,), (99,)],\
             'alpha': [0.0001, 0.001, 0.01, 0.1],\
             'learning_rate_init': [0.0001, 0.001, 0.01, 0.1]
             } 

# tuning model for 'Neural Network Classifiction O'
clf_mlpO, pred_mlpO = classification_tuning(\
                    model_name, clf_mlpO, features_used, params = gs_params)

# print the best parameters setting of 'Neural Network Classifiction O'
print('\nBest Parameters:\n', clf_mlpO)

# show brief performance metrics of 'Neural Network Classifiction O'
brief_clf_metrics(model_name, y_test, pred_mlpO.y_pred, pred_mlpO.y_score)

# reform performance metrics of 'Neural Network Classifiction O'
ref_metrics_mlpO = reform_clf_metrics(model_name, y_test, pred_mlpO.y_pred, pred_mlpO.y_score)



# use best parameters which searched from "classification_tuning()" to plot metrics
clf_mlpO = MLPClassifier(learning_rate= 'adaptive', max_iter = 365, random_state = 51, hidden_layer_sizes=(99,), alpha = 0.1, learning_rate_init = 0.0001)
# plot the "ROC", "Precision-Recall" and "Calibration" Curves of 'Neural Network Classifiction O'
plot_metrics(clf_mlpO, X_train, y_train, X_test, y_test)



#------------------------------------------------------------------------------
# XGBoost Classifiction O

model_name = 'XGBoost(O)'

# set the fixed parameters and parameters for grid-searching
clf_xgbO = XGBClassifier(random_state = 51)
gs_params = {'max_depth': [3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13],\
             'learning_rate': [0.0001, 0.001, 0.01, 0.1],\
             'colsample_bytree': [0.5, 1.0]\
             } 

# tuning model for 'XGBoost Classifiction O'
clf_xgbO, pred_xgbO = classification_tuning(model_name, clf_xgbO, features_used, params = gs_params)

# print the best parameters setting of 'XGBoost Classifiction O'
print('\nBest Parameters:\n', clf_xgbO)

# show brief performance metrics of 'XGBoost Classifiction O'
brief_clf_metrics(model_name, y_test, pred_xgbO.y_pred, pred_xgbO.y_score)

# reform performance metrics of 'XGBoost Classifiction O'
ref_metrics_xgbO = reform_clf_metrics(model_name, y_test, pred_xgbO.y_pred, pred_xgbO.y_score)



# use best parameters which searched from "classification_tuning()" to plot metrics
clf_xgbO = XGBClassifier(colsample_bytree=0.5, max_depth=4, learning_rate=0.1, random_state = 51)
# plot the "ROC", "Precision-Recall" and "Calibration" Curves of 'XGBoost Classifiction O'
plot_metrics(clf_xgbO, X_train, y_train, X_test, y_test)


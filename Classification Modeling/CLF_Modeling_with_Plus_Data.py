

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
from sklearn.ensemble import RandomForestClassifier
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
#
#features_rd = [ 'st1','st2',\
#                            's3', 's4',           's7',       's9',\
#                's11','s12',      's14','s15',     's17',           's20','s21']

features_plus = [ 'st1', 'st2','st3',\
                   's1',  's2', 's3', 's4', 's5', 's6', 's7', 's8', 's9','s10',\
                  's11', 's12','s13','s14','s15','s16','s17','s18','s19','s20','s21',\
                  
                 'avt1','avt2','avt3',\
                  'av1', 'av2', 'av3', 'av4', 'av5', 'av6', 'av7', 'av8', 'av9','av10',\
                 'av11','av12','av13','av14','av15','av16','av17','av18','av19','av20','av21',\
                 
                 'sdt1','sdt2','sdt3',\
                  'sd1', 'sd2', 'sd3', 'sd4', 'sd5', 'sd6', 'sd7', 'sd8', 'sd9','sd10',\
                 'sd11','sd12','sd13','sd14','sd15','sd16','sd17','sd18','sd19','sd20','sd21']

    
features_used = features_plus # (or features_rd, features_plus) which is used in following ML model.

X_train = df_train_lb[features_used]
y_train = df_train_lb['label_bc']

X_test = df_test_lb[features_used]
y_test = df_test_lb['label_bc']




# ----------------------------------------------------------------------------- 
# Logistic Regression P

model_name = 'Logistic Regression(P)'
# set the fixed parameters and parameters for grid-searching
clf_lgrP = LogisticRegression(random_state = 51)
gs_params = {'C': [0.01, 0.1, 1.0, 10]} 

# tuning model for 'Logistic Regression P'
clf_lgrP, pred_lgrP = classification_tuning(model_name, clf_lgrP, features_used, params = gs_params)

# print the best parameters setting of 'Logistic Regression P'
print('\nBest Parameters:\n', clf_lgrP)

# show brief performance metrics of 'Logistic Regression P'
brief_clf_metrics(model_name, y_test, pred_lgrP.y_pred, pred_lgrP.y_score)

# reform performance metrics of 'Logistic Regression P'
ref_metrics_lgrP = reform_clf_metrics(model_name, y_test, pred_lgrP.y_pred, pred_lgrP.y_score)



# use best parameters which searched from "classification_tuning()" to plot metrics
clf_lgrP = LogisticRegression(C = 10, random_state = 51)
# plot the "ROC", "Precision-Recall" and "Calibration" Curves of 'Logistic Regression P'
plot_metrics(clf_lgrP, X_train, y_train, X_test, y_test)



# ----------------------------------------------------------------------------- 
# kNN Classifiction P

model_name = 'k-NN(P)'

# set the fixed parameters and parameters for grid-searching
clf_knnP = KNeighborsClassifier(n_jobs = -1)
gs_params = {'n_neighbors': [3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19]}

# tuning model for 'k-NN P'
clf_knnP, pred_knnP = classification_tuning( model_name, clf_knnP, features_used, params = gs_params)

# print the best parameters setting of 'k-NN P'
print('\nBest Parameters:\n', clf_knnP)

# show brief performance metrics of 'k-NN P'
brief_clf_metrics(model_name, y_test, pred_knnP.y_pred, pred_knnP.y_score)

# reform performance metrics of 'k-NN P'
ref_metrics_knnP = reform_clf_metrics(model_name, y_test, pred_knnP.y_pred, pred_knnP.y_score)



# use best parameters which searched from "classification_tuning()" to plot metrics
clf_knnP = KNeighborsClassifier(n_neighbors = 15, n_jobs = -1)
# plot the "ROC", "Precision-Recall" and "Calibration" Curves of 'k-NN P'
plot_metrics(clf_knnP, X_train, y_train, X_test, y_test)



# ---------------------------------------------------------------------------- 
# SVM Classifiction P

model_name = 'SVM(P)'

# set the fixed parameters and parameters for grid-searching
clf_svcP = SVC(random_state = 51, probability = True)
gs_params = {'C': [0.01, 0.1, 1.0]} 

# tuning model for 'SVM Classifiction P'
clf_svcP, pred_svcP = classification_tuning(\
                    model_name, clf_svcP, features_used, params = gs_params\
                    )

# print the best parameters setting of 'SVM Classifiction P'
print('\nBest Parameters:\n',clf_svcP)

# show brief performance metrics of 'SVM Classifiction R'
brief_clf_metrics(model_name, y_test, pred_svcP.y_pred, pred_svcP.y_score)

# reform performance metrics of 'Logistic Regression P'
ref_metrics_svcP = reform_clf_metrics(model_name, y_test, pred_svcP.y_pred, pred_svcP.y_score)



# use best parameters which searched from "classification_tuning()" to plot metrics
clf_svcP = SVC(C = 1, random_state = 51, probability = True)
# plot the "ROC", "Precision-Recall" and "Calibration" Curves of 'SVM Classifiction P'
plot_metrics(clf_svcP, X_train, y_train, X_test, y_test)



# ----------------------------------------------------------------------------- 
# Gaussian Naive Bayesian P
model_name = 'Gaussian Naive Bayesian(P)'

# set the fixed parameters and parameters for grid-searching
clf_gnbP = GaussianNB()
gs_params = {} 

# tuning model for 'Gaussian Naive Bayesian P'
clf_gnbP, pred_gnbP = classification_tuning(model_name, clf_gnbP, features_used, params = gs_params)

# print the best parameters setting of 'Gaussian Naive Bayesian P'
print('\nBest Parameters:\n', clf_gnbP)

# show brief performance metrics of 'Gaussian Naive Bayesian P'
brief_clf_metrics(model_name, y_test, pred_gnbP.y_pred, pred_gnbP.y_score)

# reform performance metrics of 'Gaussian Naive Bayesian P'
ref_metrics_gnbP = reform_clf_metrics(model_name, y_test, pred_gnbP.y_pred, pred_gnbP.y_score)


# use best parameters which searched from "classification_tuning()" to plot metrics
# plot the "ROC", "Precision-Recall" and "Calibration" Curves of 'Gaussian Naive Bayesian P'
plot_metrics(clf_gnbP, X_train, y_train, X_test, y_test)



# ----------------------------------------------------------------------------- 
# Decision Tree P

model_name = 'Decision Tree(P)'

# set the fixed parameters and parameters for grid-searching
clf_dtrP = DecisionTreeClassifier(random_state = 51)
gs_params =  {'max_depth': [3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13],\
              'criterion': ['gini', 'entropy'],\
              'min_samples_split': [2, 3, 4, 5],\
              'max_features': [3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19]}

# tuning model for 'Decision Tree P'
clf_dtrP, pred_dtrP = classification_tuning(model_name, clf_dtrP, features_used, params = gs_params)

# print the best parameters setting of 'Decision Tree P'
print('\nBest Parameters:\n', clf_dtrP)

# show brief performance metrics of 'Decision Tree P'
brief_clf_metrics(model_name, y_test, pred_dtrP.y_pred, pred_dtrP.y_score)

# reform performance metrics of 'Decision Tree P'
ref_metrics_dtrP = reform_clf_metrics(model_name, y_test, pred_dtrP.y_pred, pred_dtrP.y_score)




# use best parameters which searched from "classification_tuning()" to plot metrics
clf_dtrP = DecisionTreeClassifier(criterion = 'gini', max_depth = 3, max_features = 10, min_samples_split=2,random_state = 51)
# plot the "ROC", "Precision-Recall" and "Calibration" Curves of 'Decision Tree P'
plot_metrics(clf_dtrP, X_train, y_train, X_test, y_test)

# ----------------------------------------------------------------------------- 
# Random Forest 

model_name = 'Random Forest(P)'

# set the fixed parameters and parameters for grid-searching
clf_rfcP = RandomForestClassifier(random_state = 51)
gs_params = {'n_estimators': [36, 49, 64, 81, 99, 144],\
             'max_depth': [3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19],\
             'criterion': ['gini', 'entropy']}

# tuning model for 'Random Forest P'
clf_rfcP, pred_rfcP = classification_tuning(model_name, clf_rfcP, features_used, params = gs_params)

# print the best parameters setting of 'Random Forest P'
print('\nBest Parameters:\n', clf_rfcP)

# show brief performance metrics of Random Forest P'
brief_clf_metrics(model_name, y_test, pred_rfcP.y_pred, pred_rfcP.y_score)

# reform performance metrics of 'Random Forest R'
ref_metrics_rfcP = reform_clf_metrics(model_name, y_test, pred_rfcP.y_pred, pred_rfcP.y_score)



# use best parameters which searched from "classification_tuning()" to plot metrics
clf_rfcP = RandomForestClassifier(criterion = 'gini', max_depth = 14, n_estimators = 36, random_state = 51)
# plot the "ROC", "Precision-Recall" and "Calibration" Curves of 'Random Forest P'
plot_metrics(clf_rfcP, X_train, y_train, X_test, y_test)



# ----------------------------------------------------------------------------- 
# Neural Network Classifiction P

model_name = 'Neural Network(P)'

# set the fixed parameters and parameters for grid-searching
clf_mlpP = MLPClassifier(learning_rate= 'adaptive', max_iter = 365, random_state = 51)
gs_params = {'hidden_layer_sizes': [(64,), (72,), (81,), (99,), (144,)],\
             'alpha': [0.0001, 0.001, 0.01, 0.1],\
             'learning_rate_init': [0.0001, 0.001, 0.01, 0.1]
             } 

# tuning model for 'Neural Network Classifiction P'
clf_mlpP, pred_mlpP = classification_tuning(\
                    model_name, clf_mlpP, features_used, params = gs_params)

# print the best parameters setting of 'Neural Network Classifiction P'
print('\nBest Parameters:\n', clf_mlpP)

# show brief performance metrics of 'Neural Network Classifiction P'
brief_clf_metrics(model_name, y_test, pred_mlpP.y_pred, pred_mlpP.y_score)

# reform performance metrics of 'Neural Network Classifiction P'
ref_metrics_mlpP = reform_clf_metrics(model_name, y_test, pred_mlpP.y_pred, pred_mlpP.y_score)



# use best parameters which searched from "classification_tuning()" to plot metrics
clf_mlpP = MLPClassifier(learning_rate= 'adaptive', max_iter = 365, random_state = 51, hidden_layer_sizes=(72,), alpha = 0.01, learning_rate_init = 0.0001)
# plot the "ROC", "Precision-Recall" and "Calibration" Curves of 'Neural Network Classifiction P'
plot_metrics(clf_mlpP, X_train, y_train, X_test, y_test)



#------------------------------------------------------------------------------
# XGBoost Classifiction P

model_name = 'XGBoost(P)'

# set the fixed parameters and parameters for grid-searching
clf_xgbP = XGBClassifier(random_state = 51)
gs_params = {'max_depth': [5, 6, 7, 8, 9, 10, 11, 12],\
             'learning_rate': [0.0001, 0.001, 0.01, 0.1],\
             'colsample_bytree': [0.5, 1.0]\
             } 

# tuning model for 'XGBoost Classifiction P'
clf_xgbP, pred_xgbP = classification_tuning(model_name, clf_xgbP, features_used, params = gs_params)

# print the best parameters setting of 'XGBoost Classifiction P'
print('\nBest Parameters:\n', clf_xgbP)

# show brief performance metrics of 'XGBoost Classifiction P'
brief_clf_metrics(model_name, y_test, pred_xgbP.y_pred, pred_xgbP.y_score)

# reform performance metrics of 'XGBoost Classifiction P'
ref_metrics_xgbP = reform_clf_metrics(model_name, y_test, pred_xgbP.y_pred, pred_xgbP.y_score)



# use best parameters which searched from "classification_tuning()" to plot metrics
clf_xgbP = XGBClassifier(colsample_bytree = 0.5, max_depth = 6, learning_rate = 0.1, random_state = 51)
# plot the "ROC", "Precision-Recall" and "Calibration" Curves of 'XGBoost Classifiction P'
plot_metrics(clf_xgbP, X_train, y_train, X_test, y_test)




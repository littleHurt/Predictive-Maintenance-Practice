

# Loading necessary module
import pandas as pd
import numpy as np
import scikitplot as skplt
import matplotlib.pyplot as plt
plt.style.use('ggplot')
%matplotlib inline  

from sklearn import metrics
from sklearn import model_selection 

from sklearn import linear_model
from sklearn.preprocessing import PolynomialFeatures
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor
from sklearn.neural_network import MLPRegressor



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

features_used = features_rd # (or features_rd, features_plus) which is used in following ML model. in following ML model.

X_train = df_train_lb[features_used]
y_train = df_train_lb['RUL']

X_test = df_test_lb[features_used]
y_test = df_test_lb['RUL']



# -----------------------------------------------------------------------------
# LASSO

lasso = linear_model.Lasso(alpha = 0.01)
lasso.fit(X_train, y_train)

y_test_predict = lasso.predict(X_test)
y_train_predict = lasso.predict(X_train)

# see brief metrics of LASSO
brief_reg_metrics('LASSO', y_test, y_test_predict)

# reform a metrics of LASSO
ref_metrics_lasso_R = reform_reg_metrics('LASSO', 'Reduced',
                   y_test, y_test_predict, y_train, y_train_predict)
ref_metrics_lasso_R

plot_features_weights('LASSO(R)', lasso.coef_, X_train.columns, 'c')
plot_residual('LASSO(R)', y_train_predict, y_train, y_test_predict, y_test)



# -----------------------------------------------------------------------------
# Ridge

rdg = linear_model.Ridge(alpha = 0.1)
rdg.fit(X_train, y_train)

y_test_predict = rdg.predict(X_test)
y_train_predict = rdg.predict(X_train)

# see brief metrics of Ridge
brief_reg_metrics('Ridge Regression', y_test, y_test_predict)

# reform a metrics of Ridge
ref_metrics_rdg_R = reform_reg_metrics('Ridge', 'Reduced',
                   y_test, y_test_predict, y_train, y_train_predict)
ref_metrics_rdg_R

plot_features_weights('Ridge(R)', rdg.coef_, X_train.columns, 'c')
plot_residual('Ridge(R)', y_train_predict, y_train, y_test_predict, y_test)



# -----------------------------------------------------------------------------
# Polynomial Regression

poly = PolynomialFeatures(degree = 2)
X_train_poly = poly.fit_transform(X_train)
X_test_poly = poly.fit_transform(X_test)

polyreg = linear_model.LinearRegression()
polyreg.fit(X_train_poly, y_train)

y_test_predict = polyreg.predict(X_test_poly)
y_train_predict = polyreg.predict(X_train_poly)

# see brief metrics of Polynomial Regression
brief_reg_metrics('Polynomial ', y_test, y_test_predict)

# reform a metrics of Polynomial Regression
ref_metrics_poly_R = reform_reg_metrics('Polynomial', 'Reduced',
                   y_test, y_test_predict, y_train, y_train_predict)
ref_metrics_poly_R

# cause the weight matrix of "weight matrix" are too large to plot by plot_features_weights()
# so we did not plot weight matrix
plot_residual('Polynomial(R)', y_train_predict, y_train, y_test_predict, y_test)



# -----------------------------------------------------------------------------
# Decision Tree regressor

dtrg = DecisionTreeRegressor(max_depth = 7, max_features = 13, random_state = 36) # best for features_rd
#dtrg = DecisionTreeRegressor(max_depth = 6 , max_features = 23, random_state = 87) # best for features_plus
#dtrg = DecisionTreeRegressor(max_depth = 8 , max_features = 23, random_state = 36) # best for features_org
dtrg.fit(X_train, y_train)

y_test_predict = dtrg.predict(X_test)
y_train_predict = dtrg.predict(X_train)

# see brief metrics of Decision Tree
brief_reg_metrics('Decision Tree', y_test, y_test_predict)

# reform a metrics of Decision Tree
ref_metrics_dtrg_R = reform_reg_metrics('Decision Tree', 'Reduced',
                   y_test, y_test_predict, y_train, y_train_predict)
ref_metrics_dtrg_R

plot_features_weights('Decision Tree(R)', dtrg.feature_importances_, X_train.columns, 't')
plot_residual('Decision Tree(R)', y_train_predict, y_train, y_test_predict, y_test)



# -----------------------------------------------------------------------------
# Random Forest

#rf = RandomForestRegressor(n_estimators = 30, max_features = 1, max_depth = 9, n_jobs= -1, random_state = 51) # feature_plus
rf = RandomForestRegressor(n_estimators = 13, max_features = 1, max_depth = 8, n_jobs= -1, random_state = 1) # features_rd
#rf = RandomForestRegressor(n_estimators = 12, max_features = 1, max_depth = 7, n_jobs= -1, random_state = 1) # features_org

rf.fit(X_train, y_train)

y_test_predict = rf.predict(X_test)
y_train_predict = rf.predict(X_train)

# see brief metrics of Random Forest
brief_reg_metrics('Random Forest', y_test, y_test_predict)

# reform a metrics of Random Forest
ref_metrics_rf_R = reform_reg_metrics('Random Forest', 'Reduced',
                   y_test, y_test_predict, y_train, y_train_predict)
ref_metrics_rf_R

plot_features_weights('Random Forest(R)', rf.feature_importances_, X_train.columns, 't')
plot_residual('Random Forest(R)', y_train_predict, y_train, y_test_predict, y_test)


# -----------------------------------------------------------------------------
# XGBoost

xgb = XGBRegressor(max_depth = 7, learning_rate = 0.01, reg_alpha = 1, reg_lambda = 0) # for all

xgb.fit(X_train, y_train)

y_test_predict = xgb.predict(X_test)
y_train_predict = xgb.predict(X_train)

# see brief metrics of Random Forest
brief_reg_metrics('XGBoost', y_test, y_test_predict)

# reform a metrics of Random Forest
ref_metrics_xgb_R = reform_reg_metrics('XGBoost', 'Reduced',
                   y_test, y_test_predict, y_train, y_train_predict)
ref_metrics_xgb_R

plot_features_weights('XGBoost(R)', xgb.feature_importances_, X_train.columns, 't')
plot_residual('XGBoost(R)', y_train_predict, y_train, y_test_predict, y_test)



# -----------------------------------------------------------------------------
# Neural Network Regression

mlp = MLPRegressor(hidden_layer_sizes=(36,), max_iter = 365,
                   alpha = 0.05, learning_rate = 'adaptive',
                   random_state = 51)
mlp.fit(X_train, y_train)

y_test_predict = mlp.predict(X_test)
y_train_predict = mlp.predict(X_train)

# see brief metrics of Neural Network
brief_reg_metrics('Neural Network', y_test, y_test_predict)

# reform a metrics of Neural Network
ref_metrics_mlp_R = reform_reg_metrics('Neural Network', 'Reduced',
                   y_test, y_test_predict, y_train, y_train_predict)
ref_metrics_mlp_R

# cause the weight matrix of "weight matrix" are too large to plot by plot_features_weights()
# so just like Polynomial Regression, we did not plot weight matrix
plot_residual('Neural Network(R)', y_train_predict, y_train, y_test_predict, y_test)



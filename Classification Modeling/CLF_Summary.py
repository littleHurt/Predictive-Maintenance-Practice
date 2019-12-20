

# importing the necessary modules
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
plt.style.use('ggplot')
%matplotlib inline 
import seaborn as sns
sns.set()
from sklearn.metrics import precision_recall_curve, roc_curve, roc_auc_score


# -----------------------------------------------------------------------------
# Compare all results of classification models

ref_metrics_lgr = pd.concat([ref_metrics_lgrO, ref_metrics_lgrR, ref_metrics_lgrP],axis = 0)
ref_metrics_knn = pd.concat([ref_metrics_knnO, ref_metrics_knnR, ref_metrics_knnP],axis = 0)
ref_metrics_svc = pd.concat([ref_metrics_svcO, ref_metrics_svcR, ref_metrics_svcP],axis = 0)
ref_metrics_gnb = pd.concat([ref_metrics_gnbO, ref_metrics_gnbR, ref_metrics_gnbP],axis = 0)
ref_metrics_dtr = pd.concat([ref_metrics_dtrO, ref_metrics_dtrR, ref_metrics_dtrP],axis = 0)
ref_metrics_rfc = pd.concat([ref_metrics_rfcO, ref_metrics_rfcR, ref_metrics_rfcP],axis = 0)
ref_metrics_xgb = pd.concat([ref_metrics_xgbO, ref_metrics_xgbR, ref_metrics_xgbP],axis = 0)
ref_metrics_mlp = pd.concat([ref_metrics_mlpO, ref_metrics_mlpR, ref_metrics_mlpP],axis = 0)

ref_clf_metrics = pd.concat([ref_metrics_lgr, ref_metrics_knn,
                             ref_metrics_svc, ref_metrics_gnb,
                             ref_metrics_dtr, ref_metrics_rfc,
                             ref_metrics_xgb, ref_metrics_mlp], axis = 0)

# see the result of combination
ref_clf_metrics



# see of ranking of Accuracy
ref_clf_metrics.sort_values( by = 'Accuracy')

# see of ranking of Precision
ref_clf_metrics.sort_values( by = 'Precision')

# see of ranking of Recall
ref_clf_metrics.sort_values( by = 'Recall')

# see of ranking of F1 Score
ref_clf_metrics.sort_values( by = 'F1 Score')

# see of ranking of ROC AUC
ref_clf_metrics.sort_values( by = 'ROC AUC')



# save "ref_clf_metrics" into CSV file
ref_clf_metrics = ref_clf_metrics.to_csv(r'C:\Users\Desktop\ref_clf_metrics.csv',\
                            index = None, header = True, encoding='utf-8')

    


# -----------------------------------------------------------------------------
# Brief Results Analyses:    
'''
    1. Most models perform well on Accuracy.
        (22/24 of them >= 90 or near 90 on Accuracy score).

    2. Most models perform well on Precision.
        (18/24 of them >= 90 or near 90 on Precision score).
        
    3. Most models perform well on ROC-AUC.
        (23/24 of them >= 90 or near 90 on ROC-AUC score)
    
    4. Unfortunately, few models perform well on Recall, and Recall should be
        deem as the most important metrics in the issue of Predictive-Maintenance
        (only 4/24 of them >= 90 or near 90 on Recall).
    
    5. In our experiment, Gaussian Naive Bayesian models almost got best performance
        with high scores on all metrics.
        
    6. Generally, {Logistics Regression, k-NN, SVM} models with original features
        perform slightly better than same models with reduced features.
    
    7. For these 8 kinds algorithm of classification models, almost over half of them perform
        better with plus features. {Logistic Regression, k-NN, SVM, Decision Tree,
        XGBoost}
    
    8. For these 8 kinds of classification models, Random Forest is the most stable
        model which the results with three kinds of features are almost same.
'''




# ----------------------------------------------------------------------------- 
# Advanced comparison
'''
We want to select some models with better performance for advanced comparison.

    1. Let's rank "ref_clf_metrics" by:
        "Recall" -> "F1 Score" -> "ROC-AUC" -> "Precision"
    
    2. Gaussian Naive Bayesian models with all kinds of features seem good candidates,
        but we decided remove the model with reduced features which got worst result in
        models of Gaussian Naive Bayesian.

    3. Neural Network model with reduced features seem a good candidate which also
        perform well on Recall (0.92), but its Precision is too low (0.426).
        So, we would not consider it.
        
    4. Follow by top of ranking list of point 1, we would select models below
        for advanced comparison:
            
        Gaussian Naive Bayesian(O)
        Gaussian Naive Bayesian(P)
        Decision Tree(P)
        Random Forest(R)
        XGBoost(O)
'''


# ----------------------------------------------------------------------------- 
# Build a customized function to create dataframe for multiple comparison

def create_compare_tb(model_name, clf, X_train, y_train, X_test, y_test):
    
    """create dataframe to compare score of confusion metrics and compute
        cost from predictive maintenance models    
    
    Args:
        model_name (str): The model name identifier
        clf (clssifier object): The used classifier 
        y_train (series): The training labels
        y_train_pred (series): Predictions on training data
        y_test (series): The test labels
        y_test_pred (series): Predictions on test data
        
    Returns:
        result_tb (dataframe): dataframe for miltiple comparision
        recall_model (array) : array to plot Precision-Recall Curves
        precision_model (array): array to plot Precision-Recall Curves
        
    """ 
    # set classification models 
    probas_model = clf.fit(X_train, y_train).predict_proba(X_test)[::,1]
    
    # got ROC-AUC scores from selected models  
    fpr_model, tpr_model, _  = roc_curve(y_test, probas_model)
    auc_model = roc_auc_score(y_test, probas_model)

    result_tb_column = ['classifiers', 'fpr','tpr','auc']
    
    result_tb_info = [[model_name, fpr_model, tpr_model, auc_model ]]
    
    result_tb = pd.DataFrame(result_tb_info, columns = result_tb_column)    
    
    # got precision and recall scores from selected models 
    precision_model, recall_model, thresh_prc_model = metrics.precision_recall_curve(y_test, probas_model)
        
    
    return result_tb, recall_model, precision_model





# ----------------------------------------------------------------------------- 
# prepare training/test data with original features for multiple comparison
    
df_train_lb = prepare_train_data( df_train_fx, 30)
df_train_lb

df_test_lb = prepare_test_data(df_test_fx, TrueRUL, 30)
df_test_lb

features_used = features_org 

X_train_O = df_train_lb[features_used]
y_train_O = df_train_lb['label_bc']

X_test_O = df_test_lb[features_used]
y_test_O = df_test_lb['label_bc']



# ----------------------------------------------------------------------------- 
# prepare training/test data with reduced features for multiple comparison
    
df_train_lb = prepare_train_data(df_train_fx, 30)
df_train_lb

df_test_lb = prepare_test_data(df_test_fx, TrueRUL, 30)
df_test_lb


features_used = features_rd # (or features_rd, features_plus) which is used in following ML model.

X_train_R = df_train_lb[features_used]
y_train_R = df_train_lb['label_bc']

X_test_R = df_test_lb[features_used]
y_test_R = df_test_lb['label_bc']



# ----------------------------------------------------------------------------- 
# prepare training/test data with plus features for multiple comparison
    
df_train_lb = prepare_train_data(df_train_fx, 30)
df_train_lb

df_test_lb = prepare_test_data(df_test_fx, TrueRUL, 30)
df_test_lb


features_used = features_plus # (or features_rd, features_plus) which is used in following ML model.

X_train_P = df_train_lb[features_used]
y_train_P = df_train_lb['label_bc']

X_test_P = df_test_lb[features_used]
y_test_P = df_test_lb['label_bc']



# ----------------------------------------------------------------------------- 
# set classification models and got metrics scores for multiple comparison

clf_gnbO = GaussianNB()
clf_gnbP = GaussianNB()
clf_dtrP = DecisionTreeClassifier(criterion = 'gini', max_depth = 3, max_features = 10, min_samples_split = 2,random_state = 51)
clf_rfcR = RandomForestClassifier(criterion = 'gini', max_depth = 11, n_estimators = 99, random_state = 51)
clf_xgbO = XGBClassifier(colsample_bytree = 0.5, max_depth = 4, learning_rate = 0.1, random_state = 51)

# got metrics scores from selected models
result_tb_gnbO, recall_gnbO, precision_gnbO = create_compare_tb('Gaussian Naive Bayesian (O)', clf_gnbO,
                                      X_train_O, y_train_O, X_test_O, y_test_O)

result_tb_gnbP, recall_gnbP, precision_gnbP = create_compare_tb('Gaussian Naive Bayesian (P)', clf_gnbP,
                                      X_train_P, y_train_P, X_test_P, y_test_P)

result_tb_dtrP, recall_dtrP, precision_dtrP = create_compare_tb('Decision Tree (O)', clf_dtrP,
                                      X_train_P, y_train_P, X_test_P, y_test_P)

result_tb_rfcR, recall_rfcR, precision_rfcR = create_compare_tb('Random Forest (R)', clf_rfcR,
                                      X_train_R, y_train_R, X_test_R, y_test_R)

result_tb_xgbO, recall_xgbO, precision_xgbO = create_compare_tb('XGBoost (O)', clf_xgbO,
                                      X_train_O, y_train_O, X_test_O, y_test_O)

result_tb = pd.concat([result_tb_gnbO,
                       result_tb_gnbP,
                       result_tb_dtrP,
                       result_tb_rfcR,
                       result_tb_xgbO], axis = 0)

# see the metrics of selected models
result_tb



    
#----------------------------------------------------------------------------- 
# plot ROC-AUC curves of selected models

# Set name of the classifiers as index labels for plot
result_tb.set_index('classifiers', inplace = True)
result_tb
    
fig = plt.figure(figsize = (14,12))

for i in result_tb.index:
    plt.plot(result_tb.loc[i]['fpr'], 
             result_tb.loc[i]['tpr'],
             label="{}, AUC = {:.3f}".format(i, result_tb.loc[i]['auc']), lw = 2)
    
plt.plot([0,1], [0,1], color = 'slategrey', lw = 3, linestyle='--')

plt.xticks(np.arange(0.0, 1.1, step = 0.1))
plt.xlabel("Flase Positive Rate", fontsize = 18)

plt.yticks(np.arange(0.0, 1.1, step = 0.1))
plt.ylabel("True Positive Rate", fontsize = 18)

plt.xticks(fontsize = 16)
plt.yticks(fontsize = 16)

plt.title('ROC Curves', fontsize = 24)
plt.legend(prop={'size':16}, loc='lower right')

plt.show()




# ----------------------------------------------------------------------------- 
# plot Precision-Recall Curves of selected models 

plt.plot(recall_gnbO, precision_gnbO, lw = 2, label = 'Gaussian Naive Bayesian (O)')
plt.plot(recall_gnbP, precision_gnbP, lw = 2, label = 'Gaussian Naive Bayesian (P)')
plt.plot(recall_dtrP, precision_dtrP, lw = 2, label = 'Decision Tree (P)')
plt.plot(recall_rfcO, precision_rfcO, lw = 2, label = 'Random Forest (R)')
plt.plot(recall_xgbO, precision_xgbO, lw = 2, label = 'XGBoost (O)')

plt.xticks(fontsize = 16)
plt.yticks(fontsize = 16)

plt.xticks(np.arange(0.0, 1.05, step = 0.1))
plt.xlabel("Recall", fontsize = 18)

plt.yticks(np.arange(0.0, 1.05, step = 0.1))
plt.ylabel("Precision", fontsize = 18)

plt.title('Precision-Recall Curves', fontsize = 24)
plt.legend(prop={'size':16}, loc='lower left')

plt.show()




# ----------------------------------------------------------------------------- 
# plot Calibration Curves of selected models by scikit-plot

# got metrics scores from selected models
probas_gnbO = clf_gnbO.fit(X_train_O, y_train_O).predict_proba(X_test_O)
probas_gnbP = clf_gnbP.fit(X_train_P, y_train_P).predict_proba(X_test_P)
probas_dtrP = clf_dtrP.fit(X_train_P, y_train_P).predict_proba(X_test_P)
probas_rfcR = clf_rfcR.fit(X_train_R, y_train_R).predict_proba(X_test_R)
probas_xgbO = clf_xgbO.fit(X_train_O, y_train_O).predict_proba(X_test_O)


probas_list = [probas_gnbO, probas_gnbP, probas_dtrP, probas_rfcO, probas_xgbO]
clf_names = ['Gaussian Naive Bayesian (O)',
             'Gaussian Naive Bayesian (P)',
             'Decision Tree (P)',
             'Random Forest (R)',
             'XGBoost (O)']

skplt.metrics.plot_calibration_curve(y_test, probas_list, clf_names)
plt.show()


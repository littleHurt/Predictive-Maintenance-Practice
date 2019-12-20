

# Loading necessary modules
import pandas as pd
import numpy as np
import scikitplot as skplt
import matplotlib.pyplot as plt
plt.style.use('ggplot')
%matplotlib inline 

from sklearn import metrics
from sklearn import model_selection

# -----------------------------------------------------------------------------
# Build a customized function to add regression and classification labels into the training data.

def prepare_train_data(input_df_test, iteration):
    
    """Add regression and classification labels to the training data.

        Regression label: RUL (run-to-failure) = each cycle# for an engine subtracted from the last cycle# of the same engine
        
        Binary classification label: label_bc = if RUL <= parameter iteration then 1 else 0 (values = 0,1)      
        
      Args:
          input_df_test (dataframe): The input training data
          interation (int)         : The number of cycles for RUL segmentation, used to derive classification labels
          
      Returns:
          dataframe: The input dataframe with regression and classification labels added
          
    """
    
    # create regression label
    
    # make a dataframe to hold the last cycle for each enginge in the dataset
    df_max_cycle = pd.DataFrame(input_df_test.groupby('id')['cycle'].max())
    df_max_cycle.reset_index(level = 0, inplace = True)
    df_max_cycle.columns = ['id', 'last_cycle']
    
    # add time-to-failure RUL as a new column - regression label
    input_df_test = pd.merge(input_df_test, df_max_cycle, on = 'id')
    input_df_test['RUL'] = input_df_test['last_cycle'] - input_df_test['cycle']
    input_df_test.drop(['last_cycle'], axis = 1, inplace = True)
    
    # create binary classification label
    input_df_test['label_bc'] = input_df_test['RUL'].apply(lambda x: 1 if x <= iteration else 0)

    return input_df_test




# -----------------------------------------------------------------------------
# Build a customized function to add regression and classification labels into the training data.

def prepare_test_data ( input_df_test, df_truth, iteration):
    
    """Add regression and classification labels to the test data.

        Regression label: RUL (run-to-failure) = each cycle# for an engine subtracted from the last cycle# of the same engine
        
        Binary classification label: label_bc = if RUL <= parameter iteration then 1 else 0 (values = 0,1)
        
      Args:
          input_df_test (dataframe): The input test data
          
          df_truth (dataframe)     : The true RUL recoreds
          
          interation (int)    : The number of cycles for RUL segmentation. Used to derive classification labels
          
      Returns:
          dataframe: The input dataframe with regression and classification labels added      
    """
    
    df_tst_last_cycle = pd.DataFrame(input_df_test.groupby('id')['cycle'].max())   
    df_tst_last_cycle.reset_index(level = 0, inplace= True )
    df_tst_last_cycle.columns = ['id', 'last_cycle']
    
    input_df_test = pd.merge(input_df_test, df_tst_last_cycle, on = 'id')
    input_df_test = input_df_test[input_df_test['cycle'] == input_df_test['last_cycle']]  
    input_df_test.drop(['last_cycle'], axis=1, inplace= True)
    
    input_df_test.reset_index(drop = True, inplace = True )
    input_df_test = pd.concat([input_df_test, df_truth ], axis = 1)
    
    # create binary classification label
    input_df_test['label_bc'] = input_df_test['RUL'].apply(lambda x: 1 if x <= iteration else 0)

    return input_df_test




# -----------------------------------------------------------------------------
# Prepare the training data:

# load the previous data
df_train_fx = pd.read_csv(r'C:\Users\Desktop\df_train_fx.csv')
df_test_fx  = pd.read_csv(r'C:\Users\Desktop\df_test_fx.csv')


# adding training labels to training data with 30 cycles for classification labing
df_train_lb = prepare_train_data( df_train_fx, 30) 
df_train_lb


# Prepare the test data:
# adding labels to test data with () cycles of  () iterations for classification labing
df_test_lb = prepare_test_data( df_test_fx, TrueRUL , 30)
df_test_lb

# We have shown that why we use 30 for classification labeling in "REG_Preprocessing" in my GitHub.


# save "df_train_fx" into CSV file for modeling
df_train_lb = df_train_lb.to_csv(r'C:\Users\Desktop\df_train_lb.csv',\
                            index = None, header = True, encoding='utf-8')

# save "df_test_fx" into CSV file for modeling
df_test_lb = df_test_lb.to_csv(r'C:\Users\Desktop\df_test_lb.csv',\
                            index = None, header = True, encoding='utf-8')  

    

# -----------------------------------------------------------------------------
# a variable to hold the set of features to experiment with
features_org = ['st1','st2','st3',\
                's1','s2','s3','s4','s5','s6','s7','s8','s9','s10',\
                's11','s12','s13','s14','s15','s16','s17','s18','s19','s20','s21']

features_rd = [ 'st1','st2',\
                            's3', 's4',           's7',       's9',\
                's11','s12',      's14','s15',     's17',           's20','s21']

features_plus = [ 'st1', 'st2','st3',\
                   's1',  's2', 's3', 's4', 's5', 's6', 's7', 's8', 's9','s10',\
                  's11', 's12','s13','s14','s15','s16','s17','s18','s19','s20','s21',\
                  
                 'avt1','avt2','avt3',\
                  'av1', 'av2', 'av3', 'av4', 'av5', 'av6', 'av7', 'av8', 'av9','av10',\
                 'av11','av12','av13','av14','av15','av16','av17','av18','av19','av20','av21',\
                 
                 'sdt1','sdt2','sdt3',\
                  'sd1', 'sd2', 'sd3', 'sd4', 'sd5', 'sd6', 'sd7', 'sd8', 'sd9','sd10',\
                 'sd11','sd12','sd13','sd14','sd15','sd16','sd17','sd18','sd19','sd20','sd21']

features_used = features_org # (or features_rd, features_plus) whcih is uesd in following ML model.

X_train = df_train_lb[features_used]
y_train = df_train_lb['label_bc']

X_test = df_test_lb[features_used]
y_test = df_test_lb['label_bc']



# -----------------------------------------------------------------------------
# Build a customized function to do cross-validation and Grid Search for
# hyper-parameter tuning on a classifier automatically

def classification_tuning(model_name, clf, features_used, params = None):
    
    """Perform cross-validation by training-set
        and then do Grid Search for hyper parameter tuning on test-set automatically.
    
    Args:
        model_name (str): The model name identifier
        clf (clssifier object): The classifier to be tuned
        features_used (list): The set of input features names
        params (dict): Grid Search parameters
        
    Returns:
        Tuned Clssifier object
        dataframe of model predictions and scores
    """    
    
    X_train = df_train_lb[features_used]
    X_test = df_test_lb[features_used] 
    
    grid_search = model_selection.GridSearchCV(
            estimator = clf, param_grid = params, cv = 6, scoring = 'recall', n_jobs = -1)

    grid_search.fit(X_train, y_train)
    y_pred = grid_search.predict(X_test)
    
    if hasattr(grid_search, 'predict_proba'):   
        y_score = grid_search.predict_proba(X_test)[:,1]
    elif hasattr(grid_search, 'decision_function'):
        y_score = grid_search.decision_function(X_test)
    else:
        y_score = y_pred
        
    predictions = {'y_pred' : y_pred, 'y_score' : y_score}
    df_predictions = pd.DataFrame.from_dict(predictions)
    
    return grid_search.best_estimator_, df_predictions




# -----------------------------------------------------------------------------
# Build a customized function to return the performance metrics
    
def brief_clf_metrics(model_name, y_test, y_pred, y_score):
    
    """Calculate main binary classifcation performance metrics
    
    Args:
        model_name (str): The model name identifier
        y_test (series): Contains the test label values
        y_pred (series): Contains the predicted values
        y_score (series): Contains the predicted scores
        
    Returns:
        dataframe: The combined metrics in single dataframe
        dataframe: ROC thresholds
        dataframe: Precision-Recall thresholds
        
    """
      
    binclass_metrics = {
                        'Accuracy' : metrics.accuracy_score(y_test, y_pred),
                        'Precision' : metrics.precision_score(y_test, y_pred),
                        'Recall' : metrics.recall_score(y_test, y_pred),
                        'F1 Score' : metrics.f1_score(y_test, y_pred),
                        'ROC AUC' : metrics.roc_auc_score(y_test, y_score)
                       }

    df_brief_clf_metrics = pd.DataFrame.from_dict(binclass_metrics, orient = 'index')
    df_brief_clf_metrics.columns = [model_name]  

    print('-----------------------------------------------------------')
    print(model_name, '\n')
    print('Confusion Matrix:')
    print(metrics.confusion_matrix(y_test, y_pred))
    print('\nClassification Report:')
    print(metrics.classification_report(y_test, y_pred))
    print('\nPerformance Metrics:')
    print(df_brief_clf_metrics)





# -----------------------------------------------------------------------------
# Build a customized function to reform metrics for comparison of all regression model
    
def reform_clf_metrics(model_name, y_test, y_pred, y_score):
    
    """Calculate main binary classifcation performance metrics
    
    Args:
        model_name (str): The model name identifier
        y_test (series): Contains the test label values
        y_pred (series): Contains the predicted values
        y_score (series): Contains the predicted scores
        
    Returns:
        dataframe: The combined metrics in single dataframe
        dataframe: ROC thresholds
        dataframe: Precision-Recall thresholds    
    """    

    clf_metrics = [[ model_name,
                round( metrics.accuracy_score(y_test, y_pred), 3),
                round( metrics.precision_score(y_test, y_pred), 3),
                round( metrics.recall_score(y_test, y_pred), 3),
                round( metrics.f1_score(y_test, y_pred), 3),
                round( metrics.roc_auc_score(y_test, y_score), 3) ]]
    
    # clf_metrics
    clf_metrics_column = ['Model','Accuracy','Precision',
                          'Recall','F1 Score','ROC AUC']    
    df_clf_metrics = pd.DataFrame(clf_metrics, columns = clf_metrics_column)
    
    return df_clf_metrics




#------------------------------------------------------------------------------
# Build a customized function to print "ROC Curve", "Precision-Recall Curve",
# "Cumulative Gains Curves", and "Calibration Curve" by scikit-plot

def plot_metrics(clf, X_train, y_train, X_test, y_test):

    """plot classifcation performance metrics
    
    Args:
        clf (clssifier object): The classifier to be tuned
        y_train (series): The training labels
        y_train_pred (series): Predictions on training data
        y_test (series): The test labels
        y_test_pred (series): Predictions on test data        
        
    Returns:
        plot: ROC Curves
        plot: Precision-Recall Curves
        plot: Cumulative Gains Curves
        plot: Calibration Curves
    """    
    
    
    clf.fit(X_train, y_train)
    y_probas = clf.fit(X_train, y_train).predict_proba(X_test)

    # plot ROC Curve by "scikit-plot"
    skplt.metrics.plot_roc(y_test, y_probas, title = model_name + ' ROC Curves',
                           title_fontsize = 24, text_fontsize = 16)
    plt.show()


    # plot Precision-Recall Curve by "scikit-plot"
    skplt.metrics.plot_precision_recall(y_test, y_probas, title = model_name + ' Precision-Recall Curves',
                                        title_fontsize = 24, text_fontsize = 16)
    plt.show()

    # plot Cumulative Gains Curves by "scikit-plot"
    skplt.metrics.plot_cumulative_gain(y_test, y_probas, title = model_name + ' Cumulative Gains Curves',
                                        title_fontsize = 24, text_fontsize = 16)
    plt.show()

    # plot Calibration Curves by "scikit-plot"
    probas_list = [y_probas]
    clf_names = [model_name]    
    skplt.metrics.plot_calibration_curve(y_test, probas_list, clf_names, title = model_name + ' Calibration Curves Curves',
                           title_fontsize = 24, text_fontsize = 16)
    plt.show()



# -----------------------------------------------------------------------------
# Label of modeles name:
"""
Before we test every classification model, we must the mark of {O, R, P}
    O stands for fit the model with original features [features_org]
    R stands for fit the model with  reduced features [features_rd]
    P stands for fit the model with     plus features [features_plus]
"""

# Classifiction models
"""
In this stage (modeling with validateion), we would try 8 kinds of classification models,
    and find its best parameters:
    
    Logistics Regression
    k-NN
    SVM
    Gaussian Naive Bayesian Classification
    Decision Tree Classification
    Random Forest Classification
    XGBoost Classification
    Neural Network Classification

After total comparison, we would select the best 5 models (and its related training-set)
to perform the final test
"""

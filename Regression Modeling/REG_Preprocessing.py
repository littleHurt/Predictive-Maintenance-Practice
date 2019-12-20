
# Loading necessary module
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
plt.style.use('ggplot')
%matplotlib inline  

from sklearn import metrics
from sklearn import model_selection



# -----------------------------------------------------------------------------
# Build a customized function to add regression and classification labels into the training data.

def prepare_train_data(input_df_test, iteration):
    
    """Add regression and classification labels to the training data.

        Regression label: RUL (Remaining useful life) = 
            each cycle# for an engine subtracted from the last cycle# of the same engine
        
        Binary classification label: label_bc = 
            if RUL <= parameter iteration then 1 else 0 (values = 0,1)      
        
      Args:
          input_df_test (dataframe): The input training data
          interation (int)         : The number of cycles for RUL segmentation, used to derive classification labels
          
      Returns:
          dataframe: The input dataframe with regression and classification labels added.         
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

def prepare_test_data (input_df_test, df_truth, iteration):
    
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
# Build a customized function to calculate brief regression metrics
def brief_reg_metrics(model_name, actual, predicted):
    
    """Calculate brief regression metrics.
    
    Args:
        model_name (str): The model name identifier
        actual (series): Contains the test label values
        predicted (series): Contains the predicted values
        
    Returns:
        dataframe: The combined metrics in single dataframe    
    
    """
    brief_metrics = {
                    'Root Mean Squared Error' : metrics.mean_squared_error(actual, predicted)**0.5,
                    'Mean Absolute Error' : metrics.mean_absolute_error(actual, predicted),
                    'R^2' : metrics.r2_score(actual, predicted),
                    'Explained Variance' : metrics.explained_variance_score(actual, predicted),
                    }

    # return brief_metrics
    df_brief_metrics = pd.DataFrame.from_dict(brief_metrics, orient = 'index')
    df_brief_metrics.columns = [model_name]
    
    return df_brief_metrics





# -----------------------------------------------------------------------------
# Build a customized function to reform metrics for comparison of all regression models

def reform_reg_metrics(model_name, features_used,
               test_actual, test_predicted, train_actual, train_predicted):
    
    """Calculate more regression metrics for comparason of all regresson model
    
    Args:
        model_name (str): The model name identifier
        features_used (str): {Origin, Ruduced, Plus}
        test_aactual (series): Contains the test label values
        test_apredicted (series): Contains the predicted values
        train_aactual (series): Contains the test label values
        train_apredicted (series): Contains the predicted values
        
    Returns:
        dataframe: 'df_reg_metrics' for compare metrics of all regresson model
    
    """    
    reg_metrics = [[ model_name,
                features_used,
                round( metrics.mean_squared_error(y_test, y_test_predict)**0.5, 3),
                round( metrics.mean_absolute_error(y_test, y_test_predict), 3),
                round( metrics.r2_score(y_test, y_test_predict), 3),
                round( metrics.explained_variance_score(y_test, y_test_predict), 3),
                round( (y_test - y_test_predict).mean(), 3),
                round( (metrics.r2_score(y_train, y_train_predict)), 3)
                ]]
    
    # reg_metrics
    reg_metrics_column = ['Model','Features','RMSE','MAE','R^2',
                              'EV','mean(Residuals)','R^2 (Train)']    
    df_reg_metrics = pd.DataFrame(reg_metrics, columns = reg_metrics_column)
    
    return df_reg_metrics
    




# -----------------------------------------------------------------------------
# Build a customized function to plot the coefficients weights or feature importance   

def plot_features_weights(model_name, weights, feature_names, weights_type = 'c'):
    
    """Plot regression coefficients weights or feature importance.
    
    Args:
        model_name (str): The model name identifier
        weights (array): Contains the regression coefficients weights or feature importance
        feature_names (list): Contains the corresponding features names
        weights_type (str): 'c' for 'coefficients weights', otherwise is 'feature importance'
        
    Returns:
        plot of either regression coefficients weights or feature importance      
    
    """
    
    W = pd.DataFrame({'Weights':weights}, feature_names)
    W.sort_values( by = 'Weights', ascending = True).plot(
            kind = 'barh', color = 'dimgrey', figsize = (14,12)) # the figure size are
    label = ' Coefficients' if weights_type == 'c' else ' Features Importance'
    plt.xlabel(model_name + label, fontsize = 20)
    plt.xticks(fontsize = 14)
    plt.yticks(fontsize = 14)
    plt.gca().legend_ = None




# -----------------------------------------------------------------------------
# Build a customized function to plot residuals of regression models
    
def plot_residual(model_name, y_train, y_train_pred, y_test, y_test_pred):
    
    """Print the regression residuals.
    
    Args:
        model_name (str): The model name identifier
        y_train (series): The training labels
        y_train_pred (series): Predictions on training data
        y_test (series): The test labels
        y_test_pred (series): Predictions on test data
        
    Returns:
        Plot of regression residuals
    
    """
    # plot Predicted vs Remaining Useful Life Actual
    fig, ax = plt.subplots()
    ax.scatter(y_test_pred, y_test, edgecolors=(0, 0, 0))
    ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw = 2)
    ax.set_xlabel('Predicted RUL', fontsize = 18)
    ax.set_ylabel('Actual RUL', fontsize = 18)
    ax.tick_params(axis='both', which='major', labelsize = 14)
    ax.set_title( model_name + ' Predicted RUL vs Actual RUL', fontsize = 20)
    plt.show()

    # plot Predicted vs Residuals
    plt.scatter(y_train_pred, y_train_pred - y_train, c = 'lightseagreen', marker='o', label='Training data')
    plt.scatter(y_test_pred, y_test_pred - y_test, c = 'lightcoral', marker='s', label='Test data')
    plt.xlabel('Predicted RUL', fontsize = 18)
    plt.ylabel('Residuals',  fontsize = 18)
    plt.xticks(fontsize = 14)
    plt.yticks(fontsize = 14)
    plt.legend(loc='upper left')
    plt.hlines(y = 0, xmin = 0, xmax = 375, color = 'dimgrey', lw = 2, linestyles = 'dashed')
    plt.title(model_name + ' Predicted value vs Residuals', fontsize = 20)
    plt.show()    
    




# -----------------------------------------------------------------------------
# Prepare the training data:

# load the previous data
df_train_fx = pd.read_csv(r'C:\Users\123\Desktop\df_train_fx.csv')
df_test_fx  = pd.read_csv(r'C:\Users\123\Desktop\df_test_fx.csv')


# adding training labels to training data with 30 cycles for classification modeling
df_train_lb = prepare_train_data( df_train_fx, 30) 
df_train_lb


# Prepare the test data:
# adding labels to test data with () cycles of  () iterations for classification
df_test_lb = prepare_test_data( df_test_fx, TrueRUL , 30)
df_test_lb

# We would explain why we use 30 for classification labeling in
# "REG_Preprocessing" in my github.



# save "df_train_fx" into CSV file for modeling
df_train_lb = df_train_lb.to_csv(r'C:\Users\123\Desktop\df_train_lb.csv',\
                            index = None, header = True, encoding='utf-8')

# save "df_test_fx" into CSV file for modeling
df_test_lb = df_test_lb.to_csv(r'C:\Users\123\Desktop\df_test_lb.csv',\
                            index = None, header = True, encoding='utf-8')  


    
# -----------------------------------------------------------------------------
# three kinds of features to use in this practice project

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

features_used = features_org 

X_train = df_train_lb[features_used]
y_train = df_train_lb['RUL']

X_test = df_test_lb[features_used]
y_test = df_test_lb['RUL']




# -----------------------------------------------------------------------------
# Labels of models' name:
"""
Before we test every classification model, we have to know the mark of {O, R, P}
    O stands for fit the model with original features [features_org]
    R stands for fit the model with  reduced features [features_rd]
    P stands for fit the model with     plus features [features_plus]
"""

# Regression models
"""
In this stage (modeling with validation), we would try 6 kinds of regression models:
    
    LASSO
    Ridge Regression
    Polynomial Regression
    Decision Tree Regression
    Random Forest Regression
    XGBoost Regression
    Neural Network Regression

After total compassion, we would use the results to proceed classification model
"""
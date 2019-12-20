

# importing the necessary modules
import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix



# ----------------------------------------------------------------------------- 
# calculate expected cost 
'''
According to this paper:
    《Machine Learning for Predictive Maintenance - A Multiple Classifier Approach》
    https://pureadmin.qub.ac.uk/ws/portalfiles/portal/17844756/machine.pdf

We could calculate benefit of Predictive Maintenance models from a customized
cost-based formula. The similar idea is also shown in the textbook:    
    《Data Science for Business: What You Need to Know about Data Mining and Data-Analytic Thinking》
    https://www.amazon.com/Data-Science-Business-Data-Analytic-Thinking/dp/1449361323
    
According to the above idea, we would build a cost-based formula to estimate the 
benefit from the predictive maintenance models
'''


# ----------------------------------------------------------------------------- 
# Build a customized function to create a dataframe including 
# confusion metrics scores and cost of selected models

def create_df_cost(model_name, y_test, y_pred):

    """Calculate main binary classifcation performance metrics
    
    Args:
        model_name (str): The model name identifier
        y_test (series): Contains the test label values
        y_pred (series): Contains the predicted values
        
    Returns:
        df_cost (dataframe): dataframe including confusion metrics scores
                                and cost of selected models.
    """
    
    cf_score = metrics.confusion_matrix(y_test, y_pred)
        
  
    # Local parameters
    #--------------------------------------------------  
    # amounts of unexploited cycles
    # were it should be assumed from results of regression models
    m_ul = 25
    
    # percentage of overpredicted safe sample
    p_ul = (cf_score[0,1] / len(y_test))
    
    # percnettage of unexploited break
    p_ub = (cf_score[1,0] / len(y_test))

    # related cost of p_ul, with $USD unit
    c_bl = 5000 
    
    # related cost of p_ub, with $USD unit
    c_ub = 200 * c_bl   
    #--------------------------------------------------    
    
    # the cost-based formula
    cost = ( m_ul * p_ul * c_bl) + (p_ub * c_ub)
    
    df_cf_cost = [[model_name, round(cost,0),\
                   cf_score[0,0], cf_score[0,1], cf_score[1,0], cf_score[1,1] ]]
    
    df_cost_column = ['Model', 'Cost', 'TP', 'FP', 'FN', 'TN']    
    df_cost = pd.DataFrame(df_cf_cost, columns = df_cost_column)
    
    return df_cost




# ----------------------------------------------------------------------------- 
# got confuson metrics scores and cost of selected models

'''
    where {pred_gnbO, pred_gnbP, pred_dtrP, pred_rfcR, pred_xgbO}
    are generated from experiment of classification modeling.
'''
  
df_cost_gnbO = create_df_cost('Gaussian Naive Bayesian(O)', y_test, pred_gnbO.y_pred)
df_cost_gnbP = create_df_cost('Gaussian Naive Bayesian(P)', y_test, pred_gnbP.y_pred)
df_cost_dtrP = create_df_cost('Decision Tree(P)', y_test, pred_dtrP.y_pred)
df_cost_rfcR = create_df_cost('Random Forest(R)', y_test, pred_rfcR.y_pred)
df_cost_xgbO = create_df_cost('XGBoost(O)', y_test, pred_xgbO.y_pred)

# combine the above results
df_selected = [df_cost_gnbO, df_cost_gnbP, df_cost_dtrP, df_cost_rfcR, df_cost_xgbO]
df_cost_selected = pd.concat(df_selected, axis = 0)
df_cost_selected

# rank dataframe by "cost" column to see most beneficial model
df_cost_selected.sort_values( by = 'Cost', ascending = True)


'''
According to the above results, Gaussian Naive Bayesian model with plus features
bring the best benefit from our predictive maintenance models
'''

# save "df_cost_selected" into CSV file
df_cost_selected = df_cost_selected.to_csv(r'C:\Users\Desktop\df_cost_selected.csv',\
                            index = None, header = True, encoding = 'utf-8')
    
    
    
# -----------------------------------------------------------------------------
# Summary
    '''
    We have built a set of predictive maintenance modules by data science
    approach on case of maintenance of aircraft engines and used a customed function
    to calculate the expected cost to selected the best module.
    
    Our results could answer of two key questions:
        
        1. Whether the engine will break in specific running period?
        2. How many unexploited cycles does the engine remain?

    '''


# Extension of Future
    '''
    This case is just a simple off-line practice. The following actions may be
    helpful for extension of this project:    
        
        1. Consult to domain experts of aircraft mechanical engineers
        2. Try more feature engineering and experiments cross-validation
        3. More detailed hyperparameters tuning for ML models
        4. Try to build an ensemble model by multiple classification threshold
    '''
    
    
    

# Reference
    '''
    1.
    【Samimust/predictive-maintenance】GitHub
    https://github.com/Samimust/predictive-maintenance
    
    2.
    【scikit-learn: machine learning in Python — scikit-learn 0.21.3 documentation】
    https://scikit-learn.org/stable/index.html
        
    3.
    【Drawing multiple ROC-Curves in a single plot · Abdullah Al Imran】
    https://abdalimran.github.io/2019-06-01/Drawing-multiple-ROC-Curves-in-a-single-plot
    
    4.
    【Metrics Module (API Reference) 】— Scikit-plot documentation
    https://scikit-plot.readthedocs.io/en/stable/metrics.html
    
    5.
    《Machine Learning for Predictive Maintenance - A Multiple Classifier Approach》
    https://pureadmin.qub.ac.uk/ws/portalfiles/portal/17844756/machine.pdf
    
    6.
    《Data Science for Business: What You Need to Know about Data Mining and Data-Analytic Thinking》
    https://www.amazon.com/Data-Science-Business-Data-Analytic-Thinking/dp/1449361323
    '''
    
    
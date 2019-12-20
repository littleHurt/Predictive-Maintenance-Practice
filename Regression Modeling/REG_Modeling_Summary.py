
# Loading necessary modules
import pandas as pd
import numpy as np

# -----------------------------------------------------------------------------
# Compare all results of regression models

# Let's combine all results of regression models in a dataframe
ref_metrics_mlp = pd.concat([ref_metrics_mlp_O, ref_metrics_mlp_R, ref_metrics_mlp_P],axis = 0)
ref_metrics_dtrg = pd.concat([ref_metrics_dtrg_O, ref_metrics_dtrg_R, ref_metrics_dtrg_P],axis = 0)
ref_metrics_lasso = pd.concat([ref_metrics_lasso_O, ref_metrics_lasso_R, ref_metrics_lasso_P],axis = 0)
ref_metrics_poly = pd.concat([ref_metrics_poly_O, ref_metrics_poly_R, ref_metrics_poly_P],axis = 0)
ref_metrics_rdg = pd.concat([ref_metrics_rdg_O, ref_metrics_rdg_R, ref_metrics_rdg_P],axis = 0)
ref_metrics_rf = pd.concat([ref_metrics_rf_O, ref_metrics_rf_R, ref_metrics_rf_P],axis = 0)
ref_metrics_xgb = pd.concat([ref_metrics_xgb_O, ref_metrics_xgb_R, ref_metrics_xgb_P],axis = 0)

ref_reg_metrics = pd.concat([ref_metrics_lasso,
                             ref_metrics_rdg,
                             ref_metrics_poly,
                             ref_metrics_dtrg,
                             ref_metrics_rf,
                             ref_metrics_xgb,
                             ref_metrics_mlp], axis = 0)

# see the combination of result
ref_reg_metrics




# The definition of columns in "ref_reg_metrics":
'''
    RMSE: (Root Mean Squared Error) of test-set. (lower is better)
    MAE : (Mean Absolute Error) of test-set. (lower is better)
    R^2 : (R Square) score of test-set. (higher is better)
    EV  : (Explained Variance) score of test-set. (higher is better)
    mean(Residuals): mean of Residuals of test-set. (close to 0 is better)
    R^2(Train) : (R Square) score of training-set. (just for reference)
'''



# see of ranking of RMSE
ref_reg_metrics.sort_values( by = 'RMSE')

# see of ranking of MAE
ref_reg_metrics.sort_values( by = 'MAE')

# see of ranking of R^2
ref_reg_metrics.sort_values( by = 'R^2')

# see of ranking of EV 
ref_reg_metrics.sort_values( by = 'EV'))

# see of ranking of mean(Residuals)
ref_reg_metrics.sort_values( by = 'mean(Residuals)')

# see of ranking of R^2(Train)
ref_reg_metrics.sort_values( by = 'R^2(Train)')




# -----------------------------------------------------------------------------
# Brief Results Analyses:    
'''
    1. XGBoost has the best regression performance on metrics
        of {RMSE, MAE, R^2, EV}, Neural Network and Random Forest rank
        2nd and 3rd in these metrics.
        
    2. Polynomial Regression, Decision Tree have better performance
        on R^2(Train), but they did not perform good scores on the metrics
        of test-set.
    
    3. Neural Network has the best performance on mean(Residuals), which mean
        the whole prediction from model may not so distant from true records.
    
    4. Linear Regression model like LASSO and Ridge Regression
        did not perform well (based on the metrics of {RMSE, MAE, R^2, EV})
        on this case, but they are not the worst model.
        
    5. results with original features perform better than results
        of reduced features and plus features except from XGBoost and Neural Network.
        This may be resulted from non-optimal parameters tuning.
        
    6. The hyper-parameter tuning of these results just are done manually.
        Grid Search or Random Search with Cross Validation could be
        helpful for better processing.
'''




# -----------------------------------------------------------------------------
# Determine classification labeling from results of regression models
'''
The classification label of this case is "the last N cycles of
Remaining Useful Life" of each engine. So, the question is:
    
    "How should we determine N for classification labeling"
    
We have got RMSE and MAE for regression models, all of these numbers could be
consider as "N" in for classification labeling. (where the results with higher
R^2 and EV could be better).

Generally, we hope that N could be as small as it could be.
Based on this goal, results from XGBoost seemed the best one to use.
But! let's see the figures of "Predicted RUL vs Actual RUL" and
"Predicted value vs Residuals"

Though the RMSE and MAE are the smallest one. But We could see that
most prediction from XGBoost regression are greater than true RUL,
which also resulted positive numbers of residuals. 

It means that the results of XGBoost regression may be too optimistic for
our goal. So, let's abandon the results of XGBoost regression.

Based on the above perspective, the results of Neural Network and Random Forest
may be much worthy to consider where most of their residuals are negative but 
much close to 0.

The results' numbers of RMSE and MAE of Neural Network and Random Forest 
ranged from 21.427 to 29.379. For a conservative perspective, we set N as 30
based on the above analyses.
'''




# -----------------------------------------------------------------------------
# Summary
'''
We successfully built some regression models to predict Remaining Useful Life.
And we would use these results to determine N as 30 for classification labeling
for next phrase.
'''

# save "ref_reg_metrics" into CSV file
ref_reg_metrics = ref_reg_metrics.to_csv(r'C:\Users\123\Desktop\ref_reg_metrics.csv',\
                            index = None, header = True, encoding='utf-8')

ref_reg_metrics = pd.read_csv(r'C:\Users\Desktop\ref_reg_metrics.csv')
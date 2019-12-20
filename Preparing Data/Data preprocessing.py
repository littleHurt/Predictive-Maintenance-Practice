
# Loading necessary module
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
plt.style.use('ggplot')
%matplotlib inline  




# -----------------------------------------------------------------------------
# loading train_FD001.txt as "df_train_raw" by pandas read_csv
df_train_raw = pd.read_csv(r'C:\Users\PdM\Dataset\NASA Turbofan Engine Degradation Simulation Data Set\train_FD001.txt', sep = " ", header = None)

# drop NA columns
df_train_raw.drop([26,27], axis = 1, inplace = True )

# see the loading result
df_train_raw 



# Define columns' names:
"""
    # id   : the engine ID, range from 1 to 100
    # cycle: the cycle number per engine sequence where failure had happened, start from 1 to 100
    # st1 to st3: engine settings of operation, start from 1 to 3.
    # s1 to s21: measurements of engine sensor No.1 to No.21
""""
col_names = ['id','cycle','st1','st2','st3',\
             's1','s2','s3','s4','s5','s6','s7','s8','s9','s10',\
             's11','s12','s13','s14','s15','s16','s17','s18','s19','s20','s21']

features = ['st1','st2','st3',\
             's1','s2','s3','s4','s5','s6','s7','s8','s9','s10',\
             's11','s12','s13','s14','s15','s16','s17','s18','s19','s20','s21']
    
# rename the columns of training-set
df_train_raw.columns = col_names

# see the result of replacement
df_train_raw 




# loading test_FD001.txt as "df_test_raw" by pandas read_csv
df_test_raw = pd.read_csv(r'C:\Users\PdM\Dataset\NASA Turbofan Engine Degradation Simulation Data Set\test_FD001.txt', sep = " ", header = None)

# drop NA columns
df_test_raw.drop([26,27], axis = 1, inplace = True )

# rename the columns of test-set
df_test_raw.columns = col_names 

# see the result of loading and replacement
df_test_raw




# loading RUL_FD001.txt as "TrueRUL" by pandas read_csv
TrueRUL = pd.read_csv(r"C:\Users\PdM\Dataset\NASA Turbofan Engine Degradation Simulation Data Set\RUL_FD001.txt", sep = "\s", header = None)

# rename the columns of "TrueRUL" column
TrueRUL.columns = ['RUL'] 

#see the loading result
TrueRUL 






# -----------------------------------------------------------------------------
# Exploratory Data Analysis
# Let's do some simple Exploratory Data Analysis to understand the dataset

# see the simple statistics summary of "df_train_raw", "df_test_raw", and "TrueRUL"
stat_train = df_train_raw.describe()
stat_train

stat_test = df_test_raw.describe()
stat_test

stat_TrueRUL = TrueRUL.describe()
stat_TrueRUL



# Let's see the variation of every setting and sensor with running cycles by engines

# plot the variation of every setting and sensors with running cycles
# by engines.1 to engines.10 "df_train_raw"

sns.set(font_scale = 1.5) # set the font scale = 1.5
variation_sor = sns.PairGrid(data = df_train_raw.query('id <= 10') ,
                 x_vars = "cycle",
                 y_vars = features,
                 hue = "id", height = 3, aspect = 3) #set the figures size
variation_sor  = variation_sor.map(plt.scatter, alpha = 0.5)
variation_sor  = variation_sor.set(xlim = (0,300) )
variation_sor  = variation_sor.add_legend()

"""
 As the above simple statistics summary and visualization results,
 we could see that some features did not vary at all with running cycles.
 Thsee features may be needless for modeling.
 So, we may consider removing these features from training-set and test-set
"""





# -----------------------------------------------------------------------------
# Dimension Reduction

# STEP1
# calculate sd (standard deviation) for each variable and rank it.
featurs_top_var = df_train_raw[features].std().sort_values(ascending = False)

# see the result of ranking
featurs_top_var

# plot and compare the standard deviation of all features:
featurs_top_var.plot(kind = 'bar', figsize = (14,18), title = "Features Standard Deviation")

"""
 Let's see "featurs_top_var" first, the sd of {st3, s1, s5, s10, s16, s18, s19} are (almost) zero.
 It means that these variables may not influence the result of modeling.
 so we remove these variables from "df_train_raw" in STEP1,
 We can also see that the std of {st1, st2, s6} are also very tiny ( sd < 0.01)
 Should we remove {st1, st2, s6}? let's see the advanced tracking in STEP2.
"""



# STEP2
# calculate "cv.abs" (Absolute value of Coefficient of Variation) for each variable
featurs_top_cv = abs( df_train_raw[features].std() / df_train_raw[features].mean() ).sort_values(ascending = False)
featurs_top_cv = featurs_top_cv.sort_values(ascending = False)
featurs_top_cv

# plot the ranking of "cv.abs"
featurs_top_cv.plot( kind = 'bar', figsize = (12,10),\
                     title = "Features Absolute value of Coefficient of Variation")

"""
 Let's see "featurs_top_cv" first
 Beside of {st3, s01, s05, s10, s16, s18, s19}, the "cv.abs" of {s2, s6, s8, s13} are tiny too ( < 0.001)
 By contrast, "cv.abs" of {st1, st2} are top large one in this ranking list.
 Due to the above result, we remove {s2, s6, s8, s13} from "df_train_raw" in STEP2.
"""



# STEP3
"""
 Based on the above results,
 we remove {st3, s01, s02, s05, s06, s08, s10, a13, s16, s18, s19} from "df_train_raw"
 and set reduced columns/features as columns_rd, "features_rd"
"""

# build the reduced data-set "df_train_rd" and "df_test_rd"
columns_rd = ['id','cycle','st1','st2',\
                                  's3', 's4',           's7',       's9',\
                     's11','s12',      's14','s15',     's17',           's20','s21']

features_rd = [      'st1','st2',\
                                  's3', 's4',           's7',       's9',\
                     's11','s12',      's14','s15',     's17',           's20','s21']





# -----------------------------------------------------------------------------
# Feature Construction
    
"""
 On the contrast to Dimension Reduction, we would like to add some
 artificial features to find a better potential model.
 In this phrase, we would add "Moving Average", and "Rolling Standard Deviation"
 into both training-set and test-set.
"""

# Build a customized function to add additional features (moving average and rolling standard deviation)
def add_features(input_df, rolling_iteration):
    
    """
    Adding moving average and rolling standard deviation for each sensor.
    
    Args:
            input_df (dataframe)   : The input dataframe to be proccessed (training-set or test-set).
            rolling_iteration (int): iterations of cycles for applying the rolling function.
        
    Reurns:
            dataframe: contains the input dataframe with additional moving average 
                        and rolling standard deviation for each sensor.    
    """
    
    variables = features

    sensors_cav_cols = [nm.replace('s' , 'av') for nm in variables]    
    sensors_csd_cols = [nm.replace('s' , 'sd') for nm in variables]   
    
    output_df = pd.DataFrame()
    ri = rolling_iteration
    
    # calculating cumulative iteration for each engine id
    
    for m_id in pd.unique(input_df.id):
    
        # get a subset for each engine sensors
        df_engine = input_df[input_df['id'] == m_id]
        df_sub = df_engine[variables]
 
    
        # get moving average for the subset
        av = df_sub.rolling(ri, min_periods = 1).mean()
        av.columns = sensors_cav_cols

        # get rolling standard deviation for the subset
        sd = df_sub.rolling(ri, min_periods = 1).std().fillna(0)
        sd.columns = sensors_csd_cols
 
    
        new_ftrs = pd.concat([df_engine, av, sd], axis = 1)
    
        # add the new features rows to the output dataframe
        output_df = pd.concat([output_df, new_ftrs])
        
    return output_df





# -----------------------------------------------------------------------------

# Determine the iteration for moving average and rolling standard deviation
"""
 There are lot of idea to determine the iteration for moving average and rolling sd
 Let's see the statistics summary of last operation cycle of each engine of training-set
 """
 
max_cycle_train = pd.DataFrame(df_train_raw.groupby('id')['cycle'].max()).describe() 
max_cycle_train


"""
 We hope that the moving average and rolling standard deviation we used
 would be helpful to predicting the last cycle, so this number of iterations
 may be close to the records of last cycle of each engine.
 The sd (standard deviation) of "max_cycle_train" seemed an useful option
 But! Let's see the statistics summary of last operation cycle of each engine of test-set
 (In fact, it right equals to "TrueRUL")
"""

TrueRUL.describe() 

"""
 Every statistical index in "TrueRUL" are smaller than "max_cycle_train"
 46.34 (the sd of "max_cycle_train") even greater than 7 (minimum of "TrueRUL")
 So, how should we do? 
 
 Don't forget that our goal is to build a generalized model for PdM.
 So, in this project, we decided to apply the 80/20 rule (Pareto principle)
 to sd of "max_cycle_train" with 

 46.34 (the sd of "max_cycle_train") * 0.2 ~= 9.268 
 So we decide to use "10" interation to calculate the the moving average 
 and rolling standard deviation for Feature Combination
 
 In order to prevent too complicated of practice, we would just use
 10 for calculation of moving average and rolling standard deviation
"""


# adding extracted features to training-set and test-set
df_train_fx = add_features( df_train_raw, 10)

# see the adding result
df_train_fx

# add extracted features to test data
df_test_fx = add_features(df_test_raw, 10)

# see the adding result
df_test_fx




# -----------------------------------------------------------------------------
# Summary
"""
 We did some simple EDA and data wrangling in this phrase and fixed 
 dataset and get three kind of features : {features_org, features_rd, features_plus} and 
 We would use these set and features for the following modeling.
"""


# save "df_train_fx" into CSV file for modeling
df_train_fx = df_train_fx.to_csv(r'C:\Users\Desktop\df_train_fx.csv',\
                            index = None, header = True, encoding='utf-8')


# save "df_test_fx" into CSV file for modeling
df_test_fx = df_test_fx.to_csv(r'C:\Users\esktop\df_test_fx.csv',\
                            index = None, header = True, encoding='utf-8')    
#!/usr/bin/env python
# coding: utf-8

# In[74]:


import numpy as np
import matplotlib.pyplot as plt 

import pandas as pd  
import seaborn as sns 

get_ipython().run_line_magic('matplotlib', 'inline')


# In[75]:


import pandas as pd
import numpy as np
from sklearn.datasets import load_boston
#載入資料集
boston_dataset = load_boston()


# In[76]:


#We will now load the data into a pandas dataframe using pd.DataFrame. We then print the first 5 rows of the data using head()
#可以看出並沒有預測的目標MEDV欄位的數值
boston = pd.DataFrame(boston_dataset.data, columns=boston_dataset.feature_names)
boston.head()


# In[77]:


#在 boston DataFrame 中新增'MEDV'，並將'MEDV'的值設定為從 boston_dataset.target獲取的房屋中位數價格

boston['MEDV'] = boston_dataset.target


# In[78]:


#資料集中沒有缺失值
boston.isnull().sum()


# In[79]:


import xgboost as xgb
from sklearn.model_selection import KFold, cross_val_score
from sklearn.metrics import mean_squared_error, r2_score

X = boston_dataset.data
y = boston_dataset.target

kf = KFold(n_splits=5, shuffle=True, random_state=42)


# In[94]:


import numpy as np
from sklearn.metrics import mean_squared_error, r2_score
import xgboost as xgb
from sklearn.model_selection import KFold

#定義 MAPE 函數
def mean_absolute_percentage_error(y_true, y_pred): 
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

rmse_scores = []
r2_scores = []
mape_scores = []

#K-fold cross validation
kf = KFold(n_splits=5, shuffle=True, random_state=42)
for train_index, test_index in kf.split(X):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]

    model = xgb.XGBRegressor(objective ='reg:squarederror')
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)
    mape = mean_absolute_percentage_error(y_test, y_pred)

    rmse_scores.append(rmse)
    r2_scores.append(r2)
    mape_scores.append(mape)

# 計算平均
avg_rmse = np.mean(rmse_scores)
avg_r2 = np.mean(r2_scores)
avg_mape = np.mean(mape_scores)

print("RMSE scores for each fold:", rmse_scores)
print("R2 scores for each fold:", r2_scores)
print("MAPE scores for each fold:", mape_scores)
print("Average RMSE:", avg_rmse)
print("Average R2:", avg_r2)
print("Average MAPE:", avg_mape)


# In[83]:


import xgboost as xgb
from sklearn.datasets import load_boston
import matplotlib.pyplot as plt

feature_importances = model.feature_importances_
features = boston_dataset.feature_names
sorted_indices = feature_importances.argsort()[::-1]

plt.figure(figsize=(12, 6))
plt.bar(range(len(feature_importances)), feature_importances[sorted_indices])
plt.xticks(range(len(feature_importances)), features[sorted_indices], rotation=90)
plt.xlabel('Features')
plt.ylabel('Importance Score')
plt.title('Feature Importance in Boston Housing Dataset (Sorted)')
plt.show()


# In[92]:


boston_2 = pd.DataFrame(boston_dataset.data, columns=boston_dataset.feature_names)
boston_2['MEDV'] = boston_dataset.target

# 移除重要性倒數三個的特徵
boston_2 = boston_2.drop(['ZN', 'CHAS', 'RAD'], axis=1)

X_2 = boston_2.drop('MEDV', axis=1)
y_2 = boston_2['MEDV']
boston_2.head()


# In[93]:


def mean_absolute_percentage_error(y_true, y_pred): 
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100


rmse_scores_2 = []
r2_scores_2 = []
mape_scores_2 = []

kf = KFold(n_splits=5, shuffle=True, random_state=42)
for train_index, test_index in kf.split(X_2):
    X_train_fold, X_test_fold = X_2.iloc[train_index], X_2.iloc[test_index]
    y_train_fold, y_test_fold = y_2.iloc[train_index], y_2.iloc[test_index]

    model_2 = xgb.XGBRegressor(objective ='reg:squarederror')
    model_2.fit(X_train_fold, y_train_fold)
    y_pred_fold = model_2.predict(X_test_fold)

    rmse_2 = np.sqrt(mean_squared_error(y_test_fold, y_pred_fold))
    r2_2 = r2_score(y_test_fold, y_pred_fold)
    mape_2 = mean_absolute_percentage_error(y_test_fold, y_pred_fold)

    rmse_scores_2.append(rmse_2)
    r2_scores_2.append(r2_2)
    mape_scores_2.append(mape_2)

#計算平均
avg_rmse_2 = np.mean(rmse_scores_2)
avg_r2_2 = np.mean(r2_scores_2)
avg_mape_2 = np.mean(mape_scores_2)

print("RMSE scores for each fold:", rmse_scores_2)
print("R2 scores for each fold:", r2_scores_2)
print("MAPE scores for each fold:", mape_scores_2)
print("Average RMSE:", avg_rmse_2)
print("Average R2:", avg_r2_2)
print("Average MAPE:", avg_mape_2)


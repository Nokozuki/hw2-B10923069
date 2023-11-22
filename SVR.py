import pandas as pd
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GridSearchCV
# 用於驗證
param_grid = {
    'C': [0.1, 1],  
    'epsilon': [0.1, 1]
}

df_train = pd.read_csv('/data/notebook_files/adult_train_processed.csv')
df_test = pd.read_csv('/data/notebook_files/adult_test_processed.csv')

X_train = df_train.drop(df_train.columns[-3], axis=1)
y_train = df_train.iloc[:, -3]
X_test = df_test.drop(df_test.columns[-3], axis=1)
y_test = df_test.iloc[:, -3]
# 交叉驗證
grid_search = GridSearchCV(SVR(kernel='rbf'), param_grid, cv=5, scoring='neg_mean_squared_error', verbose=2)

grid_search.fit(X_train, y_train)

# 最佳參數
print("最佳參數:", grid_search.best_params_)

# 最佳模型
best_svr = grid_search.best_estimator_

predictions = best_svr.predict(X_test)

mse = mean_squared_error(y_test, predictions)
print(f"MSE: {mse}")
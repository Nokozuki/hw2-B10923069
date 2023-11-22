import pandas as pd
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error
# 用於驗證
df_train = pd.read_csv('/data/notebook_files/adult_train_processed.csv')
df_test = pd.read_csv('/data/notebook_files/adult_test_processed.csv')

X_train = df_train.drop(df_train.columns[-3], axis=1)
y_train = df_train.iloc[:, -3]
X_test = df_test.drop(df_test.columns[-3], axis=1)
y_test = df_test.iloc[:, -3]
# 交叉驗證
svr = SVR(C= 1,epsilon=1)
svr.fit(X_train,y_train)
predictions = svr.predict(X_test)

mse = mean_squared_error(y_test, predictions)
print(f"MSE: {mse}")
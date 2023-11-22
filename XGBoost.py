import xgboost as xgb
from sklearn.metrics import mean_squared_error
import pandas as pd
df_train = pd.read_csv('/data/notebook_files/adult_train_processed.csv')
df_test = pd.read_csv('/data/notebook_files/adult_test_processed.csv')

X_train = df_train.drop(df_train.columns[-3], axis=1)
y_train = df_train.iloc[:, -3]
X_test = df_test.drop(df_test.columns[-3], axis=1)
y_test = df_test.iloc[:, -3]

model = xgb.XGBRegressor(objective='reg:squarederror', random_state=42)

model.fit(X_train, y_train)

y_pred = model.predict(X_test)


mse = mean_squared_error(y_test, y_pred)
print(f"MSE: {mse}")
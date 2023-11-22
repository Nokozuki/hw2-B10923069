from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import accuracy_score
import pandas as pd
import numpy as np

# 用於驗證
param_dist = {
    'n_estimators': [100, 150 , 200],
    'max_features': ['auto', 'sqrt', 'log2'],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'bootstrap': [True, False]
}

rf = RandomForestClassifier(random_state=42)

rf_random = RandomizedSearchCV(estimator=rf, param_distributions=param_dist, n_iter=20, cv=3, verbose=2, random_state=42, n_jobs=-1)

df_train = pd.read_csv('/data/notebook_files/adult_train_processed.csv')
df_test = pd.read_csv('/data/notebook_files/adult_test_processed.csv')

X_train = df_train.drop(df_train.columns[-3], axis=1)
y_train = df_train.iloc[:, -3]
X_test = df_test.drop(df_test.columns[-3], axis=1)
y_test = df_test.iloc[:, -3]

rf_random.fit(X_train, y_train)

best_rf = rf_random.best_estimator_
predictions = best_rf.predict(X_test)

accuracy = accuracy_score(y_test, predictions)
print(f"最佳參數: {rf_random.best_params_}")
print(f"準確度: {accuracy}")

# Best Parameters: {'n_estimators': 150, 'min_samples_split': 5, 'min_samples_leaf': 1, 'max_features': 'log2', 'max_depth': 10, 'bootstrap': False}
# Accuracy with Best Parameters: 0.4759944219403679
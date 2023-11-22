from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import pandas as pd

rf = RandomForestClassifier(n_estimators= 150,min_samples_split = 5,min_samples_leaf= 1
, max_features = 'log2' ,max_depth= 10,bootstrap=False)
df_train = pd.read_csv('/data/notebook_files/adult_train_processed.csv')
df_test = pd.read_csv('/data/notebook_files/adult_test_processed.csv')

X_train = df_train.drop(df_train.columns[-3], axis=1)
y_train = df_train.iloc[:, -3]
X_test = df_test.drop(df_test.columns[-3], axis=1)
y_test = df_test.iloc[:, -3]

rf.fit(X_train, y_train)
predictions = rf.predict(X_test)

accuracy = accuracy_score(y_test, predictions)
print(f"準確度: {accuracy}")
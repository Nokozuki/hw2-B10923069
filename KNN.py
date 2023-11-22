from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import pandas as pd
df_train = pd.read_csv('/data/notebook_files/adult_train_processed.csv')
df_test = pd.read_csv('/data/notebook_files/adult_test_processed.csv')

X_train = df_train.drop(df_train.columns[-3], axis=1)
y_train = df_train.iloc[:, -3]
X_test = df_test.drop(df_test.columns[-3], axis=1)
y_test = df_test.iloc[:, -3]

knn = KNeighborsClassifier(n_neighbors=50)

# 訓練模型
knn.fit(X_train, y_train)

# 預測
predictions = knn.predict(X_test)

# 準確率
accuracy = accuracy_score(y_test, predictions)
print(f"Accuracy: {accuracy}")
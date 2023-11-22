import pandas as pd
from sklearn.preprocessing import LabelEncoder

dataset = pd.read_csv('/data/notebook_files/adult_test_processed.csv')

dataset = dataset.replace(' ?',None)

dataset.columns = ['age','workclass','fnlwgt','education','education-num',
                   'marital-status','occupation','relationship','race',
                   'sex','capital-gain','capital-loss','hours-per-week','native-country',
                   'income']

dataset = dataset.fillna({'native-country':' United-States'})
dataset = dataset.dropna()
dataset.info()

label_encoder = LabelEncoder()
dataset['income'] = label_encoder.fit_transform(dataset['income'])
dataset['workclass']=label_encoder.fit_transform(dataset['workclass'])
dataset['education']=label_encoder.fit_transform(dataset['education'])
dataset['marital-status']=label_encoder.fit_transform(dataset['marital-status'])
dataset['occupation']=label_encoder.fit_transform(dataset['occupation'])
dataset['relationship']=label_encoder.fit_transform(dataset['relationship'])
dataset['race']=label_encoder.fit_transform(dataset['race'])
dataset['sex']=label_encoder.fit_transform(dataset['sex'])
dataset['native-country']=label_encoder.fit_transform(dataset['native-country'])

dataset.to_csv('/data/notebook_files/adult_test_processed.csv',index=False)


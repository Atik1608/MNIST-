import matplotlib.pyplot as plt
from sklearn.neural_network import MLPClassifier
from sklearn.cross_validation import train_test_split
from sklearn.metrics import accuracy_score
import pandas as pd
from sklearn.externals import joblib


data = pd.read_csv('mnist.csv')

df_features = data.iloc[:,1:]
df_labels = data.iloc[:,0]

features_train, features_test, labels_train, labels_test = train_test_split(df_features, df_labels, test_size=0.2, random_state=4)

clf = MLPClassifier()

clf.fit(features_train,labels_train)

pred = clf.predict(features_test)

print('Predictions: ' , pred)

accuracy = accuracy_score(pred,labels_test)
print("The accuracy of the model is: ", accuracy)

filename = 'finalized_model_neural.sav'
joblib.dump(clf,filename)

"""loaded_model = joblib.load(filename)
result = loaded_model.score(features_test,labels_test)
print(result)"""

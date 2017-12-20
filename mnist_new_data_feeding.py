"""Load pre trained model to check the accuracy by feeding the new data - using scikit learn built-in joblib"""

from sklearn.externals import joblib
import pandas as pd

data = pd.read_csv('mnist.csv')
features_test = data.iloc[:20,1:]
labels_test = data.iloc[:20,0]



load_model = joblib.load('finalized_model.sav')
new_data_result = load_model.score(features_test,labels_test)

print("After feeding some new data the score still remains: " , new_data_result)

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
data = pd.read_excel('iris-train.xlsx')
print(data.head())
# All columns except the last as features
x = data.iloc[:, :-1]
y = data.iloc[:, -1]   # Last column as label   
print(x.values)
print(y.values)
model = RandomForestClassifier()
model.fit(x.values, y.values)
print(model.predict([[5.4,3, 4.5, 1.5]]))  # Example prediction for a new sample
data_test = pd.read_excel('iris-test.xlsx')
x_test = data_test.iloc[:, :-1]
predictions = model.predict(x_test.values)  # Predict on the test data
print(predictions)  # Display predictions

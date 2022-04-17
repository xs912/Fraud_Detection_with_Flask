import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

data = pd.read_csv("transactions.csv")
type = data["type"].value_counts()
transactions = type.index
quantity = type.values

# Check for correlation between the amount and the type of transaction
correlation = data.corr()
print(correlation["isFraud"].sort_values(ascending=False))

data["type"] = data["type"].map({"CASH_OUT": 1, "PAYMENT": 2, "CASH_IN": 3, "TRANSFER": 4, "DEBIT": 5})
data["isFraud"] = data["isFraud"].map({0: "No Fraud", 1: "Fraud"})
print(data.head())

# Splitting the data into train and test
x = np.array(data[["type", "amount", "oldbalanceOrg", "newbalanceOrig"]])
y = np.array(data[["isFraud"]])

#Training the model
xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size=0.10, random_state=42)
model = DecisionTreeClassifier()
model.fit(xtrain, ytrain)
print(model.score(xtest, ytest))
import pickle
pickle.dump(model, open('model.pkl','wb'))

#Prediction on new data
features = np.array([[4, 3260.54, 2670.54, 1000.0]])
prediction = model.predict(features)
print(prediction[0])

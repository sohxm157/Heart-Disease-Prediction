import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Loading the csv data to pandas DataFrame
heart_disease = pd.read_csv("heart_disease_data.csv")

# Splitting data into training and testing sets (common split is 80% training, 20% testing)
X = heart_disease.drop(columns='target', axis=1)  # Features
Y = heart_disease['target']                        # Target variable
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

# Training the Logistic Regression with Training Data
model = LogisticRegression()
model.fit(X_train, Y_train)

# Accuracy on Training Data
X_train_prediction = model.predict(X_train)
training_data_accuracy = accuracy_score(X_train_prediction, Y_train)

print('Accuracy on Training data :', training_data_accuracy)

# Accuracy on Test Data
X_test_prediction = model.predict(X_test)
testing_data_accuracy = accuracy_score(X_test_prediction, Y_test)

print('Accuracy on Testing data :', testing_data_accuracy)

# Sample input data
input_data = (56,1,1,120,236,0,1,178,0,0.8,2,0,2)

# Change the input data to numpy Array
input_data_as_numpy_array= np.asarray(input_data)

# Reshape the numpy array as we are predicting for only on instance
input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)

prediction = model.predict(input_data_reshaped)
print(prediction)

if(prediction == [0]):
  print('The Person does not have a Heart Disease')
else:
  print('The Person has Heart Disease')

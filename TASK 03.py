#!/usr/bin/env python
# coding: utf-8

# In[5]:


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder
from zipfile import ZipFile
import urllib.request
from io import BytesIO

# Download the Bank Marketing dataset zip file
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00222/bank-additional.zip"
response = urllib.request.urlopen(url)
zip_file = ZipFile(BytesIO(response.read()))

# Extract the zip file and load the dataset
bank_data = pd.read_csv(zip_file.open('bank-additional/bank-additional-full.csv'), sep=';')

# Display the first few rows of the dataset to understand its structure
print(bank_data.head())

# Data preprocessing
# Assuming 'y' is the target variable indicating purchase ('yes' or 'no')
# Encoding categorical variables
label_encoder = LabelEncoder()
bank_data['y'] = label_encoder.fit_transform(bank_data['y'])

# Select features and target
X = bank_data.drop('y', axis=1)  # Features
y = bank_data['y']  # Target

# Convert categorical variables into dummy/indicator variables
X = pd.get_dummies(X)

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model training (using Decision Tree Classifier)
model = DecisionTreeClassifier(random_state=42)
model.fit(X_train, y_train)

# Model evaluation
predictions = model.predict(X_test)
accuracy = accuracy_score(y_test, predictions)
print(f"Accuracy: {accuracy:.2f}")

report = classification_report(y_test, predictions)
print("Classification Report:")
print(report)


# In[ ]:





# In[ ]:





# In[ ]:





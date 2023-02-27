
#!/usr/bin/env python
# CY83R-3X71NC710N Copyright 2023

# Import Statements
import pandas as pd
import numpy as np
import sklearn as sk
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt

# Main Code
# Load the data
data = pd.read_csv('https://raw.githubusercontent.com/cy83r-3x71nc710n/Cyb3rf3nZ3/master/data.csv')

# Split the data into features and labels
X = data.iloc[:, :-1].values
y = data.iloc[:, -1].values

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Scale the data
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Train the model
kmeans = KMeans(n_clusters=2, random_state=0)
kmeans.fit(X_train)

# Make predictions
y_pred = kmeans.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)

# Visualize the results
plt.scatter(X_test[:, 0], X_test[:, 1], c=y_pred, cmap='rainbow')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.title('Cyb3rf3nZ3')
plt.show()

# Finishing Touches
print('Accuracy:', accuracy)
print('Confusion Matrix:')
print(conf_matrix)

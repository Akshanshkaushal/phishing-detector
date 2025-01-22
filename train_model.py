import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
import pickle
import os

# Load the dataset
df = pd.read_csv('phishing.csv')

# Separate features and target
X = df.drop(['class', 'Index'], axis=1)
y = df['class']

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and train the model
gbc = GradientBoostingClassifier(random_state=42)
gbc.fit(X_train, y_train)

# Create pickle directory if it doesn't exist
if not os.path.exists('pickle'):
    os.makedirs('pickle')

# Save the model
with open('pickle/model.pkl', 'wb') as file:
    pickle.dump(gbc, file)

# Optional: Print model accuracy
print(f"Model accuracy on test set: {gbc.score(X_test, y_test):.2f}") 
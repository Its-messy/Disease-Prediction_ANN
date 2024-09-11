# import tensorflow as tf
# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import Dense
# import pandas as pd
# import numpy as np
# import joblib
# from sklearn.metrics import accuracy_score
# from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import StandardScaler, LabelEncoder
# from tensorflow.keras.utils import to_categorical
# model = joblib.load('Keras ANN.pkl')

# test_data = pd.read_csv('Training.csv')
# from sklearn.preprocessing import LabelEncoder
# # Drop the unnecessary column
# test_data = test_data.drop(columns=['Unnamed: 133'])
# # Separate features (X) and target (y)
# X = test_data.drop(columns=['prognosis'])
# y = test_data['prognosis']
# # Split the dataset into training and testing sets
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# # Assuming X_test is a pandas DataFrame or a column with categorical data
# label_encoder = LabelEncoder()
# # Fit the encoder on the training data and transform both training and testing data
# y_train_encoded = label_encoder.fit_transform(y_train)
# y_test_encoded = label_encoder.transform(y_test)

# # Now use to_categorical on the encoded labels
# y_train_encoded = to_categorical(y_train_encoded)
# y_test_encoded = to_categorical(y_test_encoded)

# X_test = test_data.iloc[:, :-1]
# y_test = test_data.iloc[:, -1]

# y_pred = model.predict(X_test)
# y_pred_classes = np.argmax(y_pred, axis=1) 


# accuracy= accuracy_score(y_test, y_pred_classes)

# with open('accuracy.txt', 'w') as file:
#     file.write(f'Accuracy: {accuracy}')
 
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import pandas as pd
import numpy as np
import joblib
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from tensorflow.keras.utils import to_categorical

# Load the model (Ensure it is a Keras model and not a joblib model)
model = joblib.load('Keras ANN.pkl')  # Change to your model file format

# Load and preprocess the test data
test_data = pd.read_csv('Training.csv')
test_data = test_data.drop(columns=['Unnamed: 133'])
X = test_data.drop(columns=['prognosis'])
y = test_data['prognosis']

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize the features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Encode the labels
label_encoder = LabelEncoder()
y_train_encoded = label_encoder.fit_transform(y_train)
y_test_encoded = label_encoder.transform(y_test)

# Convert labels to categorical format for Keras
y_train_encoded = to_categorical(y_train_encoded)
y_test_encoded = to_categorical(y_test_encoded)

# Make predictions
y_pred = model.predict(X_test)
y_pred_classes = np.argmax(y_pred, axis=1)

# Convert y_test_encoded back to original labels for accuracy calculation
y_test_classes = np.argmax(y_test_encoded, axis=1)

# Calculate accuracy
accuracy = accuracy_score(y_test_classes, y_pred_classes)

# Save the accuracy to a file
with open('accuracy.txt', 'w') as file:
    file.write(f'Accuracy: {accuracy}\n')

print(f'Accuracy: {accuracy}')

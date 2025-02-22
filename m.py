import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from keras.layers import Dense, Input, BatchNormalization
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from imblearn.over_sampling import SMOTE
df = pd.read_csv("landslide_risk_dataset.csv")
print(df.shape)

# Feature selection
X = df[['Temperature', 'Precipitation', 'Humidity', 'Soil Moisture', 'Elevation']]
y = df['Landslide Risk Prediction']  # Categorical variable

# Normalize numerical features
scaler = MinMaxScaler()
X = scaler.fit_transform(X)

# Encode target variable
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)  # Converts categorical labels to numbers

# Handle class imbalance using SMOTE
smote = SMOTE()
X, y = smote.fit_resample(X, y)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Convert labels to categorical (one-hot encoding)
y_train = tf.keras.utils.to_categorical(y_train, num_classes=4)
y_test = tf.keras.utils.to_categorical(y_test, num_classes=4)

# Save label classes
np.save("label_classes.npy", label_encoder.classes_)

# Build Deep Learning Model
model = Sequential([
    Dense(128, activation="relu", input_shape=(X.shape[1],)),  
    BatchNormalization(),  
    Dropout(0.3),  

    Dense(64, activation="relu"),
    BatchNormalization(),
    Dropout(0.3),

    Dense(32, activation="relu"),
    BatchNormalization(),

    Dense(4, activation="softmax")  # Output layer for 4 risk levels
])

# Compile the model
model.compile(optimizer=Adam(learning_rate=0.0001), loss="categorical_crossentropy", metrics=["accuracy"])

# Train the model
history = model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=50, batch_size=32)

# Save the trained model
model.save("landslide_model.keras")  # Recommended format

# Predict on test data
y_pred = model.predict(X_test)
predicted_classes = np.argmax(y_pred, axis=1)

# Save predictions
np.savetxt("predictions.csv", predicted_classes, delimiter=",", fmt="%d", header="Predicted Risk Level")

print("Model training complete. Accuracy results will be evaluated separately.")



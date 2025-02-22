import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.callbacks import ReduceLROnPlateau
#from tensorflow.keras.optimizers import Adam
import os

os.environ["TF_DETERMINISTIC_OPS"] = "1"


# Load dataset
df = pd.read_csv("landslide_risk_dataset.csv")  # Change to your actual dataset path

# Encode 'Risk' if it's categorical
le = LabelEncoder()
df['Landslide Risk Prediction'] = le.fit_transform(df['Landslide Risk Prediction'])  # Convert labels to 0, 1, 2 (Low, Medium, High)

# Select Features & Target
X = df[['Temperature', 'Precipitation', 'Humidity', 'Soil Moisture', 'Elevation']]
y = df['Landslide Risk Prediction']

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
np.save("X_test.npy", X_test)
np.save("y_test.npy", y_test)

# Feature Scaling
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Define Neural Network Model
model = Sequential([
    Dense(64, activation='relu', input_shape=(X_train.shape[1],)),  # Input Layer
    Dropout(0.15),  # Regularization
    Dense(32, activation='relu'),  # Hidden Layer
    Dropout(0.15),
    Dense(16, activation='relu'),  # Another Hidden Layer
    Dense(len(le.classes_), activation='softmax')  # Output Layer (for Multi-Class Classification)
])

# Compile Model
#model.compile(optimizer=Adam(learning_rate=0.001), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
optimizer = SGD(learning_rate=0.01, momentum=0.9)
model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy'])
# Model Summary
model.summary()

# Train the Model
history = model.fit(X_train, y_train, epochs=150, batch_size=24, validation_data=(X_test, y_test), shuffle=False)  
lr_reduction = ReduceLROnPlateau(monitor='val_accuracy', patience=10, factor=0.5, min_lr=1e-6)
# Evaluate Accuracy
test_loss, test_acc = model.evaluate(X_test, y_test)
print(f"Test Accuracy: {test_acc:.4f}")

# Plot Training Performance
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.legend()
plt.show()

# Save model and label encoder
model.save("lspmodel.keras")
np.save("label_classes.npy", le.classes_)  # Save label encoder classes
np.save("scaler.npy", scaler.mean_)  # Save scaler for transformation
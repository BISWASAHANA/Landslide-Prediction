import numpy as np
import tensorflow as tf
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Load the trained model
model = tf.keras.models.load_model("lspmodel.keras")

# Load test dataset
X_test = np.load("X_test.npy")
y_test = np.load("y_test.npy")

# Load label classes
label_classes = np.load("label_classes.npy", allow_pickle=True)

# Make predictions
y_pred_probs = model.predict(X_test)
y_pred_classes = np.argmax(y_pred_probs, axis=1)

# Convert categorical labels back if needed
y_test_labels = np.array([label_classes[i] for i in y_test])
y_pred_labels = np.array([label_classes[i] for i in y_pred_classes])

# Accuracy calculation
accuracy = accuracy_score(y_test_labels, y_pred_labels)
classification_rep = classification_report(y_test_labels, y_pred_labels)
cm = confusion_matrix(y_test_labels, y_pred_labels)

# Save results to a file
with open("er.txt", "w") as f:
    f.write(f"Model Accuracy: {accuracy:.4f}\n\n")
    f.write("Classification Report:\n")
    f.write(classification_rep)
    f.write("\nConfusion Matrix:\n")
    np.savetxt(f, cm, fmt="%d")

# Plot Confusion Matrix
plt.figure(figsize=(6,4))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=label_classes, yticklabels=label_classes)
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.savefig("confusion_matrix.png")  # Save confusion matrix as an image
plt.show()

print("Evaluation completed! Results saved to 'er.txt' and 'confusion_matrix2.png'")

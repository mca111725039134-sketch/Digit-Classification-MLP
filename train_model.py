import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
# Load dataset
df = pd.read_csv("data/mnist_train.csv")

# Features and labels
X = df.drop("label", axis=1)
y = df["label"]

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# Create MLP model
mlp = MLPClassifier(
    hidden_layer_sizes=(128, 64),
    max_iter=50,
    random_state=42
)

print("Training started...")

# Train model
mlp.fit(X_train, y_train)

joblib.dump(mlp, "model/digit_model.pkl")

print("Model Saved Successfully!")
print("Training completed!")

# Predict test data
y_pred = mlp.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)

print("Model Accuracy:", accuracy * 100, "%")

print("\nClassification Report:\n")
print(classification_report(y_test, y_pred))
# Create Confusion Matrix
cm = confusion_matrix(y_test, y_pred)

# Display Confusion Matrix
plt.figure(figsize=(8, 6))
plt.imshow(cm, cmap="Blues")
plt.title("Confusion Matrix")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.colorbar()

plt.savefig("screenshots/confusion_matrix.png")
plt.show()
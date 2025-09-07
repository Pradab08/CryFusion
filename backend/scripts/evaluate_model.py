import os
import joblib
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from tensorflow.keras.models import load_model
from sklearn.model_selection import train_test_split

# -----------------------------
# Paths & Config
# -----------------------------
OUTPUT_PATH = "backend/data/Processed Baby Cry Sence Dataset"
MODEL_PATH = "models/cnn_lstm_frozen.keras"
CLASSES = ["belly_pain", "burping", "discomfort", "hungry", "tired"]

# -----------------------------
# Load Data & Model
# -----------------------------
# Load dataset
X, y = joblib.load(os.path.join(OUTPUT_PATH, "features.pkl"))
print("✅ Loaded dataset:", X.shape, y.shape)

# Train-validation split (same as training)
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)
print("✅ Validation set:", X_val.shape, y_val.shape)

# Expand dims (same preprocessing as training)
X_val = np.expand_dims(X_val, -1)

# Load trained model
model = load_model(MODEL_PATH)
print(f"✅ Loaded model from {MODEL_PATH}")

# -----------------------------
# Evaluate on Validation Set
# -----------------------------
# Predictions
y_pred_probs = model.predict(X_val, verbose=0)
y_pred = np.argmax(y_pred_probs, axis=1)

# Accuracy
acc = accuracy_score(y_val, y_pred)
print(f"\n🔍 Validation Accuracy: {acc:.4f} ({acc*100:.1f}%)\n")

# Classification report
print("📋 Classification Report (Validation Set):\n")
print(classification_report(y_val, y_pred, target_names=CLASSES, digits=4))

# Prediction distribution
unique, counts = np.unique(y_pred, return_counts=True)
print("\n📊 Prediction Distribution (Validation Set):")
for cls, cnt in zip(unique, counts):
    print(f"  {CLASSES[cls]:<12}: {cnt}/{len(y_pred)} ({(cnt/len(y_pred))*100:.1f}%)")

# Confusion matrix
cm = confusion_matrix(y_val, y_pred)
plt.figure(figsize=(7, 6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=CLASSES, yticklabels=CLASSES)
plt.xlabel("Predicted")
plt.ylabel("True")
plt.title("Confusion Matrix (Validation Set)")
plt.tight_layout()
plt.show()

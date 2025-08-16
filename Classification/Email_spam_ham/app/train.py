# -----------------------------
# Email Spam/Ham Classification
# Production-ready Pipeline
# -----------------------------

import os
import pandas as pd
import numpy as np
import joblib
from sentence_transformers import SentenceTransformer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix
)

# -----------------------------
# Config & Paths
# -----------------------------
DATA_PATH = "data/processed_data.csv"
MODEL_DIR = "app/model"
EMBEDDER_PATH = os.path.join(MODEL_DIR, "sentence_transformer_model")
CLASSIFIER_PATH = os.path.join(MODEL_DIR, "spam_classifier.pkl")

os.makedirs(MODEL_DIR, exist_ok=True)

# -----------------------------
# Load Data
# -----------------------------
df = pd.read_csv(DATA_PATH)
x = df["cleaning_text"]
y = df["Category"].values

# -----------------------------
# Embedding
# -----------------------------
print("[INFO] Loading sentence transformer...")
embedder = SentenceTransformer('all-MiniLM-L6-v2')
X_embeddings = embedder.encode(x.astype(str).tolist(), convert_to_tensor=False)

# -----------------------------
# Train/Test Split
# -----------------------------
x_train, x_test, y_train, y_test = train_test_split(
    X_embeddings, y, test_size=0.25, random_state=42
)

# -----------------------------
# Model Training
# -----------------------------
print("[INFO] Training classifier...")
classifier = LogisticRegression(max_iter=1000)
classifier.fit(x_train, y_train)

# -----------------------------
# Evaluation
# -----------------------------
y_pred = classifier.predict(x_test)

print("📊 Accuracy:", accuracy_score(y_test, y_pred))
print("🎯 Precision (macro):", precision_score(y_test, y_pred, average='macro'))
print("📈 Recall (macro):", recall_score(y_test, y_pred, average='macro'))
print("📏 F1 Score (macro):", f1_score(y_test, y_pred, average='macro'))

print("\n📝 Classification Report:\n", classification_report(y_test, y_pred))

cm = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:\n", cm)

# -----------------------------
# Save Models
# -----------------------------
print("[INFO] Saving models...")
joblib.dump(classifier, CLASSIFIER_PATH)
embedder.save(EMBEDDER_PATH)

print("[INFO] Training pipeline complete. Models saved!")

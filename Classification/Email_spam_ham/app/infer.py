import os
import joblib
from sentence_transformers import SentenceTransformer

# Get directory of THIS file (infer.py)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Paths relative to infer.py
clf_path = os.path.join(BASE_DIR, "model", "spam_classifier.pkl")
embed_path = os.path.join(BASE_DIR, "model", "sentence_transformer_model")

# Load models
clf = joblib.load(clf_path)
embedder = SentenceTransformer(embed_path)

def predict_email(text: str):
    embedding = embedder.encode([text])
    prediction = clf.predict(embedding)[0]
    label_map = {0: "Ham (Not Spam)", 1: "Spam"}
    print("text:",text, "result: ", label_map[int(prediction)])
    return label_map.get(prediction,  label_map[int(prediction)])

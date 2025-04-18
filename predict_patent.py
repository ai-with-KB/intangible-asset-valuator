import pandas as pd
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier

# Load data for fitting the vectorizer
df = pd.read_csv("ai_finance_patents_cleaned.csv")
df["text"] = df["title"].fillna("") + " " + df["abstract"].fillna("")

# Fit vectorizer
vectorizer = TfidfVectorizer(max_features=3000)
vectorizer.fit(df["text"])

# Train model again (optional, for demo)
label_map = {'low': 0, 'medium': 1, 'high': 2}
df["label"] = df["citation_category"].map(label_map)
X_vec = vectorizer.transform(df["text"])
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_vec, df["label"])

# Save both model and vectorizer (optional)
joblib.dump(model, "model.pkl")
joblib.dump(vectorizer, "vectorizer.pkl")

# Get input
title = input("Enter patent title: ")
abstract = input("Enter patent abstract: ")

# Predict
text = title + " " + abstract
text_vec = vectorizer.transform([text])
pred = model.predict(text_vec)

# Reverse map
reverse_map = {0: "low", 1: "medium", 2: "high"}
print("🚀 Predicted Citation Category:", reverse_map[pred[0]])

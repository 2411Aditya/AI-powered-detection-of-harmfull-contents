import os
import pickle
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score

# Import features and labels
from tf_idf import X_vec, y

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X_vec, y, test_size=0.2, random_state=42)

# Initialize and train the model
model = LogisticRegression()
model.fit(X_train, y_train)

# Predict on test data
y_pred = model.predict(X_test)

# Evaluation
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))

# Define path to save model outside the utils folder
current_dir = os.path.dirname(os.path.abspath(__file__))           # path to ml_layer1.py
parent_dir = os.path.dirname(current_dir)                          # go one level up (project root)
model_dir = os.path.join(parent_dir, "models")                     # project_root/models
os.makedirs(model_dir, exist_ok=True)

# Save the model
model_path = os.path.join(model_dir, "logistic_model.pkl")
with open(model_path, "wb") as f:
    pickle.dump(model, f)

print(f"Model saved to {model_path}")

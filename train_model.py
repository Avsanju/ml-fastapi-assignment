# train_model.py
import joblib
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# 1. Load dataset
iris = load_iris()
X, y = iris.data, iris.target
class_names = iris.target_names.tolist()

# 2. Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 3. Train model
model = LogisticRegression(max_iter=200)
model.fit(X_train, y_train)

# 4. Evaluate
y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {acc:.2f}")

# 5. Save model + metadata
joblib.dump({
    "model": model,
    "class_names": class_names,
    "features": iris.feature_names,
    "accuracy": acc
}, "model.pkl")

print("Model saved as model.pkl")

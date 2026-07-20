"""
Project 2: Data Classification Using AI
DecodeLabs — Artificial Intelligence Industrial Training Kit

Goal: Build a basic classification model using the Iris dataset.
Pipeline (IPO Framework from the training deck):
  INPUT   -> Load Iris dataset + Feature Scaling
  PROCESS -> Train-Test Split + KNN Algorithm
  OUTPUT  -> Confusion Matrix + F1 Score
"""

import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    accuracy_score,
    f1_score,
)

# -------------------------------------------------------------
# STEP 1: INPUT — Load and understand the dataset
# -------------------------------------------------------------
iris = load_iris()
X = pd.DataFrame(iris.data, columns=iris.feature_names)   # 4 features
y = pd.Series(iris.target, name="species")                # 3 classes

print("Dataset shape:", X.shape)
print("Classes:", list(iris.target_names))
print(X.head(), "\n")

# -------------------------------------------------------------
# STEP 2: PROCESS — Train/Test split (shuffle to remove order bias)
# -------------------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,      # 80% train / 20% test, as in "THE FULL ARCHITECTURE" slide
    random_state=42,    # reproducibility
    stratify=y          # keep class balance across splits
)

# -------------------------------------------------------------
# STEP 3: INPUT (Gatekeeper Rule) — Feature scaling
# StandardScaler -> mean = 0, variance = 1
# -------------------------------------------------------------
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)   # fit only on train data
X_test_scaled = scaler.transform(X_test)         # transform test with same scaler

# -------------------------------------------------------------
# STEP 4: PROCESS — Apply KNN classification algorithm
# -------------------------------------------------------------
model = KNeighborsClassifier(n_neighbors=5)   # INSTANTIATE
model.fit(X_train_scaled, y_train)            # FIT (memorize the map)
predictions = model.predict(X_test_scaled)    # PREDICT (apply logic)

# -------------------------------------------------------------
# STEP 5: OUTPUT — Validation (never trust accuracy alone)
# -------------------------------------------------------------
print("Accuracy:", accuracy_score(y_test, predictions))
print("F1 Score (macro):", f1_score(y_test, predictions, average="macro"))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, predictions))

print("\nClassification Report:")
print(classification_report(y_test, predictions, target_names=iris.target_names))

# -------------------------------------------------------------
# BONUS: Finding the optimal K ("Tuning the Engine" slide)
# -------------------------------------------------------------
print("\nOptimal K search:")
best_k, best_score = 1, 0
for k in range(1, 21):
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train_scaled, y_train)
    score = knn.score(X_test_scaled, y_test)
    if score > best_score:
        best_k, best_score = k, score

print(f"Best K = {best_k} with accuracy = {best_score:.4f}")
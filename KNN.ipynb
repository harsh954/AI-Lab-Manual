import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (
    classification_report,
    accuracy_score,
    confusion_matrix
)
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

# ---------------------------------------------------
# CONFIG
# ---------------------------------------------------
CSV_FILE = r"c:\Users\samsu\Downloads\AI Lab\AI Lab\cleaned.csv"

# Target to predict (you can change this if needed)
TARGET = "file_type_guess"

# Use subset for TRAINING (for speed), but predict on ALL rows later
MAX_ROWS_TRAIN = 5000

MODEL_FILE = rf"c:\Users\samsu\Downloads\AI Lab\AI Lab\knn_{TARGET}.pkl"
PREDICTIONS_FILE = rf"c:\Users\samsu\Downloads\AI Lab\AI Lab\cleaned_with_{TARGET}_knn_pred.csv"
CONFUSION_PNG = rf"c:\Users\samsu\Downloads\AI Lab\AI Lab\knn_{TARGET}_confusion_matrix.png"

# ---------------------------------------------------
# 1. LOAD DATA FOR TRAINING
# ---------------------------------------------------
df = pd.read_csv(CSV_FILE)

print(f"Loaded: {CSV_FILE}")
print("Original shape:", df.shape)
print("\nColumns:", df.columns.tolist())

# Optional downsample for training (KNN is slow on 1M+ rows)
if len(df) > MAX_ROWS_TRAIN:
    df_train = df.sample(n=MAX_ROWS_TRAIN, random_state=42).copy()
    print(f"\nDownsampled for training to: {df_train.shape}")
else:
    df_train = df.copy()

# ---------------------------------------------------
# 2. BASIC CLEANING / FEATURE ENGINEERING (TRAINING DATA)
# ---------------------------------------------------
# Parse datetime
df_train["first_seen_utc"] = pd.to_datetime(df_train["first_seen_utc"], errors="coerce")

# Drop rows with invalid timestamp or missing target
df_train = df_train.dropna(subset=["first_seen_utc", TARGET])

# Time-based numeric features
df_train["fs_year"] = df_train["first_seen_utc"].dt.year
df_train["fs_month"] = df_train["first_seen_utc"].dt.month
df_train["fs_day"] = df_train["first_seen_utc"].dt.day
df_train["fs_hour"] = df_train["first_seen_utc"].dt.hour

# Target
y = df_train[TARGET].astype(str)

# Categorical features to use (like first KNN script: mixed meta fields)
cat_features = ["reporter", "file_name", "mime_type", "signature"]
cat_features = [c for c in cat_features if c in df_train.columns]

# Numeric features
num_features = ["fs_year", "fs_month", "fs_day", "fs_hour"]

feature_cols = cat_features + num_features
X = df_train[feature_cols].copy()

print("\nUsing target:", TARGET)
print("Categorical features:", cat_features)
print("Numeric features:", num_features)
print("X shape:", X.shape, "y length:", len(y))
print("Unique classes in target:", y.nunique())

# ---------------------------------------------------
# 3. PREPROCESSOR + KNN MODEL (LIKE FIRST KNN CODE)
# ---------------------------------------------------
# Numeric pipeline: impute + scale
num_pipe = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

# Categorical pipeline: impute + one-hot
cat_pipe = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("ohe", OneHotEncoder(handle_unknown="ignore"))
])

preprocessor = ColumnTransformer(
    transformers=[
        ("num", num_pipe, num_features),
        ("cat", cat_pipe, cat_features),
    ],
    remainder="drop"
)

knn = KNeighborsClassifier(
    n_neighbors=3,      # k (reduced for speed)
    weights="distance", # closer neighbors weigh more
    metric="minkowski", # Euclidean
    n_jobs=-1           # use all CPU cores
)

model = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("knn", knn),
    ]
)

# ---------------------------------------------------
# 4. TRAIN / TEST SPLIT
# ---------------------------------------------------
value_counts = y.value_counts()
min_count = value_counts.min()
print("\nNumber of classes:", len(value_counts), "Min class count:", min_count)

if min_count < 2:
    print("Not using stratify because some classes have only 1 sample.")
    stratify_arg = None
else:
    stratify_arg = y

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=stratify_arg,
)

print(f"Training samples: {len(y_train)}, Test samples: {len(y_test)}")

# ---------------------------------------------------
# 5. TRAIN KNN
# ---------------------------------------------------
print("\nTraining KNN classifier...")
model.fit(X_train, y_train)

# ---------------------------------------------------
# 6. EVALUATE (ACCURACY, REPORT, CONFUSION MATRIX, CLASS DISTRIBUTION)
# ---------------------------------------------------
y_pred = model.predict(X_test)

acc = accuracy_score(y_test, y_pred)
print("\nAccuracy on test set:", acc)
print("\nClassification report:")
print(classification_report(y_test, y_pred, zero_division=0))

# Confusion matrix
print("\nComputing confusion matrix...")
labels = sorted(y.unique())
cm = confusion_matrix(y_test, y_pred, labels=labels)

plt.figure(figsize=(10, 8))
sns.heatmap(
    cm,
    annot=False,
    fmt="d",
    cmap="Blues",
    xticklabels=labels,
    yticklabels=labels
)
plt.title(f"KNN Confusion Matrix - {TARGET}")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.tight_layout()
plt.savefig(CONFUSION_PNG, dpi=150)
plt.close()
print(f"Confusion matrix heatmap saved as: {CONFUSION_PNG}")

# Class distribution in test set (true vs predicted) – like first code
plt.figure(figsize=(8, 4))
sns.countplot(x=y_test, order=labels)
plt.title("True Class Distribution in Test Set")
plt.xticks(rotation=45, ha="right")
plt.tight_layout()
plt.show()

plt.figure(figsize=(8, 4))
sns.countplot(x=y_pred, order=labels)
plt.title("Predicted Class Distribution")
plt.xticks(rotation=45, ha="right")
plt.tight_layout()
plt.show()

# ---------------------------------------------------
# 7. SAVE MODEL
# ---------------------------------------------------
joblib.dump(model, MODEL_FILE)
print(f"\nSaved KNN model to: {MODEL_FILE}")

# ---------------------------------------------------
# 8. USE MODEL TO PREDICT FOR SAMPLE OF CSV (SKIP FULL FOR SPEED)
# ---------------------------------------------------
print("\nPredicting on a small sample (1000 rows) for demonstration...")
df_sample = pd.read_csv(CSV_FILE, nrows=1000)

# Feature engineering on sample data
df_sample["first_seen_utc"] = pd.to_datetime(df_sample["first_seen_utc"], errors="coerce")

# Fill missing time parts with 0 (still usable after scaling)
df_sample["fs_year"] = df_sample["first_seen_utc"].dt.year.fillna(0).astype(int)
df_sample["fs_month"] = df_sample["first_seen_utc"].dt.month.fillna(0).astype(int)
df_sample["fs_day"] = df_sample["first_seen_utc"].dt.day.fillna(0).astype(int)
df_sample["fs_hour"] = df_sample["first_seen_utc"].dt.hour.fillna(0).astype(int)

# Make sure all categorical feature columns exist
for col in cat_features:
    if col not in df_sample.columns:
        df_sample[col] = "missing"

X_sample = df_sample[feature_cols].copy()

print("Making predictions on sample...")
sample_pred = model.predict(X_sample)

# Add predictions column
pred_col_name = f"{TARGET}_knn_pred"
df_sample[pred_col_name] = sample_pred

# Save to CSV
df_sample.to_csv(PREDICTIONS_FILE, index=False)
print(f"\nSample predictions saved to: {PREDICTIONS_FILE}")

# ---------------------------------------------------
# 9. SHOW SAMPLE PREDICTIONS
# ---------------------------------------------------
print("\nSample predictions (first 10 rows):")
print(df_sample[[TARGET, pred_col_name]].head(10))

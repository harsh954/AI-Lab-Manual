import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline        # <--- THIS ONE IS MISSING
from sklearn.impute import SimpleImputer

np.random.seed(42)
n_samples = 100   
rows = []
src_ips = [f"10.0.0.{i}" for i in range(1, 6)]
dst_ips = [f"192.168.1.{i}" for i in range(1, 6)]
dst_ports = [21, 22, 53, 80, 443, 8080]

for _ in range(n_samples):
    rows.append({
        "src_ip":     np.random.choice(src_ips),
        "dst_ip":     np.random.choice(dst_ips),
        "src_port":   np.random.randint(1024, 65535),
        "dst_port":   np.random.choice(dst_ports),
        "protocol":   np.random.choice(["tcp", "udp", "icmp"]),
        "packet_size": np.random.normal(1000, 300),       # bytes
        "duration_ms": abs(np.random.normal(50, 20)),     # ms
        "flag":       np.random.choice(["SYN", "ACK", "FIN", "RST"]),
        "malicious":  1     # all packets are malicious in this lab
    })

df = pd.DataFrame(rows)
print("Raw collected packet data:")
print(df.head())
# =========================================================
# 2. DATA CLEANING  (introduce + fix missing & dirty values)
# =========================================================

# --- simulate missing values in numeric columns ---
for col in ["packet_size", "duration_ms"]:
    mask = np.random.rand(len(df)) < 0.05   # ~5% missing
    df.loc[mask, col] = np.nan

# --- simulate dirty / invalid values ---
df.loc[df.sample(frac=0.05).index, "packet_size"] = -999            # impossible size
df.loc[df.sample(frac=0.05).index, "protocol"] = "unknown_protocol" # invalid protocol

# --- add some duplicate rows for realism ---
df = pd.concat([df, df.iloc[:3]], ignore_index=True)

print("\nAfter adding noise (missing, dirty, duplicates):")
print(df.head())
# ---------- CLEANING STEPS ----------
# 1) Remove duplicates
df = df.drop_duplicates().reset_index(drop=True)

# 2) Replace clearly invalid values with NaN (so we can impute)
df["packet_size"] = df["packet_size"].replace(-999, np.nan)
df["protocol"] = df["protocol"].replace("unknown_protocol", np.nan)

print("\nAfter basic cleaning (duplicates dropped, invalid -> NaN):")
print(df.head())
print("\nMissing values per column:")
print(df.isna().sum())
# =========================================================
# 3. DATA INTEGRATION  (merge with extra info about IPs)
# =========================================================

# Simulated external table: reputation of source IPs
ip_reputation = pd.DataFrame({
    "src_ip": src_ips,
    "src_ip_reputation": ["suspicious", "clean", "suspicious", "clean", "unknown"]
})

df = pd.merge(df, ip_reputation, on="src_ip", how="left")
print("\nAfter integrating IP reputation information:")
print(df[["src_ip", "src_ip_reputation"]].head())
# =========================================================
# 4. DATA ENRICHMENT  (feature engineering)
# =========================================================

# Derived feature: bytes per millisecond
df["bytes_per_ms"] = df["packet_size"] / (df["duration_ms"] + 1e-3)

# Derived feature: is destination port privileged (<1024)
df["is_dst_privileged"] = (df["dst_port"] < 1024).astype(int)

print("\nAfter enrichment (new engineered features):")
print(df[["packet_size", "duration_ms", "bytes_per_ms",
          "dst_port", "is_dst_privileged"]].head())


# =========================================================
# 5. DATA TRANSFORMATION + TRAIN/TEST SPLIT (70/30)
# =========================================================

# ----- define features and target -----
target = "malicious"
feature_cols_numeric = ["packet_size", "duration_ms", "bytes_per_ms",
                        "src_port", "dst_port"]
feature_cols_categorical = ["protocol", "flag", "src_ip_reputation"]

X = df[feature_cols_numeric + feature_cols_categorical]
y = df[target]

# ----- train-test split -----
X_train, X_test, y_train, y_test = train_test_split(
    X, y, train_size=0.70, random_state=42
)

# ----- pipelines for transformation -----
numeric_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore"))
])

preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, feature_cols_numeric),
        ("cat", categorical_transformer, feature_cols_categorical)
    ]
)

# ----- fit on training data and transform both sets -----
preprocessor.fit(X_train)
X_train_processed = preprocessor.transform(X_train)
X_test_processed = preprocessor.transform(X_test)

print("\n--- Final shapes ---")
print("Raw dataset shape:", df.shape)
print("Training set shape (after transform):", X_train_processed.shape)
print("Testing set shape  (after transform):", X_test_processed.shape)

print("\nProcessed packet data is ready for machine learning models.")
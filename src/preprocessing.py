import os
import joblib
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler

df = pd.read_csv("data/raw/student_placement_data.csv")

print(df.head())

print("\nMissing Values")
print(df.isnull().sum())

print("\nDuplicate Records")
print(df.duplicated().sum())

df = df.drop_duplicates()

df = df.drop(
    columns=[
        "Student_ID",
        "Name",
        "Placement_Score"
    ]
)

label = LabelEncoder()

df["Gender"] = label.fit_transform(df["Gender"])

df["Department"] = label.fit_transform(df["Department"])

X = df.drop("Placement_Status", axis=1)

y = df["Placement_Status"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print(X_train.shape)
print(X_test.shape)

scaler = StandardScaler()

# Save original column names
feature_names = X_train.columns

# Scale data
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Convert back to DataFrame with original column names
X_train = pd.DataFrame(X_train, columns=feature_names)
X_test = pd.DataFrame(X_test, columns=feature_names)

os.makedirs("models", exist_ok=True)
os.makedirs("data/processed", exist_ok=True)

joblib.dump(
    scaler,
    "models/scaler.pkl"
)

X_train.to_csv(
    "data/processed/X_train.csv",
    index=False
)

X_test.to_csv(
    "data/processed/X_test.csv",
    index=False
)

y_train.to_frame().to_csv(
    "data/processed/y_train.csv",
    index=False
)

y_test.to_frame().to_csv(
    "data/processed/y_test.csv",
    index=False
)

print("\nPreprocessing Completed Successfully!")
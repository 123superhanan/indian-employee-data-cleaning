# -----------------------------
# IMPORTS
# -----------------------------
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split

# -----------------------------
# STEP 0: DATASET CREATION
# -----------------------------
data = {
    'student_id': [101, 102, 103, 104, 105, 106, 107, 108, 109, 110],
    'name': ['Ali', 'Sara', 'Ahmed', 'Fatima', 'Bilal', 'Ayesha', 'Usman', 'Zara', 'Hassan', 'Mariam'],
    'age': [20, np.nan, 22, 21, 19, np.nan, 23, 20, 22, 19],
    'gender': ['M', 'F', 'M', 'F', 'M', 'F', 'Male', 'Female', 'M', 'F'],
    'gpa': [3.5, 3.8, 3.2, 3.9, 2.9, 3.6, 2.8, 5.0, 3.1, 3.4],
    'study_hours': [3, 5, 2, 6, 1, 4, 1, 50, 3, 4],
    'internship': ['Yes', 'Yes', 'No', 'Yes', 'No', 'Yes', 'No', 'Yes', 'No', 'Yes'],
    'passed': [1, 1, 1, 1, 0, 1, 0, 1, 1, 1]
}

df = pd.DataFrame(data)

# -----------------------------
# STEP 1: DATA CLEANING
# -----------------------------
print("STEP 1: DATA CLEANING\n")

# Fill missing values and replace infinities
df.replace([np.inf, -np.inf], np.nan )
df['age'].fillna(df['age'].mean())

# Fix GPA outlier
df['gpa'] = np.where(df['gpa'] > 4.0, df['gpa'].median(), df['gpa'])
print(f"After fixing GPA outlier: Student S012 GPA = {df.loc[7, 'gpa']}")
df.loc[7, 'gpa'] = 3.5
df.loc[7, 'study_hours'] = 5
print(f"After fixing study_hours outlier: Student S012 study_hours = {df.loc[7, 'study_hours']}")

# Standardize gender values
df['gender'] = df['gender'].map({'M': 'Male', 'Male': 'Male', 'F': 'Female', 'Female': 'Female'})
print("\nData after cleaning (first 3 rows):")
print(df.head(3))

# -----------------------------
# STEP 2: FEATURE ENCODING
# -----------------------------
print("\nSTEP 2: FEATURE ENCODING")

Le = LabelEncoder()
df['gender_encoded'] = Le.fit_transform(df['gender'])
df['internship_encoded'] = Le.fit_transform(df['internship'])

print("\nDataset after encoding (first 3 rows):")
print(df[['name', 'gender', 'gender_encoded', 'internship', 'internship_encoded']].head(3))

# -----------------------------
# STEP 3: PREPARE FOR ML
# -----------------------------
print("\n" + "="*60)
print("STEP 3: PREPARE FOR ML")
print("="*60)

# Features (X) and target (y)
X = df[['age', 'gpa', 'study_hours', 'gender_encoded', 'internship_encoded']]
y = df['passed']

# -----------------------------
# STEP 4: FEATURE SCALING
# -----------------------------
print("\nSTEP 4: FEATURE SCALING")

# Standard Scaling
print('\nStandard Scaling\n')
scaler_std = StandardScaler()
X_scaled = scaler_std.fit_transform(X)
print("\nStandard Scaled Features:")
print(pd.DataFrame(X_scaled, columns=X.columns))

print('\nMin-Max Scaling\n')
# Min-Max Scaling
scaler_mm = MinMaxScaler()
X_minmax = scaler_mm.fit_transform(X)
print("\nMin-Max Scaled Features:")
print(pd.DataFrame(X_minmax, columns=X.columns))

# -----------------------------
# STEP 5: TRAIN/TEST SPLIT
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print("\nTraining and Testing Data Shapes:")
print("X_train:", X_train.shape)
print("X_test:", X_test.shape)
print("y_train:", y_train.shape)
print("y_test:", y_test.shape)

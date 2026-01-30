# -----------------------------
# IMPORTS
# -----------------------------
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# -----------------------------
# DATASET
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
# Fill missing values (Copy-on-Write safe)
df['age'] = df['age'].fillna(df['age'].mean())

# Replace infinities if any
df = df.replace([np.inf, -np.inf], np.nan)

# Fix outliers
df['gpa'] = np.where(df['gpa'] > 4.0, df['gpa'].median(), df['gpa'])
df.loc[7, 'gpa'] = 3.5
df.loc[7, 'study_hours'] = 5

# Standardize gender values
df['gender'] = df['gender'].map({'M': 'Male', 'Male': 'Male', 'F': 'Female', 'Female': 'Female'})

# Verify no missing values
assert df.isnull().sum().sum() == 0, "There are still missing values!"

# -----------------------------
# STEP 2: FEATURE ENCODING
# -----------------------------
# Fit encoders once
gender_encoder = LabelEncoder()
gender_encoder.fit(df['gender'])

internship_encoder = LabelEncoder()
internship_encoder.fit(df['internship'])

# Transform columns
df['gender_encoded'] = gender_encoder.transform(df['gender'])
df['internship_encoded'] = internship_encoder.transform(df['internship'])

# -----------------------------
# STEP 3: FEATURES & TARGET
# -----------------------------
X = df[['age', 'gpa', 'study_hours', 'gender_encoded', 'internship_encoded']]
y = df['passed']

# -----------------------------
# STEP 4: STRATIFIED TRAIN / TEST SPLIT
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# -----------------------------
# STEP 5: FEATURE SCALING
# -----------------------------
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# -----------------------------
# STEP 6: MODEL TRAINING
# -----------------------------
model = LogisticRegression()
model.fit(X_train_scaled, y_train)

# -----------------------------
# STEP 7: PREDICTION & EVALUATION
# -----------------------------
y_pred = model.predict(X_test_scaled)
y_prob = model.predict_proba(X_test_scaled)[:, 1]

print("\nAccuracy:", accuracy_score(y_test, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# -----------------------------
# STEP 8: SAMPLE STUDENT PREDICTION
# -----------------------------
def predict_student(age, gpa, study_hours, gender, internship):
    student_df = pd.DataFrame({
        'age': [age],
        'gpa': [gpa],
        'study_hours': [study_hours],
        'gender_encoded': gender_encoder.transform([gender]),
        'internship_encoded': internship_encoder.transform([internship])
    })
    student_scaled = scaler.transform(student_df)
    pred = model.predict(student_scaled)[0]
    prob = model.predict_proba(student_scaled)[0][1]
    return pred, prob

# Example usage
pred, prob = predict_student(21, 3.6, 4, 'Female', 'Yes')
print("\nSample Student Prediction:")
print(f"Passed? {'Yes' if pred==1 else 'No'} (Probability: {prob:.2f})")

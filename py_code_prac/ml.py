from sklearn.metrics import accuracy_score, precision_score,recall_score,f1_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import mean_absolute_error,mean_squared_error
import numpy as np


# # ture answers what actaully happend
# y_true = [0, 1, 0, 0, 1, 1, 0, 0]
# # model prediction
# y_predict = [0, 0, 1, 1, 0, 1, 1, 0]
# #evaluation
# print('\n classification matrix \n')
# print('accuracy score:', accuracy_score(y_true,y_predict))
# print('prescision: ',precision_score(y_true,y_predict))
# print('recall: ',recall_score(y_true,y_predict))
# print('f1 score: ',f1_score(y_true,y_predict))
# print("="*20)
# print('\n confusion matrix')
# cm = confusion_matrix(y_true,y_predict)
# print(cm) 

#mse 
#real score
y_true = [90, 60,45,78,90]
#model
y_predict = [85,78,100,67,43]

mbe=mean_absolute_error(y_true,y_predict)
mse = mean_squared_error(y_true,y_predict)
rmse= np.sqrt(mse)
print('Mbe: ',mbe)
print('mse: ',mse)
print('rmse: ',rmse)






















# from sklearn.tree import DecisionTreeClassifier
# import numpy as np

# # Features: [fever, cough, headache, fatigue]
# X = np.array([
#     [37.5, 1, 0, 0],   # Cold
#     [38.0, 1, 1, 1],   # Flu
#     [39.0, 1, 1, 1],   # COVID
#     [36.8, 0, 0, 0],   # Healthy / cold
#     [38.5, 1, 1, 1],   # Flu
#     [39.5, 1, 1, 1],   # COVID
#     [37.2, 1, 0, 0],   # Cold
# ])

# # Labels
# y = np.array([0, 1, 2, 0, 1, 2, 0])

# disease_map = {
#     0: "Common Cold",
#     1: "Flu",
#     2: "COVID-like Infection"
# }

# advice = {
#     0: "Rest, fluids, monitor symptoms",
#     1: "Consult doctor, possible medication",
#     2: "Isolate and seek medical testing"
# }

# # Train Decision Tree
# model = DecisionTreeClassifier(
#     criterion="entropy",
#     max_depth=4,
#     random_state=42
# )

# model.fit(X, y)

# # User input
# fever = float(input("Enter body temperature (°C): "))
# cough = int(input("Cough? (1 = Yes, 0 = No): "))
# headache = int(input("Headache? (1 = Yes, 0 = No): "))
# fatigue = int(input("Fatigue? (1 = Yes, 0 = No): "))

# prediction = model.predict([[fever, cough, headache, fatigue]])[0]

# print("\n--- Disease Prediction Result ---")
# print(f"Predicted condition: {disease_map[prediction]}")
# print(f"Advice: {advice[prediction]}")













# X = [
#     [7,2]
#     ,[8,3],
#     [9,8],
#     [10,9]
# ]
# y = [0, 0, 1, 1]

# model = DecisionTreeClassifier()
# model.fit(X,y)
# size = float(input('enter the size: '))
# shade = float(input('enter the shade(1-10): '))

# prediction = model.predict([[size,shade]])[0]
# if prediction == 0:
#     print("This is likely an Apple")
# else:
#     print("This is likely an Orange")














#KNN

#Fruit Prediction ai model using KNN 

# from sklearn.neighbors import KNeighborsClassifier
# from sklearn.preprocessing import StandardScaler
# import numpy as np

# X = np.array([
#     [180, 7],
#     [200, 7.5],
#     [250, 8],
#     [300, 8.5],
#     [330, 9],
#     [350, 9.5]
# ])

# y = [0, 0, 0, 1, 1, 1]

# scaler = StandardScaler()
# X_scaled = scaler.fit_transform(X)

# model = KNeighborsClassifier(n_neighbors=3)
# model.fit(X_scaled, y)

# weight = float(input("enter the weight in GRAMES:\n"))
# size = float(input("enter the size in CM:\n"))

# new_point = scaler.transform([[weight, size]])
# prediction = model.predict(new_point)[0]

# if prediction == 0:
#     print("This is likely an Apple")
# else:
#     print("This is likely an Orange")


# from sklearn.neighbors import KNeighborsClassifier
# from sklearn.preprocessing import StandardScaler
# import numpy as np

# # Dataset: [weight, size]
# X = np.array([
#     [180, 7],
#     [200, 7.5],
#     [250, 8],
#     [300, 8.5],
#     [330, 9],
#     [350, 9.5]
# ])

# # 0 = Apple, 1 = Orange
# y = np.array([0, 0, 0, 1, 1, 1])

# fruit_names = ["Small Apple", "Medium Apple", "Large Apple",
#                "Small Orange", "Medium Orange", "Large Orange"]

# recommendations = {
#     0: "Best for eating raw or baking pies",
#     1: "Best for juice or citrus salads"
# }

# # Scale features
# scaler = StandardScaler()
# X_scaled = scaler.fit_transform(X)

# # KNN model
# model = KNeighborsClassifier(n_neighbors=3)
# model.fit(X_scaled, y)

# # User input
# weight = float(input("Enter the weight in grams: "))
# size = float(input("Enter the size in cm: "))

# user_point = scaler.transform([[weight, size]])

# # Prediction
# prediction = model.predict(user_point)[0]
# neighbors_dist, neighbors_idx = model.kneighbors(user_point)

# # Output
# fruit_type = "Apple" if prediction == 0 else "Orange"

# print("\n--- Recommendation Engine Output ---")
# print(f"Predicted fruit: {fruit_type}")
# print(f"Recommendation: {recommendations[prediction]}")

# print("\nSimilar fruits:")
# for idx in neighbors_idx[0]:
#     print("-", fruit_names[idx])






























# from sklearn.linear_model import LinearRegression


# # Study hours
# X = [[1], [2], [3], [4], [5], [6], [7], [8], [9], [10]]

# # Corresponding marks (more realistic, roughly linear-ish with small randomness)
# y = [35, 40, 50, 55, 60, 65, 70, 75, 80, 85]


# model = LinearRegression()
# model.fit(X,y)
# hours = float(input('Enter your study hours: '))
# predicted_marks = model.predict([[hours]])
# print(f'based on your hours {hours} you may score around {predicted_marks}')


# from sklearn.linear_model import LogisticRegression

# X = [[1], [2], [3], [4], [5]]
# y = [0, 0, 1, 1, 1]
# model = LogisticRegression()
# model.fit(X,y)
# hours = float(input('Enter your study hours: '))
# result = model.predict([[hours]])[0]
# if result==1:
#     print(f'based on your hours {hours} you are Likely to pass')
# else:
#     print(f'based on your hours {hours} you are Likely not to pass')










# # -----------------------------
# # IMPORTS
# # -----------------------------
# import pandas as pd
# import numpy as np
# from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler
# from sklearn.model_selection import train_test_split

# # -----------------------------
# # STEP 0: DATASET CREATION
# # -----------------------------
# data = {
#     'student_id': [101, 102, 103, 104, 105, 106, 107, 108, 109, 110],
#     'name': ['Ali', 'Sara', 'Ahmed', 'Fatima', 'Bilal', 'Ayesha', 'Usman', 'Zara', 'Hassan', 'Mariam'],
#     'age': [20, np.nan, 22, 21, 19, np.nan, 23, 20, 22, 19],
#     'gender': ['M', 'F', 'M', 'F', 'M', 'F', 'Male', 'Female', 'M', 'F'],
#     'gpa': [3.5, 3.8, 3.2, 3.9, 2.9, 3.6, 2.8, 5.0, 3.1, 3.4],
#     'study_hours': [3, 5, 2, 6, 1, 4, 1, 50, 3, 4],
#     'internship': ['Yes', 'Yes', 'No', 'Yes', 'No', 'Yes', 'No', 'Yes', 'No', 'Yes'],
#     'passed': [1, 1, 1, 1, 0, 1, 0, 1, 1, 1]
# }

# df = pd.DataFrame(data)

# # -----------------------------
# # STEP 1: DATA CLEANING
# # -----------------------------
# print("STEP 1: DATA CLEANING\n")

# # Fill missing values and replace infinities
# df.replace([np.inf, -np.inf], np.nan )
# df['age'].fillna(df['age'].mean())

# # Fix GPA outlier
# df['gpa'] = np.where(df['gpa'] > 4.0, df['gpa'].median(), df['gpa'])
# print(f"After fixing GPA outlier: Student S012 GPA = {df.loc[7, 'gpa']}")
# df.loc[7, 'gpa'] = 3.5
# df.loc[7, 'study_hours'] = 5
# print(f"After fixing study_hours outlier: Student S012 study_hours = {df.loc[7, 'study_hours']}")

# # Standardize gender values
# df['gender'] = df['gender'].map({'M': 'Male', 'Male': 'Male', 'F': 'Female', 'Female': 'Female'})
# print("\nData after cleaning (first 3 rows):")
# print(df.head(3))

# # -----------------------------
# # STEP 2: FEATURE ENCODING
# # -----------------------------
# print("\nSTEP 2: FEATURE ENCODING")

# Le = LabelEncoder()
# df['gender_encoded'] = Le.fit_transform(df['gender'])
# df['internship_encoded'] = Le.fit_transform(df['internship'])

# print("\nDataset after encoding (first 3 rows):")
# print(df[['name', 'gender', 'gender_encoded', 'internship', 'internship_encoded']].head(3))

# # -----------------------------
# # STEP 3: PREPARE FOR ML
# # -----------------------------
# print("\n" + "="*60)
# print("STEP 3: PREPARE FOR ML")
# print("="*60)

# # Features (X) and target (y)
# X = df[['age', 'gpa', 'study_hours', 'gender_encoded', 'internship_encoded']]
# y = df['passed']

# # -----------------------------
# # STEP 4: FEATURE SCALING
# # -----------------------------
# print("\nSTEP 4: FEATURE SCALING")

# # Standard Scaling
# print('\nStandard Scaling\n')
# scaler_std = StandardScaler()
# X_scaled = scaler_std.fit_transform(X)
# print("\nStandard Scaled Features:")
# print(pd.DataFrame(X_scaled, columns=X.columns))

# print('\nMin-Max Scaling\n')
# # Min-Max Scaling
# scaler_mm = MinMaxScaler()
# X_minmax = scaler_mm.fit_transform(X)
# print("\nMin-Max Scaled Features:")
# print(pd.DataFrame(X_minmax, columns=X.columns))

# # -----------------------------
# # STEP 5: TRAIN/TEST SPLIT
# # -----------------------------
# X_train, X_test, y_train, y_test = train_test_split(
#     X, y, test_size=0.2, random_state=42
# )

# print("\nTraining and Testing Data Shapes:")
# print("X_train:", X_train.shape)
# print("X_test:", X_test.shape)
# print("y_train:", y_train.shape)
# print("y_test:", y_test.shape)
# # from sklearn.preprocessing import LabelEncoder
# # from sklearn.preprocessing import StandardScaler,MinMaxScaler
# # from sklearn.model_selection import train_test_split
# # import pandas as pd
# # import numpy as np 
# # # df =pd.read_csv('students_data.csv')
# # # df_Label = df.copy()
# # # Le = LabelEncoder()

# # #feature Scaling
# # data = {
# # "Study_hours" : [1,2,3,5,6],
# # "testingScore" :[10,68,56,120,56]
# # }
# # df = pd.DataFrame(data)
# # Scaler = StandardScaler()
# # Scaled = Scaler.fit_transform(df)
# # print("\nstandard scaled output (mean to 0 ,std to 1)")
# # print(pd.DataFrame(Scaled,columns=['Study_hours','testingScore']))


# # scaler_two=MinMaxScaler()
# # MinMaxScaled = scaler_two.fit_transform(df)
# # print("min max scaled output (values ranges from (0-1)) ")
# # print(pd.DataFrame(MinMaxScaled, columns=['Study_hours','testingScore']))


# # x = df[['Study_hours']]
# # y = df['testingScore']
# # x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=42)
# # print("\nTraining data:")
# # print("\nX_train:\n",x_train)
# # print("\nTesting data:")
# # print("\nX_test:\n",x_test)

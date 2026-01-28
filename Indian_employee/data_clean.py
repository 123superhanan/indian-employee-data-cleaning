import pandas as pd
import numpy as np

df = pd.read_csv("Indian_employee/indian_employee_data.csv")

# replace inf first
df = df.replace([np.inf, -np.inf], np.nan)

# fix dates
df["JoinDate"] = pd.to_datetime(df["JoinDate"], errors="coerce")
df["JoinDate"] = df["JoinDate"].fillna(pd.Timestamp("2000-01-01"))

# fill categorical
df["City"] = df["City"].fillna(df["City"].mode()[0])
df["Name"] = df["Name"].fillna("Unknown")

# fix numeric
df["Salary"] = df["Salary"].fillna(df["Salary"].median())
df["ExperienceYears"] = df["ExperienceYears"].fillna(df["ExperienceYears"].median())
df["Age"] = df["Age"].fillna(df["Age"].median())

# remove negatives
df["Salary"] = np.where(df["Salary"] < 0, df["Salary"].median(), df["Salary"])

# remove outliers
mean = df["Salary"].mean()
std = df["Salary"].std()
df = df[(df["Salary"] >= mean - 3*std) & (df["Salary"] <= mean + 3*std)]

# normalize text
df["Department"] = df["Department"].str.strip().str.title()
df["City"] = df["City"].str.strip().str.title()

# remove logical duplicates
df.drop_duplicates(subset="EmployeeID", inplace=True)

# reset index
df.reset_index(drop=True, inplace=True)

df.to_csv("Indian_employee/cleaned.csv", index=False)

print("Data cleaning completed successfully.")



# import pandas as pd
# import numpy as np

# #loading data from csv file
# df = pd.read_csv('Indian_employee/indian_employee_data.csv')

# #handling missing values by filling with mean salary
# df['Salary'] = df['Salary'].fillna(df['Salary'].mean())
# df['ExperienceYears'] = df['ExperienceYears'].fillna(df['ExperienceYears'].median())
# df['Age'] = df['Age'].fillna(df['Age'].mean())
# df['City'] = df['City'].fillna(df['City'].mode()[0])
# df['Name'] = df['Name'].fillna('Unknown')
# df['JoinDate'] = pd.to_datetime(df['JoinDate'], errors='coerce')
# df['JoinDate'] = df['JoinDate'].fillna(pd.to_datetime('2000-01-01'))

# #handling inconsistent data or infinite values
# df = df.replace([np.inf, -np.inf], np.nan)
# df = df.fillna(df.mean(numeric_only=True))


# #removing duplicates records
# df=df.drop_duplicates(inplace=True)

# #negative salary correction
# df['Salary'] =np.where(df['Salary'] < 0,df['Salary'].mean(),df['Salary'])

# salary_mean = df['Salary'].mean()
# salary_std = df['Salary'].std()
# lower_bound = salary_mean -(3* salary_std)
# upper_bound = salary_mean +(3* salary_std)

# #removing outliers in salary so high and low salary
# df = df[(df['Salary']>=lower_bound)&(df['Salary']<=upper_bound)]
# #standardizing department names to title case
# df['Department'] = df['Department'].str.title() 
# #standardizing city names to title case
# df['City'] = df['City'].str.title()
# #reset index after cleaning
# df.reset_index(drop=True,inplace=True)
# df.to_csv('Indian_employee/cleaned.csv',index=False)
# print("Data cleaning completed. Cleaned data saved to 'cleaned.csv'.")
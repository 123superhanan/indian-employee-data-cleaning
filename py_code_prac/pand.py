import pandas as pd

customers = {
    'CustomerID': [1, 2, 3, 4, 5],
    'CustomerName': [
        "Ali Khan", "Sara Malik", "Ahmed Raza",
        "Ayesha Noor", "Usman Tariq"
    ],
    'City': [
        "Lahore", "Karachi", "Islamabad",
        "Lahore", "Rawalpindi"
    ]
}

orders = {
    'OrderID': [101, 102, 103],
    'CustomerID': [1, 3, 4],
    'Product': [
        "Laptop", "Mobile Phone", "Headphones"
    ],
    'OrderAmount': [120000, 80000, 15000]
}

df_customers = pd.DataFrame(customers)
df_orders = pd.DataFrame(orders)
 
print("Customers DataFrame:")
print(df_customers)
print("\nOrders DataFrame:")
print(df_orders)

print("merging based on orders and customers dataframes:")

df_merged = pd.merge(df_customers,df_orders,on='CustomerID',how='inner')
print("Merged DataFrame (Inner Join):")
print(df_merged)
df_concate= pd.concat([df_customers,df_orders],axis=0,ignore_index=True)
print("\nConcatenated DataFrame:")
print(df_concate)

















# office_data = {
#     'EmployeeID': [
#         1001, 1002, 1003, 1004, 1005,
#         1006, 1007, 1008, 1009, 1010
#     ],
#     'Name': [
#         "Ali Khan", "Sara Malik", "Ahmed Raza", "Ayesha Noor", "Usman Tariq",
#         "Bilal Ahmed", "Hina Fatima", "Zain Abbas", "Noor Hassan", "Hamza Iqbal"
#     ],
#     'Department': [
#         "IT", "HR", "IT", "Finance", "Operations",
#         "IT", "HR", "Marketing", "Finance", "Operations"
#     ],
#     'Role': [
#         "Software Engineer", "HR Executive", "Backend Developer", "Accountant", "Operations Manager",
#         "Data Analyst", "Recruiter", "Marketing Executive", "Financial Analyst", "Operations Associate"
#     ],
#     'Age': [
#         24, 29, 26, 31, 35,
#         27, 28, 25, 30, 34
#     ],
#     'Experience_Years': [
#         2, 5, 3, 7, 10,
#         4, 6, 2, 8, 9
#     ],
#     'Salary': [
#         85000, 70000, 90000, 75000, 95000,
#         88000, 72000, 68000, 92000, 97000
#     ],
#     'Performance_Rating': [
#         4.2, 3.8, 4.5, 4.0, 4.1,
#         4.3, 3.9, 3.6, 4.4, 4.0
#     ]
# }



# df = pd.DataFrame(office_data)
# print(df)

#grouping data
# avg_perf_by_dept = df.groupby(['EmployeeID','Department'])['Performance_Rating'].mean()

# max_salary_by_employee = df.groupby(['EmployeeID','Name'])['Salary'].max()

# min_age_by_role = df.groupby('Role')['Age'].min()

# min_perf_by_employee = df.groupby('Name')['Performance_Rating'].min()

# print("Average Performance Rating by Department")
# print(avg_perf_by_dept)

# print("\nMaximum Salary by Employee")
# print(max_salary_by_employee)

# print("\nMinimum Age by Role")
# print(min_age_by_role)

# print("\nMinimum Performance Rating by Employee")
# print(min_perf_by_employee)

#sorting data
# print("after sorting by GPA in descending order:")
# df.sort_values(by=['GPA'], ascending=False,inplace=True)
# print(df)

#missing data handling
# Fill numeric columns with mean
# df['Age'] = df['Age'].fillna(df['Age'].mean())
# df['GPA'] = df['GPA'].fillna(df['GPA'].mean())
# print(df)
# Fill categorical columns with mode
# df['Name'] = df['Name'].fillna(df['Name'].mode()[0])
# df['Department'] = df['Department'].fillna(df['Department'].mode()[0])  
# df['Age'] = df['Age'].interpolate(method='linear')
# df['GPA'] = df['GPA'].interpolate(method='linear')
# print(df)

#adding daya
# df.insert(0, 'ID', [101,102,103,104,105])
# print(df.isnull())  #check missing values
# print(df.isnull().sum())

#removing data
# print("after removing missing data:")
# df.dropna( inplace=True) #remove rows with missing data
# print(df)















# data={
#     'Name':["Ali","Sara","Ahmed","Ayesha","Usman"],
#     'Age':[20,21,19,22,20],
#     'Department':["CS","SE","CS","AI","SE"],
#     'GPA':[3.5,3.8,3.4,3.2,2.9]
# }

# df = pd.DataFrame(data)
# df.insert(0, 'ID', [101,102,103,104,105])
# print(df)

# df.loc[0,'Name'] = 'Ali reza'
# df.drop(columns=['Age'], inplace=True)
# print(df)
# print('award for students with GPA > 3.5:')
# df["award"] = df["GPA"]  * 10
# print(df)
# df["Toppers"] = df["GPA"]> 3.5
# print('the data with column toppers added:')



# high_Gpa= df[df['GPA'] > 3.5]
# print(high_Gpa)
# filtered_df = df[(df['Department'] == 'CS') & (df['GPA'] > 3.0)]
# print(filtered_df)
# df = pd.DataFrame(data)
# df.to_csv("output.csv" ,index=False)
# df.to_excel("output.xlsx" ,index=False)


# df = pd.read_csv('students_data.csv',encoding="latin-1") #encoding="UTF-8" or "latin-1"
# # df = pd.read_excel('sales_report.xlsx')
# #df = pd.read_json('employees.json')

# print(df.head())  # Display first 5 rows
# df.info()  # Summary of the DataFrame 
# set = print(df.tail())  # Display last 5 rows 
# # if set is None:
# #     print("yes")
# # elif set is not None:
# #     print("no")


# import math

# x = math.sqrt(64)

# print(x)

# thistuple = ("apple", "banana", "cherry")
# # y = list(thistuple)
# # y.append("orange")
# # thistuple = tuple(y)
# for i in range(len(thistuple)):
#   print(thistuple[i])
# class Calculator:
#   def add(self, a, b):
#     return a + b

#   def multiply(self, a, b):
#     return a * b

# calc = Calculator()
# print(calc.add(5, 3))
# print(calc.multiply(4, 7))



# import pandas as pd

# data={
#     'Name':["Ali","Sara","Ahmed","Ayesha","Usman"],
#     'Age':[20,21,19,22,20],
#     'Department':["CS","SE","CS","AI","SE"],
#     'GPA':[3.5,3.8,3.4,3.2,2.9]
# }

# df = pd.DataFrame(data)
# df.insert(0, 'ID', [101,102,103,104,105])
# # df["Toppers"] = df["GPA"]> 3.5
# # print('the data with column toppers added:')
# print(df)
# # print('award for students with GPA > 3.5:')
# # df["award"] = df["GPA"]  * 10
# print(df)




# high_Gpa= df[df['GPA'] > 3.5]
# print(high_Gpa)
# filtered_df = df[(df['Department'] == 'CS') & (df['GPA'] > 3.0)]
# print(filtered_df)
# df = pd.DataFrame(data)
# #df.to_csv("output.csv" ,index=False)
# df.to_excel("output.xlsx" ,index=False)


 




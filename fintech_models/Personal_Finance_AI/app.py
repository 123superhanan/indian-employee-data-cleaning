# import streamlit as st
# import pandas as pd
# import joblib
# import plotly.express as px

# st.title("💰 My Simple Expense Tracker")

# # Load data
# df = pd.read_csv('cleaned_expenses.csv')
# df['date_clean'] = pd.to_datetime(df['date_clean'])

# # Sidebar
# st.sidebar.header("Filters")
# month_filter = st.sidebar.selectbox("Select Month", df['date_clean'].dt.strftime('%Y-%m').unique())

# # Main Dashboard
# col1, col2, col3 = st.columns(3)
# col1.metric("Total Spent", f"₹{df['amount'].sum():,.0f}")
# col2.metric("Avg per Transaction", f"₹{df['amount'].mean():.0f}")
# col3.metric("Total Transactions", len(df))

# # Category Spending
# st.subheader("Spending by Category")
# fig = px.bar(df.groupby('category')['amount'].sum().reset_index(), 
#              x='category', y='amount', title="Category-wise Spending")
# st.plotly_chart(fig)

# # Monthly Trend
# st.subheader("Monthly Spending Trend")
# monthly = df.groupby(df['date_clean'].dt.to_period('M'))['amount'].sum().reset_index()
# monthly['date_clean'] = monthly['date_clean'].astype(str)
# fig2 = px.line(monthly, x='date_clean', y='amount', title="Monthly Trend")
# st.plotly_chart(fig2)

# # Future Prediction Section
# st.subheader("🔮 Predict Next Expense")
# category = st.selectbox("Category", df['category'].unique())
# payment = st.selectbox("Payment Method", df['payment_method'].unique())
# day = st.slider("Day of Week (0=Mon, 6=Sun)", 0, 6, 3)
# month = st.slider("Month", 1, 12, 3)
# weekend = st.checkbox("Is Weekend?")

# if st.button("Predict Amount"):
#     model = joblib.load('expense_predictor_model.pkl')
#     input_df = pd.DataFrame({
#         'category': [category],
#         'payment_method': [payment],
#         'day_of_week': [day],
#         'month': [month],
#         'is_weekend': [1 if weekend else 0]
#     })
#     pred = model.predict(input_df)[0]
#     st.success(f"Predicted Expense: ₹{pred:.2f}")
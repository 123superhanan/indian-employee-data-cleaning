import streamlit as st
import pandas as pd
import joblib
import plotly.express as px

st.set_page_config(page_title="My Expense Tracker", layout="centered")
st.title(" My Simple Expense Tracker & Predictor")

# Load model
@st.cache_resource
def load_model():
    return joblib.load('expense_predictor_model.pkl')

model = load_model()

# Load data
try:
    df = pd.read_csv('clean_data.csv')        
    df['date_clean'] = pd.to_datetime(df['date_clean'])
except:
    st.error("Could not find 'clean_data.csv'. Please check the filename.")
    st.stop()

# Sidebar
st.sidebar.header("Make a Prediction")
category = st.sidebar.selectbox("Category", df['category'].unique())
payment = st.sidebar.selectbox("Payment Method", df['payment_method'].unique())
day = st.sidebar.slider("Day of Week (0=Mon, 6=Sun)", 0, 6, 3)
month = st.sidebar.slider("Month", 1, 12, 3)
weekend = st.sidebar.checkbox("Is Weekend?")

if st.sidebar.button("Predict Expense"):
    input_data = pd.DataFrame({
        'category': [category],
        'payment_method': [payment],
        'day_of_week': [day],
        'month': [month],
        'is_weekend': [1 if weekend else 0]
    })
    
    pred = model.predict(input_data)[0]
    st.success(f"🔮 Predicted Expense: **₹{pred:.2f}**")

# Main Dashboard
st.header("Your Spending Overview")

col1, col2, col3 = st.columns(3)
col1.metric("Total Spent", f"₹{df['amount'].sum():,.0f}")
col2.metric("Avg per Transaction", f"₹{df['amount'].mean():.0f}")
col3.metric("Total Transactions", len(df))

# Category Spending Chart
st.subheader("Spending by Category")
fig = px.bar(df.groupby('category')['amount'].sum().reset_index(), 
             x='category', y='amount', title="Category-wise Spending")
st.plotly_chart(fig, use_container_width=True)

# Monthly Trend
st.subheader("Monthly Spending Trend")
monthly = df.groupby(df['date_clean'].dt.to_period('M'))['amount'].sum().reset_index()
monthly['Month'] = monthly['date_clean'].astype(str)
fig2 = px.line(monthly, x='Month', y='amount', title="Monthly Trend", markers=True)
st.plotly_chart(fig2, use_container_width=True)

# Food Insight
st.subheader("Food Spending Insight")
food_avg = df[df['category'] == 'Food']['amount'].sum() / df['date_clean'].dt.to_period('M').nunique()
st.write(f"Average Monthly Food Expense: **₹{food_avg:.0f}**")

if food_avg > 8000:
    st.warning("Your Food spending is quite high. Try cooking at home more.")
else:
    st.success(" Food spending is under control.")

st.caption("Made with ❤️ using Streamlit + tensorflow model by Hanan")
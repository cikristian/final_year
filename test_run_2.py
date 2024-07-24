import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

st.title('Number of Consultants')
st.header('Consultant Forecasting BIZ Analytic Pro')
st.text("Visualizations of the company datasets")

# Add navigation links in the sidebar
st.sidebar.selectbox("Navigation", ["Home", "About", "Contact"])

# Load the model
try:
    model = joblib.load('Gradient_Boost_Model4.pkl')
except Exception as e:
    st.error(f"Error loading the model: {e}")
    model = None

# Define the prediction function
def predict(cons_last_month, forecast_cons_12m, mean_6m_price_peak,
            mean_3m_price_peak, churn, date_trained_diff, channel_sales_encoded):
    if model is not None:
        try:
            prediction = model.predict([[cons_last_month, forecast_cons_12m, mean_6m_price_peak,
                                         mean_3m_price_peak, churn, date_trained_diff, channel_sales_encoded]])
            return prediction
        except Exception as e:
            st.error(f"Error during prediction: {e}")
            return None
    else:
        st.error("Model is not loaded.")
        return None

# Input fields for prediction
st.sidebar.header("Prediction Input")
channel_sales_encoded = st.sidebar.number_input('Channel Sales Encoded', 0, 5)
date_trained_diff = st.sidebar.number_input('Date Trained Difference', 0, 10000)
cons_last_month = st.sidebar.number_input('Consumption Last Month', 0, 1000)
forecast_cons_12m = st.sidebar.number_input('Forecast Consumption 12M', 0, 1000)
mean_6m_price_peak = st.sidebar.number_input('Mean 6M Price Peak', 0.00, 100000.00)
mean_3m_price_peak = st.sidebar.number_input('Mean 3M Price Peak', 0.00, 100000.00)
churn = st.sidebar.number_input('Churn', 0.00, 100.00)

if st.sidebar.button('Predict number of consultants'):
    a_q = predict(cons_last_month, forecast_cons_12m, mean_6m_price_peak,
                  mean_3m_price_peak, churn, date_trained_diff, channel_sales_encoded)
    if a_q is not None:
        st.sidebar.success(f'The predicted number of consultants is {a_q[0]:.2f} Consults')

# Dataset upload
st.sidebar.header("Dataset Upload")
uploaded_file = st.sidebar.file_uploader("Choose a CSV file", type="csv")

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.header("Dataset")
    st.write(df)

    # Visualizations
    st.header("Visualizations")
    
    st.subheader("Histogram")
    column = st.selectbox("Select a column for histogram", df.columns)
    fig, ax = plt.subplots()
    ax.hist(df[column], bins=20, edgecolor='k')
    st.pyplot(fig)
    
    st.subheader("Scatter Plot")
    x_col = st.selectbox("Select X-axis for scatter plot", df.columns)
    y_col = st.selectbox("Select Y-axis for scatter plot", df.columns)
    fig, ax = plt.subplots()
    ax.scatter(df[x_col], df[y_col])
    ax.set_xlabel(x_col)
    ax.set_ylabel(y_col)
    st.pyplot(fig)
    
    st.subheader("Line Chart")
    line_col = st.selectbox("Select a column for line chart", df.columns)
    st.line_chart(df[line_col])

    st.subheader("Correlation Heatmap")
    # Filter only numeric columns
    numeric_df = df.select_dtypes(include=[np.number])
    if numeric_df.empty:
        st.warning("No numeric columns found in the dataset.")
    else:
        fig, ax = plt.subplots()
        sns.heatmap(numeric_df.corr(), annot=True, cmap='coolwarm', ax=ax)
        st.pyplot(fig)
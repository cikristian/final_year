import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

st.title('Market Forcasting')
st.header('Consultant Forecasting BIZ Analytic Pro')
st.text("Visualizations of the company datasets")

# Add navigation links in the sidebar
st.sidebar.selectbox("Navigation", ["Home", "Contact"])

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
st.sidebar.header("Market forcast")
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

    # Prepare numeric columns for visualizations
    numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()

    # Visualizations
    st.header("Visualizations")
    
    # Histogram
    st.subheader("Histogram")
    for column in numeric_columns:
        fig, ax = plt.subplots()
        ax.hist(df[column], bins=20, edgecolor='k')
        ax.set_title(f'Histogram of {column}')
        st.pyplot(fig)

    # Scatter Plot
    st.subheader("Scatter Plot")
    if len(numeric_columns) >= 2:
        x_col = st.selectbox("Select X-axis for scatter plot", numeric_columns, key='x_col')
        y_col = st.selectbox("Select Y-axis for scatter plot", numeric_columns, key='y_col')
        fig, ax = plt.subplots()
        ax.scatter(df[x_col], df[y_col])
        ax.set_xlabel(x_col)
        ax.set_ylabel(y_col)
        ax.set_title(f'Scatter Plot: {x_col} vs {y_col}')
        st.pyplot(fig)
    else:
        st.warning("Not enough numeric columns for scatter plot.")
    
    # Line Chart
    st.subheader("Line Chart")
    for column in numeric_columns:
        fig, ax = plt.subplots()
        ax.plot(df[column])
        ax.set_title(f'Line Chart of {column}')
        st.pyplot(fig)

    # Correlation Heatmap
    st.subheader("Correlation Heatmap")
    if not numeric_columns:
        st.warning("No numeric columns found in the dataset.")
    else:
        fig, ax = plt.subplots()
        sns.heatmap(df[numeric_columns].corr(), annot=True, cmap='coolwarm', ax=ax)
        ax.set_title('Correlation Heatmap')
        st.pyplot(fig)
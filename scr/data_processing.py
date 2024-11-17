import pandas as pd
import streamlit as st

def upload_data(file):
    if file.name.endswith('.csv'):
        df = pd.read_csv(file)
    elif file.name.endswith('.xlsx'):
        df = pd.read_excel(file)
    else:
        st.error("Unsupported file format. Please upload a CSV or Excel file.")
        return None
    return df

def preview_data(df):
    st.dataframe(df.head())

def basic_analysis(df):
    # Placeholder functions - replace with actual analysis
    discharge_capacity = df['Discharge Capacity'].rolling(window=5).mean()
    capacity_retention = df['Discharge Capacity'] / df['Discharge Capacity'].iloc[0] * 100
    coulombic_efficiency = df['Charge Capacity'] / df['Discharge Capacity'] * 100
    
    return discharge_capacity, capacity_retention, coulombic_efficiency

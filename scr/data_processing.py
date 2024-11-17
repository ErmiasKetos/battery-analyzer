import pandas as pd
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import io

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

def calculate_discharge_capacity(df):
    st.subheader("Discharge Capacity Calculation")
    
    if 'Discharge Current (A)' in df.columns and 'Discharge Time (h)' in df.columns:
        df['Discharge Capacity (mAh)'] = df['Discharge Current (A)'] * df['Discharge Time (h)'] * 1000
        st.success("Discharge Capacity calculated successfully.")
    else:
        st.warning("Unable to calculate Discharge Capacity. Required columns not found.")
    
    return df

def basic_analysis(df):
    st.subheader("Basic Analysis")
    
    if 'Discharge Capacity (mAh)' not in df.columns:
        df = calculate_discharge_capacity(df)
    
    discharge_capacity = df['Discharge Capacity (mAh)'].rolling(window=5).mean()
    capacity_retention = df['Discharge Capacity (mAh)'] / df['Discharge Capacity (mAh)'].iloc[0] * 100
    
    if 'Charge Capacity (mAh)' in df.columns:
        coulombic_efficiency = df['Discharge Capacity (mAh)'] / df['Charge Capacity (mAh)'] * 100
    else:
        st.warning("'Charge Capacity (mAh)' column not found. Coulombic Efficiency cannot be calculated.")
        coulombic_efficiency = None
    
    # Display results
    st.write("Discharge Capacity:")
    st.line_chart(discharge_capacity)
    
    st.write("Capacity Retention:")
    st.line_chart(capacity_retention)
    
    if coulombic_efficiency is not None:
        st.write("Coulombic Efficiency:")
        st.line_chart(coulombic_efficiency)
    
    return discharge_capacity, capacity_retention, coulombic_efficiency

def voltage_analysis(df):
    st.subheader("Voltage Analysis")
    
    if 'Voltage (V)' in df.columns and 'Discharge Capacity (mAh)' in df.columns:
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(df['Discharge Capacity (mAh)'], df['Voltage (V)'])
        ax.set_xlabel('Discharge Capacity (mAh)')
        ax.set_ylabel('Voltage (V)')
        ax.set_title('Voltage vs Discharge Capacity')
        st.pyplot(fig)
    else:
        st.warning("Unable to perform voltage analysis. Required columns not found.")

def dq_de_analysis(df):
    st.subheader("dQ/dE Analysis")
    
    if 'Voltage (V)' in df.columns and 'Discharge Capacity (mAh)' in df.columns:
        # Calculate dQ/dE
        dq = np.diff(df['Discharge Capacity (mAh)'])
        de = np.diff(df['Voltage (V)'])
        dq_de = dq / de
        
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(df['Voltage (V)'][1:], dq_de)
        ax.set_xlabel('Voltage (V)')
        ax.set_ylabel('dQ/dE')
        ax.set_title('dQ/dE vs Voltage')
        st.pyplot(fig)
    else:
        st.warning("Unable to perform dQ/dE analysis. Required columns not found.")

def generate_downloadable_csv(df):
    csv = df.to_csv(index=False)
    return csv

def data_processing_main(df):
    st.header("Data Processing and Analysis")
    
    # Calculate Discharge Capacity if not present
    if 'Discharge Capacity (mAh)' not in df.columns:
        df = calculate_discharge_capacity(df)
    
    # Perform basic analysis
    basic_analysis(df)
    
    # Perform voltage analysis
    voltage_analysis(df)
    
    # Perform dQ/dE analysis
    dq_de_analysis(df)
    
    # Generate downloadable CSV
    csv = generate_downloadable_csv(df)
    st.download_button(
        label="Download processed data as CSV",
        data=csv,
        file_name="processed_battery_data.csv",
        mime="text/csv"
    )
    
    return df

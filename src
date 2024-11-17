import pandas as pd
import streamlit as st
import numpy as np

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
    discharge_capacity = df['Discharge Capacity'].rolling(window=5).mean()
    capacity_retention = df['Discharge Capacity'] / df['Discharge Capacity'].iloc[0] * 100
    coulombic_efficiency = df['Charge Capacity'] / df['Discharge Capacity'] * 100
    
    return discharge_capacity, capacity_retention, coulombic_efficiency

def calculate_discharge_capacity(df):
    st.subheader("Discharge Capacity Analysis")
    
    # Calculate discharge capacity
    df['Discharge Capacity (Ah)'] = df['Discharge Current (A)'] * df['Discharge Time (h)']
    
    # Calculate C-rate
    nominal_capacity = df['Discharge Capacity (Ah)'].max()
    df['C-rate'] = df['Discharge Current (A)'] / nominal_capacity
    
    # Display results
    st.write("Discharge Capacity Calculation:")
    st.write(f"Nominal Capacity: {nominal_capacity:.2f} Ah")
    st.write(f"Average Discharge Current: {df['Discharge Current (A)'].mean():.2f} A")
    st.write(f"Average Discharge Time: {df['Discharge Time (h)'].mean():.2f} hours")
    st.write(f"Average C-rate: {df['C-rate'].mean():.2f}C")
    
    # Plot Discharge Capacity vs Cycle Number
    st.subheader("Discharge Capacity vs Cycle Number")
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(df['Cycle Number'], df['Discharge Capacity (Ah)'], marker='o')
    ax.set_xlabel('Cycle Number')
    ax.set_ylabel('Discharge Capacity (Ah)')
    ax.set_title('Discharge Capacity vs Cycle Number')
    st.pyplot(fig)
    
    # C-rate explanation
    st.subheader("Understanding C-rate")
    st.write("""
    The C-rate is a measure of the rate at which a battery is discharged relative to its maximum capacity. 
    A 1C rate means that the discharge current will discharge the entire battery in 1 hour.
    
    - 1C rate: Fully discharge in 1 hour
    - 2C rate: Fully discharge in 0.5 hours (30 minutes)
    - C/2 rate: Fully discharge in 2 hours
    """)
    
    # Battery chemistry information
    st.subheader("Battery Chemistry and Discharge Rates")
    st.write("""
    Different battery chemistries have different typical discharge rates:
    
    1. Lead-acid batteries: Usually rated at low discharge rates (e.g., 0.05C or 20-hour rate)
    2. Lithium-ion batteries: Can tolerate higher C rates, often 1C or higher
    3. Lithium-Sulfur (Li-S) batteries: Can typically handle moderate discharge rates, but optimal performance is often at lower rates (e.g., C/5 to C/2)
    """)
    
    return df

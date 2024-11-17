import pandas as pd
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

def basic_analysis(df):
    st.subheader("Basic Analysis")
    
    if 'Discharge Capacity' not in df.columns:
        st.warning("'Discharge Capacity' column not found. Please ensure your data includes this information.")
        return None, None, None
    
    discharge_capacity = df['Discharge Capacity'].rolling(window=5).mean()
    capacity_retention = df['Discharge Capacity'] / df['Discharge Capacity'].iloc[0] * 100
    
    if 'Charge Capacity' in df.columns:
        coulombic_efficiency = df['Charge Capacity'] / df['Discharge Capacity'] * 100
    else:
        st.warning("'Charge Capacity' column not found. Coulombic Efficiency cannot be calculated.")
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

def calculate_discharge_capacity(df):
    st.subheader("Discharge Capacity Analysis")
    
    required_columns = ['Discharge Current (A)', 'Discharge Time (h)']
    missing_columns = [col for col in required_columns if col not in df.columns]
    
    if missing_columns:
        st.warning(f"Missing columns: {', '.join(missing_columns)}. Cannot calculate Discharge Capacity.")
        return df
    
    # Calculate discharge capacity
    df['Discharge Capacity (Ah)'] = df['Discharge Current (A)'] * df['Discharge Time (h)']
    
    # Calculate C-rate
    nominal_capacity = df['Discharge Capacity (Ah)'].max()
    df['C-rate'] = df['Discharge Current (A)'] / nominal_capacity
    
    # Display results
    st.write(f"Nominal Capacity: {nominal_capacity:.2f} Ah")
    st.write(f"Average Discharge Current: {df['Discharge Current (A)'].mean():.2f} A")
    st.write(f"Average Discharge Time: {df['Discharge Time (h)'].mean():.2f} hours")
    st.write(f"Average C-rate: {df['C-rate'].mean():.2f}C")
    
    # Plot Discharge Capacity vs Cycle Number
    if 'Cycle Number' in df.columns:
        st.subheader("Discharge Capacity vs Cycle Number")
        st.line_chart(df.set_index('Cycle Number')['Discharge Capacity (Ah)'])
    else:
        st.warning("'Cycle Number' column not found. Cannot plot Discharge Capacity vs Cycle Number.")
    
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

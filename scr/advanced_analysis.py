import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def voltage_profiles(df):
    st.subheader("Voltage Profiles")
    
    required_columns = ['Capacity', 'Voltage']
    missing_columns = [col for col in required_columns if col not in df.columns]
    
    if missing_columns:
        st.warning(f"Missing columns: {', '.join(missing_columns)}. Cannot plot Voltage Profiles.")
        return
    
    fig, ax = plt.subplots()
    ax.plot(df['Capacity'], df['Voltage'])
    ax.set_xlabel('Capacity')
    ax.set_ylabel('Voltage')
    st.pyplot(fig)

def polarization_analysis(df):
    st.subheader("Polarization Analysis")
    
    required_columns = ['Charge Voltage', 'Discharge Voltage']
    missing_columns = [col for col in required_columns if col not in df.columns]
    
    if missing_columns:
        st.warning(f"Missing columns: {', '.join(missing_columns)}. Cannot perform Polarization Analysis.")
        return
    
    polarization = df['Charge Voltage'] - df['Discharge Voltage']
    st.line_chart(polarization)

def dq_de_analysis(df):
    st.subheader("dQ/dE Analysis")
    
    required_columns = ['Capacity', 'Voltage']
    missing_columns = [col for col in required_columns if col not in df.columns]
    
    if missing_columns:
        st.warning(f"Missing columns: {', '.join(missing_columns)}. Cannot perform dQ/dE Analysis.")
        return
    
    dq_de = np.gradient(df['Capacity'], df['Voltage'])
    fig, ax = plt.subplots()
    ax.plot(df['Voltage'], dq_de)
    ax.set_xlabel('Voltage')
    ax.set_ylabel('dQ/dE')
    st.pyplot(fig)

def kinetics_analysis(df):
    st.subheader("Kinetics Analysis")
    st.write("Kinetics analysis not implemented yet.")

def degradation_rate(df):
    st.subheader("Degradation Rate")
    
    if 'Discharge Capacity' not in df.columns:
        st.warning("'Discharge Capacity' column not found. Cannot calculate Degradation Rate.")
        return
    
    capacity_fade_rate = (df['Discharge Capacity'].iloc[0] - df['Discharge Capacity'].iloc[-1]) / len(df)
    st.write(f"Capacity fade rate: {capacity_fade_rate:.4f} mAh/cycle")

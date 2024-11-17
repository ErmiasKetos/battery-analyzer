import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def calculate_missing_columns(df):
    st.subheader("Calculating Missing Columns")
    
    if 'Charge Voltage' not in df.columns and 'Voltage (V)' in df.columns and 'Current (A)' in df.columns:
        df['Charge Voltage'] = df.loc[df['Current (A)'] > 0, 'Voltage (V)']
        st.write("Calculated 'Charge Voltage' based on positive current values.")
    
    if 'Discharge Voltage' not in df.columns and 'Voltage (V)' in df.columns and 'Current (A)' in df.columns:
        df['Discharge Voltage'] = df.loc[df['Current (A)'] < 0, 'Voltage (V)']
        st.write("Calculated 'Discharge Voltage' based on negative current values.")
    
    if 'Discharge Capacity (mAh)' not in df.columns and 'Current (A)' in df.columns and 'Time (s)' in df.columns:
        discharge_mask = df['Current (A)'] < 0
        df['Discharge Capacity (mAh)'] = (df.loc[discharge_mask, 'Current (A)'].abs() * df.loc[discharge_mask, 'Time (s)'] / 3600).cumsum()
        st.write("Calculated 'Discharge Capacity (mAh)' based on current and time.")
    
    # Provide downloadable CSV
    csv = df.to_csv(index=False)
    st.download_button(
        label="Download updated data as CSV",
        data=csv,
        file_name="updated_battery_data.csv",
        mime="text/csv"
    )
    
    return df

def voltage_profiles(df):
    st.subheader("Voltage Profiles")
    
    if 'Discharge Capacity (mAh)' in df.columns and 'Voltage (V)' in df.columns:
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(df['Discharge Capacity (mAh)'], df['Voltage (V)'])
        ax.set_xlabel('Discharge Capacity (mAh)')
        ax.set_ylabel('Voltage (V)')
        ax.set_title('Voltage vs Discharge Capacity')
        st.pyplot(fig)
    else:
        st.warning("Unable to plot Voltage Profiles. Make sure 'Discharge Capacity (mAh)' and 'Voltage (V)' columns are present.")

def polarization_analysis(df):
    st.subheader("Polarization Analysis")
    
    df = calculate_missing_columns(df)
    
    if 'Charge Voltage' in df.columns and 'Discharge Voltage' in df.columns:
        polarization = df['Charge Voltage'] - df['Discharge Voltage']
        
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(df['Cycle Number'], polarization)
        ax.set_xlabel('Cycle Number')
        ax.set_ylabel('Polarization (V)')
        ax.set_title('Polarization vs Cycle Number')
        st.pyplot(fig)
        
        st.write(f"Average Polarization: {polarization.mean():.4f} V")
        st.write(f"Maximum Polarization: {polarization.max():.4f} V")
    else:
        st.warning("Unable to perform Polarization Analysis. Make sure 'Charge Voltage' and 'Discharge Voltage' columns are present.")

def dq_de_analysis(df):
    st.subheader("dQ/dE Analysis")
    
    if 'Discharge Capacity (mAh)' in df.columns and 'Voltage (V)' in df.columns:
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
        st.warning("Unable to perform dQ/dE analysis. Make sure 'Discharge Capacity (mAh)' and 'Voltage (V)' columns are present.")

def kinetics_analysis(df):
    st.subheader("Kinetics Analysis")
    st.write("Kinetics analysis not implemented yet.")

def degradation_rate(df):
    st.subheader("Degradation Rate")
    
    df = calculate_missing_columns(df)
    
    if 'Discharge Capacity (mAh)' in df.columns and 'Cycle Number' in df.columns:
        initial_capacity = df['Discharge Capacity (mAh)'].iloc[0]
        final_capacity = df['Discharge Capacity (mAh)'].iloc[-1]
        total_cycles = df['Cycle Number'].max() - df['Cycle Number'].min()
        
        capacity_fade_rate = (initial_capacity - final_capacity) / total_cycles
        capacity_fade_percentage = (initial_capacity - final_capacity) / initial_capacity * 100
        
        st.write(f"Initial Capacity: {initial_capacity:.2f} mAh")
        st.write(f"Final Capacity: {final_capacity:.2f} mAh")
        st.write(f"Total Cycles: {total_cycles}")
        st.write(f"Capacity Fade Rate: {capacity_fade_rate:.4f} mAh/cycle")
        st.write(f"Total Capacity Fade: {capacity_fade_percentage:.2f}%")
        
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(df['Cycle Number'], df['Discharge Capacity (mAh)'])
        ax.set_xlabel('Cycle Number')
        ax.set_ylabel('Discharge Capacity (mAh)')
        ax.set_title('Discharge Capacity vs Cycle Number')
        st.pyplot(fig)
    else:
        st.warning("Unable to calculate Degradation Rate. Make sure 'Discharge Capacity (mAh)' and 'Cycle Number' columns are present.")

def advanced_analysis_main(df):
    st.header("Advanced Analysis")
    
    df = calculate_missing_columns(df)
    
    tab1, tab2, tab3, tab4, tab5 = st.tabs(["Voltage Profiles", "Polarization", "dQ/dE Analysis", "Kinetics", "Degradation Rate"])
    
    with tab1:
        voltage_profiles(df)
    
    with tab2:
        polarization_analysis(df)
    
    with tab3:
        dq_de_analysis(df)
    
    with tab4:
        kinetics_analysis(df)
    
    with tab5:
        degradation_rate(df)
    
    # Save updated data
    st.session_state.df = df

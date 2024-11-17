import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
from scipy.signal import savgol_filter

st.set_page_config(page_title="Battery Data Analyzer", layout="wide")

def calculate_dqdv(voltage, capacity, window_length=21, polyorder=3):
    """
    Calculate dQ/dV using Savitzky-Golay filter for smoothing
    """
    # Sort by voltage to ensure proper differentiation
    sort_idx = np.argsort(voltage)
    voltage_sorted = voltage[sort_idx]
    capacity_sorted = capacity[sort_idx]
    
    # Remove duplicate voltage values
    _, unique_idx = np.unique(voltage_sorted, return_index=True)
    voltage_unique = voltage_sorted[unique_idx]
    capacity_unique = capacity_sorted[unique_idx]
    
    # Smooth the capacity data
    capacity_smoothed = savgol_filter(capacity_unique, window_length, polyorder)
    
    # Calculate derivative
    dqdv = np.gradient(capacity_smoothed, voltage_unique)
    
    # Smooth the derivative
    dqdv_smoothed = savgol_filter(dqdv, window_length, polyorder)
    
    return voltage_unique, dqdv_smoothed

# [Previous code remains the same until the tabs section]

            # Create analysis tabs
            tab1, tab2, tab3, tab4 = st.tabs(["📈 Capacity Analysis", 
                                            "⚡ Voltage Analysis", 
                                            "🔄 Differential Capacity",
                                            "🔍 Detailed Metrics"])
            
            # [Previous tab1 and tab2 code remains the same]
            
            with tab3:
                st.subheader("Differential Capacity Analysis (dQ/dE)")
                
                # Add explanation
                with st.expander("ℹ️ About Differential Capacity Analysis"):
                    st.write("""
                    The differential capacity (dQ/dE) analysis helps identify:
                    - Phase transitions in electrode materials
                    - Changes in reaction mechanisms
                    - Battery degradation patterns
                    
                    **Interpretation:**
                    - Peaks indicate voltage plateaus where phase transitions occur
                    - Peak height relates to the amount of charge stored
                    - Peak shifts or changes in shape can indicate degradation
                    """)
                
                # Controls for dQ/dV analysis
                col1, col2 = st.columns(2)
                with col1:
                    cycle_number = st.selectbox(
                        "Select cycle for analysis",
                        options=sorted(df['Cycle'].unique()),
                        index=0
                    )
                    
                with col2:
                    smoothing_window = st.slider(
                        "Smoothing window size (must be odd)",
                        min_value=5,
                        max_value=51,
                        value=21,
                        step=2
                    )
                
                # Filter data for selected cycle
                cycle_data = df[df['Cycle'] == cycle_number]
                
                # Calculate dQ/dV for charge and discharge
                charge_voltage = cycle_data['Charge_Voltage'].values
                charge_capacity = cycle_data['Charge_Capacity'].values
                discharge_voltage = cycle_data['Discharge_Voltage'].values
                discharge_capacity = cycle_data['Discharge_Capacity'].values
                
                # Calculate derivatives
                charge_v, charge_dqdv = calculate_dqdv(
                    charge_voltage, 
                    charge_capacity, 
                    smoothing_window
                )
                discharge_v, discharge_dqdv = calculate_dqdv(
                    discharge_voltage, 
                    discharge_capacity, 
                    smoothing_window
                )
                
                # Create dQ/dV plot
                fig_dqdv = go.Figure()
                
                # Add charge curve
                fig_dqdv.add_trace(go.Scatter(
                    x=charge_v,
                    y=charge_dqdv,
                    name='Charge',
                    line=dict(color='red')
                ))
                
                # Add discharge curve
                fig_dqdv.add_trace(go.Scatter(
                    x=discharge_v,
                    y=-discharge_dqdv,  # Negative for conventional plotting
                    name='Discharge',
                    line=dict(color='blue')
                ))
                
                fig_dqdv.update_layout(
                    title=f'Differential Capacity Analysis - Cycle {cycle_number}',
                    xaxis_title='Voltage (V)',
                    yaxis_title='dQ/dV (mAh/V)',
                    hovermode='x unified'
                )
                
                st.plotly_chart(fig_dqdv, use_container_width=True)
                
                # Peak analysis
                with st.expander("🔍 Peak Analysis"):
                    # Find peaks in charge and discharge curves
                    from scipy.signal import find_peaks
                    
                    charge_peaks, _ = find_peaks(charge_dqdv, prominence=0.1)
                    discharge_peaks, _ = find_peaks(-discharge_dqdv, prominence=0.1)
                    
                    st.write("### Charge Peaks")
                    if len(charge_peaks) > 0:
                        peak_data = pd.DataFrame({
                            'Voltage (V)': charge_v[charge_peaks],
                            'dQ/dV (mAh/V)': charge_dqdv[charge_peaks]
                        })
                        st.write(peak_data)
                    else:
                        st.write("No significant peaks found")
                    
                    st.write("### Discharge Peaks")
                    if len(discharge_peaks) > 0:
                        peak_data = pd.DataFrame({
                            'Voltage (V)': discharge_v[discharge_peaks],
                            'dQ/dV (mAh/V)': -discharge_dqdv[discharge_peaks]
                        })
                        st.write(peak_data)
                    else:
                        st.write("No significant peaks found")
                
                # Cycle comparison
                with st.expander("📊 Compare Cycles"):
                    cycles_to_compare = st.multiselect(
                        "Select cycles to compare",
                        options=sorted(df['Cycle'].unique()),
                        default=[df['Cycle'].iloc[0], df['Cycle'].iloc[-1]]
                    )
                    
                    if cycles_to_compare:
                        fig_compare = go.Figure()
                        
                        for cycle in cycles_to_compare:
                            cycle_data = df[df['Cycle'] == cycle]
                            
                            # Calculate dQ/dV
                            charge_v, charge_dqdv = calculate_dqdv(
                                cycle_data['Charge_Voltage'].values,
                                cycle_data['Charge_Capacity'].values,
                                smoothing_window
                            )
                            
                            fig_compare.add_trace(go.Scatter(
                                x=charge_v,
                                y=charge_dqdv,
                                name=f'Cycle {cycle}',
                                mode='lines'
                            ))
                        
                        fig_compare.update_layout(
                            title='Cycle Comparison - Charge dQ/dV',
                            xaxis_title='Voltage (V)',
                            yaxis_title='dQ/dV (mAh/V)',
                            hovermode='x unified'
                        )
                        
                        st.plotly_chart(fig_compare, use_container_width=True)
                
                # Add insights about degradation
                if len(cycles_to_compare) >= 2:
                    st.write("### Degradation Analysis")
                    st.write("""
                    Changes in the dQ/dV curves between cycles can indicate:
                    - Loss of active material (decrease in peak height)
                    - Changes in reaction mechanisms (peak shift)
                    - Formation of new phases (new peaks)
                    - Loss of crystallinity (peak broadening)
                    """)

# [Rest of the code remains the same]

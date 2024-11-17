import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
from scipy.signal import savgol_filter, find_peaks

st.set_page_config(page_title="Battery Data Analyzer", layout="wide")

# Utility functions
def calculate_dqdv(voltage, capacity, window_length=21, polyorder=3):
    """Calculate dQ/dV using Savitzky-Golay filter for smoothing"""
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

# Title and description
st.title("🔋 Advanced Battery Data Analyzer")
st.write("Upload your battery test data and map your columns for detailed analysis.")

# File upload
st.subheader("📤 Upload Data")
uploaded_file = st.file_uploader("Choose your CSV file", type="csv")

if uploaded_file is not None:
    try:
        # Read CSV
        df_original = pd.read_csv(uploaded_file)
        
        # Show column mapping section
        st.subheader("🔄 Map Your Columns")
        st.write("Please match your columns to the required data types:")
        
        # Create column mapping
        col_mapping = {}
        required_columns = {
            'Cycle Number': 'Cycle',
            'Discharge Capacity': 'Discharge_Capacity',
            'Charge Capacity': 'Charge_Capacity',
            'Discharge Voltage': 'Discharge_Voltage',
            'Charge Voltage': 'Charge_Voltage'
        }
        
        # Display available columns
        col1, col2 = st.columns(2)
        with col1:
            st.write("Your file contains these columns:")
            st.write(", ".join(df_original.columns))
        
        # Create mapping dropdowns
        for display_name, internal_name in required_columns.items():
            col_mapping[internal_name] = st.selectbox(
                f"Select column for {display_name}:",
                options=[''] + list(df_original.columns),
                key=internal_name
            )
        
        # Process button
        if st.button("Process Data"):
            # Validate all columns are selected
            if '' in col_mapping.values():
                st.error("Please select all required columns!")
                st.stop()
            
            # Create renamed dataframe
            df = df_original.rename(columns={v: k for k, v in col_mapping.items()})
            
            # Create analysis tabs
            tabs = st.tabs([
                "📈 Capacity Analysis", 
                "⚡ Voltage Analysis", 
                "🔄 Differential Capacity",
                "🔍 Detailed Metrics"
            ])
            
            # Tab 1: Capacity Analysis
            with tabs[0]:
                st.subheader("Capacity Analysis")
                
                # Plot 1: Capacity vs Cycle
                fig_capacity = px.line(df, x='Cycle', 
                                     y=['Discharge_Capacity', 'Charge_Capacity'],
                                     title='Capacity vs Cycle Number',
                                     labels={'value': 'Capacity (mAh/g)',
                                            'variable': 'Type'})
                st.plotly_chart(fig_capacity, use_container_width=True)
                
                # Plot 2: Coulombic Efficiency
                df['Coulombic_Efficiency'] = (df['Discharge_Capacity'] / df['Charge_Capacity'] * 100)
                fig_efficiency = px.line(df, x='Cycle', 
                                       y='Coulombic_Efficiency',
                                       title='Coulombic Efficiency vs Cycle Number',
                                       labels={'Coulombic_Efficiency': 'Efficiency (%)'})
                st.plotly_chart(fig_efficiency, use_container_width=True)
                
                # Capacity metrics
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Initial Capacity", 
                             f"{df['Discharge_Capacity'].iloc[0]:.2f} mAh/g")
                    st.metric("Final Capacity", 
                             f"{df['Discharge_Capacity'].iloc[-1]:.2f} mAh/g")
                    st.metric("Capacity Retention", 
                             f"{(df['Discharge_Capacity'].iloc[-1]/df['Discharge_Capacity'].iloc[0]*100):.1f}%")
                
                with col2:
                    st.metric("Average Efficiency", 
                             f"{df['Coulombic_Efficiency'].mean():.2f}%")
                    st.metric("Minimum Efficiency", 
                             f"{df['Coulombic_Efficiency'].min():.2f}%")
                    st.metric("Maximum Efficiency", 
                             f"{df['Coulombic_Efficiency'].max():.2f}%")
            
            # Tab 2: Voltage Analysis
            with tabs[1]:
                st.subheader("Voltage Analysis")
                
                # Voltage vs Cycle plot
                fig_voltage = px.line(df, x='Cycle',
                                    y=['Charge_Voltage', 'Discharge_Voltage'],
                                    title='Voltage Profiles',
                                    labels={'value': 'Voltage (V)',
                                           'variable': 'Type'})
                st.plotly_chart(fig_voltage, use_container_width=True)
                
                # Voltage gap analysis
                df['Voltage_Gap'] = df['Charge_Voltage'] - df['Discharge_Voltage']
                fig_gap = px.line(df, x='Cycle',
                                 y='Voltage_Gap',
                                 title='Voltage Gap vs Cycle Number',
                                 labels={'Voltage_Gap': 'Voltage Gap (V)'})
                st.plotly_chart(fig_gap, use_container_width=True)
                
                # Voltage metrics
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Average Charge Voltage", 
                             f"{df['Charge_Voltage'].mean():.3f} V")
                    st.metric("Average Discharge Voltage", 
                             f"{df['Discharge_Voltage'].mean():.3f} V")
                
                with col2:
                    st.metric("Average Voltage Gap", 
                             f"{df['Voltage_Gap'].mean():.3f} V")
                    st.metric("Maximum Voltage Gap", 
                             f"{df['Voltage_Gap'].max():.3f} V")
            
            # Tab 3: Differential Capacity
            with tabs[2]:
                st.subheader("Differential Capacity Analysis (dQ/dE)")
                
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
                
                # Calculate and plot dQ/dV
                cycle_data = df[df['Cycle'] == cycle_number]
                
                charge_v, charge_dqdv = calculate_dqdv(
                    cycle_data['Charge_Voltage'].values,
                    cycle_data['Charge_Capacity'].values,
                    smoothing_window
                )
                
                discharge_v, discharge_dqdv = calculate_dqdv(
                    cycle_data['Discharge_Voltage'].values,
                    cycle_data['Discharge_Capacity'].values,
                    smoothing_window
                )
                
                # Create dQ/dV plot
                fig_dqdv = go.Figure()
                
                fig_dqdv.add_trace(go.Scatter(
                    x=charge_v,
                    y=charge_dqdv,
                    name='Charge',
                    line=dict(color='red')
                ))
                
                fig_dqdv.add_trace(go.Scatter(
                    x=discharge_v,
                    y=-discharge_dqdv,
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
                            
                            charge_v, charge_dqdv = calculate_dqdv(
                                cycle_data['Charge_Voltage'].values,
                                cycle_data['Charge_Capacity'].values,
                                smoothing_window
                            )
                            
                            fig_compare.add_trace(go.Scatter(
                                x=charge_v,
                                y=charge_dqdv,
                                name=f'Cycle {cycle}'
                            ))
                        
                        fig_compare.update_layout(
                            title='Cycle Comparison - Charge dQ/dV',
                            xaxis_title='Voltage (V)',
                            yaxis_title='dQ/dV (mAh/V)',
                            hovermode='x unified'
                        )
                        
                        st.plotly_chart(fig_compare, use_container_width=True)
                
                if len(cycles_to_compare) >= 2:
                    st.write("### Degradation Analysis")
                    st.write("""
                    Changes in the dQ/dV curves between cycles can indicate:
                    - Loss of active material (decrease in peak height)
                    - Changes in reaction mechanisms (peak shift)
                    - Formation of new phases (new peaks)
                    - Loss of crystallinity (peak broadening)
                    """)
            
            # Tab 4: Detailed Metrics
            with tabs[3]:
                st.subheader("Detailed Analysis")
                
                # Calculate additional metrics
                cycles = len(df)
                capacity_fade_rate = ((df['Discharge_Capacity'].iloc[0] - df['Discharge_Capacity'].iloc[-1]) / 
                                    (cycles * df['Discharge_Capacity'].iloc[0]) * 100)
                
                with st.expander("Capacity Analysis"):
                    st.write(f"""
                    - Total Cycles: {cycles}
                    - Initial Discharge Capacity: {df['Discharge_Capacity'].iloc[0]:.2f} mAh/g
                    - Final Discharge Capacity: {df['Discharge_Capacity'].iloc[-1]:.2f} mAh/g
                    - Capacity Retention: {(df['Discharge_Capacity'].iloc[-1]/df['Discharge_Capacity'].iloc[0]*100):.1f}%
                    - Capacity Fade Rate: {capacity_fade_rate:.4f}% per cycle
                    """)
                
                with st.expander("Statistical Summary"):
                    st.write("Discharge Capacity Statistics:")
                    st.write(df['Discharge_Capacity'].describe())
                    st.write("\nCoulombic Efficiency Statistics:")
                    st.write(df['Coulombic_Efficiency'].describe())
            
            # Download button for processed data
            st.download_button(
                label="📥 Download Analyzed Data as CSV",
                data=df.to_csv(index=False).encode('utf-8'),
                file_name='analyzed_battery_data.csv',
                mime='text/csv',
            )
            
    except Exception as e:
        st.error(f"Error analyzing file: {str(e)}")
        st.write("Please check your column mappings and data format.")

# Instructions
with st.expander("📖 How to Use"):
    st.write("""
    1. Upload your CSV file using the file uploader
    2. Map your columns to the required data types
    3. Click 'Process Data' to generate analysis
    4. Explore different tabs for various analyses
    5. Download processed data if needed
    
    Required data types:
    - Cycle Number: The cycle number of the test
    - Discharge Capacity: The discharge capacity in mAh/g
    - Charge Capacity: The charge capacity in mAh/g
    - Discharge Voltage: The discharge voltage in V
    - Charge Voltage: The charge voltage in V
    """)

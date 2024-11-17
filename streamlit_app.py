import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
from scipy.interpolate import interp1d
from scipy.signal import find_peaks
import plotly.subplots as sp

# Set page configuration first, before any other st commands
st.set_page_config(
    page_title="Battery Data Analyzer",
    page_icon="🔋",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Utility functions
def calculate_dqdv_proper(voltage, capacity, num_points=1000):
    try:
        sort_idx = np.argsort(voltage)
        voltage_sorted = voltage[sort_idx]
        capacity_sorted = capacity[sort_idx]
        
        _, unique_idx = np.unique(voltage_sorted, return_index=True)
        voltage_unique = voltage_sorted[unique_idx]
        capacity_unique = capacity_sorted[unique_idx]
        
        if len(voltage_unique) < 3:
            return None, None
            
        f = interp1d(voltage_unique, capacity_unique, kind='cubic', bounds_error=False)
        v_interp = np.linspace(voltage_unique.min(), voltage_unique.max(), num_points)
        q_interp = f(v_interp)
        dqdv = np.gradient(q_interp, v_interp)
        
        return v_interp, dqdv
        
    except Exception as e:
        st.error(f"Error in dQ/dV calculation: {str(e)}")
        return None, None

def calculate_capacity_metrics(df):
    metrics = {
        'Initial Discharge Capacity': df['Discharge_Capacity'].iloc[0],
        'Final Discharge Capacity': df['Discharge_Capacity'].iloc[-1],
        'Capacity Retention': (df['Discharge_Capacity'].iloc[-1] / df['Discharge_Capacity'].iloc[0] * 100),
        'Average Discharge Capacity': df['Discharge_Capacity'].mean(),
        'Capacity Loss Rate': ((df['Discharge_Capacity'].iloc[0] - df['Discharge_Capacity'].iloc[-1]) / 
                             (len(df) * df['Discharge_Capacity'].iloc[0]) * 100)
    }
    return metrics

def calculate_efficiency_metrics(df):
    df['Coulombic_Efficiency'] = (df['Discharge_Capacity'] / df['Charge_Capacity'] * 100)
    metrics = {
        'Average Efficiency': df['Coulombic_Efficiency'].mean(),
        'Minimum Efficiency': df['Coulombic_Efficiency'].min(),
        'Maximum Efficiency': df['Coulombic_Efficiency'].max(),
        'Efficiency Stability': df['Coulombic_Efficiency'].std()
    }
    return metrics

def calculate_voltage_metrics(df):
    df['Voltage_Gap'] = df['Charge_Voltage'] - df['Discharge_Voltage']
    metrics = {
        'Average Charge Voltage': df['Charge_Voltage'].mean(),
        'Average Discharge Voltage': df['Discharge_Voltage'].mean(),
        'Average Voltage Gap': df['Voltage_Gap'].mean(),
        'Maximum Voltage Gap': df['Voltage_Gap'].max(),
        'Voltage Stability': df['Voltage_Gap'].std()
    }
    return metrics

# Main application header
st.title("🔋 Advanced Battery Data Analyzer")
st.write("Upload your battery cycling data for comprehensive analysis and visualization.")

# Sidebar for global settings
with st.sidebar:
    st.header("Analysis Settings")
    
    plot_theme = st.selectbox(
        "Plot Theme",
        ["plotly", "plotly_white", "plotly_dark"],
        index=1
    )
    
    smoothing_factor = st.slider(
        "Data Smoothing",
        min_value=0,
        max_value=10,
        value=3,
        help="Higher values = smoother curves"
    )
    
    st.divider()
    
    with st.expander("📖 About This Tool"):
        st.write("""
        This tool provides comprehensive analysis of battery cycling data, including:
        - Capacity fade analysis
        - Coulombic efficiency tracking
        - Voltage profile analysis
        - Differential capacity analysis (dQ/dV)
        - Peak detection and phase transition analysis
        
        For best results, ensure your data includes:
        - Cycle numbers
        - Charge/Discharge capacity values
        - Charge/Discharge voltage values
        """)

# File upload section
st.subheader("📤 Upload Data")
uploaded_file = st.file_uploader("Choose your CSV file", type="csv")

if uploaded_file is not None:
    try:
        # Read CSV
        df_original = pd.read_csv(uploaded_file)
        
        # Column mapping section
        st.subheader("🔄 Map Your Columns")
        
        col1, col2 = st.columns([1, 2])
        with col1:
            st.write("Your file contains these columns:")
            st.info(", ".join(df_original.columns))
        
        # Create column mapping
        col_mapping = {}
        required_columns = {
            'Cycle Number': 'Cycle',
            'Discharge Capacity': 'Discharge_Capacity',
            'Charge Capacity': 'Charge_Capacity',
            'Discharge Voltage': 'Discharge_Voltage',
            'Charge Voltage': 'Charge_Voltage'
        }
        
        with col2:
            st.write("Map your columns to the required data types:")
            # Create mapping dropdowns
            for display_name, internal_name in required_columns.items():
                col_mapping[internal_name] = st.selectbox(
                    f"Select column for {display_name}:",
                    options=[''] + list(df_original.columns),
                    key=internal_name
                )
        
        # Process button
        if st.button("🔍 Process Data"):
            # Validate all columns are selected
            if '' in col_mapping.values():
                st.error("⚠️ Please select all required columns!")
                st.stop()
            
            # Create renamed dataframe
            df = df_original.rename(columns={v: k for k, v in col_mapping.items()})
            
            # Create analysis tabs
            tabs = st.tabs([
                "📈 Capacity Analysis",
                "⚡ Voltage Analysis",
                "🔄 Differential Capacity",
                "📊 Statistical Analysis",
                "📋 Raw Data"
            ])
            
      # Replace the "# Rest of your tab code..." section with this code:

            # Tab 1: Capacity Analysis
            with tabs[0]:
                st.subheader("Capacity Analysis")
                
                # Display metrics
                metrics = calculate_capacity_metrics(df)
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric("Initial Capacity", f"{metrics['Initial Discharge Capacity']:.2f} mAh/g")
                    st.metric("Final Capacity", f"{metrics['Final Discharge Capacity']:.2f} mAh/g")
                
                with col2:
                    st.metric("Capacity Retention", f"{metrics['Capacity Retention']:.1f}%")
                    st.metric("Average Capacity", f"{metrics['Average Discharge Capacity']:.2f} mAh/g")
                
                with col3:
                    st.metric("Capacity Loss Rate", f"{metrics['Capacity Loss Rate']:.4f}%/cycle")
                
                # Create capacity plots
                fig1 = go.Figure()
                
                # Add discharge capacity
                fig1.add_trace(go.Scatter(
                    x=df['Cycle'],
                    y=df['Discharge_Capacity'],
                    name='Discharge Capacity',
                    line=dict(color='blue')
                ))
                
                # Add charge capacity
                fig1.add_trace(go.Scatter(
                    x=df['Cycle'],
                    y=df['Charge_Capacity'],
                    name='Charge Capacity',
                    line=dict(color='red')
                ))
                
                fig1.update_layout(
                    title='Capacity vs Cycle Number',
                    xaxis_title='Cycle Number',
                    yaxis_title='Capacity (mAh/g)',
                    height=500
                )
                
                st.plotly_chart(fig1, use_container_width=True)
                
                # Calculate and plot efficiency
                df['Coulombic_Efficiency'] = (df['Discharge_Capacity'] / df['Charge_Capacity'] * 100)
                
                fig2 = go.Figure()
                fig2.add_trace(go.Scatter(
                    x=df['Cycle'],
                    y=df['Coulombic_Efficiency'],
                    name='Coulombic Efficiency',
                    line=dict(color='green')
                ))
                
                fig2.update_layout(
                    title='Coulombic Efficiency vs Cycle Number',
                    xaxis_title='Cycle Number',
                    yaxis_title='Efficiency (%)',
                    height=400
                )
                
                st.plotly_chart(fig2, use_container_width=True)
            
            # Tab 2: Voltage Analysis
            with tabs[1]:
                st.subheader("Voltage Analysis")
                
                voltage_metrics = calculate_voltage_metrics(df)
                col1, col2 = st.columns(2)
                
                with col1:
                    st.metric("Average Charge Voltage", f"{voltage_metrics['Average Charge Voltage']:.3f} V")
                    st.metric("Average Discharge Voltage", f"{voltage_metrics['Average Discharge Voltage']:.3f} V")
                
                with col2:
                    st.metric("Average Voltage Gap", f"{voltage_metrics['Average Voltage Gap']:.3f} V")
                    st.metric("Maximum Voltage Gap", f"{voltage_metrics['Maximum Voltage Gap']:.3f} V")
                
                # Voltage profile plots
                fig3 = go.Figure()
                
                fig3.add_trace(go.Scatter(
                    x=df['Cycle'],
                    y=df['Charge_Voltage'],
                    name='Charge Voltage',
                    line=dict(color='red')
                ))
                
                fig3.add_trace(go.Scatter(
                    x=df['Cycle'],
                    y=df['Discharge_Voltage'],
                    name='Discharge Voltage',
                    line=dict(color='blue')
                ))
                
                fig3.update_layout(
                    title='Voltage Profiles',
                    xaxis_title='Cycle Number',
                    yaxis_title='Voltage (V)',
                    height=500
                )
                
                st.plotly_chart(fig3, use_container_width=True)
            
            # Tab 3: Differential Capacity
            with tabs[2]:
                st.subheader("Differential Capacity Analysis (dQ/dE)")
                
                # Select cycle for analysis
                cycle_number = st.selectbox(
                    "Select cycle for analysis",
                    options=sorted(df['Cycle'].unique())
                )
                
                # Get data for selected cycle
                cycle_data = df[df['Cycle'] == cycle_number]
                
                # Calculate dQ/dV
                charge_v, charge_dqdv = calculate_dqdv_proper(
                    cycle_data['Charge_Voltage'].values,
                    cycle_data['Charge_Capacity'].values
                )
                
                if charge_v is not None:
                    fig4 = go.Figure()
                    
                    fig4.add_trace(go.Scatter(
                        x=charge_v,
                        y=charge_dqdv,
                        name='dQ/dV',
                        line=dict(color='blue')
                    ))
                    
                    fig4.update_layout(
                        title=f'Differential Capacity Analysis - Cycle {cycle_number}',
                        xaxis_title='Voltage (V)',
                        yaxis_title='dQ/dV (mAh/V)',
                        height=500
                    )
                    
                    st.plotly_chart(fig4, use_container_width=True)
                else:
                    st.warning("Could not calculate dQ/dV for this cycle")
            
            # Tab 4: Statistical Analysis
            with tabs[3]:
                st.subheader("Statistical Analysis")
                
                # Basic statistics
                st.write("### Basic Statistics")
                stats_df = df[['Discharge_Capacity', 'Charge_Capacity', 'Coulombic_Efficiency']].describe()
                st.write(stats_df)
                
                # Correlation analysis
                st.write("### Correlation Analysis")
                corr_matrix = df[['Discharge_Capacity', 'Charge_Capacity', 'Coulombic_Efficiency']].corr()
                
                fig5 = px.imshow(
                    corr_matrix,
                    text=corr_matrix.values.round(3),
                    aspect='auto',
                    color_continuous_scale='RdBu'
                )
                
                fig5.update_layout(
                    title='Correlation Matrix',
                    height=400
                )
                
                st.plotly_chart(fig5, use_container_width=True)
            
            # Tab 5: Raw Data
            with tabs[4]:
                st.subheader("Raw Data")
                st.write(df)
                
                # Download button
                csv = df.to_csv(index=False)
                st.download_button(
                    label="Download Processed Data",
                    data=csv,
                    file_name="processed_battery_data.csv",
                    mime="text/csv"
                )
    
            
    except Exception as e:
        st.error(f"Error processing file: {str(e)}")
        st.write("Please check your data format and try again.")

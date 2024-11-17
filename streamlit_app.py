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
            
            # Rest of your tab code...
            
    except Exception as e:
        st.error(f"Error processing file: {str(e)}")
        st.write("Please check your data format and try again.")

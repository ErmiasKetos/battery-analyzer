import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import plotly.subplots as sp
from scipy.interpolate import interp1d
from scipy.signal import find_peaks
import sklearn
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor, IsolationForest
from sklearn.cluster import KMeans

# Page config must be the first Streamlit command
st.set_page_config(page_title="Battery Data Analyzer", page_icon="🔋", layout="wide")

# Define utility functions
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

def calculate_metrics(df):
    """Calculate all metrics in one go"""
    try:
        metrics = {
            # Capacity metrics
            'Initial_Capacity': df['Discharge_Capacity'].iloc[0],
            'Final_Capacity': df['Discharge_Capacity'].iloc[-1],
            'Capacity_Retention': df['Discharge_Capacity'].iloc[-1] / df['Discharge_Capacity'].iloc[0] * 100,
            'Average_Capacity': df['Discharge_Capacity'].mean(),
            'Capacity_Loss_Rate': ((df['Discharge_Capacity'].iloc[0] - df['Discharge_Capacity'].iloc[-1]) / 
                                 (len(df) * df['Discharge_Capacity'].iloc[0]) * 100),
            
            # Efficiency metrics
            'Average_Efficiency': (df['Discharge_Capacity'] / df['Charge_Capacity'] * 100).mean(),
            'Min_Efficiency': (df['Discharge_Capacity'] / df['Charge_Capacity'] * 100).min(),
            'Max_Efficiency': (df['Discharge_Capacity'] / df['Charge_Capacity'] * 100).max(),
            
            # Voltage metrics
            'Average_Charge_V': df['Charge_Voltage'].mean(),
            'Average_Discharge_V': df['Discharge_Voltage'].mean(),
            'Voltage_Gap': (df['Charge_Voltage'] - df['Discharge_Voltage']).mean()
        }
        return metrics
    except Exception as e:
        st.error(f"Error calculating metrics: {str(e)}")
        return None

# Main app
try:
    st.title("🔋 Advanced Battery Data Analyzer")
    
    # File upload
    uploaded_file = st.file_uploader("Upload your battery data (CSV)", type="csv")
    
    if uploaded_file is not None:
        # Read data
        df = pd.read_csv(uploaded_file)
        
        # Create tabs
        tabs = st.tabs([
            "📈 Basic Analysis",
            "🔋 Capacity Analysis",
            "⚡ Voltage Analysis", 
            "🔄 dQ/dV Analysis",
            "📊 Statistics",
            "🤖 ML Analysis"
        ])
        
        # Basic Analysis tab
        with tabs[0]:
            st.subheader("Basic Data Analysis")
            st.write("Raw data preview:")
            st.write(df)
            
            if st.button("Calculate Basic Metrics"):
                metrics = calculate_metrics(df)
                if metrics:
                    cols = st.columns(3)
                    with cols[0]:
                        st.metric("Capacity Retention", f"{metrics['Capacity_Retention']:.1f}%")
                    with cols[1]:
                        st.metric("Average Efficiency", f"{metrics['Average_Efficiency']:.1f}%")
                    with cols[2]:
                        st.metric("Average Voltage Gap", f"{metrics['Voltage_Gap']:.3f}V")

        # Add other tabs...

except Exception as e:
    st.error(f"An error occurred: {str(e)}")
    st.write("Please check your data format and try again.")

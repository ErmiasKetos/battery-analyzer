import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np

st.set_page_config(page_title="Battery Data Analyzer", layout="wide")

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
        
        # Create two columns for mapping
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
            
            # General Analysis
            st.subheader("📊 Battery Performance Analysis")
            
            # Create analysis tabs
            tab1, tab2, tab3 = st.tabs(["📈 Capacity Analysis", "⚡ Voltage Analysis", "🔍 Detailed Metrics"])
            
            with tab1:
                # Capacity Analysis
                discharge_capacity = df['Discharge_Capacity'].dropna()
                charge_capacity = df['Charge_Capacity'].dropna()
                
                # Create subplot with shared x-axis
                fig = make_subplots(rows=2, cols=1, shared_xaxis=True,
                                  vertical_spacing=0.1,
                                  subplot_titles=('Capacity vs Cycle', 'Coulombic Efficiency'))
                
                # Add capacity traces
                fig.add_trace(
                    go.Scatter(x=df['Cycle'], y=df['Discharge_Capacity'],
                              name='Discharge Capacity', line=dict(color='blue')),
                    row=1, col=1
                )
                fig.add_trace(
                    go.Scatter(x=df['Cycle'], y=df['Charge_Capacity'],
                              name='Charge Capacity', line=dict(color='red')),
                    row=1, col=1
                )
                
                # Calculate and add efficiency
                efficiency = (df['Discharge_Capacity'] / df['Charge_Capacity'] * 100)
                fig.add_trace(
                    go.Scatter(x=df['Cycle'], y=efficiency,
                              name='Coulombic Efficiency', line=dict(color='green')),
                    row=2, col=1
                )
                
                fig.update_layout(height=800)
                fig.update_yaxes(title_text='Capacity (mAh/g)', row=1, col=1)
                fig.update_yaxes(title_text='Efficiency (%)', row=2, col=1)
                fig.update_xaxes(title_text='Cycle Number', row=2, col=1)
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Capacity metrics
                metrics_col1, metrics_col2 = st.columns(2)
                with metrics_col1:
                    st.metric("Initial Capacity", f"{discharge_capacity.iloc[0]:.2f} mAh/g")
                    st.metric("Final Capacity", f"{discharge_capacity.iloc[-1]:.2f} mAh/g")
                    st.metric("Capacity Retention", f"{(discharge_capacity.iloc[-1]/discharge_capacity.iloc[0]*100):.1f}%")
                
                with metrics_col2:
                    st.metric("Average Efficiency", f"{efficiency.mean():.2f}%")
                    st.metric("Minimum Efficiency", f"{efficiency.min():.2f}%")
                    st.metric("Maximum Efficiency", f"{efficiency.max():.2f}%")
            
            with tab2:
                # Voltage Analysis
                fig_voltage = go.Figure()
                
                # Voltage vs Cycle
                fig_voltage.add_trace(go.Scatter(x=df['Cycle'], y=df['Charge_Voltage'],
                                               name='Charge Voltage', line=dict(color='red')))
                fig_voltage.add_trace(go.Scatter(x=df['Cycle'], y=df['Discharge_Voltage'],
                                               name='Discharge Voltage', line=dict(color='blue')))
                
                # Calculate voltage gap
                voltage_gap = df['Charge_Voltage'] - df['Discharge_Voltage']
                fig_voltage.add_trace(go.Scatter(x=df['Cycle'], y=voltage_gap,
                                               name='Voltage Gap', line=dict(color='green')))
                
                fig_voltage.update_layout(
                    title='Voltage Profiles',
                    xaxis_title='Cycle Number',
                    yaxis_title='Voltage (V)',
                    height=600
                )
                
                st.plotly_chart(fig_voltage, use_container_width=True)
                
                # Voltage metrics
                v_col1, v_col2 = st.columns(2)
                with v_col1:
                    st.metric("Average Charge Voltage", f"{df['Charge_Voltage'].mean():.3f} V")
                    st.metric("Average Discharge Voltage", f"{df['Discharge_Voltage'].mean():.3f} V")
                
                with v_col2:
                    st.metric("Average Voltage Gap", f"{voltage_gap.mean():.3f} V")
                    st.metric("Maximum Voltage Gap", f"{voltage_gap.max():.3f} V")
            
            with tab3:
                # Detailed Analysis
                st.subheader("Detailed Performance Metrics")
                
                # Calculate additional metrics
                cycles = len(df)
                capacity_fade_rate = ((discharge_capacity.iloc[0] - discharge_capacity.iloc[-1]) / 
                                    (cycles * discharge_capacity.iloc[0]) * 100)
                
                # Display metrics in expandable sections
                with st.expander("Capacity Metrics"):
                    st.write(f"""
                    - Total Cycles: {cycles}
                    - Initial Discharge Capacity: {discharge_capacity.iloc[0]:.2f} mAh/g
                    - Final Discharge Capacity: {discharge_capacity.iloc[-1]:.2f} mAh/g
                    - Capacity Retention: {(discharge_capacity.iloc[-1]/discharge_capacity.iloc[0]*100):.1f}%
                    - Capacity Fade Rate: {capacity_fade_rate:.4f}% per cycle
                    """)
                
                with st.expander("Efficiency Metrics"):
                    st.write(f"""
                    - Average Coulombic Efficiency: {efficiency.mean():.2f}%
                    - Minimum Efficiency: {efficiency.min():.2f}%
                    - Maximum Efficiency: {efficiency.max():.2f}%
                    - Efficiency Standard Deviation: {efficiency.std():.2f}%
                    """)
                
                with st.expander("Voltage Metrics"):
                    st.write(f"""
                    - Average Charge Voltage: {df['Charge_Voltage'].mean():.3f} V
                    - Average Discharge Voltage: {df['Discharge_Voltage'].mean():.3f} V
                    - Average Voltage Gap: {voltage_gap.mean():.3f} V
                    - Maximum Voltage Gap: {voltage_gap.max():.3f} V
                    """)
                
                # Add statistical analysis
                with st.expander("Statistical Analysis"):
                    st.write("Discharge Capacity Statistics:")
                    st.write(discharge_capacity.describe())
                    
                    st.write("\nCoulombic Efficiency Statistics:")
                    st.write(efficiency.describe())
            
            # Download processed data
            st.download_button(
                label="📥 Download Analyzed Data as CSV",
                data=df.to_csv(index=False).encode('utf-8'),
                file_name='analyzed_battery_data.csv',
                mime='text/csv',
            )
            
    except Exception as e:
        st.error(f"Error analyzing file: {str(e)}")
        st.write("Please check your column mappings and data format.")

# Add instructions at the bottom
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

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
            
            # Create analysis tabs
            tab1, tab2, tab3 = st.tabs(["📈 Capacity Analysis", "⚡ Voltage Analysis", "🔍 Detailed Metrics"])
            
            with tab1:
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
                
                # Metrics
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
            
            with tab2:
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
            
            with tab3:
                st.subheader("Detailed Analysis")
                
                # Calculate additional metrics
                cycles = len(df)
                capacity_fade_rate = ((df['Discharge_Capacity'].iloc[0] - df['Discharge_Capacity'].iloc[-1]) / 
                                    (cycles * df['Discharge_Capacity'].iloc[0]) * 100)
                
                # Display detailed metrics
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

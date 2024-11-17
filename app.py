import streamlit as st
import pandas as pd
from scr.data_processing import upload_data, preview_data, data_processing_main
from scr.advanced_analysis import voltage_profiles, polarization_analysis, kinetics_analysis, degradation_rate
from scr.visualization import plot_capacity_vs_cycle, plot_voltage_vs_capacity, plot_dq_de_curves
from scr.li_s_features import polysulfide_shuttle_assessment, lithium_metal_anode_monitoring
from scr.ml_analysis import predict_capacity, detect_anomalies, estimate_rul

st.set_page_config(page_title="Li-S Battery Analyzer", layout="wide")

st.title("Li-S Battery Charge-Discharge Cycle Analyzer")

# Initialize session state
if 'df' not in st.session_state:
    st.session_state.df = None
if 'column_mapping' not in st.session_state:
    st.session_state.column_mapping = {}

# Sidebar for data upload and analysis options
st.sidebar.header("Data Upload & Analysis Options")

uploaded_file = st.sidebar.file_uploader("Upload CSV/Excel file", type=['csv', 'xlsx'])

if uploaded_file:
    df = upload_data(uploaded_file)
    
    if df is not None:
        st.session_state.df = df
        
        st.sidebar.header("Column Matching")
        
        required_columns = [
            "Cycle Number", "Discharge Capacity (mAh)", "Charge Capacity (mAh)",
            "Voltage (V)", "Current (A)", "Time (s)"
        ]
        
        for required_col in required_columns:
            st.session_state.column_mapping[required_col] = st.sidebar.selectbox(
                f"Match '{required_col}' to:",
                [""] + list(df.columns),
                key=f"match_{required_col}"
            )
        
        if st.sidebar.button("Apply Column Mapping"):
            # Rename columns based on user mapping
            reverse_mapping = {v: k for k, v in st.session_state.column_mapping.items() if v}
            st.session_state.df = st.session_state.df.rename(columns=reverse_mapping)
            st.success("Column mapping applied successfully!")
        
        st.sidebar.header("Analysis Options")
        analysis_type = st.sidebar.selectbox("Select Analysis Type", 
                                             ["Data Processing and Basic Analysis", "Advanced Analysis", "Li-S Specific Analysis", "ML Analysis"])
        
        # Preview raw data
        st.subheader("Raw Data Preview")
        preview_data(st.session_state.df)
        
        # Check if required columns are mapped
        missing_columns = [col for col in required_columns if col not in st.session_state.df.columns]
        
        if missing_columns:
            st.warning(f"The following required columns are not mapped: {', '.join(missing_columns)}. Please complete the column mapping before proceeding with analysis.")
        else:
            if analysis_type == "Data Processing and Basic Analysis":
                st.session_state.df = data_processing_main(st.session_state.df)
            
            elif analysis_type == "Advanced Analysis": advanced_analysis_main(st.session_state.df)
                
                tab1, tab2, tab3, tab4 = st.tabs(["Voltage Profiles", "Polarization", "Kinetics", "Degradation Rate"])
                
                with tab1:
                    voltage_profiles(st.session_state.df)
                
                with tab2:
                    polarization_analysis(st.session_state.df)
                
                with tab3:
                    kinetics_analysis(st.session_state.df)
                
                with tab4:
                    degradation_rate(st.session_state.df)
            
            elif analysis_type == "Li-S Specific Analysis":
                st.header("Li-S Specific Analysis")
                
                col1, col2 = st.columns(2)
                with col1:
                    st.subheader("Polysulfide Shuttle Assessment")
                    polysulfide_shuttle_assessment(st.session_state.df)
                
                with col2:
                    st.subheader("Lithium Metal Anode Monitoring")
                    lithium_metal_anode_monitoring(st.session_state.df)
            
            elif analysis_type == "ML Analysis":
                st.header("Machine Learning Analysis")
                
                tab1, tab2, tab3 = st.tabs(["Capacity Prediction", "Anomaly Detection", "RUL Estimation"])
                
                with tab1:
                    predict_capacity(st.session_state.df)
                
                with tab2:
                    detect_anomalies(st.session_state.df)
                
                with tab3:
                    estimate_rul(st.session_state.df)

else:
    st.info("Please upload a CSV or Excel file to begin analysis.")

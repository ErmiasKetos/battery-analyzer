import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scr.data_processing import upload_data, preview_data, basic_analysis, calculate_discharge_capacity
from scr.advanced_analysis import voltage_profiles, polarization_analysis, dq_de_analysis, kinetics_analysis, degradation_rate
from scr.visualization import plot_capacity_vs_cycle, plot_voltage_vs_capacity, plot_dq_de_curves
from scr.li_s_features import polysulfide_shuttle_assessment, lithium_metal_anode_monitoring
from scr.ml_analysis import predict_capacity, detect_anomalies, estimate_rul
from scr.ml_analysis import predict_capacity, detect_anomalies, estimate_rul


st.set_page_config(page_title="Advanced Battery Data Analyzer", layout="wide")

st.title("Advanced Battery Data Analyzer")

# Sidebar for data upload and analysis options
st.sidebar.header("Data Upload & Analysis Options")

uploaded_files = st.sidebar.file_uploader("Upload CSV/Excel files", accept_multiple_files=True, type=['csv', 'xlsx'])

if uploaded_files:
    datasets = {}
    for file in uploaded_files:
        datasets[file.name] = upload_data(file)
    
    selected_dataset = st.sidebar.selectbox("Select Dataset", list(datasets.keys()))
    df = datasets[selected_dataset]
    
    st.sidebar.header("Analysis Options")
    analysis_type = st.sidebar.selectbox("Select Analysis Type", 
                                         ["Basic Analysis", "Discharge Capacity Analysis", "Advanced Analysis", "Li-S Specific Analysis", "ML Analysis"])
    
    # Preview raw data
    st.subheader("Raw Data Preview")
    preview_data(df)
    
    if analysis_type == "Basic Analysis":
        st.header("Basic Analysis")
        discharge_capacity, capacity_retention, coulombic_efficiency = basic_analysis(df)
        
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Discharge Capacity")
            st.line_chart(discharge_capacity)
        with col2:
            st.subheader("Capacity Retention")
            st.line_chart(capacity_retention)
        
        st.subheader("Coulombic Efficiency")
        st.line_chart(coulombic_efficiency)
    
    elif analysis_type == "Discharge Capacity Analysis":
        st.header("Discharge Capacity Analysis")
        df = calculate_discharge_capacity(df)
    
    elif analysis_type == "Advanced Analysis":
        st.header("Advanced Analysis")
        
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
    
    elif analysis_type == "Li-S Specific Analysis":
        st.header("Li-S Specific Analysis")
        
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Polysulfide Shuttle Assessment")
            polysulfide_shuttle_assessment(df)
        
        with col2:
            st.subheader("Lithium Metal Anode Monitoring")
            lithium_metal_anode_monitoring(df)

elif analysis_type == "ML Analysis":
    st.header("Machine Learning Analysis")
    
    tab1, tab2, tab3 = st.tabs(["Capacity Prediction", "Anomaly Detection", "RUL Estimation"])
    
    with tab1:
        predict_capacity(df)
    
    with tab2:
        detect_anomalies(df)
    
    with tab3:
        estimate_rul(df)
    
    # Data Export
    st.sidebar.header("Data Export")
    if st.sidebar.button("Export Processed Data"):
        # Add export functionality here
        st.sidebar.success("Data exported successfully!")
    
    if st.sidebar.button("Export Visualizations"):
        # Add export functionality for visualizations here
        st.sidebar.success("Visualizations exported successfully!")

else:
    st.info("Please upload one or more CSV/Excel files to begin analysis.")

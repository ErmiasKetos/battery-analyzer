import streamlit as st
import pandas as pd
import plotly.express as px

st.set_page_config(page_title="Battery Data Analyzer", layout="wide")

# Title and description
st.title("🔋 Battery Data Analyzer")
st.write("Upload your battery test data to analyze performance metrics and visualize results.")

# File upload with instructions
st.subheader("📤 Upload Data")
st.write("""
Your CSV file should contain these columns:
- Cycle
- Discharge_Capacity
- Charge_Capacity
- Discharge_Voltage
- Charge_Voltage
""")

uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

if uploaded_file is not None:
    try:
        # Read CSV
        df = pd.read_csv(uploaded_file)
        
        # Check required columns
        required_columns = ['Cycle', 'Discharge_Capacity', 'Charge_Capacity', 
                          'Discharge_Voltage', 'Charge_Voltage']
        
        missing_columns = [col for col in required_columns if col not in df.columns]
        
        if missing_columns:
            st.error(f"Missing required columns: {', '.join(missing_columns)}")
            st.stop()
        
        # Display basic info in two columns
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("📊 General Analysis")
            discharge_capacity = df['Discharge_Capacity'].dropna()
            charge_capacity = df['Charge_Capacity'].dropna()
            
            metrics = {
                "Total Cycles": len(discharge_capacity),
                "Avg Discharge Capacity (mAh/g)": discharge_capacity.mean().round(2),
                "Max Discharge Capacity (mAh/g)": discharge_capacity.max().round(2),
                "Min Discharge Capacity (mAh/g)": discharge_capacity.min().round(2),
                "Capacity Retention (%)": (discharge_capacity.iloc[-1] / discharge_capacity.iloc[0] * 100).round(2)
            }
            
            for label, value in metrics.items():
                st.metric(label, value)
        
        with col2:
            st.subheader("⚡ Voltage Analysis")
            voltage_metrics = {
                "Avg Charge Voltage (V)": df['Charge_Voltage'].mean().round(3),
                "Avg Discharge Voltage (V)": df['Discharge_Voltage'].mean().round(3),
                "Voltage Gap (V)": (df['Charge_Voltage'].mean() - df['Discharge_Voltage'].mean()).round(3)
            }
            
            for label, value in voltage_metrics.items():
                st.metric(label, value)
        
        # Plotting
        st.subheader("📈 Capacity vs Cycle Number")
        
        fig = px.line(df, x='Cycle', y=['Discharge_Capacity', 'Charge_Capacity'],
                      title='Capacity vs Cycle',
                      labels={'value': 'Capacity (mAh/g)', 'variable': 'Type'})
        st.plotly_chart(fig, use_container_width=True)
        
        # Efficiency plot
        df['Coulombic_Efficiency'] = (df['Discharge_Capacity'] / df['Charge_Capacity'] * 100)
        
        fig_eff = px.line(df, x='Cycle', y='Coulombic_Efficiency',
                          title='Coulombic Efficiency vs Cycle',
                          labels={'Coulombic_Efficiency': 'Efficiency (%)'})
        st.plotly_chart(fig_eff, use_container_width=True)
        
        # Download processed data
        st.download_button(
            label="📥 Download Analyzed Data as CSV",
            data=df.to_csv(index=False).encode('utf-8'),
            file_name='analyzed_battery_data.csv',
            mime='text/csv',
        )
        
    except Exception as e:
        st.error(f"Error analyzing file: {str(e)}")
        st.write("Please make sure your CSV file is properly formatted.")

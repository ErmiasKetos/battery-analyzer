import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import plotly.subplots as sp
from scipy.interpolate import interp1d
from scipy.signal import find_peaks
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor, IsolationForest
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from sklearn.cluster import KMeans

# Set page configuration
st.set_page_config(
    page_title="Battery Data Analyzer",
    page_icon="🔋",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Utility functions
def create_features(df, window_size=5):
    """Create features for ML models"""
    features = []
    targets = []

    for i in range(len(df) - window_size):
        features.append(df.iloc[i:i+window_size][
            ['Discharge_Capacity', 'Charge_Capacity', 'Coulombic_Efficiency']
        ].values.flatten())
        targets.append(df.iloc[i+window_size]['Discharge_Capacity'])

    return np.array(features), np.array(targets)

def train_ml_model(X_train, X_test, y_train, y_test, model_type="Random Forest"):
    """Train and evaluate ML model"""
    if model_type == "Random Forest":
        model = RandomForestRegressor(n_estimators=100, random_state=42)
    elif model_type == "Support Vector Machine":
        model = SVR(kernel='rbf')
    else:
        model = MLPRegressor(hidden_layer_sizes=(100, 50), max_iter=1000)

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    return model, y_pred, {
        'mse': mean_squared_error(y_test, y_pred),
        'r2': r2_score(y_test, y_pred)
    }

def calculate_dqdv(voltage, capacity, num_points=1000):
    """Calculate dQ/dV using proper numerical differentiation"""
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
        dv = v_interp[1] - v_interp[0]
        dqdv = np.gradient(q_interp, dv)

        return v_interp, dqdv

    except Exception as e:
        st.error(f"Error in dQ/dV calculation: {str(e)}")
        return None, None

# Analysis functions
def capacity_analysis(df):
    st.subheader("Capacity Analysis")

    # Capacity metrics
    initial_capacity = df['Discharge_Capacity'].iloc[0]
    final_capacity = df['Discharge_Capacity'].iloc[-1]
    capacity_retention = (final_capacity / initial_capacity) * 100
    
    col1, col2, col3 = st.columns(3)
    col1.metric("Initial Capacity", f"{initial_capacity:.2f} mAh/g")
    col2.metric("Final Capacity", f"{final_capacity:.2f} mAh/g")
    col3.metric("Capacity Retention", f"{capacity_retention:.2f}%")

    # Capacity plot
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df['Cycle'], y=df['Discharge_Capacity'], mode='lines+markers', name='Discharge Capacity'))
    fig.add_trace(go.Scatter(x=df['Cycle'], y=df['Charge_Capacity'], mode='lines+markers', name='Charge Capacity'))
    fig.update_layout(title='Capacity vs Cycle', xaxis_title='Cycle', yaxis_title='Capacity (mAh/g)')
    st.plotly_chart(fig, use_container_width=True)

def voltage_analysis(df):
    st.subheader("Voltage Analysis")

    # Voltage metrics
    avg_discharge_voltage = df['Discharge_Voltage'].mean()
    avg_charge_voltage = df['Charge_Voltage'].mean()
    
    col1, col2 = st.columns(2)
    col1.metric("Avg Discharge Voltage", f"{avg_discharge_voltage:.2f} V")
    col2.metric("Avg Charge Voltage", f"{avg_charge_voltage:.2f} V")

    # Voltage plot
    cycle = st.selectbox("Select Cycle", df['Cycle'].unique())
    cycle_data = df[df['Cycle'] == cycle]
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=cycle_data['Discharge_Capacity'], y=cycle_data['Discharge_Voltage'], mode='lines', name='Discharge'))
    fig.add_trace(go.Scatter(x=cycle_data['Charge_Capacity'], y=cycle_data['Charge_Voltage'], mode='lines', name='Charge'))
    fig.update_layout(title=f'Voltage Profile - Cycle {cycle}', xaxis_title='Capacity (mAh/g)', yaxis_title='Voltage (V)')
    st.plotly_chart(fig, use_container_width=True)

def differential_capacity_analysis(df):
    st.subheader("Differential Capacity Analysis")

    cycle = st.selectbox("Select Cycle", df['Cycle'].unique())
    cycle_data = df[df['Cycle'] == cycle]

    v_charge, dqdv_charge = calculate_dqdv(cycle_data['Charge_Voltage'].values, cycle_data['Charge_Capacity'].values)
    v_discharge, dqdv_discharge = calculate_dqdv(cycle_data['Discharge_Voltage'].values, cycle_data['Discharge_Capacity'].values)

    if v_charge is not None and v_discharge is not None:
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=v_charge, y=dqdv_charge, mode='lines', name='Charge'))
        fig.add_trace(go.Scatter(x=v_discharge, y=-dqdv_discharge, mode='lines', name='Discharge'))
        fig.update_layout(title=f'dQ/dV Analysis - Cycle {cycle}', xaxis_title='Voltage (V)', yaxis_title='dQ/dV')
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("Unable to calculate dQ/dV for the selected cycle.")

def statistical_analysis(df):
    st.subheader("Statistical Analysis")

    # Basic statistics
    st.write(df.describe())

    # Correlation heatmap
    corr = df[['Discharge_Capacity', 'Charge_Capacity', 'Coulombic_Efficiency']].corr()
    fig = px.imshow(corr, text_auto=True, aspect="equal")
    fig.update_layout(title='Correlation Heatmap')
    st.plotly_chart(fig, use_container_width=True)

def machine_learning_analysis(df):
    st.subheader("Machine Learning Analysis")

    analysis_type = st.radio("Select Analysis Type", ["Capacity Prediction", "Anomaly Detection", "Pattern Recognition"])

    if analysis_type == "Capacity Prediction":
        capacity_prediction(df)
    elif analysis_type == "Anomaly Detection":
        anomaly_detection(df)
    else:
        pattern_recognition(df)

def capacity_prediction(df):
    st.write("### Capacity Prediction Model")

    model_type = st.selectbox("Select Model Type", ["Random Forest", "Support Vector Machine", "Neural Network"])
    window_size = st.slider("Window Size", 3, 10, 5)

    if st.button("Train Model"):
        X, y = create_features(df, window_size)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        model, y_pred, metrics = train_ml_model(X_train_scaled, X_test_scaled, y_train, y_test, model_type)

        col1, col2 = st.columns(2)
        col1.metric("Mean Squared Error", f"{metrics['mse']:.4f}")
        col2.metric("R² Score", f"{metrics['r2']:.4f}")

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=range(len(y_test)), y=y_test, mode='lines', name="Actual"))
        fig.add_trace(go.Scatter(x=range(len(y_pred)), y=y_pred, mode='lines', name="Predicted"))
        fig.update_layout(title="Model Predictions vs Actual Values", xaxis_title="Sample", yaxis_title="Capacity")
        st.plotly_chart(fig, use_container_width=True)

def anomaly_detection(df):
    st.write("### Anomaly Detection")

    contamination = st.slider("Contamination", 0.01, 0.2, 0.1)

    if st.button("Detect Anomalies"):
        features = df[['Discharge_Capacity', 'Charge_Capacity', 'Coulombic_Efficiency']]
        scaler = StandardScaler()
        features_scaled = scaler.fit_transform(features)

        iso_forest = IsolationForest(contamination=contamination, random_state=42)
        anomalies = iso_forest.fit_predict(features_scaled)

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df[anomalies == 1]['Cycle'], y=df[anomalies == 1]['Discharge_Capacity'], mode='markers', name='Normal'))
        fig.add_trace(go.Scatter(x=df[anomalies == -1]['Cycle'], y=df[anomalies == -1]['Discharge_Capacity'], mode='markers', name='Anomaly', marker=dict(color='red')))
        fig.update_layout(title='Anomaly Detection', xaxis_title='Cycle', yaxis_title='Discharge Capacity')
        st.plotly_chart(fig, use_container_width=True)

def pattern_recognition(df):
    st.write("### Pattern Recognition")

    required_columns = ['Discharge_Capacity', 'Coulombic_Efficiency']
    missing_columns = [col for col in required_columns if col not in df.columns]

    if missing_columns:
        st.error(f"Error: The following required columns are missing from your data: {', '.join(missing_columns)}")
        st.write("Please ensure your data includes these columns and try again.")
        return

    n_clusters = st.slider("Number of patterns", 2, 10, 3)

    if st.button("Identify Patterns"):
        features = df[required_columns]
        
        if not features.apply(lambda x: pd.to_numeric(x, errors='coerce')).notna().all().all():
            st.error("Error: Non-numeric data found in the selected columns. Please ensure all values are numeric.")
            return

        scaler = StandardScaler()
        features_scaled = scaler.fit_transform(features)

        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        clusters = kmeans.fit_predict(features_scaled)

        fig = go.Figure()
        for i in range(n_clusters):
            mask = clusters == i
            fig.add_trace(go.Scatter(x=df[mask]['Cycle'], y=df[mask]['Discharge_Capacity'], mode='markers', name=f'Pattern {i+1}'))
        
        fig.update_layout(title='Battery Discharge Capacity Patterns', xaxis_title='Cycle', yaxis_title='Discharge Capacity')
        st.plotly_chart(fig, use_container_width=True)

        st.write("### Cluster Statistics")
        for i in range(n_clusters):
            mask = clusters == i
            cluster_data = df[mask]
            st.write(f"Cluster {i+1}:")
            st.write(f"  - Number of points: {mask.sum()}")
            st.write(f"  - Average Discharge Capacity: {cluster_data['Discharge_Capacity'].mean():.2f}")
            st.write(f"  - Average Coulombic Efficiency: {cluster_data['Coulombic_Efficiency'].mean():.2f}%")

def main():
    st.title("🔋 Advanced Battery Data Analyzer")
    st.write("Upload your battery cycling data for comprehensive analysis.")

    uploaded_file = st.file_uploader("Choose your CSV file", type="csv")

    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)

            required_columns = ['Cycle', 'Discharge_Capacity', 'Charge_Capacity', 'Discharge_Voltage', 'Charge_Voltage', 'Coulombic_Efficiency']
            missing_columns = [col for col in required_columns if col not in df.columns]

            if missing_columns:
                st.error(f"Error: The following required columns are missing from your data: {', '.join(missing_columns)}")
                st.write("Please ensure your CSV file includes these columns and try again.")
                return

            tabs = st.tabs([
                "📈 Capacity Analysis",
                "⚡ Voltage Analysis",
                "🔄 Differential Capacity",
                "📊 Statistical Analysis",
                "🤖 Machine Learning",
                "📋 Raw Data"
            ])

            with tabs[0]:
                capacity_analysis(df)
            
            with tabs[1]:
                voltage_analysis(df)
            
            with tabs[2]:
                differential_capacity_analysis(df)
            
            with tabs[3]:
                statistical_analysis(df)
            
            with tabs[4]:
                machine_learning_analysis(df)
            
            with tabs[5]:
                st.subheader("Raw Data")
                st.write(df)

        except Exception as e:
            st.error(f"An error occurred while processing your data: {str(e)}")
            st.write("Please check your data format and try again.")

if __name__ == "__main__":
    main()

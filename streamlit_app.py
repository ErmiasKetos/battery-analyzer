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
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from sklearn.cluster import KMeans
from sklearn.model_selection import cross_val_score

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

# Main application
def main():
    st.title("🔋 Advanced Battery Data Analyzer")
    st.write("Upload your battery cycling data for comprehensive analysis.")

    # File upload
    uploaded_file = st.file_uploader("Choose your CSV file", type="csv")

    if uploaded_file is not None:
        # Read and process data
        df = pd.read_csv(uploaded_file)

        # Create tabs
        tabs = st.tabs([
            "📈 Capacity Analysis",
            "⚡ Voltage Analysis",
            "🔄 Differential Capacity",
            "📊 Statistical Analysis",
            "🤖 Machine Learning",
            "📋 Raw Data"
        ])

        # ML Analysis Tab
        with tabs[4]:
            st.subheader("🤖 Machine Learning Analysis")

            ml_analysis_type = st.radio(
                "Select Analysis Type",
                ["Capacity Prediction", "Anomaly Detection", "Pattern Recognition", "RUL Estimation"],
                horizontal=True
            )

            if ml_analysis_type == "Capacity Prediction":
                capacity_prediction(df)
            elif ml_analysis_type == "Anomaly Detection":
                anomaly_detection(df)
            elif ml_analysis_type == "Pattern Recognition":
                pattern_recognition(df)
            else:  # RUL Estimation
                rul_estimation(df)

def capacity_prediction(df):
    st.write("### Capacity Prediction Model")

    # Model selection
    model_type = st.selectbox(
        "Select Model Type",
        ["Random Forest", "Support Vector Machine", "Neural Network"]
    )

    window_size = st.slider("Window Size", 3, 10, 5)

    if st.button("Train Model"):
        with st.spinner("Training model..."):
            # Prepare data
            X, y = create_features(df, window_size)
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42
            )

            # Scale features
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)

            # Train and evaluate model
            model, y_pred, metrics = train_ml_model(
                X_train_scaled, X_test_scaled, y_train, y_test, model_type
            )

            # Display results
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Mean Squared Error", f"{metrics['mse']:.4f}")
            with col2:
                st.metric("R² Score", f"{metrics['r2']:.4f}")

            # Plot predictions
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=range(len(y_test)), y=y_test, name="Actual"))
            fig.add_trace(go.Scatter(x=range(len(y_pred)), y=y_pred, name="Predicted"))
            fig.update_layout(title="Model Predictions vs Actual Values")
            st.plotly_chart(fig, use_container_width=True)

def anomaly_detection(df):
    st.write("### Anomaly Detection")

    contamination = st.slider("Contamination", 0.01, 0.2, 0.1)

    if st.button("Detect Anomalies"):
        # Prepare data
        features = df[['Discharge_Capacity', 'Charge_Capacity', 'Coulombic_Efficiency']]
        scaler = StandardScaler()
        features_scaled = scaler.fit_transform(features)

        # Detect anomalies
        iso_forest = IsolationForest(contamination=contamination, random_state=42)
        anomalies = iso_forest.fit_predict(features_scaled)

        # Plot results
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=df[anomalies == 1]['Cycle'],
            y=df[anomalies == 1]['Discharge_Capacity'],
            mode='markers',
            name='Normal'
        ))
        fig.add_trace(go.Scatter(
            x=df[anomalies == -1]['Cycle'],
            y=df[anomalies == -1]['Discharge_Capacity'],
            mode='markers',
            name='Anomaly',
            marker=dict(color='red')
        ))
        st.plotly_chart(fig, use_container_width=True)

def pattern_recognition(df):
    st.write("### Pattern Recognition")

    n_clusters = st.slider("Number of patterns", 2, 10, 3)

    if st.button("Identify Patterns"):
        # Prepare data
        features = df[['Discharge_Capacity', 'Coulombic_Efficiency']]
        scaler = StandardScaler()
        features_scaled = scaler.fit_transform(features)

        # Perform clustering
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        clusters = kmeans.fit_predict(features_scaled)

        # Plot results
        fig = go.Figure()
        for i in range(n_clusters):
            mask = clusters == i
            fig.add_trace(go.Scatter(
                x=df[mask]['Cycle'],
                y=df[mask]['Discharge_Capacity'],
                mode='markers',
                name=f'Pattern {i+1}'
            ))
        st.plotly_chart(fig, use_container_width=True)

def rul_estimation(df):
    st.write("### Remaining Useful Life Estimation")

    threshold = st.slider("Capacity Threshold (%)", 60, 90, 80)

    if st.button("Estimate RUL"):
        # Prepare data
        initial_capacity = df['Discharge_Capacity'].iloc[0]
        threshold_capacity = initial_capacity * threshold / 100

        # Train model
        X = df[['Cycle']]
        y = df['Discharge_Capacity']
        model = RandomForestRegressor(n_estimators=100)
        model.fit(X, y)

        # Predict future cycles
        future_cycles = np.arange(df['Cycle'].max(), df['Cycle'].max() + 100)
        future_capacity = model.predict(future_cycles.reshape(-1, 1))

        # Plot results
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df['Cycle'], y=df['Discharge_Capacity'], name='Historical'))
        fig.add_trace(go.Scatter(x=future_cycles, y=future_capacity, name='Predicted'))
        fig.add_hline(y=threshold_capacity, line_dash="dash", line_color="red")
        st.plotly_chart(fig, use_container_width=True)

if __name__ == "__main__":
    main()

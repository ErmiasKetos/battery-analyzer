import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler

def predict_capacity(df):
    st.subheader("Capacity Prediction")
    
    required_columns = ['Cycle', 'Charge Capacity', 'Discharge Capacity', 'Coulombic Efficiency']
    missing_columns = [col for col in required_columns if col not in df.columns]
    
    if missing_columns:
        st.warning(f"Missing columns: {', '.join(missing_columns)}. Cannot perform Capacity Prediction.")
        return
    
    # Prepare data
    X = df[required_columns]
    y = df['Discharge Capacity'].shift(-1)  # Predict next cycle's capacity
    X = X[:-1]  # Remove last row
    y = y[:-1]  # Remove last row
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train model
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train_scaled, y_train)
    
    # Make predictions
    y_pred = model.predict(X_test_scaled)
    
    # Calculate RMSE
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    st.write(f"Root Mean Squared Error: {rmse:.4f}")
    
    # Plot actual vs predicted
    fig, ax = plt.subplots()
    ax.scatter(y_test, y_pred)
    ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
    ax.set_xlabel('Actual Capacity')
    ax.set_ylabel('Predicted Capacity')
    ax.set_title('Actual vs Predicted Capacity')
    st.pyplot(fig)

def detect_anomalies(df):
    st.subheader("Anomaly Detection")
    st.write("Anomaly detection not implemented yet.")

def estimate_rul(df):
    st.subheader("Remaining Useful Life (RUL) Estimation")
    st.write("RUL estimation not implemented yet.")

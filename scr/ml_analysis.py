import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

def clean_and_prepare_data(df):
    st.subheader("Data Cleaning and Preparation")
    
    # Calculate missing metrics
    if 'Charge Capacity' not in df.columns and 'Charge Current' in df.columns and 'Charge Time' in df.columns:
        df['Charge Capacity'] = df['Charge Current'] * df['Charge Time']
        st.write("Calculated Charge Capacity from Charge Current and Charge Time.")
    
    if 'Discharge Capacity' not in df.columns and 'Discharge Current' in df.columns and 'Discharge Time' in df.columns:
        df['Discharge Capacity'] = df['Discharge Current'] * df['Discharge Time']
        st.write("Calculated Discharge Capacity from Discharge Current and Discharge Time.")
    
    if 'Coulombic Efficiency' not in df.columns and 'Charge Capacity' in df.columns and 'Discharge Capacity' in df.columns:
        df['Coulombic Efficiency'] = (df['Discharge Capacity'] / df['Charge Capacity']) * 100
        st.write("Calculated Coulombic Efficiency from Charge Capacity and Discharge Capacity.")
    
    # Generate missing data if needed
    required_columns = ['Cycle', 'Charge Capacity', 'Discharge Capacity', 'Coulombic Efficiency']
    missing_columns = [col for col in required_columns if col not in df.columns]
    
    if missing_columns:
        st.warning(f"Missing columns: {', '.join(missing_columns)}. Generating synthetic data for these columns.")
        df = generate_missing_data(df, missing_columns)
    
    return df

def generate_missing_data(df, missing_columns):
    for col in missing_columns:
        if col == 'Cycle':
            df[col] = range(1, len(df) + 1)
        elif col in ['Charge Capacity', 'Discharge Capacity']:
            # Generate synthetic capacity data with some degradation
            initial_capacity = 1000  # mAh
            degradation_rate = 0.05  # 5% per 100 cycles
            noise = np.random.normal(0, 10, len(df))
            df[col] = initial_capacity * (1 - degradation_rate * df['Cycle'] / 100) + noise
        elif col == 'Coulombic Efficiency':
            # Generate synthetic Coulombic Efficiency data
            df[col] = 98 + np.random.normal(0, 0.5, len(df))
            df[col] = df[col].clip(0, 100)  # Ensure values are between 0 and 100
    
    return df

def interpret_data(df):
    st.subheader("Data Interpretation")
    
    # Simulating AI-based interpretation
    avg_coulombic_efficiency = df['Coulombic Efficiency'].mean()
    capacity_fade = (df['Discharge Capacity'].iloc[0] - df['Discharge Capacity'].iloc[-1]) / df['Discharge Capacity'].iloc[0] * 100
    
    interpretation = f"""
    Based on the analysis of the battery data:
    
    1. The average Coulombic Efficiency is {avg_coulombic_efficiency:.2f}%. 
       {
       'This indicates excellent charge-discharge efficiency.' if avg_coulombic_efficiency > 99 else
       'This suggests good overall performance, but there might be room for improvement.' if avg_coulombic_efficiency > 95 else
       'This indicates potential issues with the charge-discharge process that need investigation.'
       }
    
    2. The capacity fade over the measured cycles is {capacity_fade:.2f}%. 
       {
       'This suggests minimal degradation and excellent cycle life.' if capacity_fade < 10 else
       'This indicates moderate degradation, typical for many Li-ion batteries.' if capacity_fade < 20 else
       'This shows significant capacity loss, suggesting accelerated aging or potential issues with the battery.'
       }
    
    3. {
       'The battery shows consistent performance across cycles.' if df['Discharge Capacity'].std() / df['Discharge Capacity'].mean() < 0.05 else
       'There is noticeable variation in discharge capacity across cycles, which may indicate inconsistent performance or measurement issues.'
       }
    
    Recommendations:
    1. {
       'Continue monitoring the battery performance to maintain its excellent characteristics.' if avg_coulombic_efficiency > 99 and capacity_fade < 10 else
       'Investigate factors affecting Coulombic Efficiency to improve overall performance.' if avg_coulombic_efficiency <= 99 else
       'Conduct a thorough review of the battery design and usage conditions to address the significant capacity fade.' if capacity_fade >= 20 else
       'Optimize charging protocols to improve consistency across cycles.'
       }
    2. Consider conducting additional tests such as rate capability and long-term cycling to further characterize the battery performance.
    """
    
    st.write(interpretation)

def predict_capacity(df):
    st.subheader("Capacity Prediction")
    
    df = clean_and_prepare_data(df)
    interpret_data(df)
    
    # Prepare data for ML model
    X = df[['Cycle', 'Charge Capacity', 'Discharge Capacity', 'Coulombic Efficiency']]
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
    ax.scatter(y_test, y_pred, alpha=0.5)
    ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
    ax.set_xlabel('Actual Capacity')
    ax.set_ylabel('Predicted Capacity')
    ax.set_title('Actual vs Predicted Capacity')
    st.pyplot(fig)
    
    # Feature importance
    feature_importance = pd.DataFrame({'feature': X.columns, 'importance': model.feature_importances_})
    feature_importance = feature_importance.sort_values('importance', ascending=False)
    st.write("Feature Importance:")
    st.write(feature_importance)
    
    # Predict future cycles
    last_cycle = df['Cycle'].max()
    future_cycles = pd.DataFrame({'Cycle': range(last_cycle + 1, last_cycle + 51)})
    future_X = pd.concat([X.tail(1)] * 50, ignore_index=True)
    future_X['Cycle'] = future_cycles['Cycle']
    future_X_scaled = scaler.transform(future_X)
    future_predictions = model.predict(future_X_scaled)
    
    # Plot future predictions
    fig, ax = plt.subplots()
    ax.plot(df['Cycle'], df['Discharge Capacity'], label='Historical Data')
    ax.plot(future_cycles['Cycle'], future_predictions, label='Predicted', linestyle='--')
    ax.set_xlabel('Cycle')
    ax.set_ylabel('Discharge Capacity')
    ax.set_title('Discharge Capacity Prediction')
    ax.legend()
    st.pyplot(fig)

def detect_anomalies(df):
    st.subheader("Anomaly Detection")
    
    df = clean_and_prepare_data(df)
    
    # Calculate rolling mean and standard deviation
    window = 10
    df['rolling_mean'] = df['Discharge Capacity'].rolling(window=window).mean()
    df['rolling_std'] = df['Discharge Capacity'].rolling(window=window).std()
    
    # Define anomalies as points outside 3 standard deviations
    df['lower_bound'] = df['rolling_mean'] - 3 * df['rolling_std']
    df['upper_bound'] = df['rolling_mean'] + 3 * df['rolling_std']
    df['is_anomaly'] = (df['Discharge Capacity'] < df['lower_bound']) | (df['Discharge Capacity'] > df['upper_bound'])
    
    # Plot results
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(df['Cycle'], df['Discharge Capacity'], label='Discharge Capacity')
    ax.plot(df['Cycle'], df['rolling_mean'], label='Rolling Mean', linestyle='--')
    ax.fill_between(df['Cycle'], df['lower_bound'], df['upper_bound'], alpha=0.2, label='Normal Range')
    ax.scatter(df[df['is_anomaly']]['Cycle'], df[df['is_anomaly']]['Discharge Capacity'], color='red', label='Anomalies')
    ax.set_xlabel('Cycle')
    ax.set_ylabel('Discharge Capacity')
    ax.set_title('Anomaly Detection in Discharge Capacity')
    ax.legend()
    st.pyplot(fig)
    
    st.write(f"Detected {df['is_anomaly'].sum()} anomalies out of {len(df)} cycles.")

def estimate_rul(df):
    st.subheader("Remaining Useful Life (RUL) Estimation")
    
    df = clean_and_prepare_data(df)
    
    # Define end-of-life capacity (e.g., 80% of initial capacity)
    initial_capacity = df['Discharge Capacity'].iloc[0]
    eol_capacity = 0.8 * initial_capacity
    
    # Fit a linear regression model to estimate capacity fade
    from sklearn.linear_model import LinearRegression
    model = LinearRegression()
    model.fit(df[['Cycle']], df['Discharge Capacity'])
    
    # Predict future cycles until EOL
    future_cycles = pd.DataFrame({'Cycle': range(df['Cycle'].max() + 1, df['Cycle'].max() + 1000)})
    future_capacity = model.predict(future_cycles)
    
    # Find cycle where capacity reaches EOL
    eol_cycle = future_cycles[future_capacity <= eol_capacity]['Cycle'].min()
    
    if pd.isna(eol_cycle):
        st.write("Based on the current trend, the battery is not expected to reach end-of-life within the next 1000 cycles.")
    else:
        rul = eol_cycle - df['Cycle'].max()
        st.write(f"Estimated Remaining Useful Life: {rul} cycles")
    
    # Plot RUL estimation
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(df['Cycle'], df['Discharge Capacity'], label='Historical Data')
    ax.plot(future_cycles['Cycle'], future_capacity, linestyle='--', label='Projected Capacity')
    ax.axhline(y=eol_capacity, color='r', linestyle=':', label='End-of-Life Capacity')
    if not pd.isna(eol_cycle):
        ax.axvline(x=eol_cycle, color='g', linestyle=':', label='Estimated EOL')
    ax.set_xlabel('Cycle')
    ax.set_ylabel('Discharge Capacity')
    ax.set_title('Remaining Useful Life Estimation')
    ax.legend()
    st.pyplot(fig)

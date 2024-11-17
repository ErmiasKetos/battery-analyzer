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
from scipy.stats import gaussian_kde

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
# Capacity Analysis tab
        with tabs[0]:
            st.subheader("🔋 Detailed Capacity Analysis")
            
            # Create expandable sections for different analyses
            with st.expander("📊 Capacity Metrics", expanded=True):
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric(
                        "Initial Capacity", 
                        f"{df['Discharge_Capacity'].iloc[0]:.1f} mAh/g",
                        help="First cycle discharge capacity"
                    )
                    st.metric(
                        "Final Capacity", 
                        f"{df['Discharge_Capacity'].iloc[-1]:.1f} mAh/g",
                        delta=f"{(df['Discharge_Capacity'].iloc[-1] - df['Discharge_Capacity'].iloc[0]):.1f}",
                        help="Last cycle discharge capacity"
                    )
                
                with col2:
                    retention = (df['Discharge_Capacity'].iloc[-1] / df['Discharge_Capacity'].iloc[0] * 100)
                    st.metric(
                        "Capacity Retention",
                        f"{retention:.1f}%",
                        help="Percentage of initial capacity retained"
                    )
                    avg_capacity = df['Discharge_Capacity'].mean()
                    st.metric(
                        "Average Capacity",
                        f"{avg_capacity:.1f} mAh/g",
                        help="Mean discharge capacity across all cycles"
                    )
                
                with col3:
                    fade_rate = ((df['Discharge_Capacity'].iloc[0] - df['Discharge_Capacity'].iloc[-1]) / 
                               (len(df) * df['Discharge_Capacity'].iloc[0]) * 100)
                    st.metric(
                        "Capacity Fade Rate",
                        f"{fade_rate:.4f}%/cycle",
                        help="Average capacity loss per cycle"
                    )
                    stability = df['Discharge_Capacity'].std() / df['Discharge_Capacity'].mean() * 100
                    st.metric(
                        "Capacity Stability",
                        f"{stability:.2f}% CV",
                        help="Coefficient of variation in capacity"
                    )
            
            # Capacity vs Cycle plot with customization options
            with st.expander("📈 Capacity vs Cycle", expanded=True):
                col1, col2 = st.columns([3, 1])
                
                with col2:
                    # Plot customization options
                    show_charge = st.checkbox("Show Charge Capacity", value=True)
                    show_efficiency = st.checkbox("Show Coulombic Efficiency", value=True)
                    normalize = st.checkbox("Normalize Capacity", value=False)
                    rolling_avg = st.checkbox("Show Moving Average", value=False)
                    
                    if rolling_avg:
                        window = st.slider("Window Size", 3, 20, 5)
                
                with col1:
                    fig = go.Figure()
                    
                    # Discharge capacity
                    y_discharge = (df['Discharge_Capacity'] / df['Discharge_Capacity'].iloc[0] * 100) if normalize \
                                else df['Discharge_Capacity']
                    
                    if rolling_avg:
                        y_discharge_smooth = y_discharge.rolling(window=window).mean()
                        fig.add_trace(go.Scatter(
                            x=df['Cycle'],
                            y=y_discharge_smooth,
                            name='Discharge (MA)',
                            line=dict(color='blue', width=2)
                        ))
                        
                    fig.add_trace(go.Scatter(
                        x=df['Cycle'],
                        y=y_discharge,
                        name='Discharge Capacity',
                        line=dict(color='blue', width=1 if rolling_avg else 2),
                        opacity=0.5 if rolling_avg else 1
                    ))
                    
                    # Charge capacity
                    if show_charge:
                        y_charge = (df['Charge_Capacity'] / df['Charge_Capacity'].iloc[0] * 100) if normalize \
                                 else df['Charge_Capacity']
                        
                        if rolling_avg:
                            y_charge_smooth = y_charge.rolling(window=window).mean()
                            fig.add_trace(go.Scatter(
                                x=df['Cycle'],
                                y=y_charge_smooth,
                                name='Charge (MA)',
                                line=dict(color='red', width=2)
                            ))
                            
                        fig.add_trace(go.Scatter(
                            x=df['Cycle'],
                            y=y_charge,
                            name='Charge Capacity',
                            line=dict(color='red', width=1 if rolling_avg else 2),
                            opacity=0.5 if rolling_avg else 1
                        ))
                    
                    # Coulombic efficiency
                    if show_efficiency:
                        efficiency = df['Discharge_Capacity'] / df['Charge_Capacity'] * 100
                        fig.add_trace(go.Scatter(
                            x=df['Cycle'],
                            y=efficiency,
                            name='Coulombic Efficiency',
                            yaxis='y2',
                            line=dict(color='green', width=1)
                        ))
                    
                    # Update layout
                    fig.update_layout(
                        title='Capacity and Efficiency vs Cycle Number',
                        xaxis_title='Cycle Number',
                        yaxis_title='Capacity (%)' if normalize else 'Capacity (mAh/g)',
                        yaxis2=dict(
                            title='Coulombic Efficiency (%)',
                            overlaying='y',
                            side='right',
                            range=[90, 101] if show_efficiency else None
                        ) if show_efficiency else None,
                        height=500,
                        hovermode='x unified'
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
            
            # Capacity fade analysis
            with st.expander("📉 Capacity Fade Analysis"):
                # Calculate capacity retention at different intervals
                intervals = st.slider(
                    "Select analysis intervals",
                    min_value=5,
                    max_value=50,
                    value=10,
                    help="Number of intervals for fade analysis"
                )
                
                # Create intervals
                cycle_points = np.linspace(0, len(df)-1, intervals, dtype=int)
                fade_data = []
                
                for i, cycle_idx in enumerate(cycle_points):
                    cycle_num = df['Cycle'].iloc[cycle_idx]
                    retention = (df['Discharge_Capacity'].iloc[cycle_idx] / 
                               df['Discharge_Capacity'].iloc[0] * 100)
                    fade_rate = (100 - retention) / cycle_num if cycle_num > 0 else 0
                    
                    fade_data.append({
                        'Cycle': cycle_num,
                        'Retention (%)': retention,
                        'Fade Rate (%/cycle)': fade_rate
                    })
                
                fade_df = pd.DataFrame(fade_data)
                
                # Plot fade analysis
                fig_fade = go.Figure()
                
                # Retention curve
                fig_fade.add_trace(go.Scatter(
                    x=fade_df['Cycle'],
                    y=fade_df['Retention (%)'],
                    name='Retention',
                    mode='lines+markers',
                    line=dict(color='blue')
                ))
                
                # Fade rate curve
                fig_fade.add_trace(go.Scatter(
                    x=fade_df['Cycle'],
                    y=fade_df['Fade Rate (%/cycle)'],
                    name='Fade Rate',
                    mode='lines+markers',
                    yaxis='y2',
                    line=dict(color='red')
                ))
                
                fig_fade.update_layout(
                    title='Capacity Fade Analysis',
                    xaxis_title='Cycle Number',
                    yaxis_title='Retention (%)',
                    yaxis2=dict(
                        title='Fade Rate (%/cycle)',
                        overlaying='y',
                        side='right'
                    ),
                    height=400,
                    showlegend=True
                )
                
                st.plotly_chart(fig_fade, use_container_width=True)
                
                # Display fade data table
                st.write("### Detailed Fade Analysis")
                st.dataframe(fade_df.style.format({
                    'Retention (%)': '{:.2f}',
                    'Fade Rate (%/cycle)': '{:.4f}'
                }))
            
            # Capacity distribution analysis
            with st.expander("📊 Capacity Distribution"):
                fig_dist = go.Figure()
                
                # Add histogram
                fig_dist.add_trace(go.Histogram(
                    x=df['Discharge_Capacity'],
                    name='Frequency',
                    nbinsx=30,
                    histnorm='probability'
                ))
                
                # Add kernel density estimation
                kde_points = np.linspace(
                    df['Discharge_Capacity'].min(),
                    df['Discharge_Capacity'].max(),
                    100
                )
                kde = gaussian_kde(df['Discharge_Capacity'])
                
                fig_dist.add_trace(go.Scatter(
                    x=kde_points,
                    y=kde(kde_points),
                    name='KDE',
                    line=dict(color='red')
                ))
                
                fig_dist.update_layout(
                    title='Capacity Distribution',
                    xaxis_title='Discharge Capacity (mAh/g)',
                    yaxis_title='Probability Density',
                    height=400,
                    barmode='overlay'
                )
                
                st.plotly_chart(fig_dist, use_container_width=True)
                
                # Distribution statistics
                dist_stats = df['Discharge_Capacity'].describe()
                st.write("### Distribution Statistics")
                st.write(pd.DataFrame({
                    'Statistic': dist_stats.index,
                    'Value': dist_stats.values
                }).set_index('Statistic').round(3))

# Voltage Analysis Tab
            with tabs[1]:
                st.subheader("⚡ Voltage Analysis")
                
                # Basic voltage metrics
                with st.expander("📊 Voltage Metrics", expanded=True):
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.metric(
                            "Average Charge Voltage",
                            f"{df['Charge_Voltage'].mean():.3f} V",
                            help="Mean charging voltage"
                        )
                        st.metric(
                            "Max Charge Voltage",
                            f"{df['Charge_Voltage'].max():.3f} V",
                            help="Maximum charging voltage"
                        )
                    
                    with col2:
                        st.metric(
                            "Average Discharge Voltage",
                            f"{df['Discharge_Voltage'].mean():.3f} V",
                            help="Mean discharging voltage"
                        )
                        st.metric(
                            "Min Discharge Voltage",
                            f"{df['Discharge_Voltage'].min():.3f} V",
                            help="Minimum discharging voltage"
                        )
                    
                    with col3:
                        avg_gap = df['Charge_Voltage'].mean() - df['Discharge_Voltage'].mean()
                        st.metric(
                            "Average Voltage Gap",
                            f"{avg_gap:.3f} V",
                            help="Average difference between charge and discharge voltage"
                        )
                        st.metric(
                            "Voltage Stability",
                            f"{df['Charge_Voltage'].std():.3f} V",
                            help="Standard deviation of charge voltage"
                        )
                
                # Voltage evolution plot
                with st.expander("📈 Voltage Evolution", expanded=True):
                    col1, col2 = st.columns([3, 1])
                    
                    with col2:
                        show_gap = st.checkbox("Show Voltage Gap", value=True)
                        show_ma = st.checkbox("Show Moving Average", value=False)
                        
                        if show_ma:
                            ma_window = st.slider(
                                "Moving Average Window",
                                min_value=3,
                                max_value=20,
                                value=5
                            )
                    
                    with col1:
                        fig_voltage = go.Figure()
                        
                        # Add charge voltage
                        if show_ma:
                            ma_charge = df['Charge_Voltage'].rolling(window=ma_window).mean()
                            fig_voltage.add_trace(go.Scatter(
                                x=df['Cycle'],
                                y=ma_charge,
                                name='Charge V (MA)',
                                line=dict(color='red', width=2)
                            ))
                        
                        fig_voltage.add_trace(go.Scatter(
                            x=df['Cycle'],
                            y=df['Charge_Voltage'],
                            name='Charge Voltage',
                            line=dict(color='red', width=1 if show_ma else 2),
                            opacity=0.5 if show_ma else 1
                        ))
                        
                        # Add discharge voltage
                        if show_ma:
                            ma_discharge = df['Discharge_Voltage'].rolling(window=ma_window).mean()
                            fig_voltage.add_trace(go.Scatter(
                                x=df['Cycle'],
                                y=ma_discharge,
                                name='Discharge V (MA)',
                                line=dict(color='blue', width=2)
                            ))
                        
                        fig_voltage.add_trace(go.Scatter(
                            x=df['Cycle'],
                            y=df['Discharge_Voltage'],
                            name='Discharge Voltage',
                            line=dict(color='blue', width=1 if show_ma else 2),
                            opacity=0.5 if show_ma else 1
                        ))
                        
                        # Add voltage gap
                        if show_gap:
                            voltage_gap = df['Charge_Voltage'] - df['Discharge_Voltage']
                            fig_voltage.add_trace(go.Scatter(
                                x=df['Cycle'],
                                y=voltage_gap,
                                name='Voltage Gap',
                                line=dict(color='green'),
                                yaxis='y2'
                            ))
                        
                        # Update layout
                        fig_voltage.update_layout(
                            title='Voltage Evolution',
                            xaxis_title='Cycle Number',
                            yaxis_title='Voltage (V)',
                            yaxis2=dict(
                                title='Voltage Gap (V)',
                                overlaying='y',
                                side='right'
                            ) if show_gap else None,
                            height=500,
                            hovermode='x unified'
                        )
                        
                        st.plotly_chart(fig_voltage, use_container_width=True)
                
                # Voltage profile comparison
                with st.expander("🔍 Voltage Profile Analysis"):
                    # Select cycles to compare
                    cycles_to_compare = st.multiselect(
                        "Select cycles to compare",
                        options=sorted(df['Cycle'].unique()),
                        default=[df['Cycle'].iloc[0], df['Cycle'].iloc[-1]]
                    )
                    
                    if cycles_to_compare:
                        fig_profile = go.Figure()
                        
                        for cycle in cycles_to_compare:
                            cycle_data = df[df['Cycle'] == cycle]
                            
                            # Add charge profile
                            fig_profile.add_trace(go.Scatter(
                                x=cycle_data['Charge_Capacity'],
                                y=cycle_data['Charge_Voltage'],
                                name=f'Cycle {cycle} (Charge)',
                                line=dict(dash='solid')
                            ))
                            
                            # Add discharge profile
                            fig_profile.add_trace(go.Scatter(
                                x=cycle_data['Discharge_Capacity'],
                                y=cycle_data['Discharge_Voltage'],
                                name=f'Cycle {cycle} (Discharge)',
                                line=dict(dash='dot')
                            ))
                        
                        fig_profile.update_layout(
                            title='Voltage vs Capacity Profiles',
                            xaxis_title='Capacity (mAh/g)',
                            yaxis_title='Voltage (V)',
                            height=500,
                            hovermode='x unified'
                        )
                        
                        st.plotly_chart(fig_profile, use_container_width=True)
                
                # Statistical analysis
                with st.expander("📊 Voltage Statistics"):
                    # Calculate voltage statistics for each cycle
                    voltage_stats = df.groupby('Cycle').agg({
                        'Charge_Voltage': ['mean', 'std', 'min', 'max'],
                        'Discharge_Voltage': ['mean', 'std', 'min', 'max']
                    }).round(3)
                    
                    # Rename columns for clarity
                    voltage_stats.columns = [
                        'Charge V (mean)', 'Charge V (std)', 'Charge V (min)', 'Charge V (max)',
                        'Discharge V (mean)', 'Discharge V (std)', 'Discharge V (min)', 'Discharge V (max)'
                    ]
                    
                    # Display statistics
                    st.write("### Cycle-by-Cycle Voltage Statistics")
                    st.dataframe(voltage_stats)
                    
                    # Plot voltage distributions
                    fig_dist = go.Figure()
                    
                    # Add charge voltage distribution
                    fig_dist.add_trace(go.Box(
                        y=df['Charge_Voltage'],
                        name='Charge Voltage',
                        boxpoints='outliers'
                    ))
                    
                    # Add discharge voltage distribution
                    fig_dist.add_trace(go.Box(
                        y=df['Discharge_Voltage'],
                        name='Discharge Voltage',
                        boxpoints='outliers'
                    ))
                    
                    fig_dist.update_layout(
                        title='Voltage Distributions',
                        yaxis_title='Voltage (V)',
                        height=400,
                        showlegend=True
                    )
                    
                    st.plotly_chart(fig_dist, use_container_width=True)
                    
                    # Download statistics
                    csv = voltage_stats.to_csv()
                    st.download_button(
                        label="Download Voltage Statistics",
                        data=csv,
                        file_name="voltage_statistics.csv",
                        mime="text/csv"
                    )
           # dQ/dV Analysis Tab
            with tabs[2]:
                st.subheader("🔄 Differential Capacity Analysis (dQ/dV)")
                
                with st.expander("ℹ️ About dQ/dV Analysis", expanded=False):
                    st.write("""
                    The differential capacity (dQ/dV) analysis reveals:
                    - **Phase transitions:** Peaks indicate voltage plateaus where phase transitions occur
                    - **Reaction mechanisms:** Peak positions show reaction voltages
                    - **Degradation:** Changes in peak height/position indicate material degradation
                    - **Material characteristics:** Peak shape relates to reaction kinetics
                    """)
                
                # Analysis settings
                with st.expander("⚙️ Analysis Settings", expanded=True):
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        selected_cycles = st.multiselect(
                            "Select cycles for analysis",
                            options=sorted(df['Cycle'].unique()),
                            default=[df['Cycle'].iloc[0], df['Cycle'].iloc[-1]]
                        )
                    
                    with col2:
                        points_dqdv = st.slider(
                            "Interpolation points",
                            min_value=100,
                            max_value=2000,
                            value=1000,
                            step=100,
                            help="More points = smoother curve but slower calculation"
                        )
                    
                    with col3:
                        plot_type = st.radio(
                            "Analysis type",
                            ["Charge", "Discharge", "Both"],
                            horizontal=True
                        )
                
                if selected_cycles:
                    # Main dQ/dV plot
                    with st.expander("📈 dQ/dV Analysis", expanded=True):
                        fig_dqdv = go.Figure()
                        
                        # Process each selected cycle
                        for cycle in selected_cycles:
                            cycle_data = df[df['Cycle'] == cycle]
                            
                            if plot_type in ["Charge", "Both"]:
                                v_charge, dqdv_charge = calculate_dqdv_proper(
                                    cycle_data['Charge_Voltage'].values,
                                    cycle_data['Charge_Capacity'].values,
                                    points_dqdv
                                )
                                
                                if v_charge is not None:
                                    fig_dqdv.add_trace(go.Scatter(
                                        x=v_charge,
                                        y=dqdv_charge,
                                        name=f'Cycle {cycle} (Charge)',
                                        line=dict(dash='solid')
                                    ))
                            
                            if plot_type in ["Discharge", "Both"]:
                                v_discharge, dqdv_discharge = calculate_dqdv_proper(
                                    cycle_data['Discharge_Voltage'].values,
                                    cycle_data['Discharge_Capacity'].values,
                                    points_dqdv
                                )
                                
                                if v_discharge is not None:
                                    fig_dqdv.add_trace(go.Scatter(
                                        x=v_discharge,
                                        y=-dqdv_discharge,  # Negative for conventional plotting
                                        name=f'Cycle {cycle} (Discharge)',
                                        line=dict(dash='dot')
                                    ))
                        
                        fig_dqdv.update_layout(
                            title='Differential Capacity Analysis',
                            xaxis_title='Voltage (V)',
                            yaxis_title='dQ/dV (mAh/V)',
                            height=600,
                            hovermode='x unified'
                        )
                        
                        st.plotly_chart(fig_dqdv, use_container_width=True)
                    
                    # Peak analysis
                    with st.expander("🔍 Peak Analysis"):
                        # Peak detection settings
                        col1, col2 = st.columns(2)
                        with col1:
                            prominence = st.slider(
                                "Peak prominence",
                                min_value=0.01,
                                max_value=1.0,
                                value=0.1,
                                step=0.01,
                                help="Higher value = fewer peaks detected"
                            )
                        with col2:
                            width = st.slider(
                                "Minimum peak width",
                                min_value=1,
                                max_value=50,
                                value=5,
                                help="Minimum number of points for peak"
                            )
                        
                        # Analyze peaks for each cycle
                        peak_data = []
                        
                        for cycle in selected_cycles:
                            cycle_data = df[df['Cycle'] == cycle]
                            
                            # Analyze charge peaks
                            v_charge, dqdv_charge = calculate_dqdv_proper(
                                cycle_data['Charge_Voltage'].values,
                                cycle_data['Charge_Capacity'].values,
                                points_dqdv
                            )
                            
                            if v_charge is not None:
                                peaks, properties = find_peaks(
                                    dqdv_charge,
                                    prominence=prominence,
                                    width=width
                                )
                                
                                for i, peak in enumerate(peaks):
                                    peak_data.append({
                                        'Cycle': cycle,
                                        'Process': 'Charge',
                                        'Peak Voltage': v_charge[peak],
                                        'Peak Height': dqdv_charge[peak],
                                        'Prominence': properties['prominences'][i],
                                        'Width': properties['widths'][i]
                                    })
                            
                            # Analyze discharge peaks
                            v_discharge, dqdv_discharge = calculate_dqdv_proper(
                                cycle_data['Discharge_Voltage'].values,
                                cycle_data['Discharge_Capacity'].values,
                                points_dqdv
                            )
                            
                            if v_discharge is not None:
                                peaks, properties = find_peaks(
                                    -dqdv_discharge,  # Negative for consistent peak direction
                                    prominence=prominence,
                                    width=width
                                )
                                
                                for i, peak in enumerate(peaks):
                                    peak_data.append({
                                        'Cycle': cycle,
                                        'Process': 'Discharge',
                                        'Peak Voltage': v_discharge[peak],
                                        'Peak Height': -dqdv_discharge[peak],
                                        'Prominence': properties['prominences'][i],
                                        'Width': properties['widths'][i]
                                    })
                        
                        if peak_data:
                            # Create peak data DataFrame
                            peak_df = pd.DataFrame(peak_data)
                            
                            # Display peak data
                            st.write("### Detected Peaks")
                            st.dataframe(peak_df.round(3))
                            
                            # Plot peak evolution
                            fig_peaks = go.Figure()
                            
                            # Plot charge peaks
                            charge_peaks = peak_df[peak_df['Process'] == 'Charge']
                            if not charge_peaks.empty:
                                fig_peaks.add_trace(go.Scatter(
                                    x=charge_peaks['Cycle'],
                                    y=charge_peaks['Peak Voltage'],
                                    mode='markers',
                                    name='Charge Peaks',
                                    marker=dict(
                                        size=charge_peaks['Peak Height'].abs() * 20,
                                        color='red'
                                    )
                                ))
                            
                            # Plot discharge peaks
                            discharge_peaks = peak_df[peak_df['Process'] == 'Discharge']
                            if not discharge_peaks.empty:
                                fig_peaks.add_trace(go.Scatter(
                                    x=discharge_peaks['Cycle'],
                                    y=discharge_peaks['Peak Voltage'],
                                    mode='markers',
                                    name='Discharge Peaks',
                                    marker=dict(
                                        size=discharge_peaks['Peak Height'].abs() * 20,
                                        color='blue'
                                    )
                                ))
                            
                            fig_peaks.update_layout(
                                title='Peak Evolution',
                                xaxis_title='Cycle Number',
                                yaxis_title='Peak Voltage (V)',
                                height=400
                            )
                            
                            st.plotly_chart(fig_peaks, use_container_width=True)
                            
                            # Download peak data
                            csv = peak_df.to_csv(index=False)
                            st.download_button(
                                label="Download Peak Data",
                                data=csv,
                                file_name="peak_analysis.csv",
                                mime="text/csv"
                            )
                        else:
                            st.warning("No peaks detected with current settings")
                    
                    # Peak comparison
                    with st.expander("📊 Peak Evolution Analysis"):
                        if len(selected_cycles) > 1:
                            # Calculate peak changes
                            peak_changes = []
                            baseline_cycle = selected_cycles[0]
                            
                            for cycle in selected_cycles[1:]:
                                baseline_peaks = peak_df[
                                    (peak_df['Cycle'] == baseline_cycle) &
                                    (peak_df['Process'] == 'Charge')
                                ]['Peak Voltage'].values
                                
                                current_peaks = peak_df[
                                    (peak_df['Cycle'] == cycle) &
                                    (peak_df['Process'] == 'Charge')
                                ]['Peak Voltage'].values
                                
                                # Find closest peaks
                                for bp in baseline_peaks:
                                    if len(current_peaks) > 0:
                                        closest_peak = current_peaks[
                                            np.abs(current_peaks - bp).argmin()
                                        ]
                                        
                                        peak_changes.append({
                                            'Baseline Cycle': baseline_cycle,
                                            'Compare Cycle': cycle,
                                            'Baseline Voltage': bp,
                                            'Current Voltage': closest_peak,
                                            'Voltage Shift': closest_peak - bp
                                        })
                            
                            if peak_changes:
                                changes_df = pd.DataFrame(peak_changes)
                                
                                st.write("### Peak Position Changes")
                                st.dataframe(changes_df.round(4))
                                
                                # Plot peak shifts
                                fig_shifts = go.Figure()
                                
                                for peak_v in changes_df['Baseline Voltage'].unique():
                                    peak_data = changes_df[
                                        changes_df['Baseline Voltage'] == peak_v
                                    ]
                                    
                                    fig_shifts.add_trace(go.Scatter(
                                        x=peak_data['Compare Cycle'],
                                        y=peak_data['Voltage Shift'],
                                        name=f'Peak at {peak_v:.2f}V',
                                        mode='lines+markers'
                                    ))
                                
                                fig_shifts.update_layout(
                                    title='Peak Voltage Shifts',
                                    xaxis_title='Cycle Number',
                                    yaxis_title='Voltage Shift (V)',
                                    height=400
                                )
                                
                                st.plotly_chart(fig_shifts, use_container_width=True)
                            else:
                                st.warning("No comparable peaks found between cycles")
                        else:
                            st.info("Select multiple cycles to analyze peak evolution")         
                
        # Add other tabs...

except Exception as e:
    st.error(f"An error occurred: {str(e)}")
    st.write("Please check your data format and try again.")

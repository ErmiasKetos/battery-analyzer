import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
from scipy.interpolate import interp1d
from scipy.signal import find_peaks
import plotly.subplots as sp

# Set page configuration
st.set_page_config(
    page_title="Battery Data Analyzer",
    page_icon="🔋",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Utility functions
def calculate_dqdv_proper(voltage, capacity, num_points=1000):
    """
    Calculate dQ/dV using proper numerical differentiation
    """
    try:
        # Sort by voltage to ensure proper interpolation
        sort_idx = np.argsort(voltage)
        voltage_sorted = voltage[sort_idx]
        capacity_sorted = capacity[sort_idx]
        
        # Remove duplicates
        _, unique_idx = np.unique(voltage_sorted, return_index=True)
        voltage_unique = voltage_sorted[unique_idx]
        capacity_unique = capacity_sorted[unique_idx]
        
        if len(voltage_unique) < 3:
            return None, None
            
        # Create interpolation function
        f = interp1d(voltage_unique, capacity_unique, kind='cubic', bounds_error=False)
        
        # Create evenly spaced voltage points for interpolation
        v_interp = np.linspace(voltage_unique.min(), voltage_unique.max(), num_points)
        
        # Get interpolated capacity values
        q_interp = f(v_interp)
        
        # Calculate numerical derivative using central difference
        dv = v_interp[1] - v_interp[0]
        dqdv = np.gradient(q_interp, dv)
        
        return v_interp, dqdv
        
    except Exception as e:
        st.error(f"Error in dQ/dV calculation: {str(e)}")
        return None, None

def calculate_capacity_metrics(df):
    """Calculate various capacity-related metrics"""
    metrics = {
        'Initial Discharge Capacity': df['Discharge_Capacity'].iloc[0],
        'Final Discharge Capacity': df['Discharge_Capacity'].iloc[-1],
        'Capacity Retention': (df['Discharge_Capacity'].iloc[-1] / df['Discharge_Capacity'].iloc[0] * 100),
        'Average Discharge Capacity': df['Discharge_Capacity'].mean(),
        'Capacity Loss Rate': ((df['Discharge_Capacity'].iloc[0] - df['Discharge_Capacity'].iloc[-1]) / 
                             (len(df) * df['Discharge_Capacity'].iloc[0]) * 100)
    }
    return metrics

def calculate_efficiency_metrics(df):
    """Calculate efficiency-related metrics"""
    df['Coulombic_Efficiency'] = (df['Discharge_Capacity'] / df['Charge_Capacity'] * 100)
    metrics = {
        'Average Efficiency': df['Coulombic_Efficiency'].mean(),
        'Minimum Efficiency': df['Coulombic_Efficiency'].min(),
        'Maximum Efficiency': df['Coulombic_Efficiency'].max(),
        'Efficiency Stability': df['Coulombic_Efficiency'].std()
    }
    return metrics

def calculate_voltage_metrics(df):
    """Calculate voltage-related metrics"""
    df['Voltage_Gap'] = df['Charge_Voltage'] - df['Discharge_Voltage']
    metrics = {
        'Average Charge Voltage': df['Charge_Voltage'].mean(),
        'Average Discharge Voltage': df['Discharge_Voltage'].mean(),
        'Average Voltage Gap': df['Voltage_Gap'].mean(),
        'Maximum Voltage Gap': df['Voltage_Gap'].max(),
        'Voltage Stability': df['Voltage_Gap'].std()
    }
    return metrics

# Main application header
st.title("🔋 Advanced Battery Data Analyzer")
st.write("Upload your battery cycling data for comprehensive analysis and visualization.")

# Sidebar for global settings
with st.sidebar:
    st.header("Analysis Settings")
    
    plot_theme = st.selectbox(
        "Plot Theme",
        ["plotly", "plotly_white", "plotly_dark"],
        index=1
    )
    
    smoothing_factor = st.slider(
        "Data Smoothing",
        min_value=0,
        max_value=10,
        value=3,
        help="Higher values = smoother curves"
    )
    
    st.divider()
    
    with st.expander("📖 About This Tool"):
        st.write("""
        This tool provides comprehensive analysis of battery cycling data, including:
        - Capacity fade analysis
        - Coulombic efficiency tracking
        - Voltage profile analysis
        - Differential capacity analysis (dQ/dV)
        - Peak detection and phase transition analysis
        
        For best results, ensure your data includes:
        - Cycle numbers
        - Charge/Discharge capacity values
        - Charge/Discharge voltage values
        """)

# File upload section
st.subheader("📤 Upload Data")
uploaded_file = st.file_uploader("Choose your CSV file", type="csv")

# Continue with the main application logic...
# Continue from previous part...

if uploaded_file is not None:
    try:
        # Read CSV
        df_original = pd.read_csv(uploaded_file)
        
        # Column mapping section
        st.subheader("🔄 Map Your Columns")
        
        col1, col2 = st.columns([1, 2])
        with col1:
            st.write("Your file contains these columns:")
            st.info(", ".join(df_original.columns))
        
        # Create column mapping
        col_mapping = {}
        required_columns = {
            'Cycle Number': 'Cycle',
            'Discharge Capacity': 'Discharge_Capacity',
            'Charge Capacity': 'Charge_Capacity',
            'Discharge Voltage': 'Discharge_Voltage',
            'Charge Voltage': 'Charge_Voltage'
        }
        
        with col2:
            st.write("Map your columns to the required data types:")
            # Create mapping dropdowns
            for display_name, internal_name in required_columns.items():
                col_mapping[internal_name] = st.selectbox(
                    f"Select column for {display_name}:",
                    options=[''] + list(df_original.columns),
                    key=internal_name
                )
        
        # Process button
        if st.button("🔍 Process Data"):
            # Validate all columns are selected
            if '' in col_mapping.values():
                st.error("⚠️ Please select all required columns!")
                st.stop()
            
            # Create renamed dataframe
            df = df_original.rename(columns={v: k for k, v in col_mapping.items()})
            
            # Create analysis tabs
            tabs = st.tabs([
                "📈 Capacity Analysis",
                "⚡ Voltage Analysis",
                "🔄 Differential Capacity",
                "📊 Statistical Analysis",
                "📋 Raw Data"
            ])
            
            # Tab 1: Capacity Analysis
            with tabs[0]:
                st.subheader("Capacity Analysis")
                
                # Capacity metrics
                metrics = calculate_capacity_metrics(df)
                
                # Display metrics in columns
                metric_cols = st.columns(3)
                with metric_cols[0]:
                    st.metric("Initial Capacity", 
                             f"{metrics['Initial Discharge Capacity']:.2f} mAh/g")
                    st.metric("Capacity Retention", 
                             f"{metrics['Capacity Retention']:.1f}%")
                
                with metric_cols[1]:
                    st.metric("Final Capacity", 
                             f"{metrics['Final Discharge Capacity']:.2f} mAh/g")
                    st.metric("Average Capacity", 
                             f"{metrics['Average Discharge Capacity']:.2f} mAh/g")
                
                with metric_cols[2]:
                    st.metric("Capacity Loss Rate", 
                             f"{metrics['Capacity Loss Rate']:.4f}%/cycle")
                
                # Capacity plots
                capacity_fig = sp.make_subplots(
                    rows=2, cols=1,
                    subplot_titles=('Capacity vs Cycle', 'Coulombic Efficiency'),
                    vertical_spacing=0.15
                )
                
                # Add capacity traces
                capacity_fig.add_trace(
                    go.Scatter(
                        x=df['Cycle'],
                        y=df['Discharge_Capacity'],
                        name='Discharge Capacity',
                        line=dict(color='blue')
                    ),
                    row=1, col=1
                )
                
                capacity_fig.add_trace(
                    go.Scatter(
                        x=df['Cycle'],
                        y=df['Charge_Capacity'],
                        name='Charge Capacity',
                        line=dict(color='red')
                    ),
                    row=1, col=1
                )
                
                # Calculate and plot efficiency
                df['Coulombic_Efficiency'] = (df['Discharge_Capacity'] / df['Charge_Capacity'] * 100)
                
                capacity_fig.add_trace(
                    go.Scatter(
                        x=df['Cycle'],
                        y=df['Coulombic_Efficiency'],
                        name='Coulombic Efficiency',
                        line=dict(color='green')
                    ),
                    row=2, col=1
                )
                
                # Update layout
                capacity_fig.update_layout(
                    height=800,
                    showlegend=True,
                    title_text="Battery Performance Analysis"
                )
                
                capacity_fig.update_xaxes(title_text='Cycle Number', row=2, col=1)
                capacity_fig.update_yaxes(title_text='Capacity (mAh/g)', row=1, col=1)
                capacity_fig.update_yaxes(title_text='Efficiency (%)', row=2, col=1)
                
                st.plotly_chart(capacity_fig, use_container_width=True)
                
                # Capacity fade analysis
                with st.expander("📉 Capacity Fade Analysis"):
                    # Calculate capacity retention at different cycle ranges
                    cycle_ranges = [
                        (1, 10), (1, 50), (1, 100),
                        (1, int(len(df) * 0.25)),
                        (1, int(len(df) * 0.5)),
                        (1, int(len(df) * 0.75)),
                        (1, len(df))
                    ]
                    
                    fade_data = []
                    for start, end in cycle_ranges:
                        if end <= len(df):
                            retention = (df['Discharge_Capacity'].iloc[end-1] / 
                                      df['Discharge_Capacity'].iloc[0] * 100)
                            fade_data.append({
                                'Range': f'Cycles 1-{end}',
                                'Retention (%)': retention,
                                'Fade Rate (%/cycle)': (100 - retention) / end
                            })
                    
                    fade_df = pd.DataFrame(fade_data)
                    st.write(fade_df)
                    
                    # Plot retention vs cycle range
                    fig_fade = px.line(fade_df, 
                                     x='Range', 
                                     y=['Retention (%)', 'Fade Rate (%/cycle)'],
                                     title='Capacity Retention Analysis')
                    st.plotly_chart(fig_fade, use_container_width=True)
            
            # Continue with other tabs...
            # Tab 2: Voltage Analysis
            with tabs[1]:
                st.subheader("Voltage Analysis")
                
                # Calculate voltage metrics
                voltage_metrics = calculate_voltage_metrics(df)
                
                # Display voltage metrics
                v_cols = st.columns(3)
                with v_cols[0]:
                    st.metric("Average Charge Voltage", 
                             f"{voltage_metrics['Average Charge Voltage']:.3f} V")
                    st.metric("Average Discharge Voltage", 
                             f"{voltage_metrics['Average Discharge Voltage']:.3f} V")
                
                with v_cols[1]:
                    st.metric("Average Voltage Gap", 
                             f"{voltage_metrics['Average Voltage Gap']:.3f} V")
                    st.metric("Maximum Voltage Gap", 
                             f"{voltage_metrics['Maximum Voltage Gap']:.3f} V")
                
                with v_cols[2]:
                    st.metric("Voltage Stability", 
                             f"{voltage_metrics['Voltage Stability']:.3f} V")
                
                # Voltage profile plots
                voltage_fig = sp.make_subplots(
                    rows=2, cols=1,
                    subplot_titles=('Voltage Profiles', 'Voltage Gap Evolution'),
                    vertical_spacing=0.15
                )
                
                # Add voltage traces
                voltage_fig.add_trace(
                    go.Scatter(
                        x=df['Cycle'],
                        y=df['Charge_Voltage'],
                        name='Charge Voltage',
                        line=dict(color='red')
                    ),
                    row=1, col=1
                )
                
                voltage_fig.add_trace(
                    go.Scatter(
                        x=df['Cycle'],
                        y=df['Discharge_Voltage'],
                        name='Discharge Voltage',
                        line=dict(color='blue')
                    ),
                    row=1, col=1
                )
                
                # Add voltage gap trace
                voltage_fig.add_trace(
                    go.Scatter(
                        x=df['Cycle'],
                        y=df['Voltage_Gap'],
                        name='Voltage Gap',
                        line=dict(color='green')
                    ),
                    row=2, col=1
                )
                
                voltage_fig.update_layout(
                    height=800,
                    showlegend=True,
                    title_text="Voltage Analysis"
                )
                
                voltage_fig.update_xaxes(title_text='Cycle Number', row=2, col=1)
                voltage_fig.update_yaxes(title_text='Voltage (V)', row=1, col=1)
                voltage_fig.update_yaxes(title_text='Voltage Gap (V)', row=2, col=1)
                
                st.plotly_chart(voltage_fig, use_container_width=True)
                
                # Advanced voltage analysis
                with st.expander("🔍 Advanced Voltage Analysis"):
                    # Select specific cycles for comparison
                    cycles_to_compare = st.multiselect(
                        "Select cycles to compare",
                        options=sorted(df['Cycle'].unique()),
                        default=[df['Cycle'].iloc[0], df['Cycle'].iloc[-1]]
                    )
                    
                    if cycles_to_compare:
                        # Create voltage profile comparison
                        fig_compare = go.Figure()
                        
                        for cycle in cycles_to_compare:
                            cycle_data = df[df['Cycle'] == cycle]
                            
                            fig_compare.add_trace(go.Scatter(
                                x=cycle_data['Charge_Voltage'],
                                y=cycle_data['Charge_Capacity'],
                                name=f'Cycle {cycle} (Charge)',
                                line=dict(dash='solid')
                            ))
                            
                            fig_compare.add_trace(go.Scatter(
                                x=cycle_data['Discharge_Voltage'],
                                y=cycle_data['Discharge_Capacity'],
                                name=f'Cycle {cycle} (Discharge)',
                                line=dict(dash='dot')
                            ))
                        
                        fig_compare.update_layout(
                            title='Voltage Profile Comparison',
                            xaxis_title='Voltage (V)',
                            yaxis_title='Capacity (mAh/g)',
                            hovermode='x unified'
                        )
                        
                        st.plotly_chart(fig_compare, use_container_width=True)
                
                # Voltage stability analysis
                with st.expander("📊 Voltage Stability Analysis"):
                    # Calculate voltage statistics for each cycle
                    voltage_stats = df.groupby('Cycle').agg({
                        'Charge_Voltage': ['mean', 'std'],
                        'Discharge_Voltage': ['mean', 'std'],
                        'Voltage_Gap': ['mean', 'std']
                    }).round(3)
                    
                    voltage_stats.columns = [
                        'Charge V (mean)', 'Charge V (std)',
                        'Discharge V (mean)', 'Discharge V (std)',
                        'Gap (mean)', 'Gap (std)'
                    ]
                    
                    st.write(voltage_stats)
            
            # Tab 3: Differential Capacity Analysis
            with tabs[2]:
                st.subheader("Differential Capacity Analysis (dQ/dE)")
                
                # Analysis settings
                dqdv_cols = st.columns([1, 1, 1])
                with dqdv_cols[0]:
                    selected_cycles = st.multiselect(
                        "Select cycles for analysis",
                        options=sorted(df['Cycle'].unique()),
                        default=[df['Cycle'].iloc[0], df['Cycle'].iloc[-1]]
                    )
                
                with dqdv_cols[1]:
                    points_dqdv = st.slider(
                        "Interpolation points",
                        min_value=100,
                        max_value=2000,
                        value=1000,
                        step=100
                    )
                
                with dqdv_cols[2]:
                    plot_type = st.radio(
                        "Plot type",
                        ["Charge", "Discharge", "Both"],
                        horizontal=True
                    )
                
                if selected_cycles:
                    # Create dQ/dV plot
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
                                    y=-dqdv_discharge,
                                    name=f'Cycle {cycle} (Discharge)',
                                    line=dict(dash='dot')
                                ))
                    
                    fig_dqdv.update_layout(
                        title='Differential Capacity Analysis',
                        xaxis_title='Voltage (V)',
                        yaxis_title='dQ/dV (mAh/V)',
                        hovermode='x unified'
                    )
                    
                    st.plotly_chart(fig_dqdv, use_container_width=True)
                    
                    # Peak analysis
                    if st.checkbox("Show Peak Analysis"):
                        st.write("### Peak Analysis")
                        
                        for cycle in selected_cycles:
                            st.write(f"#### Cycle {cycle}")
                            cycle_data = df[df['Cycle'] == cycle]
                            
                            # Analyze charge peaks
                            v_charge, dqdv_charge = calculate_dqdv_proper(
                                cycle_data['Charge_Voltage'].values,
                                cycle_data['Charge_Capacity'].values,
                                points_dqdv
                            )
                            
                            if v_charge is not None:
                                peaks, properties = find_peaks(dqdv_charge, prominence=0.1)
                                
                                if len(peaks) > 0:
                                    peak_data = pd.DataFrame({
                                        'Voltage (V)': v_charge[peaks],
                                        'dQ/dV (mAh/V)': dqdv_charge[peaks],
                                        'Prominence': properties['prominences']
                                    })
                                    
                                    st.write("Charge peaks:")
                                    st.write(peak_data)

# Continue with the Statistical Analysis tab and Raw Data tab...
# Tab 4: Statistical Analysis
            with tabs[3]:
                st.subheader("Statistical Analysis")
                
                # Create sections for different types of analysis
                analysis_type = st.radio(
                    "Select Analysis Type",
                    ["Overview Statistics", "Cycle-by-Cycle Analysis", "Correlation Analysis", "Distribution Analysis"],
                    horizontal=True
                )
                
                if analysis_type == "Overview Statistics":
                    # Overview statistics for all parameters
                    st.write("### Overview Statistics")
                    
                    # Calculate statistics for main parameters
                    stats_df = pd.DataFrame({
                        'Discharge Capacity': df['Discharge_Capacity'].describe(),
                        'Charge Capacity': df['Charge_Capacity'].describe(),
                        'Discharge Voltage': df['Discharge_Voltage'].describe(),
                        'Charge Voltage': df['Charge_Voltage'].describe(),
                        'Coulombic Efficiency': df['Coulombic_Efficiency'].describe(),
                        'Voltage Gap': df['Voltage_Gap'].describe()
                    }).round(3)
                    
                    # Display statistics in an expandable section
                    with st.expander("📊 Detailed Statistics", expanded=True):
                        st.write(stats_df)
                    
                    # Calculate and display stability metrics
                    stability_metrics = {
                        'Capacity Stability (CV%)': (df['Discharge_Capacity'].std() / df['Discharge_Capacity'].mean() * 100),
                        'Efficiency Stability (CV%)': (df['Coulombic_Efficiency'].std() / df['Coulombic_Efficiency'].mean() * 100),
                        'Voltage Stability (CV%)': (df['Voltage_Gap'].std() / df['Voltage_Gap'].mean() * 100)
                    }
                    
                    # Display stability metrics
                    st.write("### Stability Metrics")
                    cols = st.columns(3)
                    for i, (metric, value) in enumerate(stability_metrics.items()):
                        cols[i].metric(metric, f"{value:.2f}%")
                
                elif analysis_type == "Cycle-by-Cycle Analysis":
                    st.write("### Cycle-by-Cycle Analysis")
                    
                    # Select cycles range
                    cycle_range = st.slider(
                        "Select Cycle Range",
                        min_value=int(df['Cycle'].min()),
                        max_value=int(df['Cycle'].max()),
                        value=(int(df['Cycle'].min()), int(df['Cycle'].max()))
                    )
                    
                    # Filter data based on selected range
                    mask = df['Cycle'].between(cycle_range[0], cycle_range[1])
                    cycle_data = df[mask]
                    
                    # Calculate cycle-by-cycle changes
                    cycle_analysis = pd.DataFrame({
                        'Cycle': cycle_data['Cycle'],
                        'Capacity Change (%)': cycle_data['Discharge_Capacity'].pct_change() * 100,
                        'Efficiency Change (%)': cycle_data['Coulombic_Efficiency'].pct_change() * 100,
                        'Voltage Gap Change (%)': cycle_data['Voltage_Gap'].pct_change() * 100
                    })
                    
                    # Plot cycle-by-cycle changes
                    fig_changes = go.Figure()
                    
                    for col in ['Capacity Change (%)', 'Efficiency Change (%)', 'Voltage Gap Change (%)']:
                        fig_changes.add_trace(go.Scatter(
                            x=cycle_analysis['Cycle'],
                            y=cycle_analysis[col],
                            name=col,
                            mode='lines+markers'
                        ))
                    
                    fig_changes.update_layout(
                        title='Cycle-by-Cycle Changes',
                        xaxis_title='Cycle Number',
                        yaxis_title='Change (%)',
                        hovermode='x unified'
                    )
                    
                    st.plotly_chart(fig_changes, use_container_width=True)
                    
                    # Statistical summary of changes
                    with st.expander("📊 Change Statistics"):
                        st.write(cycle_analysis.describe().round(3))
                
                elif analysis_type == "Correlation Analysis":
                    st.write("### Correlation Analysis")
                    
                    # Calculate correlation matrix
                    correlation_cols = [
                        'Discharge_Capacity', 'Charge_Capacity', 
                        'Discharge_Voltage', 'Charge_Voltage',
                        'Coulombic_Efficiency', 'Voltage_Gap'
                    ]
                    
                    corr_matrix = df[correlation_cols].corr().round(3)
                    
                    # Create heatmap
                    fig_corr = go.Figure(data=go.Heatmap(
                        z=corr_matrix.values,
                        x=corr_matrix.columns,
                        y=corr_matrix.index,
                        colorscale='RdBu',
                        zmin=-1,
                        zmax=1,
                        text=np.round(corr_matrix.values, 3),
                        texttemplate='%{text}',
                        textfont={"size": 10},
                        hoverongaps=False
                    ))
                    
                    fig_corr.update_layout(
                        title='Parameter Correlation Matrix',
                        height=600
                    )
                    
                    st.plotly_chart(fig_corr, use_container_width=True)
                    
                    # Detailed correlation analysis
                    with st.expander("📊 Detailed Correlation Analysis"):
                        # Select parameters for detailed analysis
                        param1 = st.selectbox("Select first parameter", correlation_cols)
                        param2 = st.selectbox("Select second parameter", correlation_cols)
                        
                        if param1 != param2:
                            # Create scatter plot
                            fig_scatter = px.scatter(
                                df,
                                x=param1,
                                y=param2,
                                color='Cycle',
                                title=f'{param1} vs {param2}',
                                trendline="ols"
                            )
                            
                            st.plotly_chart(fig_scatter, use_container_width=True)
                            
                            # Calculate regression statistics
                            from scipy import stats
                            slope, intercept, r_value, p_value, std_err = stats.linregress(
                                df[param1], df[param2]
                            )
                            
                            st.write("Regression Statistics:")
                            stats_cols = st.columns(3)
                            stats_cols[0].metric("R²", f"{r_value**2:.3f}")
                            stats_cols[1].metric("Slope", f"{slope:.3e}")
                            stats_cols[2].metric("P-value", f"{p_value:.3e}")
                
                else:  # Distribution Analysis
                    st.write("### Distribution Analysis")
                    
                    # Select parameter for distribution analysis
                    param = st.selectbox(
                        "Select Parameter",
                        ['Discharge_Capacity', 'Charge_Capacity', 
                         'Discharge_Voltage', 'Charge_Voltage',
                         'Coulombic_Efficiency', 'Voltage_Gap']
                    )
                    
                    # Create distribution plot
                    fig_dist = go.Figure()
                    
                    # Add histogram
                    fig_dist.add_trace(go.Histogram(
                        x=df[param],
                        name="Histogram",
                        nbinsx=30,
                        histnorm='probability'
                    ))
                    
                    # Add KDE
                    from scipy.stats import gaussian_kde
                    kde = gaussian_kde(df[param])
                    x_range = np.linspace(df[param].min(), df[param].max(), 100)
                    fig_dist.add_trace(go.Scatter(
                        x=x_range,
                        y=kde(x_range),
                        name="KDE",
                        line=dict(color='red')
                    ))
                    
                    fig_dist.update_layout(
                        title=f'Distribution of {param}',
                        xaxis_title=param,
                        yaxis_title='Probability Density',
                        barmode='overlay'
                    )
                    
                    st.plotly_chart(fig_dist, use_container_width=True)
                    
                    # Statistical tests
                    with st.expander("📊 Distribution Statistics"):
                        from scipy.stats import normaltest, skew, kurtosis
                        
                        # Calculate statistics
                        stat, p_value = normaltest(df[param])
                        skewness = skew(df[param])
                        kurt = kurtosis(df[param])
                        
                        # Display results
                        st.write("Normality Test (D'Agostino-Pearson):")
                        st.write(f"p-value: {p_value:.3e}")
                        st.write(f"Is Normal Distribution? {'Yes' if p_value > 0.05 else 'No'}")
                        
                        stats_cols = st.columns(2)
                        stats_cols[0].metric("Skewness", f"{skewness:.3f}")
                        stats_cols[1].metric("Kurtosis", f"{kurt:.3f}")

# Would you like to see the Raw Data tab next?
# Tab 4: Enhanced Statistical Analysis
            with tabs[3]:
                st.subheader("Advanced Statistical Analysis")
                
                analysis_type = st.radio(
                    "Select Analysis Type",
                    ["Overview Statistics", "Degradation Analysis", "Cycle-by-Cycle Analysis", 
                     "Advanced Correlation", "Distribution Analysis", "Time Series Analysis"],
                    horizontal=True
                )
                
                if analysis_type == "Overview Statistics":
                    # Previous overview statistics code...
                    
                    # Add Moving Window Analysis
                    with st.expander("🔄 Moving Window Analysis"):
                        window_size = st.slider(
                            "Select window size (cycles)",
                            min_value=5,
                            max_value=50,
                            value=10
                        )
                        
                        # Calculate rolling statistics
                        rolling_stats = pd.DataFrame({
                            'Mean Capacity': df['Discharge_Capacity'].rolling(window=window_size).mean(),
                            'Std Capacity': df['Discharge_Capacity'].rolling(window=window_size).std(),
                            'Mean Efficiency': df['Coulombic_Efficiency'].rolling(window=window_size).mean(),
                            'Std Efficiency': df['Coulombic_Efficiency'].rolling(window=window_size).std()
                        })
                        
                        # Plot rolling statistics
                        fig_rolling = go.Figure()
                        
                        # Add traces for capacity
                        fig_rolling.add_trace(go.Scatter(
                            x=df['Cycle'],
                            y=rolling_stats['Mean Capacity'],
                            name='Mean Capacity',
                            line=dict(color='blue')
                        ))
                        
                        fig_rolling.add_trace(go.Scatter(
                            x=df['Cycle'],
                            y=rolling_stats['Mean Capacity'] + rolling_stats['Std Capacity'],
                            name='Capacity +1σ',
                            line=dict(dash='dash', color='lightblue')
                        ))
                        
                        fig_rolling.add_trace(go.Scatter(
                            x=df['Cycle'],
                            y=rolling_stats['Mean Capacity'] - rolling_stats['Std Capacity'],
                            name='Capacity -1σ',
                            line=dict(dash='dash', color='lightblue'),
                            fill='tonexty'
                        ))
                        
                        fig_rolling.update_layout(
                            title=f'Rolling Statistics (Window: {window_size} cycles)',
                            xaxis_title='Cycle Number',
                            yaxis_title='Capacity (mAh/g)'
                        )
                        
                        st.plotly_chart(fig_rolling, use_container_width=True)
                
                elif analysis_type == "Degradation Analysis":
                    st.write("### Advanced Degradation Analysis")
                    
                    # Fit different degradation models
                    from scipy.optimize import curve_fit
                    
                    # Define degradation models
                    def linear_model(x, a, b):
                        return a * x + b
                    
                    def exponential_model(x, a, b, c):
                        return a * np.exp(-b * x) + c
                    
                    def power_law_model(x, a, b, c):
                        return a * x**b + c
                    
                    # Normalize capacity data
                    cycles = df['Cycle'].values
                    capacity = df['Discharge_Capacity'].values / df['Discharge_Capacity'].iloc[0]
                    
                    # Fit models
                    models = {
                        'Linear': (linear_model, ['Slope', 'Intercept']),
                        'Exponential': (exponential_model, ['Amplitude', 'Decay Rate', 'Offset']),
                        'Power Law': (power_law_model, ['Scale', 'Exponent', 'Offset'])
                    }
                    
                    # Plot data and fits
                    fig_fits = go.Figure()
                    
                    # Add actual data
                    fig_fits.add_trace(go.Scatter(
                        x=cycles,
                        y=capacity,
                        name='Actual Data',
                        mode='markers'
                    ))
                    
                    # Fit and plot each model
                    fit_results = {}
                    for model_name, (model_func, param_names) in models.items():
                        try:
                            popt, pcov = curve_fit(model_func, cycles, capacity, maxfev=10000)
                            y_fit = model_func(cycles, *popt)
                            
                            fig_fits.add_trace(go.Scatter(
                                x=cycles,
                                y=y_fit,
                                name=f'{model_name} Fit'
                            ))
                            
                            # Calculate R-squared
                            residuals = capacity - y_fit
                            ss_res = np.sum(residuals**2)
                            ss_tot = np.sum((capacity - np.mean(capacity))**2)
                            r_squared = 1 - (ss_res / ss_tot)
                            
                            # Store results
                            fit_results[model_name] = {
                                'R-squared': r_squared,
                                'Parameters': dict(zip(param_names, popt)),
                                'Uncertainties': np.sqrt(np.diag(pcov))
                            }
                            
                        except RuntimeError:
                            st.warning(f"Could not fit {model_name} model")
                    
                    fig_fits.update_layout(
                        title='Degradation Model Fitting',
                        xaxis_title='Cycle Number',
                        yaxis_title='Normalized Capacity'
                    )
                    
                    st.plotly_chart(fig_fits, use_container_width=True)
                    
                    # Display fit results
                    for model_name, results in fit_results.items():
                        with st.expander(f"{model_name} Model Results"):
                            st.write(f"R-squared: {results['R-squared']:.4f}")
                            st.write("Parameters:")
                            for param_name, value in results['Parameters'].items():
                                st.write(f"{param_name}: {value:.4e}")
                
                elif analysis_type == "Advanced Correlation":
                    st.write("### Advanced Correlation Analysis")
                    
                    # Add partial correlation analysis
                    from scipy.stats import spearmanr
                    
                    def partial_correlation(data, x, y, controlling):
                        """Calculate partial correlation between x and y while controlling for other variables"""
                        x_resid = np.residuals(np.polyfit(data[controlling], data[x], 1), 
                                             data[controlling], data[x])
                        y_resid = np.residuals(np.polyfit(data[controlling], data[y], 1),
                                             data[controlling], data[y])
                        return spearmanr(x_resid, y_resid)
                    
                    # Select variables for partial correlation
                    cols = st.columns(3)
                    with cols[0]:
                        var1 = st.selectbox("Select first variable", df.columns)
                    with cols[1]:
                        var2 = st.selectbox("Select second variable", df.columns)
                    with cols[2]:
                        control_var = st.selectbox("Control for", df.columns)
                    
                    if var1 != var2 != control_var:
                        # Calculate and display partial correlation
                        corr, p_val = partial_correlation(df, var1, var2, control_var)
                        
                        st.write(f"Partial correlation controlling for {control_var}:")
                        st.write(f"Correlation coefficient: {corr:.3f}")
                        st.write(f"P-value: {p_val:.3e}")
                    
                    # Add Granger causality analysis
                    with st.expander("🔄 Granger Causality Analysis"):
                        from statsmodels.tsa.stattools import grangercausalitytests
                        
                        max_lag = st.slider("Maximum lag to test", 1, 20, 5)
                        
                        try:
                            gc_test = grangercausalitytests(df[[var1, var2]], maxlag=max_lag, verbose=False)
                            
                            # Display results
                            results_df = pd.DataFrame(
                                index=range(1, max_lag + 1),
                                columns=['F-statistic', 'p-value']
                            )
                            
                            for lag in range(1, max_lag + 1):
                                results_df.loc[lag] = [
                                    gc_test[lag][0]['ssr_ftest'][0],
                                    gc_test[lag][0]['ssr_ftest'][1]
                                ]
                            
                            st.write(results_df)
                            
                            # Plot test results
                            fig_gc = go.Figure()
                            fig_gc.add_trace(go.Scatter(
                                x=results_df.index,
                                y=-np.log10(results_df['p-value']),
                                mode='lines+markers',
                                name='-log10(p-value)'
                            ))
                            
                            fig_gc.update_layout(
                                title='Granger Causality Test Results',
                                xaxis_title='Lag',
                                yaxis_title='-log10(p-value)'
                            )
                            
                            st.plotly_chart(fig_gc, use_container_width=True)
                            
                        except:
                            st.error("Could not perform Granger causality test on selected variables")
                
                elif analysis_type == "Time Series Analysis":
                    st.write("### Time Series Analysis")
                    
                    # Select variable for analysis
                    var_to_analyze = st.selectbox(
                        "Select variable for analysis",
                        ['Discharge_Capacity', 'Charge_Capacity', 'Coulombic_Efficiency']
                    )
                    
                    # Decomposition analysis
                    from statsmodels.tsa.seasonal import seasonal_decompose
                    
                    decomposition = seasonal_decompose(
                        df[var_to_analyze],
                        period=max(10, len(df)//10),
                        extrapolate_trend='freq'
                    )
                    
                    # Plot decomposition
                    fig_decomp = sp.make_subplots(
                        rows=4, cols=1,
                        subplot_titles=['Original', 'Trend', 'Seasonal', 'Residual']
                    )
                    
                    components = [
                        df[var_to_analyze],
                        decomposition.trend,
                        decomposition.seasonal,
                        decomposition.resid
                    ]
                    
                    for i, component in enumerate(components, 1):
                        fig_decomp.add_trace(
                            go.Scatter(x=df['Cycle'], y=component),
                            row=i, col=1
                        )
                    
                    fig_decomp.update_layout(height=800, title='Time Series Decomposition')
                    st.plotly_chart(fig_decomp, use_container_width=True)
                    
                    # Add change point detection
                    with st.expander("🔍 Change Point Detection"):
                        from ruptures import Binseg
                        
                        # Perform change point detection
                        model = Binseg(model="l2").fit(df[var_to_analyze].values.reshape(-1, 1))
                        n_bkps = st.slider("Number of change points", 1, 10, 3)
                        bkps = model.predict(n_bkps=n_bkps)
                        
                        # Plot results
                        fig_cp = go.Figure()
                        
                        fig_cp.add_trace(go.Scatter(
                            x=df['Cycle'],
                            y=df[var_to_analyze],
                            name='Data'
                        ))
                        
                        # Add vertical lines for change points
                        for bkp in bkps[:-1]:
                            fig_cp.add_vline(
                                x=df['Cycle'].iloc[bkp],
                                line_dash="dash",
                                line_color="red"
                            )
                        
                        fig_cp.update_layout(
                            title='Change Point Detection',
                            xaxis_title='Cycle Number',
                            yaxis_title=var_to_analyze
                        )
                        
                        st.plotly_chart(fig_cp, use_container_width=True)
                        
                        # Display segment statistics
                        st.write("### Segment Statistics")
                        segments = np.split(df[var_to_analyze], bkps[:-1])
                        
                        for i, segment in enumerate(segments):
                            st.write(f"Segment {i+1}:")
                            st.write(segment.describe())

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
try:
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
            
            elif ml_analysis_type == "Anomaly Detection":
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
            
            elif ml_analysis_type == "Pattern Recognition":
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
            
            else:  # RUL Estimation
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

except Exception as e:
    st.error(f"An error occurred: {str(e)}")
    st.write("Please check your data format and try again.")

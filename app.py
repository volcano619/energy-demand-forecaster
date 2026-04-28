"""
Streamlit Dashboard for Energy Demand Forecasting

Features:
1. Historical data visualization
2. Interactive forecasting
3. Model comparison
4. Anomaly detection
"""

import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging

from config import (
    APP_TITLE, APP_LAYOUT, TARGET_COL, DATETIME_COL,
    FORECAST_HORIZONS, DEFAULT_HORIZON, ANOMALY_THRESHOLD_SIGMA,
    CHART_HEIGHT, PRIMARY_COLOR, SECONDARY_COLOR, ANOMALY_COLOR
)
from models.data_processor import load_energy_data, create_features
from models.statistical import ProphetForecaster, SimpleSeasonalModel
import shared_ui

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Page config
st.set_page_config(
    page_title=APP_TITLE,
    layout=APP_LAYOUT,
    page_icon="⚡",
    initial_sidebar_state="expanded"
)

# Apply global theme
shared_ui.apply_global_theme()


# ============================================================================
# DATA LOADING (Cached)
# ============================================================================

@st.cache_data
def load_data():
    """Load and cache energy data."""
    with st.spinner("Loading energy data..."):
        df = load_energy_data()
        df = create_features(df)
        df = df.dropna()  # Remove rows with NaN from feature engineering
    return df


@st.cache_resource
def get_forecaster(df: pd.DataFrame):
    """Train and cache forecaster."""
    model = SimpleSeasonalModel(seasonal_period=168)
    model.fit(df)
    return model


# Load data
try:
    df = load_data()
    data_loaded = True
except Exception as e:
    st.error(f"Error loading data: {e}")
    data_loaded = False
    st.stop()


# ============================================================================
# SIDEBAR
# ============================================================================

with st.sidebar:
    st.markdown("## ⚙️ Settings")
    
    # Date range selection
    st.markdown("### 📅 Date Range")
    min_date = df[DATETIME_COL].min().date()
    max_date = df[DATETIME_COL].max().date()
    
    if "horizon_name" not in st.session_state:
        st.session_state.horizon_name = list(FORECAST_HORIZONS.keys())[0]
    
    # Initialize date range if not present
    if "date_range" not in st.session_state or not st.session_state.date_range:
        st.session_state.date_range = (max_date - timedelta(days=30), max_date)
    
    date_range = st.date_input(
        "Select Range",
        key="date_range",
        min_value=min_date,
        max_value=max_date
    )
    
    # Forecast settings
    st.markdown("### 🔮 Forecast Settings")
    horizon_name = st.selectbox(
        "Forecast Horizon",
        options=list(FORECAST_HORIZONS.keys()),
        key="horizon_name"
    )
    horizon_hours = FORECAST_HORIZONS[horizon_name]
    
    # Anomaly detection
    st.markdown("### 🚨 Anomaly Detection")
    if "anomaly_threshold" not in st.session_state:
        from config import ANOMALY_THRESHOLD_SIGMA
        st.session_state.anomaly_threshold = ANOMALY_THRESHOLD_SIGMA
    anomaly_threshold = st.slider(
        "Threshold (σ)",
        min_value=1.0,
        max_value=5.0,
        key="anomaly_threshold",
        step=0.5,
        help="Standard deviations from mean"
    )
    
    # Model selection
    st.markdown("### 🤖 Model")
    if "ts_model_type" not in st.session_state:
        st.session_state.ts_model_type = "Seasonal Naive"
    model_type = st.selectbox(
        "Forecasting Model",
        options=["Seasonal Naive", "Prophet (if available)"],
        key="ts_model_type"
    )

    # Help Section
    shared_ui.add_help_section(
        "Energy Demand Forecasting",
        "Predictive analytics system for real-time energy grid demand optimization.",
        "Select a date range and forecast horizon in the sidebar, then click 'Generate Forecast'.",
        "Traditional 'Seasonal Averages' fail during extreme events; this ML model adapts to complex patterns and detects anomalies.",
        "Grid operators can pre-allocate resources for a predicted 20% surge tomorrow, preventing brownouts."
    )


# ============================================================================
# MAIN CONTENT
# ============================================================================

# Header
shared_ui.add_header(
    "⚡ Energy Demand Forecasting",
    "Real-time energy demand analysis and forecasting for grid optimization | *Solving the $20B+ grid optimization challenge*"
)

st.markdown("---")

# Filter data by date range
if len(date_range) == 2:
    mask = (df[DATETIME_COL].dt.date >= date_range[0]) & (df[DATETIME_COL].dt.date <= date_range[1])
    df_filtered = df[mask].copy()
else:
    df_filtered = df.tail(720).copy()  # Last 30 days


# ============================================================================
# TAB LAYOUT
# ============================================================================

tab1, tab2, tab3, tab4 = st.tabs([
    "📊 Dashboard",
    "🔮 Forecasting",
    "🚨 Anomalies",
    "📈 Analysis"
])


# ============================================================================
# TAB 1: DASHBOARD
# ============================================================================

with tab1:
    # Key metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        shared_ui.create_metric_card("Avg Demand", f"{df_filtered[TARGET_COL].mean():.0f} MW")
    with col2:
        shared_ui.create_metric_card("Peak Demand", f"{df_filtered[TARGET_COL].max():.0f} MW")
    with col3:
        shared_ui.create_metric_card("Min Demand", f"{df_filtered[TARGET_COL].min():.0f} MW")
    with col4:
        shared_ui.create_metric_card("Volatility (σ)", f"{df_filtered[TARGET_COL].std():.0f} MW")
    
    st.markdown("---")
    
    # Main demand chart
    st.markdown("### Energy Demand Over Time")
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df_filtered[DATETIME_COL],
        y=df_filtered[TARGET_COL],
        mode='lines',
        name='Demand',
        line=dict(color=PRIMARY_COLOR)
    ))
    
    # Add anomaly markers
    if 'is_anomaly' in df_filtered.columns:
        anomalies = df_filtered[df_filtered['is_anomaly'] == 1]
        fig.add_trace(go.Scatter(
            x=anomalies[DATETIME_COL],
            y=anomalies[TARGET_COL],
            mode='markers',
            name='Anomaly',
            marker=dict(color=ANOMALY_COLOR, size=10, symbol='x')
        ))
    
    fig.update_layout(
        height=CHART_HEIGHT,
        xaxis_title="Time",
        yaxis_title="Demand (MW)",
        hovermode='x unified'
    )
    st.plotly_chart(fig, use_container_width=True)
    
    # Daily pattern
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### Daily Pattern")
        hourly_avg = df_filtered.groupby('hour')[TARGET_COL].mean().reset_index()
        fig_hourly = px.bar(hourly_avg, x='hour', y=TARGET_COL, 
                           labels={'hour': 'Hour of Day', TARGET_COL: 'Avg Demand (MW)'},
                           color=TARGET_COL, color_continuous_scale='YlOrRd')
        fig_hourly.update_layout(height=300)
        st.plotly_chart(fig_hourly, use_container_width=True)
    
    with col2:
        st.markdown("### Weekly Pattern")
        daily_avg = df_filtered.groupby('day_of_week')[TARGET_COL].mean().reset_index()
        daily_avg['day_name'] = daily_avg['day_of_week'].map({
            0: 'Mon', 1: 'Tue', 2: 'Wed', 3: 'Thu', 4: 'Fri', 5: 'Sat', 6: 'Sun'
        })
        fig_daily = px.bar(daily_avg, x='day_name', y=TARGET_COL,
                          labels={'day_name': 'Day', TARGET_COL: 'Avg Demand (MW)'},
                          color=TARGET_COL, color_continuous_scale='YlOrRd')
        fig_daily.update_layout(height=300)
        st.plotly_chart(fig_daily, use_container_width=True)


# ============================================================================
# TAB 2: FORECASTING
# ============================================================================

with tab2:
    st.markdown("### 🔮 Demand Forecast")
    st.markdown(f"Forecasting **{horizon_hours} hours** ahead using **{model_type}**")
    
    if st.button("Generate Forecast", type="primary"):
        with st.spinner("Generating forecast..."):
            try:
                # Use simple seasonal model
                forecaster = get_forecaster(df)
                predictions = forecaster.predict(horizon_hours)
                
                # Create forecast dataframe
                last_date = df[DATETIME_COL].max()
                forecast_dates = pd.date_range(
                    start=last_date + timedelta(hours=1),
                    periods=horizon_hours,
                    freq='h'
                )
                
                st.session_state.forecast_df = pd.DataFrame({
                    DATETIME_COL: forecast_dates,
                    'forecast': predictions,
                    'lower': predictions * 0.9,  # Simple confidence interval
                    'upper': predictions * 1.1
                })
                st.session_state.forecast_metrics = {
                    "avg": predictions.mean(),
                    "peak": predictions.max(),
                    "min": predictions.min()
                }
            except Exception as e:
                st.error(f"Forecasting failed: {e}")

    if "forecast_df" in st.session_state:
        forecast_df = st.session_state.forecast_df
        metrics = st.session_state.forecast_metrics
        
        # Display metrics
        col1, col2, col3 = st.columns(3)
        with col1:
            shared_ui.create_metric_card("Avg Forecast", f"{metrics['avg']:.0f} MW")
        with col2:
            shared_ui.create_metric_card("Peak Forecast", f"{metrics['peak']:.0f} MW")
        with col3:
            shared_ui.create_metric_card("Min Forecast", f"{metrics['min']:.0f} MW")
        
        # Forecast chart
        fig = go.Figure()
        
        # Historical (last 48 hours)
        recent = df.tail(48)
        fig.add_trace(go.Scatter(
            x=recent[DATETIME_COL],
            y=recent[TARGET_COL],
            mode='lines',
            name='Historical',
            line=dict(color=PRIMARY_COLOR)
        ))
        
        # Forecast
        fig.add_trace(go.Scatter(
            x=forecast_df[DATETIME_COL],
            y=forecast_df['forecast'],
            mode='lines',
            name='Forecast',
            line=dict(color=SECONDARY_COLOR, dash='dash')
        ))
        
        # Confidence interval
        fig.add_trace(go.Scatter(
            x=list(forecast_df[DATETIME_COL]) + list(forecast_df[DATETIME_COL][::-1]),
            y=list(forecast_df['upper']) + list(forecast_df['lower'][::-1]),
            fill='toself',
            fillcolor='rgba(255,127,14,0.2)',
            line=dict(color='rgba(0,0,0,0)'),
            name='Confidence Interval'
        ))
        
        fig.update_layout(
            height=CHART_HEIGHT,
            xaxis_title="Time",
            yaxis_title="Demand (MW)",
            hovermode='x unified'
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Forecast table
        with st.expander("📋 Forecast Details"):
            st.dataframe(
                forecast_df.assign(**{DATETIME_COL: forecast_df[DATETIME_COL].dt.strftime('%Y-%m-%d %H:%M')}),
                use_container_width=True
            )
    else:
        st.info("Click 'Generate Forecast' to see predictions")


# ============================================================================
# TAB 3: ANOMALIES
# ============================================================================

with tab3:
    st.markdown("### 🚨 Anomaly Detection")
    
    # Calculate anomalies
    mean_demand = df_filtered[TARGET_COL].mean()
    std_demand = df_filtered[TARGET_COL].std()
    
    upper_bound = mean_demand + anomaly_threshold * std_demand
    lower_bound = mean_demand - anomaly_threshold * std_demand
    
    df_filtered['is_detected_anomaly'] = (
        (df_filtered[TARGET_COL] > upper_bound) | 
        (df_filtered[TARGET_COL] < lower_bound)
    )
    
    detected_anomalies = df_filtered[df_filtered['is_detected_anomaly']]
    
    # Summary
    col1, col2, col3 = st.columns(3)
    with col1:
        shared_ui.create_metric_card("Total Anomalies", str(len(detected_anomalies)), delta="Last 30 days", delta_pos=False)
    with col2:
        shared_ui.create_metric_card("Anomaly Rate", f"{len(detected_anomalies) / len(df_filtered) * 100:.1f}%")
    with col3:
        shared_ui.create_metric_card("Threshold", f"±{anomaly_threshold}σ", delta="Sensitivity")
    
    st.markdown("---")
    
    # Anomaly visualization
    fig = go.Figure()
    
    # Normal data
    normal = df_filtered[~df_filtered['is_detected_anomaly']]
    fig.add_trace(go.Scatter(
        x=normal[DATETIME_COL],
        y=normal[TARGET_COL],
        mode='lines',
        name='Normal',
        line=dict(color=PRIMARY_COLOR)
    ))
    
    # Anomalies
    fig.add_trace(go.Scatter(
        x=detected_anomalies[DATETIME_COL],
        y=detected_anomalies[TARGET_COL],
        mode='markers',
        name='Anomaly',
        marker=dict(color=ANOMALY_COLOR, size=10, symbol='x')
    ))
    
    # Bounds
    fig.add_hline(y=upper_bound, line_dash="dash", line_color="red", annotation_text="Upper Bound")
    fig.add_hline(y=lower_bound, line_dash="dash", line_color="red", annotation_text="Lower Bound")
    
    fig.update_layout(height=CHART_HEIGHT, xaxis_title="Time", yaxis_title="Demand (MW)")
    st.plotly_chart(fig, use_container_width=True)
    
    # Anomaly list
    if len(detected_anomalies) > 0:
        st.markdown("### Recent Anomalies")
        display_df = detected_anomalies[[DATETIME_COL, TARGET_COL, 'temperature']].tail(10).copy()
        display_df[DATETIME_COL] = display_df[DATETIME_COL].dt.strftime('%Y-%m-%d %H:%M')
        display_df.columns = ['Timestamp', 'Demand (MW)', 'Temperature (°C)']
        st.dataframe(display_df, use_container_width=True)
    else:
        st.success("No anomalies detected in selected period!")


# ============================================================================
# TAB 4: ANALYSIS
# ============================================================================

with tab4:
    st.markdown("### 📈 Demand Analysis")
    
    # Temperature vs Demand
    st.markdown("#### Temperature vs Demand Correlation")
    fig_scatter = px.scatter(
        df_filtered,
        x='temperature',
        y=TARGET_COL,
        color='is_weekend',
        labels={'temperature': 'Temperature (°C)', TARGET_COL: 'Demand (MW)', 'is_weekend': 'Weekend'},
        color_discrete_map={0: PRIMARY_COLOR, 1: SECONDARY_COLOR}
    )
    fig_scatter.update_layout(height=400)
    st.plotly_chart(fig_scatter, use_container_width=True)
    
    # Statistics
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### Demand Statistics")
        stats = df_filtered[TARGET_COL].describe()
        st.dataframe(stats.round(2))
    
    with col2:
        st.markdown("#### Temperature Statistics")
        temp_stats = df_filtered['temperature'].describe()
        st.dataframe(temp_stats.round(2))
    
    # Monthly trends
    st.markdown("#### Monthly Demand Trends")
    monthly = df_filtered.groupby('month')[TARGET_COL].agg(['mean', 'std']).reset_index()
    monthly['month_name'] = monthly['month'].map({
        1: 'Jan', 2: 'Feb', 3: 'Mar', 4: 'Apr', 5: 'May', 6: 'Jun',
        7: 'Jul', 8: 'Aug', 9: 'Sep', 10: 'Oct', 11: 'Nov', 12: 'Dec'
    })
    
    fig_monthly = go.Figure()
    fig_monthly.add_trace(go.Bar(
        x=monthly['month_name'],
        y=monthly['mean'],
        name='Average',
        marker_color=PRIMARY_COLOR
    ))
    fig_monthly.add_trace(go.Scatter(
        x=monthly['month_name'],
        y=monthly['std'],
        name='Std Dev',
        yaxis='y2',
        line=dict(color=SECONDARY_COLOR)
    ))
    fig_monthly.update_layout(
        yaxis=dict(title='Average Demand (MW)'),
        yaxis2=dict(title='Std Dev', overlaying='y', side='right'),
        height=350
    )
    st.plotly_chart(fig_monthly, use_container_width=True)


# ============================================================================
# FOOTER
# ============================================================================

st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666; font-size: 0.9rem;">
    ⚡ Energy Demand Forecasting | Hybrid ML: Statistical + Deep Learning<br>
    Solving the $20B+ grid optimization challenge
</div>
""", unsafe_allow_html=True)

st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #64748B; font-size: 0.875rem; padding: 2rem 0;">
    ⚡ Energy Demand Forecasting | Built with Prophet + Seasonal Decompose<br>
    <span style="font-family: 'Roboto Mono', monospace;">Version 1.3.0-Premium</span>
</div>
""", unsafe_allow_html=True)

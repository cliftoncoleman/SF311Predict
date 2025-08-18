import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import os
from utils.api_client import APIClient
from utils.data_processor import DataProcessor
from components.charts import ChartGenerator
from components.filters import FilterComponent

# Page configuration
st.set_page_config(
    page_title="SF311 Street & Sidewalk Cleaning Predictions",
    page_icon="🧹",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state
if 'data' not in st.session_state:
    st.session_state.data = None
if 'neighborhoods' not in st.session_state:
    st.session_state.neighborhoods = []
if 'last_refresh' not in st.session_state:
    st.session_state.last_refresh = None

# Initialize components
api_client = APIClient()
data_processor = DataProcessor()
chart_generator = ChartGenerator()
filter_component = FilterComponent()

def main():
    """Main application function"""
    
    # Header
    st.title("🧹 SF311 Street & Sidewalk Cleaning Predictions Dashboard")
    st.markdown("---")
    
    # Sidebar for filters and controls
    with st.sidebar:
        st.header("Dashboard Controls")
        
        # Refresh data button
        if st.button("🔄 Refresh Data", type="primary"):
            load_data()
        
        # Display last refresh time
        if st.session_state.last_refresh:
            st.caption(f"Last updated: {st.session_state.last_refresh.strftime('%Y-%m-%d %H:%M:%S')}")
        
        st.markdown("---")
        
        # Date range selection
        st.subheader("📅 Date Range")
        default_start = datetime.now().date()
        default_end = default_start + timedelta(days=30)
        
        date_range = st.date_input(
            "Select prediction period:",
            value=(default_start, default_end),
            min_value=datetime.now().date(),
            max_value=datetime.now().date() + timedelta(days=365)
        )
        
        # Neighborhood selection
        st.subheader("🏘️ Neighborhoods")
        if st.session_state.neighborhoods:
            selected_neighborhoods = st.multiselect(
                "Select neighborhoods:",
                options=st.session_state.neighborhoods,
                default=st.session_state.neighborhoods[:5] if len(st.session_state.neighborhoods) > 5 else st.session_state.neighborhoods,
                help="Choose specific neighborhoods to analyze"
            )
        else:
            selected_neighborhoods = []
            st.info("Load data to see available neighborhoods")
        
        # View options
        st.subheader("📊 Display Options")
        show_confidence_intervals = st.checkbox("Show confidence intervals", value=True)
        chart_type = st.selectbox(
            "Chart type:",
            ["Line Chart", "Bar Chart", "Heatmap"],
            index=0
        )
        
        # Aggregation level
        aggregation_level = st.selectbox(
            "Aggregation level:",
            ["Daily", "Weekly", "Monthly"],
            index=0
        )
        
        # Historical comparison toggle
        st.subheader("📊 Historical Comparison")
        show_historical = st.checkbox("Show actual vs predicted", value=True)
        if show_historical:
            historical_days = st.slider(
                "Historical period (days):",
                min_value=30,
                max_value=180,
                value=90,
                step=30
            )
    
    # Main content area
    if st.session_state.data is None:
        # Initial load screen
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            st.info("👋 Welcome! Click 'Refresh Data' to load SF311 predictions.")
            if st.button("Load Initial Data", type="primary"):
                load_data()
    else:
        # Filter data based on selections
        if len(date_range) == 2:
            start_date, end_date = date_range
            filtered_data = filter_data(st.session_state.data, start_date, end_date, selected_neighborhoods)
            
            if filtered_data.empty:
                st.warning("No data available for the selected filters. Please adjust your selection.")
                return
            
            # Process data for visualization
            processed_data = data_processor.process_for_visualization(
                filtered_data, 
                aggregation_level.lower()
            )
            
            # Get historical data for comparison if requested
            historical_data = None
            if 'show_historical' in locals() and show_historical:
                with st.spinner("Loading historical comparison data..."):
                    historical_data = api_client.get_historical_comparison_data(historical_days)
            
            # Main dashboard layout
            display_dashboard(processed_data, chart_type, show_confidence_intervals, aggregation_level, historical_data)
        else:
            st.error("Please select both start and end dates.")

def load_data():
    """Load data from API"""
    with st.spinner("Loading prediction data..."):
        try:
            # Fetch predictions data
            predictions = api_client.get_predictions()
            
            # Fetch neighborhoods data
            neighborhoods = api_client.get_neighborhoods()
            
            if predictions is not None and neighborhoods is not None:
                st.session_state.data = predictions
                st.session_state.neighborhoods = neighborhoods
                st.session_state.last_refresh = datetime.now()
                st.success("✅ Data loaded successfully!")
                st.rerun()
            else:
                st.error("❌ Failed to load data. Please check API connection.")
                
        except Exception as e:
            st.error(f"❌ Error loading data: {str(e)}")

def filter_data(data, start_date, end_date, neighborhoods):
    """Filter data based on user selections"""
    try:
        # Convert dates to datetime for filtering
        filtered_data = data.copy()
        
        # Filter by date range
        filtered_data['date'] = pd.to_datetime(filtered_data['date'])
        filtered_data = filtered_data[
            (filtered_data['date'].dt.date >= start_date) & 
            (filtered_data['date'].dt.date <= end_date)
        ]
        
        # Filter by neighborhoods
        if neighborhoods:
            filtered_data = filtered_data[filtered_data['neighborhood'].isin(neighborhoods)]
        
        return filtered_data
    except Exception as e:
        st.error(f"Error filtering data: {str(e)}")
        return pd.DataFrame()

def display_dashboard(data, chart_type, show_confidence_intervals, aggregation_level, historical_data=None):
    """Display the main dashboard with charts and metrics"""
    
    # Key metrics row
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        total_predictions = data['predicted_requests'].sum()
        delta_text = None
        if historical_data is not None and not historical_data.empty:
            historical_total = historical_data['actual_requests'].sum()
            delta_text = f"{((total_predictions - historical_total) / historical_total * 100):+.1f}% vs recent actual"
        st.metric(
            label="Total Predicted Requests (All Neighborhoods)",
            value=f"{total_predictions:,.0f}",
            delta=delta_text
        )
    
    with col2:
        avg_daily = data.groupby('date')['predicted_requests'].sum().mean()
        st.metric(
            label=f"Avg {aggregation_level} Requests",
            value=f"{avg_daily:.1f}"
        )
    
    with col3:
        peak_day = data.groupby('date')['predicted_requests'].sum().idxmax()
        peak_value = data.groupby('date')['predicted_requests'].sum().max()
        st.metric(
            label="Peak Day",
            value=peak_day.strftime('%m/%d') if hasattr(peak_day, 'strftime') else str(peak_day),
            delta=f"{peak_value:.0f} requests"
        )
    
    with col4:
        unique_neighborhoods = data['neighborhood'].nunique()
        st.metric(
            label="Neighborhoods",
            value=unique_neighborhoods
        )
    
    st.markdown("---")
    
    # Main visualization area
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("📈 Prediction Trends")
        
        if chart_type == "Line Chart":
            fig = chart_generator.create_line_chart(data, show_confidence_intervals, historical_data)
        elif chart_type == "Bar Chart":
            fig = chart_generator.create_bar_chart(data, historical_data)
        else:  # Heatmap
            fig = chart_generator.create_heatmap(data)
        
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("🏘️ Top Neighborhoods")
        neighborhood_totals = data.groupby('neighborhood')['predicted_requests'].sum().sort_values(ascending=False)
        
        # Create a bar chart for top neighborhoods
        fig_neighborhoods = px.bar(
            x=neighborhood_totals.head(10).values,
            y=neighborhood_totals.head(10).index,
            orientation='h',
            title="Top 10 Neighborhoods by Predicted Requests"
        )
        fig_neighborhoods.update_layout(
            height=400,
            showlegend=False,
            yaxis={'categoryorder': 'total ascending'}
        )
        st.plotly_chart(fig_neighborhoods, use_container_width=True)
    
    # Data table section
    st.markdown("---")
    st.subheader("📋 Detailed Predictions")
    
    # Display options for table
    col1, col2 = st.columns([3, 1])
    with col2:
        show_raw_data = st.checkbox("Show raw data", value=False)
    
    if show_raw_data:
        st.dataframe(
            data.sort_values(['date', 'neighborhood']),
            use_container_width=True,
            hide_index=True
        )
    else:
        # Show aggregated summary
        summary_data = data.groupby(['date', 'neighborhood']).agg({
            'predicted_requests': 'sum',
            'confidence_lower': 'mean',
            'confidence_upper': 'mean'
        }).reset_index()
        
        st.dataframe(
            summary_data.sort_values(['date', 'neighborhood']),
            use_container_width=True,
            hide_index=True
        )
    
    # Download section
    st.markdown("---")
    col1, col2 = st.columns([3, 1])
    with col2:
        csv_data = data.to_csv(index=False)
        st.download_button(
            label="📥 Download Data (CSV)",
            data=csv_data,
            file_name=f"sf311_predictions_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv"
        )

if __name__ == "__main__":
    main()

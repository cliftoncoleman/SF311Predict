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
                default=st.session_state.neighborhoods,  # Show all neighborhoods by default
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
        
        # Historical comparison settings
        st.subheader("📊 Historical Comparison")
        show_historical = st.checkbox("Show actual vs predicted", value=True)  # Default to True
        historical_days = st.slider(
            "Historical period (days):",
            min_value=30,
            max_value=180,
            value=90,
            step=30,
            help="Number of days of historical data to compare against predictions"
        )
    
    # Main content area
    if st.session_state.data is None:
        # Auto-load data on first visit
        load_data()
        
    if st.session_state.data is None:
        # If data still failed to load, show fallback
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
            if show_historical:
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
    col1, col2, col3 = st.columns(3)
    
    with col1:
        # Current day's predicted requests
        today = datetime.now().date()
        today_data = data[pd.to_datetime(data['date']).dt.date == today]
        if not today_data.empty:
            today_total = today_data['predicted_requests'].sum()
            st.metric(
                label="Today's Predicted Requests",
                value=f"{today_total:,.0f}"
            )
        else:
            # Get first available day if today's data isn't available
            first_day_total = data.groupby('date')['predicted_requests'].sum().iloc[0] if not data.empty else 0
            st.metric(
                label="Next Day's Predicted Requests", 
                value=f"{first_day_total:,.0f}"
            )
    
    with col2:
        # Top neighborhood by total predicted requests
        neighborhood_totals = data.groupby('neighborhood')['predicted_requests'].sum()
        if not neighborhood_totals.empty:
            top_neighborhood = neighborhood_totals.idxmax()
            top_value = neighborhood_totals.max()
            st.metric(
                label="Top Neighborhood",
                value=top_neighborhood,
                delta=f"{top_value:.0f} total requests"
            )
        else:
            st.metric(label="Top Neighborhood", value="No data")
    
    with col3:
        peak_day = data.groupby('date')['predicted_requests'].sum().idxmax()
        peak_value = data.groupby('date')['predicted_requests'].sum().max()
        st.metric(
            label="Peak Day",
            value=peak_day.strftime('%m/%d') if hasattr(peak_day, 'strftime') else str(peak_day),
            delta=f"{peak_value:.0f} requests"
        )
    
    st.markdown("---")
    
    # Main visualization area
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("📈 Prediction Trends")
        
        if chart_type == "Line Chart":
            fig = chart_generator.create_line_chart(data, show_confidence_intervals, historical_data if historical_data is not None else pd.DataFrame())
        elif chart_type == "Bar Chart":
            fig = chart_generator.create_bar_chart(data, historical_data if historical_data is not None else pd.DataFrame())
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
    
    # Historical data section (moved to bottom as table)
    if historical_data is not None and not historical_data.empty:
        st.markdown("---")
        st.subheader("📊 Historical vs Predicted Comparison")
        
        # Create summary table by neighborhood
        historical_summary = historical_data.groupby('neighborhood').agg({
            'actual_requests': ['sum', 'mean']
        }).round(1)
        
        # Flatten column names
        historical_summary.columns = ['Total Actual', 'Avg Daily Actual']
        historical_summary = historical_summary.reset_index()
        
        # Add predicted data for comparison
        predicted_summary = data.groupby('neighborhood').agg({
            'predicted_requests': ['sum', 'mean']
        }).round(1)
        predicted_summary.columns = ['Total Predicted', 'Avg Daily Predicted']
        predicted_summary = predicted_summary.reset_index()
        
        # Merge the data
        comparison_table = historical_summary.merge(predicted_summary, on='neighborhood', how='outer').fillna(0)
        
        # Calculate accuracy metrics
        comparison_table['Accuracy %'] = (
            (comparison_table['Avg Daily Actual'] - abs(comparison_table['Avg Daily Actual'] - comparison_table['Avg Daily Predicted'])) / 
            comparison_table['Avg Daily Actual'] * 100
        ).round(1)
        comparison_table['Accuracy %'] = comparison_table['Accuracy %'].fillna(0)
        
        st.dataframe(
            comparison_table.sort_values('Total Predicted', ascending=False),
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

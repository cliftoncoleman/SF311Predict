import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import numpy as np
from fixed_pipeline import FixedSF311Pipeline
from utils.data_processor import DataProcessor

# Page configuration
st.set_page_config(
    page_title="Enhanced SF311 Predictions - Working Version",
    page_icon="🧹",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state
if 'working_data' not in st.session_state:
    st.session_state.working_data = None
if 'working_neighborhoods' not in st.session_state:
    st.session_state.working_neighborhoods = []
if 'last_working_refresh' not in st.session_state:
    st.session_state.last_working_refresh = None

def get_working_pipeline():
    return FixedSF311Pipeline()



def get_data_processor():
    return DataProcessor()

# Create fresh instances each time to avoid caching issues
pipeline = get_working_pipeline()
processor = get_data_processor()

def create_simple_line_chart(data: pd.DataFrame) -> go.Figure:
    """Create simple working line chart"""
    fig = go.Figure()
    
    if data.empty:
        fig.add_annotation(
            text="No data available",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False
        )
        return fig
    
    # Group by neighborhood and create traces
    for neighborhood in data['neighborhood'].unique()[:10]:  # Limit to 10 for performance
        nbhd_data = data[data['neighborhood'] == neighborhood]
        fig.add_trace(go.Scatter(
            x=nbhd_data['date'],
            y=nbhd_data['predicted_requests'],
            mode='lines+markers',
            name=neighborhood,
            hovertemplate='%{y:.0f} requests<extra></extra>'
        ))
    
    fig.update_layout(
        title="SF311 Predictions by Neighborhood",
        xaxis_title="Date",
        yaxis_title="Predicted Requests",
        height=500
    )
    
    return fig

def create_simple_bar_chart(data: pd.DataFrame) -> go.Figure:
    """Create simple bar chart"""
    if data.empty:
        fig = go.Figure()
        fig.add_annotation(text="No data available", x=0.5, y=0.5, showarrow=False)
        return fig
    
    # Aggregate by neighborhood
    neighborhood_totals = data.groupby('neighborhood')['predicted_requests'].sum().sort_values(ascending=True)
    
    fig = go.Figure(data=go.Bar(
        y=neighborhood_totals.index,
        x=neighborhood_totals.values,
        orientation='h',
        marker_color='#3498DB'
    ))
    
    fig.update_layout(
        title="Total Predicted Requests by Neighborhood",
        xaxis_title="Predicted Requests",
        yaxis_title="Neighborhood",
        height=max(400, len(neighborhood_totals) * 25)
    )
    
    return fig

def load_working_data(force_refresh: bool = False):
    """Load data using smart database caching for dramatically improved performance"""
    try:
        with st.spinner("Loading SF311 data with smart caching..."):
            # Clear Streamlit caches if force refresh
            if force_refresh:
                st.cache_data.clear()
                st.cache_resource.clear()
            
            # Use smart caching pipeline for much faster data loading
            from database_cache import SmartSF311Pipeline
            smart_pipeline = SmartSF311Pipeline()
            
            # Generate predictions with intelligent caching (only fetches new data)
            predictions = smart_pipeline.generate_predictions_with_cache(
                target_days=1825,  # 5 years for robust training
                force_refresh=force_refresh
            )
            
            if not predictions.empty:
                st.session_state.working_data = predictions
                # Use specific default neighborhoods in priority order (exact case match)
                priority_neighborhoods = [
                    "South Of Market", "Tenderloin", "Hayes Valley", 
                    "Mission", "Bayview Hunters Point"
                ]
                
                # Get all available neighborhoods
                available_neighborhoods = sorted(predictions['neighborhood'].unique())
                
                # Put priority neighborhoods first, then others
                ordered_neighborhoods = []
                for neighborhood in priority_neighborhoods:
                    if neighborhood in available_neighborhoods:
                        ordered_neighborhoods.append(neighborhood)
                
                # Add remaining neighborhoods alphabetically
                for neighborhood in available_neighborhoods:
                    if neighborhood not in ordered_neighborhoods:
                        ordered_neighborhoods.append(neighborhood)
                
                st.session_state.working_neighborhoods = ordered_neighborhoods
                st.session_state.last_working_refresh = datetime.now()
                

                
                # Show success message
                st.success(f"🚀 Smart caching enabled! {len(predictions)} records from {len(st.session_state.working_neighborhoods)} neighborhoods")
            else:
                st.error("No data could be loaded")
                
    except Exception as e:
        st.error(f"Pipeline error: {str(e)}")
        
        # Provide demo data as fallback
        st.info("Loading demo data instead...")
        demo_data = create_demo_data()
        st.session_state.working_data = demo_data
        st.session_state.working_neighborhoods = sorted(demo_data['neighborhood'].unique())
        st.session_state.last_working_refresh = datetime.now()





def create_demo_data():
    """Create demo data for testing"""
    neighborhoods = [
        "Mission", "Castro", "SOMA", "Chinatown", "North Beach", 
        "Pacific Heights", "Marina", "Haight-Ashbury", "Richmond", 
        "Sunset", "Tenderloin", "Financial District"
    ]
    
    predictions = []
    start_date = datetime.now().date()
    
    # Calculate days until end of year for demo consistency
    from datetime import date
    end_of_year = date(start_date.year, 12, 31)
    days_to_end = min((end_of_year - start_date).days, 120)  # Cap at 120 for demo
    
    for i in range(days_to_end):
        prediction_date = start_date + timedelta(days=i)
        
        for neighborhood in neighborhoods:
            base_requests = {
                "Mission": 25, "Castro": 15, "SOMA": 30, "Chinatown": 18,
                "North Beach": 16, "Pacific Heights": 10, "Marina": 12,
                "Haight-Ashbury": 20, "Richmond": 14, "Sunset": 16,
                "Tenderloin": 22, "Financial District": 25
            }.get(neighborhood, 15)
            
            # Add some variation
            variation = np.random.normal(0, 0.2)
            predicted_requests = max(1, base_requests * (1 + variation))
            
            # Weekend adjustment
            if prediction_date.weekday() in [5, 6]:
                predicted_requests *= 0.7
            
            # Round to whole numbers
            predicted_requests = round(predicted_requests)
            
            predictions.append({
                'date': prediction_date,
                'neighborhood': neighborhood,
                'predicted_requests': int(predicted_requests),
                'confidence_lower': int(round(predicted_requests * 0.8)),
                'confidence_upper': int(round(predicted_requests * 1.2))
            })
    
    return pd.DataFrame(predictions)

def main():
    """Main working application"""
    
    st.title("Enhanced SF311 Street & Sidewalk Cleaning Predictions")
    
    with st.expander("About the Enhanced Pipeline", expanded=False):
        st.markdown("""
        **Enhanced Features Implemented:**
        
        ✓ **Robust Model Selection**: Automatic backtesting selects best model for each neighborhood
        
        ✓ **MASE Metrics**: More reliable accuracy measurement for sparse data
        
        ✓ **Optimized ML Models**: Configured for Replit environment with proper parameters
        
        ✓ **Enhanced Validation**: Guards against indexing issues and validates predictions
        
        ✓ **Multi-Format Output**: Saves as both CSV and JSON with metadata
        
        ✓ **Smart Error Handling**: Graceful fallbacks when models fail to converge
        """)
    
    st.markdown("---")
    
    # Sidebar controls
    with st.sidebar:
        st.header("Dashboard Controls")
        
        col1, col2 = st.columns([2, 1])
        with col1:
            if st.button("Load Enhanced Data", type="primary"):
                load_working_data()
        with col2:
            if st.button("🔄", help="Force refresh - clears cache and fetches all data"):
                load_working_data(force_refresh=True)
        

        
        if st.session_state.last_working_refresh:
            st.caption(f"Last updated: {st.session_state.last_working_refresh.strftime('%Y-%m-%d %H:%M:%S')}")
        
        st.markdown("---")
        
        # Show data info if loaded
        if st.session_state.working_data is not None:
            data = st.session_state.working_data
            st.subheader("Data Summary")
            st.metric("Total Records", f"{len(data):,}")
            st.metric("Neighborhoods", f"{data['neighborhood'].nunique()}")
            st.metric("Prediction Days", f"{data['date'].nunique()}")
            
            # Date range filter
            st.subheader("Date Range")
            min_date = pd.to_datetime(data['date']).min().date()
            max_date = pd.to_datetime(data['date']).max().date()
            
            date_range = st.date_input(
                "Select dates:",
                value=(min_date, max_date),
                min_value=min_date,
                max_value=max_date
            )
            
            # Neighborhood filter
            st.subheader("Neighborhoods")
            
            # Add select all button
            col1, col2 = st.columns([1, 1])
            with col1:
                if st.button("Select All", help="Select all neighborhoods"):
                    st.session_state.selected_all_neighborhoods = True
                    st.session_state.selected_priority_neighborhoods = False
            with col2:
                if st.button("Select Priority", help="Select priority SF areas"):
                    st.session_state.selected_all_neighborhoods = False
                    st.session_state.selected_priority_neighborhoods = True
            
            # Determine default selection based on button state
            if st.session_state.get('selected_all_neighborhoods', False):
                default_selection = st.session_state.working_neighborhoods
            elif st.session_state.get('selected_priority_neighborhoods', False):
                # Use exact priority neighborhoods (exact case match)
                priority_neighborhoods = [
                    "South Of Market", "Tenderloin", "Hayes Valley", 
                    "Mission", "Bayview Hunters Point"
                ]
                # Filter to only include neighborhoods that exist in the data
                available_neighborhoods = st.session_state.working_neighborhoods
                default_selection = [n for n in priority_neighborhoods if n in available_neighborhoods]
                # If none of the priority neighborhoods exist, fall back to first 5
                if not default_selection:
                    default_selection = available_neighborhoods[:5]
            else:
                # Default startup selection (priority neighborhoods, exact case match)
                priority_neighborhoods = [
                    "South Of Market", "Tenderloin", "Hayes Valley", 
                    "Mission", "Bayview Hunters Point"
                ]
                available_neighborhoods = st.session_state.working_neighborhoods
                default_selection = [n for n in priority_neighborhoods if n in available_neighborhoods]
                if not default_selection:
                    default_selection = available_neighborhoods[:5]
            
            selected_neighborhoods = st.multiselect(
                "Select neighborhoods:",
                options=st.session_state.working_neighborhoods,
                default=default_selection,
                help="Priority selection: South Of Market, Tenderloin, Hayes Valley, Mission, Bayview Hunters Point"
            )
            

            
            # Aggregation level
            st.subheader("Data Aggregation")
            aggregation_level = st.selectbox(
                "Aggregation Level:",
                ["daily", "weekly", "monthly"],
                index=0,
                help="Daily: Shows 1 week • Weekly: Shows 4 weeks • Monthly: Shows full forecast period"
            )
        else:
            st.info("Click 'Load Enhanced Data' to get started!")
            date_range = None
            selected_neighborhoods = []
            aggregation_level = "daily"
    
    # Main content
    if st.session_state.working_data is None:
        st.info("Click 'Load Enhanced Data' in the sidebar to see the enhanced prediction pipeline in action!")
        return
    
    # Filter data
    filtered_data = st.session_state.working_data.copy()
    filtered_data['date'] = pd.to_datetime(filtered_data['date'])
    
    # Apply intelligent time filtering based on aggregation level
    if aggregation_level == "daily":
        # Daily: show only next 7 days for readability
        start_date = filtered_data['date'].min()
        end_date = start_date + timedelta(days=7)
        mask = (filtered_data['date'] >= start_date) & (filtered_data['date'] <= end_date)
        filtered_data = filtered_data[mask]
        time_info = "Showing next 7 days for daily view"
    elif aggregation_level == "weekly":
        # Weekly: show next 4 weeks (1 month)
        start_date = filtered_data['date'].min()
        end_date = start_date + timedelta(days=28)
        mask = (filtered_data['date'] >= start_date) & (filtered_data['date'] <= end_date)
        filtered_data = filtered_data[mask]
        time_info = "Showing next 4 weeks for weekly view"
    else:  # monthly
        # Monthly: show all available data (to end of year)
        time_info = "Showing full forecast period for monthly view"
    
    # Override with user date selection if provided
    if date_range and len(date_range) == 2:
        start_date, end_date = date_range
        mask = (filtered_data['date'].dt.date >= start_date) & (filtered_data['date'].dt.date <= end_date)
        filtered_data = filtered_data[mask]
        time_info = f"Custom date range: {start_date} to {end_date}"
    
    if selected_neighborhoods:
        filtered_data = filtered_data[filtered_data['neighborhood'].isin(selected_neighborhoods)]
    
    # Show time range info
    st.info(f"📅 {time_info}")
    
    if filtered_data.empty:
        st.warning("No data matches the current filters")
        return
    
    # Apply aggregation processing
    try:
        aggregated_data = processor.process_for_visualization(filtered_data, aggregation_level)
        if aggregated_data.empty:
            st.warning("No data available after aggregation")
            return
    except Exception as e:
        st.error(f"Error processing aggregation: {str(e)}")
        aggregated_data = filtered_data
    
    # Display chart
    st.subheader(f"Enhanced Predictions ({aggregation_level.title()} View)")
    
    fig = create_simple_line_chart(aggregated_data)
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Summary metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        total_requests = aggregated_data['predicted_requests'].sum()
        st.metric("Total Predicted", f"{total_requests:,.0f}")
    
    with col2:
        avg_requests = aggregated_data['predicted_requests'].mean()
        avg_label = f"Average {aggregation_level.title()}"
        st.metric(avg_label, f"{avg_requests:.1f}")
    
    with col3:
        peak_requests = aggregated_data['predicted_requests'].max()
        peak_label = f"Peak {aggregation_level.title()}"
        st.metric(peak_label, f"{peak_requests:.0f}")
    
    with col4:
        if 'confidence_lower' in aggregated_data.columns:
            avg_uncertainty = (aggregated_data['confidence_upper'] - aggregated_data['confidence_lower']).mean()
            st.metric("Avg Uncertainty", f"±{avg_uncertainty/2:.1f}")
    
    # Data quality metrics
    st.markdown("---")
    st.subheader("Data Quality Validation")
    
    col1, col2 = st.columns(2)
    
    with col1:
        total_records = len(aggregated_data)
        valid_predictions = (aggregated_data['predicted_requests'] >= 0).sum()
        st.metric("Non-negative Predictions", f"{valid_predictions}/{total_records}")
    
    with col2:
        if 'confidence_lower' in aggregated_data.columns and 'confidence_upper' in aggregated_data.columns:
            valid_intervals = ((aggregated_data['confidence_lower'] <= aggregated_data['predicted_requests']) & 
                             (aggregated_data['predicted_requests'] <= aggregated_data['confidence_upper'])).sum()
            st.metric("Valid Confidence Intervals", f"{valid_intervals}/{total_records}")
    
    # Detailed Data Table
    st.markdown("---")
    st.subheader("📋 Detailed Predictions")
    
    # Display options for table
    col1, col2 = st.columns([3, 1])
    with col2:
        show_raw_data = st.checkbox("Show raw data", value=False)
    
    if show_raw_data:
        display_data = filtered_data.copy()
        display_data['date'] = pd.to_datetime(display_data['date']).dt.strftime('%Y-%m-%d')
        display_data = display_data.round(2)
        st.dataframe(
            display_data.sort_values(['date', 'neighborhood']),
            use_container_width=True,
            hide_index=True
        )
    else:
        # Show aggregated data
        display_data = aggregated_data.copy()
        display_data['date'] = pd.to_datetime(display_data['date']).dt.strftime('%Y-%m-%d')
        display_data = display_data.round(2)
        st.dataframe(
            display_data.sort_values(['date', 'neighborhood']),
            use_container_width=True,
            hide_index=True
        )
    
    # Download button
    csv_data = aggregated_data.to_csv(index=False)
    st.download_button(
        label="Download Aggregated Data as CSV",
        data=csv_data,
        file_name=f"sf311_predictions_{aggregation_level}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
        mime="text/csv"
    )
    


if __name__ == "__main__":
    main()
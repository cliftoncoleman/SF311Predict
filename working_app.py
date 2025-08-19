import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import numpy as np
from fixed_pipeline import FixedSF311Pipeline
from utils.data_processor import DataProcessor
from components.geospatial_map import GeospatialMapComponent

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

def get_map_component():
    return GeospatialMapComponent()

# Create fresh instances each time to avoid caching issues
pipeline = get_working_pipeline()
processor = get_data_processor()
map_component = get_map_component()

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

def load_working_data():
    """Load data using the working pipeline with smart caching"""
    try:
        # Smart session-based caching for better performance
        cache_key = f"sf311_data_{datetime.now().strftime('%Y%m%d_%H')}"  # Hourly cache key
        
        # Check if we have recent cached data
        if (hasattr(st.session_state, cache_key) and 
            hasattr(st.session_state, 'last_working_refresh') and 
            st.session_state.last_working_refresh):
            
            time_since_last = datetime.now() - st.session_state.last_working_refresh
            if time_since_last < timedelta(hours=1):
                st.info("Using cached data for faster loading...")
                st.session_state.working_data = getattr(st.session_state, cache_key)
                st.session_state.working_neighborhoods = sorted(st.session_state.working_data['neighborhood'].unique())
                st.success(f"Data loaded from cache! Forecasts for {len(st.session_state.working_neighborhoods)} neighborhoods now available")
                return
        
        # Create a compact status container for live updates
        status_container = st.container()
        with status_container:
            status_box = st.empty()
            status_box.info("🔄 **Loading Status:** Initializing SF311 pipeline with 5-year training data...")
        
        with st.spinner("Processing data..."):
            fresh_pipeline = FixedSF311Pipeline()
            
            # Update status during processing
            status_box.info("🔄 **Loading Status:** Fetching 5 years of historical data from SF311 API...")
            
            # Start prediction process
            status_box.info("🔄 **Loading Status:** Training models for 42+ neighborhoods (this may take a few minuts upon first run)...")
            
            # Add a small expandable section for technical details
            with st.expander("📊 See Technical Details", expanded=False):
                tech_info = st.empty()
                tech_info.text("Model training in progress:\n• Using 1825 days (5 years) of historical data\n• Selecting best model per neighborhood (trend/exponential/seasonal)\n• Generating predictions through end of year")
            
            # Use 5 years of training data for robust modeling
            predictions = fresh_pipeline.run_full_fixed_pipeline(
                days_back=1825,  # 5 years for robust training  
                prediction_days=None  # Full year forecast (restored)
            )
            
            # Update status when processing is complete
            status_box.success("✅ **Loading Complete:** Data processed successfully!")
            
            if not predictions.empty:
                st.session_state.working_data = predictions
                
                # Cache the data with hourly key
                cache_key = f"sf311_data_{datetime.now().strftime('%Y%m%d_%H')}"
                setattr(st.session_state, cache_key, predictions)
                
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
                
                # Update final status with results
                status_box.success(f"✅ **Complete:** Loaded {len(predictions)} predictions from {len(available_neighborhoods)} neighborhoods!")
                
                # Save predictions
                try:
                    saved_files = pipeline.save_predictions_enhanced(predictions, "output")

                    with st.expander("Files saved"):
                        st.write(f"CSV: {saved_files['csv_path']}")
                        st.write(f"JSON: {saved_files['json_path']}")
                except Exception as e:
                    st.success(f"Data loaded! {len(predictions)} records from {len(st.session_state.working_neighborhoods)} neighborhoods")
                    st.info(f"Note: File saving encountered an issue: {str(e)}")
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
    
    st.title("SF311 Street & Sidewalk Cleaning Predictions")
    
    st.markdown("---")
    
    # Sidebar controls
    with st.sidebar:
        st.header("App Controls")
        
        if st.button("Load App", type="primary"):
            load_working_data()
        
        # Add cache management controls
        if st.button("Force Refresh (Clear Cache)"):
            st.cache_data.clear()
            st.cache_resource.clear()
            st.session_state.last_working_refresh = None
            st.info("Cache cleared! Click 'Load App' for fresh data.")
        
        

        
        if st.session_state.last_working_refresh:
            st.caption(f"Last updated: {st.session_state.last_working_refresh.strftime('%Y-%m-%d %H:%M:%S')}")
        
        st.markdown("---")
        
        # Show data info if loaded
        if st.session_state.working_data is not None:
            data = st.session_state.working_data
            
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
            st.info("Click 'Load App to get started!")
            date_range = None
            selected_neighborhoods = []
            aggregation_level = "daily"
    
    # Main content
    if st.session_state.working_data is None:
        st.info("Click 'Load App' in the sidebar to see the prediction pipeline in action!")
        return
    
    # Start from full data each time
    base_data = st.session_state.working_data.copy()
    base_data['date'] = pd.to_datetime(base_data['date'])

    # Figure out if user changed the default date range
    data_min = base_data['date'].min().date()
    data_max = base_data['date'].max().date()

    custom_range = (
        date_range and len(date_range) == 2 and
        not (data_min == date_range[0] and data_max == date_range[1])
    )

    # Helper: window length by aggregation level
    def _window_days_for(level: str) -> int | None:
        if level == "daily":
            return 7
        if level == "weekly":
            return 28
        return None  # monthly = full span

    # 1) Start with either full data or the user's custom span
    if custom_range:
        user_start, user_end = date_range
        # keep only the user-selected span first
        filtered_data = base_data[
            (base_data['date'].dt.date >= user_start) &
            (base_data['date'].dt.date <= user_end)
        ]
        # 2) Inside the user's span, still apply the auto window for daily/weekly
        win_days = _window_days_for(aggregation_level)
        if win_days is not None and not filtered_data.empty:
            start_date = filtered_data['date'].min().normalize()
            end_date = min(start_date + timedelta(days=win_days), pd.to_datetime(user_end) + timedelta(days=1))
            # Note: end_date is exclusive when using >= and < to avoid off-by-one with normalize()
            filtered_data = filtered_data[(filtered_data['date'] >= start_date) & (filtered_data['date'] < end_date)]
            time_info = f"{aggregation_level.title()} view window inside custom range: {start_date.date()} → {(end_date - timedelta(days=1)).date()}"
        else:
            time_info = f"Custom date range: {user_start} → {user_end}"
    else:
        # No custom range: use the default auto windows
        win_days = _window_days_for(aggregation_level)
        if win_days is not None:
            start_date = base_data['date'].min().normalize()
            end_date = start_date + timedelta(days=win_days)
            filtered_data = base_data[(base_data['date'] >= start_date) & (base_data['date'] < end_date)]
            time_info = "Showing next 7 days (Daily)" if aggregation_level == "daily" else "Showing next 4 weeks (Weekly)"
        else:
            filtered_data = base_data.copy()
            time_info = "Showing full forecast period (Monthly)"

    # Neighborhood filter AFTER time windowing
    if selected_neighborhoods:
        filtered_data = filtered_data[filtered_data['neighborhood'].isin(selected_neighborhoods)]

    st.markdown(
        f"""
        <div style="background:#f0f2f6; padding:10px 14px; border-radius:8px;">
          <div>📅 {time_info}</div>
         <p> <div style="font-size:0.9em; color:#6b7280;">
            Use <b>Data Aggregation</b> and <b>Date Range</b> controls to adjust the view.<br>
            <i>Note:</i> 'Monthly' view shows the full forecast period (up to end of year).
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )



    
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
    
    # Summary metrics

    st.subheader(f"Summary Metrics")
    st.caption("for above date range and neighborhoods")
    
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

    
    
    # Display chart
    st.markdown("---")
    st.subheader(f"Enhanced Predictions ({aggregation_level.title()} View)")
    
    fig = create_simple_line_chart(aggregated_data)
    
    st.plotly_chart(fig, use_container_width=True)

    # Detailed Data Table
    st.markdown("---")
    st.subheader(f"📋 Detailed Predictions ({aggregation_level.title()} View)")

    # Display options for table
    col1, col2 = st.columns([3, 1])
    with col1:
        show_raw_data = st.checkbox("Display daily data while in weekly or monthly view", value=False)

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
        label="Download Data as CSV",
        data=csv_data,
        file_name=f"sf311_predictions_{aggregation_level}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
        mime="text/csv"
    )
    
    # Geospatial Heatmap — independent of sidebar filters
    st.markdown("---")
    with st.spinner("Rendering geospatial heatmap..."):
        try:
            df_for_map = st.session_state.working_data.copy()
            map_component.render_map_component(
                df_for_map,
                title=f"Heat Map — All Neighborhoods (Daily Only)",
                key="sf311_main_map",  # namespaced state so it runs independently
            )
        except Exception as e:
            st.error("Map failed to render.")
            with st.expander("Details", expanded=False):
                st.write(str(e))



    
    
   



if __name__ == "__main__":
    main()
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import os
import numpy as np
from improved_data_pipeline import EnhancedSF311Pipeline
from components.enhanced_charts import EnhancedChartGenerator
from components.filters import FilterComponent

# Page configuration
st.set_page_config(
    page_title="Enhanced SF311 Street & Sidewalk Cleaning Predictions",
    page_icon="🧹",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state
if 'enhanced_data' not in st.session_state:
    st.session_state.enhanced_data = None
if 'neighborhoods' not in st.session_state:
    st.session_state.neighborhoods = []
if 'last_enhanced_refresh' not in st.session_state:
    st.session_state.last_enhanced_refresh = None

# Initialize enhanced components
@st.cache_resource
def get_pipeline():
    return EnhancedSF311Pipeline()

@st.cache_resource  
def get_chart_generator():
    return EnhancedChartGenerator()

pipeline = get_pipeline()
chart_generator = get_chart_generator()
filter_component = FilterComponent()

def load_enhanced_data():
    """Load data using the enhanced pipeline with all improvements"""
    try:
        with st.spinner("Loading enhanced SF311 data with improved models..."):
            # Run the enhanced pipeline
            predictions = pipeline.run_full_enhanced_pipeline(
                days_back=1095,  # 3 years of history
                prediction_days=30
            )
            
            if not predictions.empty:
                st.session_state.enhanced_data = predictions
                st.session_state.neighborhoods = sorted(predictions['neighborhood'].unique())
                st.session_state.last_enhanced_refresh = datetime.now()
                
                # Save predictions in multiple formats as suggested
                saved_files = pipeline.save_predictions_enhanced(predictions, "predictions_output")
                
                st.success(f"Enhanced predictions loaded successfully! {len(predictions)} records from {len(st.session_state.neighborhoods)} neighborhoods")
                st.info(f"Predictions saved to: {saved_files['csv_path']} and {saved_files['json_path']}")
            else:
                st.error("No data could be loaded with the enhanced pipeline")
    except Exception as e:
        st.error(f"Enhanced pipeline error: {str(e)}")

def main():
    """Main enhanced application function"""
    
    # Header
    st.title("🧹 Enhanced SF311 Street & Sidewalk Cleaning Predictions Dashboard")
    
    # Add information about improvements
    with st.expander("🚀 What's New - Enhanced Features", expanded=False):
        st.markdown("""
        **Enhanced Prediction Pipeline with Industry Best Practices:**
        
        ✅ **Robust Model Selection**: Automatic backtesting chooses the best model (Seasonal Naive, ML, or SARIMAX) for each neighborhood
        
        ✅ **MASE Metrics**: More reliable accuracy measurement using Mean Absolute Scaled Error, especially for sparse data
        
        ✅ **Optimized ML Models**: HistGradientBoostingRegressor with early stopping and proper parameters for Replit
        
        ✅ **Enhanced Validation**: Guards against mis-ordered indexing and validates prediction schemas
        
        ✅ **Multi-Format Output**: Saves predictions as both CSV and compact JSON with metadata
        
        ✅ **Smart Feature Engineering**: Consistent exogenous variables and improved seasonal patterns
        
        ✅ **Error Handling**: Comprehensive validation ensures non-negative predictions and proper confidence intervals
        """)
    
    st.markdown("---")
    
    # Sidebar for controls
    with st.sidebar:
        st.header("Enhanced Dashboard Controls")
        
        # Load enhanced data button
        if st.button("🔄 Load Enhanced Data", type="primary"):
            load_enhanced_data()
        
        # Display last refresh time
        if st.session_state.last_enhanced_refresh:
            st.caption(f"Last enhanced update: {st.session_state.last_enhanced_refresh.strftime('%Y-%m-%d %H:%M:%S')}")
        
        st.markdown("---")
        
        # Model performance information
        if st.session_state.enhanced_data is not None:
            st.subheader("📊 Pipeline Statistics")
            
            data = st.session_state.enhanced_data
            total_predictions = len(data)
            unique_neighborhoods = data['neighborhood'].nunique()
            prediction_days = data['date'].nunique()
            
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Total Records", f"{total_predictions:,}")
                st.metric("Neighborhoods", unique_neighborhoods)
            with col2:
                st.metric("Prediction Days", prediction_days)
                st.metric("Avg Daily/Nbhd", f"{total_predictions/max(prediction_days*unique_neighborhoods, 1):.1f}")
        
        st.markdown("---")
        
        # Date range selection
        st.subheader("📅 Date Range")
        if st.session_state.enhanced_data is not None:
            data = st.session_state.enhanced_data
            min_date = pd.to_datetime(data['date']).min().date()
            max_date = pd.to_datetime(data['date']).max().date()
            
            date_range = st.date_input(
                "Select date range for analysis:",
                value=(min_date, max_date),
                min_value=min_date,
                max_value=max_date
            )
        else:
            st.info("Load data to select date ranges")
            date_range = None
        
        # Neighborhood selection
        st.subheader("🏘️ Neighborhoods")
        if st.session_state.neighborhoods:
            selected_neighborhoods = st.multiselect(
                "Select neighborhoods:",
                options=st.session_state.neighborhoods,
                default=st.session_state.neighborhoods[:10],  # Show top 10 by default for performance
                help="Choose specific neighborhoods to analyze"
            )
        else:
            st.info("Load data to select neighborhoods")
            selected_neighborhoods = []
        
        # Chart options
        st.subheader("📈 Chart Options")
        show_confidence = st.checkbox("Show Confidence Intervals", value=True)
        chart_type = st.selectbox(
            "Primary Chart Type:",
            ["Line Chart", "Bar Chart", "Heatmap", "Metrics Summary"]
        )
    
    # Main content area
    if st.session_state.enhanced_data is None:
        st.info("👆 Click 'Load Enhanced Data' in the sidebar to get started with the improved prediction pipeline!")
        
        # Show demo of what the enhanced pipeline offers
        st.subheader("🎯 Enhanced Pipeline Features")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown("**🤖 Smart Model Selection**")
            st.write("Automatically selects the best forecasting model for each neighborhood")
        with col2:
            st.markdown("**📏 MASE Metrics**")
            st.write("More reliable accuracy measurement than MAPE for sparse data")
        with col3:
            st.markdown("**⚡ Optimized Performance**") 
            st.write("Efficient models designed for Replit environment")
        
        return
    
    # Filter data if selections made
    filtered_data = st.session_state.enhanced_data.copy()
    
    if date_range and len(date_range) == 2:
        start_date, end_date = date_range
        filtered_data['date'] = pd.to_datetime(filtered_data['date'])
        mask = (filtered_data['date'].dt.date >= start_date) & (filtered_data['date'].dt.date <= end_date)
        filtered_data = filtered_data[mask]
    
    if selected_neighborhoods:
        filtered_data = filtered_data[filtered_data['neighborhood'].isin(selected_neighborhoods)]
    
    if filtered_data.empty:
        st.warning("No data matches the current filters. Please adjust your selections.")
        return
    
    # Display charts based on selection
    st.subheader(f"📊 {chart_type} - Enhanced Predictions")
    
    if chart_type == "Line Chart":
        fig = chart_generator.create_line_chart(filtered_data, show_confidence=show_confidence)
        st.plotly_chart(fig, use_container_width=True)
    elif chart_type == "Bar Chart":
        fig = chart_generator.create_bar_chart(filtered_data)
        st.plotly_chart(fig, use_container_width=True)
    elif chart_type == "Heatmap":
        fig = chart_generator.create_heatmap(filtered_data)
        st.plotly_chart(fig, use_container_width=True)
    elif chart_type == "Metrics Summary":
        fig = chart_generator.create_metrics_summary(filtered_data)
        st.plotly_chart(fig, use_container_width=True)
    
    # Show neighborhood comparison if multiple selected
    if len(selected_neighborhoods) > 1 and len(selected_neighborhoods) <= 5:
        st.subheader("🏘️ Neighborhood Comparison")
        comparison_fig = chart_generator.create_neighborhood_comparison(
            filtered_data, 
            selected_neighborhoods
        )
        st.plotly_chart(comparison_fig, use_container_width=True)
    
    # Data summary and validation info
    st.markdown("---")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📈 Prediction Summary")
        
        total_requests = filtered_data['predicted_requests'].sum()
        avg_requests = filtered_data['predicted_requests'].mean()
        peak_requests = filtered_data['predicted_requests'].max()
        
        st.metric("Total Predicted Requests", f"{total_requests:,.0f}")
        st.metric("Average Request Rate", f"{avg_requests:.1f}")
        st.metric("Peak Daily Rate", f"{peak_requests:.0f}")
        
        # Confidence interval statistics
        if 'confidence_lower' in filtered_data.columns:
            avg_uncertainty = (filtered_data['confidence_upper'] - filtered_data['confidence_lower']).mean()
            st.metric("Avg Uncertainty Range", f"±{avg_uncertainty/2:.1f}")
    
    with col2:
        st.subheader("✅ Data Quality Metrics")
        
        # Validation checks as suggested in improvements
        total_records = len(filtered_data)
        valid_predictions = (filtered_data['predicted_requests'] >= 0).sum()
        valid_confidence = ((filtered_data['confidence_lower'] <= filtered_data['predicted_requests']) & 
                          (filtered_data['predicted_requests'] <= filtered_data['confidence_upper'])).sum()
        no_nulls = filtered_data[['predicted_requests', 'confidence_lower', 'confidence_upper']].notna().all(axis=1).sum()
        
        st.metric("Valid Records", f"{total_records:,}")
        st.metric("Non-negative Predictions", f"{valid_predictions}/{total_records}")
        st.metric("Valid Confidence Intervals", f"{valid_confidence}/{total_records}")
        st.metric("Complete Data", f"{no_nulls}/{total_records}")
    
    # Show detailed data table
    with st.expander("📋 Detailed Predictions Data", expanded=False):
        # Format data for display
        display_data = filtered_data.copy()
        display_data['date'] = pd.to_datetime(display_data['date']).dt.strftime('%Y-%m-%d')
        display_data = display_data.round(2)
        
        st.dataframe(
            display_data,
            use_container_width=True,
            column_config={
                "predicted_requests": st.column_config.NumberColumn("Predicted Requests", format="%.0f"),
                "confidence_lower": st.column_config.NumberColumn("Lower Bound", format="%.1f"),
                "confidence_upper": st.column_config.NumberColumn("Upper Bound", format="%.1f")
            }
        )
        
        # Download button for enhanced data
        csv_data = filtered_data.to_csv(index=False)
        st.download_button(
            label="📥 Download Filtered Data as CSV",
            data=csv_data,
            file_name=f"enhanced_sf311_predictions_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv"
        )

if __name__ == "__main__":
    main()
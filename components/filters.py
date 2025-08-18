import streamlit as st
import pandas as pd
from datetime import datetime, timedelta
from typing import List, Tuple, Optional

class FilterComponent:
    """Component for handling dashboard filters and user inputs"""
    
    def __init__(self):
        pass
    
    def render_date_filter(self, 
                          default_days: int = 30,
                          max_days: int = 365,
                          key_suffix: str = "") -> Tuple[datetime, datetime]:
        """Render date range filter"""
        
        col1, col2 = st.columns(2)
        
        with col1:
            start_date = st.date_input(
                "Start Date",
                value=datetime.now().date(),
                min_value=datetime.now().date(),
                max_value=datetime.now().date() + timedelta(days=max_days),
                key=f"start_date_{key_suffix}"
            )
        
        with col2:
            end_date = st.date_input(
                "End Date",
                value=datetime.now().date() + timedelta(days=default_days),
                min_value=datetime.now().date(),
                max_value=datetime.now().date() + timedelta(days=max_days),
                key=f"end_date_{key_suffix}"
            )
        
        # Validate date range
        if start_date > end_date:
            st.error("Start date must be before end date")
            return start_date, start_date
        
        return start_date, end_date
    
    def render_neighborhood_filter(self, 
                                 neighborhoods: List[str],
                                 default_count: int = 5,
                                 key_suffix: str = "") -> List[str]:
        """Render neighborhood selection filter"""
        
        if not neighborhoods:
            st.info("No neighborhoods available")
            return []
        
        # Selection method
        selection_method = st.radio(
            "Selection method:",
            ["Select specific neighborhoods", "Select all", "Select top N"],
            key=f"selection_method_{key_suffix}"
        )
        
        if selection_method == "Select all":
            return neighborhoods
        elif selection_method == "Select top N":
            n = st.slider(
                "Number of top neighborhoods:",
                min_value=1,
                max_value=min(20, len(neighborhoods)),
                value=min(default_count, len(neighborhoods)),
                key=f"top_n_{key_suffix}"
            )
            return neighborhoods[:n]
        else:  # Select specific
            return st.multiselect(
                "Choose neighborhoods:",
                options=neighborhoods,
                default=neighborhoods[:min(default_count, len(neighborhoods))],
                key=f"specific_neighborhoods_{key_suffix}"
            )
    
    def render_aggregation_filter(self, key_suffix: str = "") -> str:
        """Render aggregation level filter"""
        return st.selectbox(
            "Time aggregation:",
            ["Daily", "Weekly", "Monthly"],
            index=0,
            key=f"aggregation_{key_suffix}"
        )
    
    def render_chart_type_filter(self, key_suffix: str = "") -> str:
        """Render chart type selection filter"""
        return st.selectbox(
            "Chart type:",
            ["Line Chart", "Bar Chart", "Heatmap", "Area Chart"],
            index=0,
            key=f"chart_type_{key_suffix}"
        )
    
    def render_confidence_filter(self, key_suffix: str = "") -> Tuple[bool, float]:
        """Render confidence interval filters"""
        show_confidence = st.checkbox(
            "Show confidence intervals",
            value=True,
            key=f"show_confidence_{key_suffix}"
        )
        
        min_confidence = st.slider(
            "Minimum confidence threshold:",
            min_value=0.1,
            max_value=1.0,
            value=0.8,
            step=0.1,
            key=f"min_confidence_{key_suffix}"
        )
        
        return show_confidence, min_confidence
    
    def render_display_options(self, key_suffix: str = "") -> dict:
        """Render various display options"""
        options = {}
        
        st.subheader("Display Options")
        
        options['show_trends'] = st.checkbox(
            "Show trend analysis",
            value=True,
            key=f"show_trends_{key_suffix}"
        )
        
        options['show_anomalies'] = st.checkbox(
            "Highlight anomalies",
            value=False,
            key=f"show_anomalies_{key_suffix}"
        )
        
        options['normalize_data'] = st.checkbox(
            "Normalize data",
            value=False,
            key=f"normalize_data_{key_suffix}"
        )
        
        options['log_scale'] = st.checkbox(
            "Use logarithmic scale",
            value=False,
            key=f"log_scale_{key_suffix}"
        )
        
        return options
    
    def render_export_options(self, data: pd.DataFrame, key_suffix: str = "") -> None:
        """Render data export options"""
        if data.empty:
            st.info("No data to export")
            return
        
        st.subheader("Export Data")
        
        col1, col2 = st.columns(2)
        
        with col1:
            export_format = st.selectbox(
                "Format:",
                ["CSV", "JSON", "Excel"],
                key=f"export_format_{key_suffix}"
            )
        
        with col2:
            include_metadata = st.checkbox(
                "Include metadata",
                value=True,
                key=f"include_metadata_{key_suffix}"
            )
        
        # Generate filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        if export_format == "CSV":
            filename = f"sf311_predictions_{timestamp}.csv"
            data_to_export = data.to_csv(index=False)
            mime_type = "text/csv"
        elif export_format == "JSON":
            filename = f"sf311_predictions_{timestamp}.json"
            data_to_export = data.to_json(orient='records', date_format='iso')
            mime_type = "application/json"
        else:  # Excel
            filename = f"sf311_predictions_{timestamp}.xlsx"
            # For Excel, we'll convert to CSV as a fallback
            data_to_export = data.to_csv(index=False)
            mime_type = "text/csv"
        
        st.download_button(
            label=f"📥 Download as {export_format}",
            data=data_to_export,
            file_name=filename,
            mime=mime_type,
            key=f"download_button_{key_suffix}"
        )
    
    def validate_filters(self, 
                        start_date: datetime, 
                        end_date: datetime, 
                        neighborhoods: List[str]) -> Tuple[bool, str]:
        """Validate filter selections"""
        
        # Check date range
        if start_date >= end_date:
            return False, "End date must be after start date"
        
        # Check if date range is too large
        if (end_date - start_date).days > 365:
            return False, "Date range cannot exceed 365 days"
        
        # Check neighborhoods
        if not neighborhoods:
            return False, "At least one neighborhood must be selected"
        
        return True, "Filters are valid"
    
    def get_filter_summary(self, 
                          start_date: datetime, 
                          end_date: datetime, 
                          neighborhoods: List[str],
                          aggregation: str) -> str:
        """Generate a summary of current filter settings"""
        
        date_range = f"{start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}"
        days_count = (end_date - start_date).days
        neighborhood_count = len(neighborhoods)
        
        summary = f"""
        **Current Filters:**
        - Date Range: {date_range} ({days_count} days)
        - Neighborhoods: {neighborhood_count} selected
        - Aggregation: {aggregation}
        """
        
        if neighborhood_count <= 5:
            summary += f"\n- Selected Areas: {', '.join(neighborhoods)}"
        
        return summary

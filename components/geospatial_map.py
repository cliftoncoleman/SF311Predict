import streamlit as st
import pandas as pd
import folium
import requests
import json
from streamlit_folium import st_folium
import numpy as np
from typing import Dict, Any

class GeospatialMapComponent:
    """Component for creating geospatial heatmaps of SF311 predictions"""
    
    def __init__(self):
        self.sf_center = [37.7749, -122.4194]
        self.neighborhoods_geojson_url = "https://data.sfgov.org/resource/pty2-tcw4.geojson"
        self.neighborhood_mapping = self._create_neighborhood_mapping()
    
    def _create_neighborhood_mapping(self) -> Dict[str, str]:
        """Create mapping between SF311 neighborhood names and official boundary names"""
        return {
            # Common variations and mappings
            "South Of Market": "South of Market",
            "SOMA": "South of Market", 
            "Bayview Hunters Point": "Bayview",
            "Hunters Point": "Bayview",
            "West Of Twin Peaks": "West of Twin Peaks",
            "Visitacion Valley": "Visitacion Valley",
            "Western Addition": "Western Addition",
            "Castro/Upper Market": "Castro",
            "Bernal Heights": "Bernal Heights",
            "Glen Park": "Glen Park",
            "Noe Valley": "Noe Valley",
            "Mission": "Mission",
            "Potrero Hill": "Potrero Hill",
            "Dogpatch": "Potrero Hill",
            "Central Waterfront": "Potrero Hill",
            "Hayes Valley": "Hayes Valley",
            "Tenderloin": "Tenderloin",
            "Chinatown": "Chinatown", 
            "North Beach": "North Beach",
            "Russian Hill": "Russian Hill",
            "Nob Hill": "Nob Hill",
            "Financial District": "Financial District/South Beach",
            "Inner Richmond": "Richmond",
            "Outer Richmond": "Richmond", 
            "Richmond": "Richmond",
            "Inner Sunset": "Sunset/Parkside",
            "Outer Sunset": "Sunset/Parkside",
            "Sunset": "Sunset/Parkside",
            "Parkside": "Sunset/Parkside",
            "Haight Ashbury": "Haight Ashbury",
            "Fillmore": "Western Addition",
            "Pacific Heights": "Pacific Heights",
            "Marina": "Marina",
            "Presidio Heights": "Presidio Heights",
            "Laurel Heights": "Presidio Heights",
            "Japantown": "Japantown",
            "Twin Peaks": "Twin Peaks",
            "Diamond Heights": "Diamond Heights",
            "Excelsior": "Excelsior",
            "Outer Mission": "Outer Mission",
            "Ingleside": "Ingleside",
            "Oceanview": "Oceanview/Merced/Ingleside",
            "Merced": "Oceanview/Merced/Ingleside",
            "Lakeshore": "Lakeshore",
            "Presidio": "Presidio",
            "Golden Gate Park": "Golden Gate Park",
            "Seacliff": "Seacliff",
            "Lake Street": "Richmond",
            "Lone Mountain": "Richmond",
            "Treasure Island": "Treasure Island"
        }
    
    @st.cache_data
    def _fetch_neighborhoods_geojson(_self):
        """Fetch SF neighborhood boundaries from official data source"""
        try:
            response = requests.get(_self.neighborhoods_geojson_url, timeout=30)
            response.raise_for_status()
            return response.json()
        except requests.RequestException as e:
            st.warning(f"Could not fetch neighborhood boundaries: {e}")
            return None
    
    def _aggregate_predictions_by_neighborhood(self, data: pd.DataFrame) -> Dict[str, float]:
        """Aggregate prediction data by neighborhood for the map"""
        if data.empty:
            return {}
        
        # Group by neighborhood and sum predictions
        neighborhood_totals = data.groupby('neighborhood')['predicted_requests'].sum()
        
        # Map neighborhood names to official boundary names
        aggregated = {}
        for neighborhood, total in neighborhood_totals.items():
            official_name = self.neighborhood_mapping.get(neighborhood, neighborhood)
            if official_name in aggregated:
                aggregated[official_name] += total
            else:
                aggregated[official_name] = total
        
        return aggregated
    
    def _get_color_scale(self, value: float, min_val: float, max_val: float) -> str:
        """Generate color based on value using a heat scale"""
        if max_val == min_val:
            return '#3498db'  # Default blue if no variation
        
        # Normalize value between 0 and 1
        normalized = (value - min_val) / (max_val - min_val)
        
        # Create color gradient from blue (low) to red (high)
        if normalized <= 0.25:
            # Blue to cyan
            r, g, b = int(52 + (0 - 52) * (normalized / 0.25)), int(152 + (255 - 152) * (normalized / 0.25)), 219
        elif normalized <= 0.5:
            # Cyan to green
            r, g, b = 0, 255, int(219 + (0 - 219) * ((normalized - 0.25) / 0.25))
        elif normalized <= 0.75:
            # Green to yellow
            r, g, b = int(0 + (255 - 0) * ((normalized - 0.5) / 0.25)), 255, 0
        else:
            # Yellow to red
            r, g, b = 255, int(255 + (0 - 255) * ((normalized - 0.75) / 0.25)), 0
        
        return f'#{r:02x}{g:02x}{b:02x}'
    
    def create_geospatial_heatmap(self, data: pd.DataFrame, title: str = "SF311 Predictions Heatmap") -> folium.Map:
        """Create a geospatial heatmap of SF311 predictions"""
        
        # Create base map centered on SF
        m = folium.Map(
            location=self.sf_center,
            zoom_start=12,
            tiles='OpenStreetMap'
        )
        
        # Fetch neighborhood boundaries
        neighborhoods_geojson = self._fetch_neighborhoods_geojson()
        if not neighborhoods_geojson:
            # Add error message to map
            folium.Marker(
                self.sf_center,
                popup="Could not load neighborhood boundaries",
                icon=folium.Icon(color='red')
            ).add_to(m)
            return m
        
        # Aggregate prediction data by neighborhood
        neighborhood_totals = self._aggregate_predictions_by_neighborhood(data)
        
        if not neighborhood_totals:
            # Add message for no data
            folium.Marker(
                self.sf_center,
                popup="No prediction data available",
                icon=folium.Icon(color='orange')
            ).add_to(m)
            return m
        
        # Get min/max values for color scaling
        values = list(neighborhood_totals.values())
        min_val, max_val = min(values), max(values)
        
        # Add neighborhoods to map with color coding
        for feature in neighborhoods_geojson['features']:
            neighborhood_name = feature['properties'].get('nhood', 'Unknown')
            
            # Get prediction total for this neighborhood
            total_predictions = neighborhood_totals.get(neighborhood_name, 0)
            
            # Determine color based on prediction volume
            if total_predictions > 0:
                color = self._get_color_scale(total_predictions, min_val, max_val)
                opacity = 0.7
            else:
                color = '#cccccc'  # Gray for no data
                opacity = 0.3
            
            # Create popup content
            popup_content = f"""
            <b>{neighborhood_name}</b><br>
            Predicted Requests: {total_predictions:,.0f}<br>
            Rank: {sorted(values, reverse=True).index(total_predictions) + 1 if total_predictions > 0 else 'N/A'} of {len(values)}
            """
            
            # Add neighborhood polygon to map
            folium.GeoJson(
                feature,
                style_function=lambda x, color=color, opacity=opacity: {
                    'fillColor': color,
                    'color': 'black',
                    'weight': 1,
                    'fillOpacity': opacity
                },
                popup=folium.Popup(popup_content, max_width=300),
                tooltip=f"{neighborhood_name}: {total_predictions:,.0f} requests"
            ).add_to(m)
        
        # Add legend
        legend_html = f'''
        <div style="position: fixed; 
                    bottom: 50px; left: 50px; width: 200px; height: 120px; 
                    background-color: white; border:2px solid grey; z-index:9999; 
                    font-size:14px; padding: 10px">
        <p><b>{title}</b></p>
        <p><i class="fa fa-square" style="color:#ff0000"></i> High: {max_val:,.0f}</p>
        <p><i class="fa fa-square" style="color:#ffff00"></i> Medium: {(max_val + min_val) / 2:,.0f}</p>
        <p><i class="fa fa-square" style="color:#0000ff"></i> Low: {min_val:,.0f}</p>
        <p><i class="fa fa-square" style="color:#cccccc"></i> No Data</p>
        </div>
        '''
        m.get_root().html.add_child(folium.Element(legend_html))
        
        return m
    
    def render_map_component(self, data: pd.DataFrame, title: str = "SF311 Predictions Heatmap"):
        """Render the geospatial map component in Streamlit"""
        
        st.subheader("🗺️ Geospatial Heatmap - All Neighborhoods")
        
        if data.empty:
            st.warning("No data available for mapping")
            return
        
        # Show data summary
        total_predictions = data['predicted_requests'].sum()
        total_neighborhoods = data['neighborhood'].nunique()
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Predictions", f"{total_predictions:,.0f}")
        with col2:
            st.metric("Neighborhoods", total_neighborhoods)
        with col3:
            st.metric("Avg per Neighborhood", f"{total_predictions / total_neighborhoods:,.0f}")
        
        # Create and display map
        with st.spinner("Loading neighborhood boundaries and creating heatmap..."):
            heatmap = self.create_geospatial_heatmap(data, title)
            
            # Display map using streamlit-folium
            map_data = st_folium(
                heatmap,
                width=700,
                height=500,
                returned_objects=["last_object_clicked"]
            )
            
            # Show clicked neighborhood info
            if map_data['last_object_clicked']:
                clicked_data = map_data['last_object_clicked']
                if 'tooltip' in clicked_data:
                    st.info(f"Selected: {clicked_data['tooltip']}")
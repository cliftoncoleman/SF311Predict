import streamlit as st
import pandas as pd
import folium
from folium.plugins import HeatMap, HeatMapWithTime
import requests
import json
from streamlit_folium import st_folium
import numpy as np
from typing import Dict, Any, List, Tuple

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
    
    def _get_neighborhood_centers(self) -> Dict[str, Tuple[float, float]]:
        """Get approximate center coordinates for SF neighborhoods"""
        # These are approximate centers for major SF neighborhoods
        return {
            "Mission": (37.7599, -122.4148),
            "Castro": (37.7609, -122.4350),
            "South Of Market": (37.7749, -122.4104),
            "SOMA": (37.7749, -122.4104),
            "Tenderloin": (37.7836, -122.4130),
            "Hayes Valley": (37.7749, -122.4252),
            "Chinatown": (37.7941, -122.4078),
            "North Beach": (37.8067, -122.4103),
            "Russian Hill": (37.8018, -122.4200),
            "Nob Hill": (37.7946, -122.4094),
            "Financial District": (37.7946, -122.3999),
            "Richmond": (37.7806, -122.4644),
            "Inner Richmond": (37.7806, -122.4644),
            "Outer Richmond": (37.7781, -122.4944),
            "Sunset": (37.7431, -122.4644),
            "Inner Sunset": (37.7431, -122.4644),
            "Outer Sunset": (37.7431, -122.4944),
            "Parkside": (37.7331, -122.4844),
            "Haight Ashbury": (37.7694, -122.4481),
            "Western Addition": (37.7819, -122.4378),
            "Fillmore": (37.7819, -122.4378),
            "Pacific Heights": (37.7919, -122.4419),
            "Marina": (37.8019, -122.4419),
            "Presidio Heights": (37.7919, -122.4519),
            "Laurel Heights": (37.7869, -122.4519),
            "Japantown": (37.7850, -122.4300),
            "Twin Peaks": (37.7544, -122.4477),
            "Glen Park": (37.7331, -122.4331),
            "Bernal Heights": (37.7431, -122.4131),
            "Potrero Hill": (37.7631, -122.3981),
            "Dogpatch": (37.7531, -122.3881),
            "Bayview": (37.7331, -122.3881),
            "Bayview Hunters Point": (37.7331, -122.3881),
            "Hunters Point": (37.7231, -122.3781),
            "Visitacion Valley": (37.7131, -122.4031),
            "Excelsior": (37.7231, -122.4331),
            "Outer Mission": (37.7181, -122.4481),
            "Ingleside": (37.7181, -122.4631),
            "Oceanview": (37.7131, -122.4731),
            "Merced": (37.7081, -122.4831),
            "Lakeshore": (37.7181, -122.4881),
            "West Of Twin Peaks": (37.7394, -122.4677),
            "Diamond Heights": (37.7444, -122.4377),
            "Noe Valley": (37.7531, -122.4331),
            "Presidio": (37.8019, -122.4719),
            "Seacliff": (37.7869, -122.4919),
            "Lake Street": (37.7819, -122.4719),
            "Lone Mountain": (37.7819, -122.4619),
            "Golden Gate Park": (37.7694, -122.4844),
            "Treasure Island": (37.8269, -122.3719),
            "Unknown": (37.7749, -122.4194)
        }

    def _generate_heat_points(self, data: pd.DataFrame) -> List[List[float]]:
        """Generate heat points for the heatmap based on prediction data"""
        neighborhood_centers = self._get_neighborhood_centers()
        heat_points = []
        
        # Aggregate predictions by neighborhood
        neighborhood_totals = data.groupby('neighborhood')['predicted_requests'].sum()
        
        for neighborhood, total_requests in neighborhood_totals.items():
            # Get center coordinates for this neighborhood
            center = neighborhood_centers.get(neighborhood)
            if not center:
                # Try mapping variations
                mapped_name = self.neighborhood_mapping.get(neighborhood)
                center = neighborhood_centers.get(mapped_name) if mapped_name else None
            
            if center and total_requests > 0:
                lat, lng = center
                
                # Create multiple points around the center to simulate density
                # Higher predictions = more points = more heat
                intensity = min(total_requests / 10, 50)  # Scale intensity
                
                for i in range(int(intensity)):
                    # Add some random variation around the center
                    lat_offset = np.random.normal(0, 0.005)  # ~500m variation
                    lng_offset = np.random.normal(0, 0.005)
                    
                    heat_points.append([
                        lat + lat_offset,
                        lng + lng_offset,
                        total_requests / 100  # Weight for heat intensity
                    ])
        
        return heat_points

    def create_geospatial_heatmap(self, data: pd.DataFrame, title: str = "SF311 Predictions Heatmap") -> folium.Map:
        """Create a true geospatial heatmap of SF311 predictions"""
        
        # Create base map centered on SF with dark tiles for better heat contrast
        m = folium.Map(
            location=self.sf_center,
            zoom_start=12,
            tiles='CartoDB dark_matter'
        )
        
        if data.empty:
            folium.Marker(
                self.sf_center,
                popup="No prediction data available",
                icon=folium.Icon(color='orange')
            ).add_to(m)
            return m
        
        # Generate heat points from prediction data
        heat_points = self._generate_heat_points(data)
        
        if not heat_points:
            folium.Marker(
                self.sf_center,
                popup="No valid location data for neighborhoods",
                icon=folium.Icon(color='red')
            ).add_to(m)
            return m
        
        # Add heatmap layer
        HeatMap(
            heat_points,
            min_opacity=0.2,
            max_zoom=18,
            radius=25,
            blur=15,
            gradient={
                0.0: 'blue',
                0.2: 'cyan', 
                0.4: 'lime',
                0.6: 'yellow',
                0.8: 'orange',
                1.0: 'red'
            }
        ).add_to(m)
        
        # Add neighborhood markers with prediction totals
        neighborhood_totals = self._aggregate_predictions_by_neighborhood(data)
        neighborhood_centers = self._get_neighborhood_centers()
        
        for neighborhood, total_requests in neighborhood_totals.items():
            center = neighborhood_centers.get(neighborhood)
            if not center:
                # Try mapping variations
                mapped_name = self.neighborhood_mapping.get(neighborhood)
                center = neighborhood_centers.get(mapped_name) if mapped_name else None
            
            if center and total_requests > 0:
                # Determine marker color based on prediction volume
                if total_requests > np.percentile(list(neighborhood_totals.values()), 75):
                    color = 'red'
                elif total_requests > np.percentile(list(neighborhood_totals.values()), 50):
                    color = 'orange'
                elif total_requests > np.percentile(list(neighborhood_totals.values()), 25):
                    color = 'green'
                else:
                    color = 'blue'
                
                folium.CircleMarker(
                    location=center,
                    radius=8,
                    popup=f"""
                    <b>{neighborhood}</b><br>
                    Predicted Requests: {total_requests:,.0f}<br>
                    Rank: {sorted(neighborhood_totals.values(), reverse=True).index(total_requests) + 1} of {len(neighborhood_totals)}
                    """,
                    color='white',
                    weight=2,
                    fillColor=color,
                    fillOpacity=0.8
                ).add_to(m)
        
        # Add legend
        legend_html = f'''
        <div style="position: fixed; 
                    bottom: 50px; left: 50px; width: 220px; height: 140px; 
                    background-color: white; border:2px solid grey; z-index:9999; 
                    font-size:12px; padding: 10px">
        <p><b>{title}</b></p>
        <p><span style="color:red">●</span> Heat intensity shows prediction density</p>
        <p><span style="color:red">●</span> Top 25% neighborhoods</p>
        <p><span style="color:orange">●</span> 50-75% neighborhoods</p>
        <p><span style="color:green">●</span> 25-50% neighborhoods</p>
        <p><span style="color:blue">●</span> Bottom 25% neighborhoods</p>
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
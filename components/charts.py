import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from typing import Optional

class ChartGenerator:
    """Class for generating interactive charts for SF311 predictions"""
    
    def __init__(self):
        self.color_palette = px.colors.qualitative.Set3
    
    def create_line_chart(self, data: pd.DataFrame, show_confidence: bool = True, historical_data: Optional[pd.DataFrame] = None) -> go.Figure:
        """Create line chart showing prediction trends over time"""
        try:
            if data.empty:
                return self._create_empty_chart("No data available")
            
            # Aggregate data by date for overall trend
            daily_totals = data.groupby('date').agg({
                'predicted_requests': 'sum',
                'confidence_lower': 'sum',
                'confidence_upper': 'sum'
            }).reset_index()
            
            fig = go.Figure()
            
            # Add confidence interval if requested
            if show_confidence and 'confidence_upper' in data.columns:
                fig.add_trace(go.Scatter(
                    x=daily_totals['date'],
                    y=daily_totals['confidence_upper'],
                    fill=None,
                    mode='lines',
                    line_color='rgba(0,100,80,0)',
                    showlegend=False,
                    name='Upper Confidence'
                ))
                
                fig.add_trace(go.Scatter(
                    x=daily_totals['date'],
                    y=daily_totals['confidence_lower'],
                    fill='tonexty',
                    mode='lines',
                    line_color='rgba(0,100,80,0)',
                    name='Confidence Interval',
                    fillcolor='rgba(68, 68, 68, 0.2)'
                ))
            
            # Add main prediction line
            fig.add_trace(go.Scatter(
                x=daily_totals['date'],
                y=daily_totals['predicted_requests'],
                mode='lines+markers',
                name='Predicted Requests',
                line=dict(color='#2E86C1', width=3),
                marker=dict(size=6)
            ))
            
            # Add historical actual data if available
            if historical_data is not None and not historical_data.empty:
                historical_daily = historical_data.groupby('date')['actual_requests'].sum().reset_index()
                historical_daily['date'] = pd.to_datetime(historical_daily['date'])
                
                fig.add_trace(go.Scatter(
                    x=historical_daily['date'],
                    y=historical_daily['actual_requests'],
                    mode='lines+markers',
                    name='Actual Requests (Historical)',
                    line=dict(color='#E74C3C', width=3, dash='dot'),
                    marker=dict(size=6, symbol='diamond')
                ))
            
            # Add individual neighborhood lines (top 5 by total requests)
            neighborhood_sums = data.groupby('neighborhood')['predicted_requests'].sum()
            top_neighborhoods = neighborhood_sums.nlargest(5)
            
            for i, neighborhood in enumerate(top_neighborhoods.index):
                neighborhood_data = data[data['neighborhood'] == neighborhood]
                neighborhood_grouped = neighborhood_data.groupby('date')['predicted_requests'].sum().reset_index()
                
                fig.add_trace(go.Scatter(
                    x=neighborhood_grouped['date'],
                    y=neighborhood_grouped['predicted_requests'],
                    mode='lines',
                    name=neighborhood,
                    line=dict(color=self.color_palette[i % len(self.color_palette)], width=2),
                    opacity=0.7
                ))
            
            fig.update_layout(
                title="SF311 Street & Sidewalk Cleaning Predictions Over Time",
                xaxis_title="Date",
                yaxis_title="Predicted Requests",
                hovermode='x unified',
                height=500,
                showlegend=True,
                legend=dict(
                    orientation="h",
                    yanchor="bottom",
                    y=1.02,
                    xanchor="right",
                    x=1
                )
            )
            
            return fig
            
        except Exception as e:
            return self._create_error_chart(f"Error creating line chart: {str(e)}")
    
    def create_bar_chart(self, data: pd.DataFrame, historical_data: Optional[pd.DataFrame] = None) -> go.Figure:
        """Create bar chart showing predictions by neighborhood"""
        try:
            if data.empty:
                return self._create_empty_chart("No data available")
            
            # Aggregate by neighborhood
            neighborhood_totals = data.groupby('neighborhood')['predicted_requests'].sum()
            neighborhood_totals = neighborhood_totals.sort_values(ascending=True)
            
            # Add historical data if available
            if historical_data is not None and not historical_data.empty:
                historical_totals = historical_data.groupby('neighborhood')['actual_requests'].sum()
                
                # Combine the data
                comparison_df = pd.DataFrame({
                    'predicted': neighborhood_totals,
                    'actual': historical_totals
                }).fillna(0).sort_values('predicted', ascending=True)
                
                fig = go.Figure()
                
                # Add predicted bars
                fig.add_trace(go.Bar(
                    y=comparison_df.index,
                    x=comparison_df['predicted'],
                    orientation='h',
                    name='Predicted',
                    marker_color='#3498DB',
                    text=comparison_df['predicted'],
                    textposition='outside',
                    texttemplate='%{text:.0f}'
                ))
                
                # Add actual bars
                fig.add_trace(go.Bar(
                    y=comparison_df.index,
                    x=comparison_df['actual'],
                    orientation='h',
                    name='Actual (Historical)',
                    marker_color='#E74C3C',
                    opacity=0.7,
                    text=comparison_df['actual'],
                    textposition='outside',
                    texttemplate='%{text:.0f}'
                ))
                
                fig.update_layout(barmode='group')
            else:
                fig = go.Figure(data=[
                    go.Bar(
                        y=neighborhood_totals.index,
                        x=neighborhood_totals.values,
                        orientation='h',
                        marker_color='#3498DB',
                        text=neighborhood_totals.values,
                        textposition='outside',
                        texttemplate='%{text:.0f}'
                    )
                ])
            
            fig.update_layout(
                title="Total Predicted Requests by Neighborhood",
                xaxis_title="Predicted Requests",
                yaxis_title="Neighborhood",
                height=max(400, len(neighborhood_totals) * 25),
                showlegend=False
            )
            
            return fig
            
        except Exception as e:
            return self._create_error_chart(f"Error creating bar chart: {str(e)}")
    
    def create_heatmap(self, data: pd.DataFrame) -> go.Figure:
        """Create heatmap showing predictions by date and neighborhood"""
        try:
            if data.empty:
                return self._create_empty_chart("No data available")
            
            # Create pivot table for heatmap
            heatmap_data = data.pivot_table(
                index='neighborhood',
                columns='date',
                values='predicted_requests',
                aggfunc='sum',
                fill_value=0
            )
            
            # Limit to top neighborhoods if too many
            if len(heatmap_data) > 20:
                neighborhood_totals = data.groupby('neighborhood')['predicted_requests'].sum()
                top_neighborhoods_series = neighborhood_totals.nlargest(20)
                heatmap_data = heatmap_data.loc[top_neighborhoods_series.index]
            
            fig = go.Figure(data=go.Heatmap(
                z=heatmap_data.values,
                x=[col.strftime('%m/%d') if hasattr(col, 'strftime') else str(col) for col in heatmap_data.columns],
                y=heatmap_data.index,
                colorscale='Blues',
                showscale=True,
                colorbar=dict(title="Predicted Requests")
            ))
            
            fig.update_layout(
                title="Prediction Heatmap: Neighborhoods vs Time",
                xaxis_title="Date",
                yaxis_title="Neighborhood",
                height=max(400, len(heatmap_data) * 20),
                xaxis=dict(tickangle=45)
            )
            
            return fig
            
        except Exception as e:
            return self._create_error_chart(f"Error creating heatmap: {str(e)}")
    
    def create_neighborhood_comparison(self, data: pd.DataFrame, neighborhoods: list) -> go.Figure:
        """Create comparison chart for selected neighborhoods"""
        try:
            if data.empty or not neighborhoods:
                return self._create_empty_chart("No data available for comparison")
            
            fig = go.Figure()
            
            for i, neighborhood in enumerate(neighborhoods):
                neighborhood_data = data[data['neighborhood'] == neighborhood].groupby('date')['predicted_requests'].sum().reset_index()
                
                if not neighborhood_data.empty:
                    fig.add_trace(go.Scatter(
                        x=neighborhood_data['date'],
                        y=neighborhood_data['predicted_requests'],
                        mode='lines+markers',
                        name=neighborhood,
                        line=dict(color=self.color_palette[i % len(self.color_palette)], width=2),
                        marker=dict(size=4)
                    ))
            
            fig.update_layout(
                title="Neighborhood Comparison",
                xaxis_title="Date",
                yaxis_title="Predicted Requests",
                hovermode='x unified',
                height=400,
                showlegend=True
            )
            
            return fig
            
        except Exception as e:
            return self._create_error_chart(f"Error creating comparison chart: {str(e)}")
    
    def create_daily_distribution(self, data: pd.DataFrame) -> go.Figure:
        """Create distribution chart showing request patterns"""
        try:
            if data.empty:
                return self._create_empty_chart("No data available")
            
            # Calculate daily totals
            daily_totals = data.groupby('date')['predicted_requests'].sum()
            
            fig = go.Figure()
            
            # Add histogram
            fig.add_trace(go.Histogram(
                x=daily_totals.values,
                nbinsx=20,
                name='Distribution',
                marker_color='#3498DB',
                opacity=0.7
            ))
            
            # Add mean line
            mean_value = daily_totals.mean()
            fig.add_vline(
                x=mean_value,
                line_dash="dash",
                line_color="red",
                annotation_text=f"Mean: {mean_value:.1f}"
            )
            
            fig.update_layout(
                title="Distribution of Daily Predicted Requests",
                xaxis_title="Predicted Requests per Day",
                yaxis_title="Frequency",
                height=400,
                showlegend=False
            )
            
            return fig
            
        except Exception as e:
            return self._create_error_chart(f"Error creating distribution chart: {str(e)}")
    
    def _create_empty_chart(self, message: str) -> go.Figure:
        """Create an empty chart with a message"""
        fig = go.Figure()
        fig.add_annotation(
            text=message,
            xref="paper", yref="paper",
            x=0.5, y=0.5,
            xanchor='center', yanchor='middle',
            showarrow=False,
            font=dict(size=16, color="gray")
        )
        fig.update_layout(
            height=400,
            xaxis=dict(showgrid=False, showticklabels=False, zeroline=False),
            yaxis=dict(showgrid=False, showticklabels=False, zeroline=False)
        )
        return fig
    
    def _create_error_chart(self, error_message: str) -> go.Figure:
        """Create an error chart with error message"""
        fig = go.Figure()
        fig.add_annotation(
            text=f"⚠️ {error_message}",
            xref="paper", yref="paper",
            x=0.5, y=0.5,
            xanchor='center', yanchor='middle',
            showarrow=False,
            font=dict(size=14, color="red")
        )
        fig.update_layout(
            height=400,
            xaxis=dict(showgrid=False, showticklabels=False, zeroline=False),
            yaxis=dict(showgrid=False, showticklabels=False, zeroline=False)
        )
        return fig

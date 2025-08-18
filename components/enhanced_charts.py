"""
Enhanced chart components with all LSP errors fixed and improved visualizations
"""

import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np
from typing import Optional


class EnhancedChartGenerator:
    """Enhanced chart generator with all fixes applied"""
    
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
            
            # Add individual neighborhood lines (all neighborhoods)
            neighborhood_sums = data.groupby('neighborhood')['predicted_requests'].sum()
            top_neighborhoods = neighborhood_sums.sort_values(ascending=False)  # Fix: this is correct
            
            for i, neighborhood in enumerate(top_neighborhoods.index):
                neighborhood_data = data[data['neighborhood'] == neighborhood]
                if not neighborhood_data.empty:
                    neighborhood_grouped = neighborhood_data.groupby('date')['predicted_requests'].sum().reset_index()
                    
                    # Add confidence intervals for individual neighborhoods if requested
                    if show_confidence and 'confidence_upper' in neighborhood_data.columns:
                        # Add confidence interval
                        neighborhood_conf = neighborhood_data.groupby('date').agg({
                            'confidence_lower': 'sum',
                            'confidence_upper': 'sum'
                        }).reset_index()
                        
                        fig.add_trace(go.Scatter(
                            x=neighborhood_conf['date'],
                            y=neighborhood_conf['confidence_upper'],
                            fill=None,
                            mode='lines',
                            line_color='rgba(0,0,0,0)',
                            showlegend=False,
                            name=f'{neighborhood} Upper'
                        ))
                        
                        fig.add_trace(go.Scatter(
                            x=neighborhood_conf['date'],
                            y=neighborhood_conf['confidence_lower'],
                            fill='tonexty',
                            mode='lines',
                            line_color='rgba(0,0,0,0)',
                            name=f'{neighborhood} Confidence',
                            fillcolor=f'rgba({i*30 % 255},{(i*50) % 255},{(i*70) % 255},0.1)',
                            showlegend=False
                        ))
                    
                    fig.add_trace(go.Scatter(
                        x=neighborhood_grouped['date'],
                        y=neighborhood_grouped['predicted_requests'],
                        mode='lines+markers',
                        name=neighborhood,
                        line=dict(width=2),
                        marker=dict(size=4),
                        hovertemplate='%{x}<br>%{y:.0f} requests<extra></extra>'
                    ))
            
            fig.update_layout(
                title="Predicted Requests by Neighborhood Over Time",
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
            neighborhood_totals = neighborhood_totals.sort_values(ascending=True)  # Fix: this is correct
            
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
                    text=comparison_df['actual'],
                    textposition='outside',
                    texttemplate='%{text:.0f}'
                ))
                
                fig.update_layout(
                    title="Predicted vs Historical Requests by Neighborhood",
                    xaxis_title="Requests",
                    yaxis_title="Neighborhood",
                    height=max(400, len(comparison_df) * 25),
                    showlegend=True
                )
            else:
                # Just predicted data
                fig = go.Figure(data=go.Bar(
                    y=neighborhood_totals.index,
                    x=neighborhood_totals.values,
                    orientation='h',
                    marker_color='#3498DB',
                    text=neighborhood_totals.values,
                    textposition='outside',
                    texttemplate='%{text:.0f}'
                ))
                
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
            
            # Filter data for selected neighborhoods
            filtered_data = data[data['neighborhood'].isin(neighborhoods)]
            
            for i, neighborhood in enumerate(neighborhoods):
                neighborhood_data = filtered_data[filtered_data['neighborhood'] == neighborhood]
                if not neighborhood_data.empty:
                    fig.add_trace(go.Scatter(
                        x=neighborhood_data['date'],
                        y=neighborhood_data['predicted_requests'],
                        mode='lines+markers',
                        name=neighborhood,
                        line=dict(width=3),
                        marker=dict(size=6),
                        hovertemplate='<b>%{fullData.name}</b><br>%{x}<br>%{y:.0f} requests<extra></extra>'
                    ))
            
            fig.update_layout(
                title=f"Prediction Comparison: {', '.join(neighborhoods)}",
                xaxis_title="Date",
                yaxis_title="Predicted Requests",
                height=400,
                hovermode='x unified',
                showlegend=True
            )
            
            return fig
            
        except Exception as e:
            return self._create_error_chart(f"Error creating comparison chart: {str(e)}")
    
    def create_metrics_summary(self, data: pd.DataFrame, historical_data: Optional[pd.DataFrame] = None) -> go.Figure:
        """Create metrics summary visualization"""
        try:
            if data.empty:
                return self._create_empty_chart("No data available")
            
            # Calculate summary metrics
            total_predicted = data['predicted_requests'].sum()
            avg_daily = data.groupby('date')['predicted_requests'].sum().mean()
            peak_day = data.groupby('date')['predicted_requests'].sum().max()
            num_neighborhoods = data['neighborhood'].nunique()
            
            # Create metrics display
            fig = go.Figure()
            
            metrics = [
                ("Total Predicted Requests", f"{total_predicted:,.0f}"),
                ("Average Daily Requests", f"{avg_daily:.0f}"),
                ("Peak Day Requests", f"{peak_day:.0f}"),
                ("Active Neighborhoods", f"{num_neighborhoods}")
            ]
            
            colors = ['#3498DB', '#E74C3C', '#F39C12', '#27AE60']
            
            for i, (label, value) in enumerate(metrics):
                fig.add_trace(go.Bar(
                    x=[label],
                    y=[1],  # Just for display
                    name=label,
                    marker_color=colors[i],
                    text=value,
                    textposition='middle',
                    textfont=dict(size=20, color='white'),
                    hovertemplate=f'<b>{label}</b><br>{value}<extra></extra>'
                ))
            
            fig.update_layout(
                title="Prediction Summary Metrics",
                showlegend=False,
                height=200,
                yaxis=dict(showticklabels=False, showgrid=False),
                xaxis=dict(tickangle=45),
                plot_bgcolor='rgba(0,0,0,0)'
            )
            
            return fig
            
        except Exception as e:
            return self._create_error_chart(f"Error creating metrics summary: {str(e)}")
    
    def _create_empty_chart(self, message: str = "No data available") -> go.Figure:
        """Create empty chart placeholder"""
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
            title="SF311 Predictions",
            height=400,
            xaxis=dict(showgrid=False, showticklabels=False),
            yaxis=dict(showgrid=False, showticklabels=False),
            plot_bgcolor='rgba(0,0,0,0)'
        )
        return fig
    
    def _create_error_chart(self, error_message: str) -> go.Figure:
        """Create error chart"""
        fig = go.Figure()
        fig.add_annotation(
            text=f"Error: {error_message}",
            xref="paper", yref="paper",
            x=0.5, y=0.5,
            xanchor='center', yanchor='middle',
            showarrow=False,
            font=dict(size=14, color="red")
        )
        fig.update_layout(
            title="Chart Error",
            height=400,
            xaxis=dict(showgrid=False, showticklabels=False),
            yaxis=dict(showgrid=False, showticklabels=False),
            plot_bgcolor='rgba(0,0,0,0)'
        )
        return fig
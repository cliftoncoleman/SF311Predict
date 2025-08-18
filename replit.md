# SF311 Street & Sidewalk Cleaning Predictions Dashboard

## Overview

This is a Streamlit-based web dashboard for visualizing SF311 street and sidewalk cleaning predictions. The application provides an interactive interface to view predictive analytics for cleaning service requests across San Francisco neighborhoods. Users can explore trends, filter data by date ranges and neighborhoods, and view confidence intervals for predictions through various chart types including line charts, heatmaps, and geographic visualizations.

## User Preferences

Preferred communication style: Simple, everyday language.

## System Architecture

### Frontend Architecture
- **Framework**: Streamlit for rapid web application development
- **UI Components**: Modular component architecture with separate classes for charts and filters
- **State Management**: Streamlit's session state for maintaining data and user selections across interactions
- **Layout**: Wide layout with expandable sidebar for controls and filters

### Data Visualization
- **Charting Library**: Plotly for interactive visualizations
- **Chart Types**: Line charts with confidence intervals, heatmaps, and geographic plots
- **Data Processing**: Custom DataProcessor class for aggregating data at daily, weekly, and monthly levels
- **Color Schemes**: Consistent color palette using Plotly's qualitative color sets

### Data Flow Architecture
- **API Client**: Dedicated APIClient class for external data retrieval
- **Data Processing Pipeline**: Multi-stage processing with validation and error handling
- **Caching Strategy**: Session state caching to minimize API calls and improve performance
- **Real-time Updates**: Manual refresh capability with timestamp tracking

### Component Structure
- **Modular Design**: Separate modules for charts, filters, and utilities
- **Filter Component**: Reusable date range and neighborhood selection filters
- **Chart Generator**: Centralized chart creation with consistent styling and error handling
- **Error Handling**: Comprehensive error handling across all components with user-friendly messages

## External Dependencies

### Core Dependencies
- **Streamlit**: Web application framework for Python
- **Pandas**: Data manipulation and analysis
- **Plotly**: Interactive plotting library (plotly.express and plotly.graph_objects)
- **NumPy**: Numerical computing support
- **Requests**: HTTP client for API communication

### API Integration
- **SF311 Prediction API**: External API service for retrieving prediction data
- **Authentication**: Bearer token-based authentication with configurable API keys
- **Environment Configuration**: API base URL and keys configured via environment variables
- **Timeout Handling**: 30-second timeout for API requests with graceful degradation

### Data Sources
- **Prediction Data**: Real-time prediction data from external API
- **Neighborhood Data**: Geographic and administrative boundary information
- **Time Series Data**: Historical and predicted cleaning request volumes with confidence intervals
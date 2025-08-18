# SF311 Street & Sidewalk Cleaning Predictions Dashboard

## Overview

This is a Streamlit-based web dashboard for visualizing SF311 street and sidewalk cleaning predictions. The application provides an interactive interface to view predictive analytics for cleaning service requests across San Francisco neighborhoods. Users can explore trends, filter data by date ranges and neighborhoods, and view confidence intervals for predictions through various chart types including line charts, heatmaps, and geographic visualizations.

## User Preferences

Preferred communication style: Simple, everyday language.

## Recent Changes

### August 18, 2025
- ✓ Built complete Streamlit dashboard for SF311 Street & Sidewalk Cleaning predictions
- ✓ Added demo data functionality for testing and preview
- ✓ User confirmed dashboard design and functionality looks good
- ✓ **MAJOR ENHANCEMENT**: Implemented comprehensive prediction pipeline improvements based on expert feedback
- ✓ Added enhanced data pipeline with robust model selection (Seasonal Naive, ML, SARIMAX)
- ✓ Implemented MASE (Mean Absolute Scaled Error) metrics for more reliable accuracy measurement
- ✓ Optimized ML models for Replit environment with early stopping and proper parameters
- ✓ Added comprehensive validation and error handling with schema checks
- ✓ Enhanced feature engineering with consistent exogenous variables for SARIMAX
- ✓ Implemented multi-format output (CSV + JSON) with metadata
- ✓ Fixed all LSP diagnostics and improved code quality
- ✓ **NEIGHBORHOOD COALESCING FIX**: Resolved micro-neighborhood fragmentation issue
- ✓ Implemented proper mapping from neighborhoods_sffind_boundaries to analysis_neighborhood
- ✓ Fixed column name issues (analysis_neighborhood vs neighborhoods_analysis)
- ✓ System now shows 42 unique analysis neighborhoods instead of fragmented micro-areas
- ✓ User confirmed proper neighborhood groupings are now working
- ✓ **MAJOR MODEL ENHANCEMENTS**: Implemented comprehensive forecast improvements
- ✓ Added expanded momentum features (wk_delta, wk_ratio) to break weekly-flat forecasts  
- ✓ Enhanced lag features (1-6, 14, 21, 28 days) and rolling statistics (3, 7, 14, 28 windows)
- ✓ Implemented week-of-year seasonality with Fourier harmonics for broader patterns
- ✓ Added weekly-flat detection with intelligent model selection (MASE + repetition score)
- ✓ Enhanced confidence intervals using quantile regression models and conformal prediction
- ✓ Added guardrails for small/volatile neighborhoods with quality thresholds
- ✓ Implemented historical data capping (max_hist + 3*std_hist) to prevent unrealistic spikes
- ✓ **FORECASTING SUCCESS**: User confirmed major improvements - forecasts now much more dynamic
- ✓ Model diversity achieved: trend, exponential_smoothing, and seasonal_naive models all competing
- ✓ Weekly-flat issue resolved with intelligent model selection and enhanced features
- ✓ 3-year historical data providing better pattern recognition
- ✓ System now produces realistic trends, seasonality, and momentum-based forecasts
- → Next: Additional UI enhancements or feature requests as needed

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

### Enhanced Data Flow Architecture
- **Enhanced SF311 Pipeline**: Production-ready pipeline with sophisticated model selection
- **Automatic Model Selection**: Backtesting-based selection between Seasonal Naive, ML, and SARIMAX models
- **Robust Validation**: Schema validation ensuring yhat_lo ≤ yhat ≤ yhat_hi and non-negativity
- **Multi-stage Processing**: Historical data fetching → feature engineering → model training → forecasting
- **Performance Optimized**: ML models configured for Replit environment with early stopping
- **Multi-format Output**: CSV and JSON exports with generation metadata and timestamps

### Enhanced Component Structure
- **Enhanced Pipeline Module**: `improved_data_pipeline.py` with production-ready forecasting
- **MASE Metrics**: More reliable accuracy measurement than MAPE for sparse neighborhood data  
- **Seasonal Naive Forecasting**: Proper implementation preventing mis-ordered validation indexing
- **Optimized ML Models**: HistGradientBoostingRegressor with max_iter=300, early stopping, random_state=42
- **SARIMAX Integration**: Consistent exogenous variable handling with assertion checks
- **Enhanced Chart Components**: Fixed all LSP diagnostics and improved visualizations
- **Comprehensive Validation**: Data quality checks and schema validation throughout pipeline

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
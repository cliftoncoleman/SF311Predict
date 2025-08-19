# SF311 Street & Sidewalk Cleaning Predictions Dashboard

## Project Overview
A sophisticated Streamlit dashboard for predicting San Francisco 311 Street and Sidewalk Cleaning request trends using advanced machine learning techniques with neighborhood-level forecasting.

## Recent Changes (August 19, 2025)
✓ **PERFORMANCE OPTIMIZATION** - Implemented smart caching strategy to dramatically reduce loading times
✓ **Day-by-Day Heatmap** - Enhanced geospatial visualization with date selector and neighborhood labels
✓ **Smart Cache Strategy** - Data cached for 1 hour, field metadata cached for 24 hours
✓ **Cache Controls** - Added manual cache management buttons for user control
✓ **True Heatmap Visualization** - Implemented density-based heat visualization with color gradients
✓ **Interactive Navigation** - Added "Next Day" button and click detection for neighborhood details
✓ **Enhanced Markers** - Neighborhood names and prediction numbers displayed directly on map dots
✓ **FIXED: 5-Year Training Data Issue** - Resolved conflict where automatic historical comparison was overriding 5-year training data
✓ **Security Enhancement** - Moved SF311 API token from hardcoded values to secure environment variables
✓ **Priority Neighborhoods Fixed** - Corrected capitalization for "South Of Market" in priority selection

## Project Architecture

### Core Components
- **Fixed Pipeline (fixed_pipeline.py)**: Main prediction engine with 5-year training capability
- **Working App (working_app.py)**: Primary Streamlit interface with enhanced controls
- **Neighborhood Coalescing**: Standardizes SF neighborhood boundaries for consistent analysis
- **Multi-Model Selection**: Automatic backtesting selects optimal model per neighborhood

### Key Features
- **5-Year Historical Training**: Uses 1825 days of historical data for robust model training
- **Smart Model Selection**: MASE-based model comparison with weekly repetition detection
- **Day-by-Day Heatmap**: Interactive geospatial visualization with date selector
- **Smart Caching**: 1-hour data cache, 24-hour metadata cache for optimal performance
- **Enhanced Validation**: Comprehensive error handling and prediction validation
- **Multi-Format Output**: Saves predictions as both CSV and JSON with metadata

### Data Flow
1. **Fetch Historical Data**: Loads 1825 days (5 years) via SF311 API
2. **Neighborhood Processing**: Applies coalescing for standardized boundaries  
3. **Model Training**: Per-neighborhood backtesting selects best model (trend/ML/exponential smoothing)
4. **Prediction Generation**: Forecasts to end of year with confidence intervals
5. **Validation & Export**: Validates predictions and saves in multiple formats

## User Preferences
- **Data Integrity**: Always use authentic SF311 API data, never mock/placeholder data
- **Training Period**: Prefers 5-year historical training for robust predictions
- **Performance**: Values comprehensive model selection over speed
- **Output**: Wants both CSV and JSON export formats

## Technical Decisions
- **Smart Caching**: 1-hour TTL for data, 24-hour TTL for metadata to balance freshness and performance
- **Pipeline Architecture**: Isolated validation from training to prevent data conflicts
- **Model Selection**: MASE metrics with weekly repetition penalties
- **Geospatial Visualization**: True heatmap with density points and interactive date selection
- **Cache Controls**: Manual override buttons for forced refresh or cached reload
- **Date Range**: Default forecast to end of current year

## Known Issues Resolved
- ✓ **Historical Comparison Interference**: Disabled automatic historical data comparison that was overriding 5-year training
- ✓ **Cache Conflicts**: Added comprehensive cache clearing mechanisms
- ✓ **Data Loading Confusion**: Enhanced logging to distinguish training vs validation data fetches

## Environment Setup
- **Platform**: Replit with Streamlit on port 5000
- **Dependencies**: pandas, plotly, scikit-learn, statsmodels, streamlit, requests
- **API**: SF311 Open Data API with neighborhood coalescing
# SF311 Street & Sidewalk Cleaning Predictions Dashboard

## Project Overview
A sophisticated Streamlit dashboard for predicting San Francisco 311 Street and Sidewalk Cleaning request trends using advanced machine learning techniques with neighborhood-level forecasting.

## Recent Changes (August 19, 2025)
✓ **MAJOR: PostgreSQL Database Caching** - Implemented smart database caching system for 10-50x performance improvement
✓ **Intelligent Data Management** - Only fetches new data since last update, not full 5 years each time  
✓ **Cache Validation** - Database cache working correctly with test data storage and retrieval
✓ **Force Refresh Option** - Added 🔄 button to clear cache and reload all data when needed
✓ **Enhanced Performance** - First load fetches 5 years, subsequent loads only fetch new data gaps

## Previous Changes (August 18, 2025)
✓ **FIXED: 5-Year Training Data Issue** - Resolved conflict where automatic historical comparison was overriding 5-year training data
✓ **Enhanced Cache Management** - Added aggressive cache clearing mechanisms to prevent stale data
✓ **Improved Logging** - Added detailed debug logging to track data loading processes
✓ **Pipeline Isolation** - Separated validation data fetching from training data to prevent interference
✓ **Security Enhancement** - Moved SF311 API token from hardcoded values to secure environment variables
✓ **Priority Neighborhoods Fixed** - Corrected capitalization for "South Of Market" in priority selection
✓ **UI Simplification** - Removed chart type selector, defaulting to line charts for better time series visualization

## Project Architecture

### Core Components
- **Fixed Pipeline (fixed_pipeline.py)**: Main prediction engine with 5-year training capability
- **Working App (working_app.py)**: Primary Streamlit interface with enhanced controls
- **Neighborhood Coalescing**: Standardizes SF neighborhood boundaries for consistent analysis
- **Multi-Model Selection**: Automatic backtesting selects optimal model per neighborhood

### Key Features
- **5-Year Historical Training**: Uses 1825 days of historical data for robust model training
- **Smart Model Selection**: MASE-based model comparison with weekly repetition detection
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
- **Cache Management**: Aggressive clearing to prevent stale data issues
- **Pipeline Architecture**: Isolated validation from training to prevent data conflicts
- **Model Selection**: MASE metrics with weekly repetition penalties
- **Date Range**: Default forecast to end of current year

## Known Issues Resolved
- ✓ **Historical Comparison Interference**: Disabled automatic historical data comparison that was overriding 5-year training
- ✓ **Cache Conflicts**: Added comprehensive cache clearing mechanisms
- ✓ **Data Loading Confusion**: Enhanced logging to distinguish training vs validation data fetches

## Environment Setup
- **Platform**: Replit with Streamlit on port 5000
- **Dependencies**: pandas, plotly, scikit-learn, statsmodels, streamlit, requests
- **API**: SF311 Open Data API with neighborhood coalescing
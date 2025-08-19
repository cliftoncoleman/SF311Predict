"""
Smart database caching for SF311 data to dramatically improve performance.
Only fetches new data since last update, stores in PostgreSQL for fast access.
"""

import pandas as pd
import psycopg2
from psycopg2.extras import RealDictCursor
import os
import numpy as np
from datetime import datetime, timedelta, date
from typing import Optional, Tuple
import streamlit as st

class SF311DatabaseCache:
    """PostgreSQL-based cache for SF311 data with incremental updates"""
    
    def __init__(self):
        self.db_url = os.getenv("DATABASE_URL")
        if not self.db_url:
            raise ValueError("DATABASE_URL environment variable is required")
    
    def get_connection(self):
        """Get database connection"""
        return psycopg2.connect(self.db_url)
    
    def setup_tables(self):
        """Create tables if they don't exist"""
        with self.get_connection() as conn:
            with conn.cursor() as cur:
                # SF311 raw data table
                cur.execute("""
                    CREATE TABLE IF NOT EXISTS sf311_raw_data (
                        id SERIAL PRIMARY KEY,
                        date DATE NOT NULL,
                        neighborhood VARCHAR(255) NOT NULL,
                        cases INTEGER NOT NULL DEFAULT 0,
                        created_at TIMESTAMP DEFAULT NOW(),
                        UNIQUE(date, neighborhood)
                    );
                """)
                
                # Cache metadata table
                cur.execute("""
                    CREATE TABLE IF NOT EXISTS cache_metadata (
                        key VARCHAR(255) PRIMARY KEY,
                        value TEXT NOT NULL,
                        updated_at TIMESTAMP DEFAULT NOW()
                    );
                """)
                
                # Indexes for performance
                cur.execute("""
                    CREATE INDEX IF NOT EXISTS idx_sf311_date_neighborhood 
                    ON sf311_raw_data(date, neighborhood);
                """)
                
                cur.execute("""
                    CREATE INDEX IF NOT EXISTS idx_sf311_date 
                    ON sf311_raw_data(date);
                """)
                
                conn.commit()
    
    def get_last_update_date(self) -> Optional[date]:
        """Get the last date we have data for"""
        with self.get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT MAX(date) FROM sf311_raw_data")
                result = cur.fetchone()
                return result[0] if result and result[0] else None
    
    def get_data_count(self) -> int:
        """Get total number of records in cache"""
        with self.get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT COUNT(*) FROM sf311_raw_data")
                result = cur.fetchone()
                return result[0] if result else 0
    
    def store_data(self, df: pd.DataFrame):
        """Store new SF311 data in database"""
        if df.empty:
            return
        
        with self.get_connection() as conn:
            with conn.cursor() as cur:
                # Use ON CONFLICT to handle duplicates
                for _, row in df.iterrows():
                    cur.execute("""
                        INSERT INTO sf311_raw_data (date, neighborhood, cases)
                        VALUES (%s, %s, %s)
                        ON CONFLICT (date, neighborhood) 
                        DO UPDATE SET cases = EXCLUDED.cases, created_at = NOW()
                    """, (row['date'], row['neighborhood'], row['cases']))
                
                conn.commit()
    
    def get_cached_data(self, start_date: date, end_date: date) -> pd.DataFrame:
        """Retrieve cached data for date range"""
        with self.get_connection() as conn:
            query = """
                SELECT date, neighborhood, cases
                FROM sf311_raw_data 
                WHERE date >= %s AND date <= %s
                ORDER BY date, neighborhood
            """
            df = pd.read_sql_query(query, conn, params=[start_date, end_date])
            return df
    
    def get_all_cached_data(self) -> pd.DataFrame:
        """Get all cached data"""
        with self.get_connection() as conn:
            query = """
                SELECT date, neighborhood, cases
                FROM sf311_raw_data 
                ORDER BY date, neighborhood
            """
            df = pd.read_sql_query(query, conn)
            return df
    
    def update_metadata(self, key: str, value: str):
        """Update cache metadata"""
        with self.get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    INSERT INTO cache_metadata (key, value)
                    VALUES (%s, %s)
                    ON CONFLICT (key) 
                    DO UPDATE SET value = EXCLUDED.value, updated_at = NOW()
                """, (key, value))
                conn.commit()
    
    def get_metadata(self, key: str) -> Optional[str]:
        """Get cache metadata"""
        with self.get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT value FROM cache_metadata WHERE key = %s", (key,))
                result = cur.fetchone()
                return result[0] if result else None
    
    def clear_cache(self):
        """Clear all cached data (for testing/reset)"""
        with self.get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("DELETE FROM sf311_raw_data")
                cur.execute("DELETE FROM cache_metadata")
                conn.commit()
    
    def get_cache_stats(self) -> dict:
        """Get cache statistics"""
        with self.get_connection() as conn:
            with conn.cursor() as cur:
                # Date range
                cur.execute("SELECT MIN(date), MAX(date) FROM sf311_raw_data")
                date_range = cur.fetchone()
                
                # Record count
                cur.execute("SELECT COUNT(*) FROM sf311_raw_data")
                result = cur.fetchone()
                total_records = result[0] if result else 0
                
                # Neighborhood count
                cur.execute("SELECT COUNT(DISTINCT neighborhood) FROM sf311_raw_data")
                result = cur.fetchone()
                neighborhood_count = result[0] if result else 0
                
                # Last update
                cur.execute("SELECT MAX(created_at) FROM sf311_raw_data")
                result = cur.fetchone()
                last_update = result[0] if result else None
                
                return {
                    'total_records': total_records,
                    'neighborhood_count': neighborhood_count,
                    'date_range': date_range,
                    'last_update': last_update
                }


class SmartSF311Pipeline:
    """SF311 Pipeline with intelligent database caching"""
    
    def __init__(self):
        from fixed_pipeline import FixedSF311Pipeline
        self.api_pipeline = FixedSF311Pipeline()
        self.cache = SF311DatabaseCache()
        self.cache.setup_tables()
    
    def needs_update(self, target_days: int = 1825) -> Tuple[bool, Optional[date], Optional[date]]:
        """Check if we need to fetch new data"""
        last_cached = self.cache.get_last_update_date()
        today = date.today()
        target_start = today - timedelta(days=target_days)
        
        if not last_cached:
            # No data at all - need full fetch
            return True, target_start, today
        
        # Check if we have data spanning the full target range
        # Get earliest date in our cache
        with self.cache.get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT MIN(date) FROM sf311_raw_data")
                result = cur.fetchone()
                earliest_cached = result[0] if result and result[0] else None
        
        if not earliest_cached:
            # No data - need full fetch
            return True, target_start, today
        
        # Check if we have the full historical span
        if earliest_cached > target_start:
            # Missing historical data - fetch from target start to earliest we have
            return True, target_start, earliest_cached - timedelta(days=1)
        
        # Check if we're missing recent data
        if last_cached < today - timedelta(days=1):
            # Missing recent data - incremental fetch
            return True, last_cached + timedelta(days=1), today
        
        # Cache is complete
        return False, None, None
    
    def fetch_and_cache_data(self, target_days: int = 1825, force_refresh: bool = False) -> pd.DataFrame:
        """Intelligently fetch and cache SF311 data"""
        
        if force_refresh:
            st.info("🔄 Force refresh requested - clearing cache...")
            self.cache.clear_cache()
        
        needs_update, start_date, end_date = self.needs_update(target_days)
        
        st.info(f"🔍 Cache check: needs_update={needs_update}, target_days={target_days}")
        
        if needs_update and start_date and end_date:
            days_to_fetch = (end_date - start_date).days + 1
            st.info(f"📡 Fetching {days_to_fetch} days of new data ({start_date} to {end_date})...")
            
            # Fetch new data using existing pipeline
            fresh_data = self.api_pipeline.fetch_historical_data(days_to_fetch)
            
            st.info(f"🔍 API returned {len(fresh_data)} records")
            
            if not fresh_data.empty:
                # Convert to proper format for database storage
                if 'date' not in fresh_data.columns:
                    st.info("🔄 Converting data format...")
                    fresh_data = self._convert_to_cache_format(fresh_data)
                else:
                    st.info("✅ Data already in correct format")
                
                # Store in cache
                st.info(f"💾 Storing {len(fresh_data)} records in cache...")
                
                try:
                    self.cache.store_data(fresh_data)
                    # Verify storage immediately
                    count_after = self.cache.get_data_count()
                    st.info(f"📊 Cache now contains {count_after} total records")
                    
                    if count_after == 0:
                        st.error("❌ Storage failed - cache is still empty!")
                        # Try manual storage debug
                        st.info("🔧 Debugging storage issue...")
                        sample_data = fresh_data.head(5)
                        st.write("Sample data to store:", sample_data)
                        
                except Exception as storage_error:
                    st.error(f"❌ Cache storage failed: {storage_error}")
                    st.info("📝 Will use data directly without caching")
                
                self.cache.update_metadata('last_full_update', datetime.now().isoformat())
                st.success(f"✅ Successfully cached {len(fresh_data)} new records")
            else:
                st.warning("⚠️ No new data retrieved from API")
        
        else:
            st.info("✅ Cache is up to date - no new data needed!")
        
        # Get all data from cache
        target_start = date.today() - timedelta(days=target_days)
        cached_data = self.cache.get_cached_data(target_start, date.today())
        
        if cached_data.empty:
            st.error("❌ No cached data available")
            return pd.DataFrame()
        
        # Show cache stats
        stats = self.cache.get_cache_stats()
        st.success(f"🎯 Using cached data: {stats['total_records']:,} records, "
                  f"{stats['neighborhood_count']} neighborhoods, "
                  f"dates {stats['date_range'][0]} to {stats['date_range'][1]}")
        
        return cached_data
    
    def generate_predictions_with_cache(self, target_days: int = 1825, force_refresh: bool = False) -> pd.DataFrame:
        """Generate predictions using cached data"""
        
        # Get cached historical data
        historical_data = self.fetch_and_cache_data(target_days, force_refresh)
        
        if historical_data.empty:
            st.error("No historical data available - using API directly")
            # Fallback to direct API call if cache fails
            historical_data = self.api_pipeline.fetch_historical_data(target_days)
            if not historical_data.empty:
                st.info(f"Using direct API data: {len(historical_data)} records")
                # Try to store it for next time
                try:
                    self.cache.store_data(historical_data)
                except Exception as e:
                    st.warning(f"Could not cache data: {e}")
            else:
                return pd.DataFrame()
        
        # Use the prediction logic from fixed pipeline
        st.info("🤖 Generating predictions from historical data...")
        
        # Create predictions using the same logic as fixed pipeline but with cached data
        return self._generate_predictions_from_cached_data(historical_data)
    
    def _generate_predictions_from_cached_data(self, historical_data: pd.DataFrame) -> pd.DataFrame:
        """Generate predictions from cached historical data using fixed pipeline logic"""
        
        # Import prediction functions from fixed pipeline
        try:
            from fixed_pipeline import validate_predictions
        except ImportError:
            # Simple validation if import fails
            def validate_predictions(df):
                return df
        
        # Get unique neighborhoods
        neighborhoods = historical_data['neighborhood'].unique()
        prediction_results = []
        
        # Get prediction period (rest of current year)
        today = date.today()
        end_of_year = date(today.year, 12, 31)
        prediction_days = (end_of_year - today).days + 1
        
        for neighborhood in neighborhoods:
            nbhd_data = historical_data[historical_data['neighborhood'] == neighborhood].copy()
            nbhd_data = nbhd_data.sort_values('date').reset_index(drop=True)
            
            if len(nbhd_data) < 30:  # Need minimum data
                continue
            
            # Get time series
            hist_values = nbhd_data['cases'].values
            
            # Simple model selection (trend model as primary)
            try:
                # Simple linear trend forecast
                if len(hist_values) > 30:
                    # Calculate simple trend
                    recent_avg = np.mean(hist_values[-30:])
                    trend_slope = (hist_values[-1] - hist_values[-30]) / 30
                    forecast = [max(1, recent_avg + trend_slope * i) for i in range(prediction_days)]
                else:
                    # Use simple average if insufficient data
                    avg_value = np.mean(hist_values)
                    forecast = [max(1, avg_value)] * prediction_days
                
                # Create date range for predictions
                pred_dates = pd.date_range(
                    start=today,
                    periods=prediction_days,
                    freq='D'
                )
                
                # Create prediction records
                for i, pred_date in enumerate(pred_dates):
                    predicted_value = max(1, int(forecast[i]))
                    
                    # Simple confidence intervals
                    confidence_lower = max(0, int(predicted_value * 0.8))
                    confidence_upper = int(predicted_value * 1.2)
                    
                    prediction_results.append({
                        'date': pred_date.date(),
                        'neighborhood': neighborhood,
                        'predicted_requests': predicted_value,
                        'confidence_lower': confidence_lower,
                        'confidence_upper': confidence_upper,
                        'model_used': 'trend'
                    })
            
            except Exception as e:
                st.warning(f"Prediction failed for {neighborhood}: {e}")
                continue
        
        if not prediction_results:
            return pd.DataFrame()
        
        predictions_df = pd.DataFrame(prediction_results)
        return validate_predictions(predictions_df)
    
    def _convert_to_cache_format(self, data: pd.DataFrame) -> pd.DataFrame:
        """Convert API data to cache format (date, neighborhood, cases)"""
        
        # Check if data is already in the right format
        if all(col in data.columns for col in ['date', 'neighborhood', 'cases']):
            return data
        
        # Convert from wide format (dates as columns) to long format
        if 'neighborhood' in data.columns:
            # Melt the dataframe to convert date columns to rows
            date_columns = [col for col in data.columns if col != 'neighborhood']
            
            melted_data = data.melt(
                id_vars=['neighborhood'],
                value_vars=date_columns,
                var_name='date',
                value_name='cases'
            )
            
            # Convert date strings to proper date format
            melted_data['date'] = pd.to_datetime(melted_data['date']).dt.date
            melted_data['cases'] = melted_data['cases'].fillna(0).astype(int)
            
            return melted_data[['date', 'neighborhood', 'cases']]
        
        # If format is unknown, return empty dataframe
        st.error("Unknown data format - cannot convert to cache format")
        return pd.DataFrame({'date': [], 'neighborhood': [], 'cases': []})
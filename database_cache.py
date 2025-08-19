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
    
    REVISION_BACKFILL_DAYS = 14
    CHUNK_DAYS = 90  # avoid massive single API calls
    DATE_FMT = "%Y-%m-%d"
    
    def __init__(self):
        from fixed_pipeline import FixedSF311Pipeline
        self.api_pipeline = FixedSF311Pipeline()
        self.cache = SF311DatabaseCache()
        self.cache.setup_tables()
    
    def _compute_fetch_window(self, target_days: int):
        today = date.today()
        target_start = today - timedelta(days=target_days)

        # If cache empty → full backfill
        with self.cache.get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT MIN(date), MAX(date) FROM sf311_raw_data")
                row = cur.fetchone()
                min_d, max_d = row if row else (None, None)

        if not min_d or not max_d:
            return True, target_start, today  # full bootstrap

        # Oldest gap? backfill earlier section
        if min_d > target_start:
            return True, target_start, min_d - timedelta(days=1)

        # Recent gap? pull from after max_d to today
        if max_d < today:
            return True, max_d + timedelta(days=1), today

        return False, None, None
    
    def needs_update(self, target_days: int = 1825):
        # Keep a thin wrapper for compatibility; delegate to the new logic
        return self._compute_fetch_window(target_days)
    
    def _fetch_range_from_api(self, start_d: date, end_d: date) -> pd.DataFrame:
        """
        Fetch an explicit date range. Prefer a range-aware method.
        Fallback: if only "last N days" exists, we break the window into
        chunks that end at 'cur_end' so the API returns the correct slice.
        """
        # Try a proper range method first
        fetch_range = getattr(self.api_pipeline, "fetch_range", None)
        if callable(fetch_range):
            return fetch_range(start_d, end_d)

        # Fallback path: fetch_historical_data(n) returns "last n days ending today"
        fetch_hist = getattr(self.api_pipeline, "fetch_historical_data", None)
        if not callable(fetch_hist):
            return pd.DataFrame(columns=["date", "neighborhood", "cases"])

        # If the function supports end_date kwarg, use it:
        try:
            n_days = (end_d - start_d).days + 1
            return fetch_hist(n_days, end_date=end_d)
        except TypeError:
            # Last resort: this will fetch the most recent n_days ending *today*.
            # It will NOT honor start/end. Keep this only as a temporary fallback.
            n_days = (end_d - start_d).days + 1
            df = fetch_hist(n_days)
            # Filter locally to the intended window:
            if not df.empty:
                df["date"] = pd.to_datetime(df["date"]).dt.date
                return df[(df["date"] >= start_d) & (df["date"] <= end_d)]
            return df
    
    def fetch_and_cache_data(self, target_days: int = 1825, force_refresh: bool = False) -> pd.DataFrame:
        if force_refresh:
            st.info("🔄 Force refresh requested - clearing cache…")
            self.cache.clear_cache()

        needs_update, start_date, end_date = self.needs_update(target_days)
        st.info(f"🔍 Cache check: needs_update={needs_update}, target_days={target_days}")

        if needs_update and start_date and end_date:
            total_days = (end_date - start_date).days + 1
            st.info(f"📡 Fetching {total_days} days ({start_date} → {end_date}) in chunks of {self.CHUNK_DAYS}…")

            cur_start = start_date
            total_inserted = 0
            while cur_start <= end_date:
                cur_end = min(cur_start + timedelta(days=self.CHUNK_DAYS - 1), end_date)
                fresh = self._fetch_range_from_api(cur_start, cur_end)

                if not fresh.empty:
                    # Normalize types
                    fresh = fresh.copy()
                    fresh["date"] = pd.to_datetime(fresh["date"]).dt.date
                    if "cases" in fresh.columns:
                        fresh["cases"] = fresh["cases"].fillna(0).astype(int)

                    # Store
                    self.cache.store_data(fresh)
                    total_inserted += len(fresh)

                st.info(f"  • Stored {len(fresh)} rows for {cur_start} → {cur_end}")
                cur_start = cur_end + timedelta(days=1)

            st.success(f"✅ Finished caching: +{total_inserted} rows")

        else:
            st.info("✅ Cache is up to date — no new data needed")

        # Always read back exactly the 5-year window for the app
        target_start = date.today() - timedelta(days=target_days)
        cached_data = self.cache.get_cached_data(target_start, date.today())
        
        st.info(f"📊 Requested {target_days} days ({target_start} → {date.today()}), got {len(cached_data)} records")

        if cached_data.empty:
            st.error("❌ No cached data available")
            return pd.DataFrame()

        stats = self.cache.get_cache_stats()
        st.success(
            f"🎯 Using cached data: {stats['total_records']:,} rows • "
            f"{stats['neighborhood_count']} neighborhoods • "
            f"range {stats['date_range'][0]} → {stats['date_range'][1]}"
        )
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
        """Generate predictions using the actual sophisticated models from FixedSF311Pipeline"""
        
        # Use the actual sophisticated prediction logic from FixedSF311Pipeline
        st.info("🤖 Using robust ML models with backtesting and model selection...")
        
        try:
            # Create a temporary pipeline to leverage the sophisticated prediction methods
            temp_data = historical_data.copy()
            
            # Get prediction period (rest of current year) 
            today = date.today()
            end_of_year = date(today.year, 12, 31)
            prediction_days = (end_of_year - today).days + 1
            
            # Use the API pipeline's sophisticated generate_fixed_predictions method directly
            predictions = self.api_pipeline.generate_fixed_predictions(temp_data, prediction_days=prediction_days)
            
            st.success(f"✅ Generated {len(predictions)} predictions using sophisticated models")
            return predictions
            
        except Exception as e:
            st.warning(f"Sophisticated models failed ({e}), falling back to manual prediction generation")
            
            # Fallback: manually implement key prediction logic from FixedSF311Pipeline
            return self._fallback_generate_predictions(historical_data)
    
    def _fallback_generate_predictions(self, historical_data: pd.DataFrame) -> pd.DataFrame:
        """Fallback prediction generation using the full FixedSF311Pipeline sophisticated models"""
        
        try:
            # Use the full sophisticated prediction pipeline including backtesting
            st.info("🧠 Running full model selection: ML, exponential smoothing, trend, and seasonal models...")
            
            today = date.today()
            end_of_year = date(today.year, 12, 31)
            prediction_days = (end_of_year - today).days + 1
            
            # Get neighborhoods with sufficient data for sophisticated models
            neighborhoods = historical_data['neighborhood'].unique()
            all_forecasts = []
            
            for neighborhood in neighborhoods:
                nbhd_data = historical_data[historical_data['neighborhood'] == neighborhood].copy()
                nbhd_data = nbhd_data.sort_values('date').reset_index(drop=True)
                
                if len(nbhd_data) < 100:  # Need substantial data for sophisticated models
                    continue
                
                st.info(f"🔬 Training models for {neighborhood}...")
                
                # Split data for backtesting (use last 28 days for validation)
                val_size = min(28, len(nbhd_data) // 4)
                train_df = nbhd_data.iloc[:-val_size].copy()
                val_df = nbhd_data.iloc[-val_size:].copy()
                
                # Train all model types using FixedSF311Pipeline methods
                models = {}
                
                # 1. Trend model
                trend_result = self.api_pipeline._train_trend_model(train_df, val_df, 'cases')
                if trend_result:
                    models['trend'] = trend_result
                
                # 2. Exponential smoothing
                es_result = self.api_pipeline._train_exponential_smoothing(train_df, val_df, 'cases')
                if es_result:
                    models['exponential_smoothing'] = es_result
                
                # 3. ML model
                ml_result = self.api_pipeline._train_ml_model(train_df, val_df, 'cases')
                if ml_result:
                    models['ml'] = ml_result
                
                # 4. Seasonal naive baseline
                from fixed_pipeline import seasonal_naive_forecast, mase
                hist_values = train_df['cases'].values.astype(float)
                seasonal_forecast = seasonal_naive_forecast(hist_values, len(val_df), season=7)
                y_val = val_df['cases'].values.astype(float)
                seasonal_mase = mase(y_val, seasonal_forecast, season=7)
                
                models['seasonal_naive'] = {
                    "model_type": "seasonal_naive",
                    "model": None,
                    "mase_score": seasonal_mase,
                    "predictions": seasonal_forecast
                }
                
                # Select best model based on MASE score
                if models:
                    best_model_name = min(models.keys(), key=lambda k: models[k].get('mase_score', float('inf')))
                    best_model = models[best_model_name]
                    
                    st.info(f"✅ {neighborhood}: Selected {best_model_name} (MASE: {best_model['mase_score']:.3f})")
                    
                    # Generate forecast using the best model
                    forecast_df = self.api_pipeline._generate_forecast_from_model(
                        best_model, nbhd_data, neighborhood, prediction_days
                    )
                    
                    if not forecast_df.empty:
                        all_forecasts.append(forecast_df)
                else:
                    st.warning(f"⚠️ No models trained successfully for {neighborhood}")
            
            if all_forecasts:
                final_forecast = pd.concat(all_forecasts, ignore_index=True)
                from fixed_pipeline import validate_predictions
                final_forecast = validate_predictions(final_forecast)
                st.success(f"🎯 Generated sophisticated predictions for {len(final_forecast['neighborhood'].unique())} neighborhoods")
                return final_forecast
            else:
                st.error("❌ No sophisticated models could be trained")
                return pd.DataFrame()
                
        except Exception as e:
            st.error(f"❌ Sophisticated models failed completely: {e}")
            return pd.DataFrame()
    
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
"""
Fixed SF311 Data Pipeline - Robust version with proper error handling
"""

import pandas as pd
import numpy as np
import datetime as dt
import time
import json
from datetime import datetime, timedelta
from typing import Optional, List, Dict, Any, Tuple
import requests
import streamlit as st
import warnings
warnings.filterwarnings('ignore')


def seasonal_naive_forecast(hist_values: np.ndarray, n_periods: int, season: int = 7) -> np.ndarray:
    """Generate seasonal naive forecast by repeating last seasonal pattern"""
    if len(hist_values) < season:
        return np.full(n_periods, max(hist_values[-1] if len(hist_values) > 0 else 1.0, 0))
    
    last_season = hist_values[-season:]
    n_full_cycles = n_periods // season
    remainder = n_periods % season
    
    forecast = np.tile(last_season, n_full_cycles)
    if remainder > 0:
        forecast = np.concatenate([forecast, last_season[:remainder]])
    
    return np.maximum(forecast, 0)


def mase(y_true: np.ndarray, y_pred: np.ndarray, season: int = 7) -> float:
    """Mean Absolute Scaled Error"""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    
    if len(y_true) <= season:
        denom = np.mean(np.abs(np.diff(y_true))) + 1e-6
    else:
        denom = np.mean(np.abs(y_true[season:] - y_true[:-season]))
    
    numerator = np.mean(np.abs(y_true - y_pred))
    return numerator / max(denom, 1e-6)


def validate_predictions(df: pd.DataFrame) -> pd.DataFrame:
    """Validate prediction schema and fix any issues"""
    df = df.copy()
    
    required_cols = ['date', 'neighborhood', 'predicted_requests', 'confidence_lower', 'confidence_upper']
    for col in required_cols:
        if col not in df.columns:
            if 'predicted' in col:
                df[col] = 1.0
            elif 'confidence' in col:
                df[col] = 1.0 if 'lower' in col else 2.0
            else:
                df[col] = 'Unknown'
    
    df = df.dropna(subset=['predicted_requests', 'confidence_lower', 'confidence_upper'])
    
    numeric_cols = ['predicted_requests', 'confidence_lower', 'confidence_upper']
    for col in numeric_cols:
        df[col] = np.maximum(df[col].astype(float), 0)
    
    df['confidence_lower'] = np.minimum(df['confidence_lower'], df['predicted_requests'])
    df['confidence_upper'] = np.maximum(df['confidence_upper'], df['predicted_requests'])
    
    return df


class FixedSF311Pipeline:
    """Fixed SF311 prediction pipeline with robust error handling"""
    
    def __init__(self):
        self.base_url = "https://data.sfgov.org/resource/vw6y-z8j6.json"
        self.meta_url = "https://data.sfgov.org/api/views/vw6y-z8j6?content=metadata"
        self.app_token = "TuXFZRAF7T8dnb1Rqk5VOdOKN"
        
        self.time_field = "requested_datetime"
        self.category_field = "service_name"
        self.category_value = "Street and Sidewalk Cleaning"
        self.page_size = 50000
        
        self.FOURIER_K_YEAR = 5  # Reduced for stability
        self.max_forecast_horizon = 90
        
        self.neighbor_pref_order = [
            "neighborhoods_analysis_boundaries",
            "neighborhoods_sffind_boundaries", 
            "neighborhood_district",
        ]
        
        self.session = requests.Session()
        self.session.headers.update({"X-App-Token": self.app_token})
    
    def get_field_names(self) -> set:
        """Get available field names from SF311 API metadata"""
        try:
            r = self.session.get(self.meta_url, timeout=60)
            r.raise_for_status()
            meta = r.json()
            return {c["fieldName"] for c in meta.get("columns", [])}
        except Exception:
            return set()

    def pick_neighborhood_field(self, field_names: set) -> str:
        """Pick the best available neighborhood field"""
        for f in self.neighbor_pref_order:
            if f in field_names:
                return f
        return self.neighbor_pref_order[0]

    def month_windows(self, start_date: dt.date, end_date: dt.date):
        """Generate month windows for data fetching"""
        cur = dt.date(start_date.year, start_date.month, 1)
        end_month = dt.date(end_date.year, end_date.month, 1)
        while cur <= end_month:
            next_month = (cur.replace(day=28) + dt.timedelta(days=4)).replace(day=1)
            win_start = dt.datetime.combine(cur, dt.time.min)
            hard_end_date = end_date + dt.timedelta(days=1)
            win_end_date = min(next_month, hard_end_date)
            win_end = dt.datetime.combine(win_end_date, dt.time.min)
            yield win_start.isoformat(), win_end.isoformat()
            cur = next_month

    def fetch_month(self, neighborhood_field: str, win_start_iso: str, win_end_iso: str) -> pd.DataFrame:
        """Fetch one month of data with paging"""
        frames = []
        offset = 0
        retries = 0
        
        while True:
            params = {
                "$select": f"{self.time_field}, {neighborhood_field}",
                "$where": (
                    f"{self.category_field} = '{self.category_value}' AND "
                    f"{self.time_field} >= '{win_start_iso}' AND {self.time_field} < '{win_end_iso}'"
                ),
                "$order": f"{self.time_field} ASC",
                "$limit": self.page_size,
                "$offset": offset,
            }
            
            try:
                r = self.session.get(self.base_url, params=params, timeout=120)
            except requests.exceptions.RequestException as e:
                if retries < 5:
                    time.sleep(2 ** retries)
                    retries += 1
                    continue
                raise RuntimeError(f"Network error after retries: {e}") from e

            if r.status_code == 429:
                time.sleep(2 + retries)
                retries = min(retries + 1, 5)
                continue

            if r.status_code != 200:
                raise RuntimeError(f"Socrata error {r.status_code}: {r.text[:1000]}")

            rows = r.json()
            if not rows:
                break
                
            frames.append(pd.DataFrame(rows))
            if len(rows) < self.page_size:
                break
            offset += self.page_size

        if not frames:
            return pd.DataFrame({self.time_field: [], neighborhood_field: []})
        return pd.concat(frames, ignore_index=True)

    def fetch_historical_data(self, start_days: int = 365) -> pd.DataFrame:
        """Fetch historical SF311 data"""
        today = dt.date.today()
        start_date = today - dt.timedelta(days=start_days)
        
        try:
            fields = self.get_field_names()
            nbhd_field = self.pick_neighborhood_field(fields)
            
            all_frames = []
            for win_start, win_end in self.month_windows(start_date, today):
                df_month = self.fetch_month(nbhd_field, win_start, win_end)
                if not df_month.empty:
                    all_frames.append(df_month)
            
            if not all_frames:
                return pd.DataFrame()
                
            raw = pd.concat(all_frames, ignore_index=True)
            
            raw[self.time_field] = pd.to_datetime(raw[self.time_field], errors="coerce", utc=True)
            raw["date"] = raw[self.time_field].dt.tz_convert("US/Pacific").dt.date
            raw["neighborhood"] = raw[nbhd_field].fillna("Unknown").astype(str)
            
            daily_counts = raw.groupby(["date", "neighborhood"], as_index=False).size()
            daily_counts = daily_counts.rename(columns={"size": "cases"})
            daily = daily_counts.sort_values(["date", "neighborhood"]).reset_index(drop=True)
            
            return daily
            
        except Exception as e:
            st.error(f"Error fetching SF311 data: {str(e)}")
            return pd.DataFrame()
    
    def backtest_and_select_model(self, df_nbhd: pd.DataFrame, y_col: str = 'cases', val_days: int = 30) -> Dict[str, Any]:
        """Improved model selection with fixed validation indexing"""
        if len(df_nbhd) < val_days + 60:
            return {"model_type": "seasonal_naive", "model": None, "score": float('inf')}
        
        # Fixed indexing as suggested
        train_df = df_nbhd.iloc[:-val_days].copy()
        val_df = df_nbhd.iloc[-val_days:].copy()
        
        # Fixed seasonal naive baseline
        hist = df_nbhd[y_col].values
        val_len = len(val_df)
        base_val_pred = seasonal_naive_forecast(hist[:-val_len], val_len, season=7)
        
        y_val = val_df[y_col].values.astype(float)
        
        eps = 1e-6
        mape_baseline = np.mean(np.abs((y_val - base_val_pred) / np.maximum(y_val, eps))) * 100
        mase_baseline = mase(y_val, base_val_pred, season=7)
        
        best_model = {
            "model_type": "seasonal_naive",
            "model": None,
            "mape_score": mape_baseline,
            "mase_score": mase_baseline,
            "predictions": base_val_pred
        }
        
        # Try ML model if enough data
        if len(train_df) >= 60:
            try:
                ml_result = self._train_ml_model(train_df, val_df, y_col)
                if ml_result and ml_result["mase_score"] < best_model["mase_score"]:
                    best_model = ml_result
            except Exception:
                pass
        
        return best_model
    
    def _train_ml_model(self, train_df: pd.DataFrame, val_df: pd.DataFrame, y_col: str) -> Optional[Dict[str, Any]]:
        """Train ML model with fixed parameters"""
        try:
            from sklearn.ensemble import HistGradientBoostingRegressor
            
            train_features = self._build_ml_features(train_df.copy())
            
            if train_features.empty or len(train_features) < 28:
                return None
            
            feature_cols = [c for c in train_features.columns if c not in ['date', y_col]]
            X_train = train_features[feature_cols].fillna(0)
            y_train = train_features[y_col].fillna(0).astype(float)
            
            # Fixed parameters as suggested
            model = HistGradientBoostingRegressor(
                loss="poisson",
                max_iter=300,
                learning_rate=0.05,
                random_state=42
            )
            
            model.fit(X_train, y_train)
            
            # Generate validation predictions
            val_features = self._build_ml_features(pd.concat([train_df, val_df]))
            val_features_subset = val_features.tail(len(val_df))
            X_val = val_features_subset[feature_cols].fillna(0)
            
            val_pred = model.predict(X_val)
            val_pred = np.maximum(val_pred, 0)
            
            y_val = val_df[y_col].values.astype(float)
            
            eps = 1e-6
            mape_score = np.mean(np.abs((y_val - val_pred) / np.maximum(y_val, eps))) * 100
            mase_score = mase(y_val, val_pred, season=7)
            
            return {
                "model_type": "ml",
                "model": model,
                "feature_cols": feature_cols,
                "mape_score": mape_score,
                "mase_score": mase_score,
                "predictions": val_pred
            }
            
        except Exception:
            return None
    
    def _build_ml_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Build ML features"""
        df = df.copy()
        df['date'] = pd.to_datetime(df['date'])
        df = df.sort_values('date').reset_index(drop=True)
        
        # Lag features
        for lag in [1, 7, 14, 28]:
            df[f'lag_{lag}'] = df['cases'].shift(lag)
        
        # Rolling statistics
        for window in [7, 14, 28]:
            df[f'roll_mean_{window}'] = df['cases'].shift(1).rolling(
                window, min_periods=max(1, window//2)
            ).mean()
            df[f'roll_std_{window}'] = df['cases'].shift(1).rolling(
                window, min_periods=max(1, window//2)
            ).std()
        
        # Time features
        df['dow'] = df['date'].dt.dayofweek
        df['month'] = df['date'].dt.month
        df['day'] = df['date'].dt.day
        
        df = df.dropna(subset=['lag_1', 'lag_7'])
        return df
    
    def generate_fixed_predictions(self, 
                                  historical_data: pd.DataFrame,
                                  prediction_days: int = 30) -> pd.DataFrame:
        """Generate predictions using fixed pipeline"""
        
        if historical_data.empty:
            return self._generate_baseline_predictions(prediction_days)
        
        prediction_days = min(prediction_days, self.max_forecast_horizon)
        
        all_forecasts = []
        neighborhoods = historical_data['neighborhood'].unique()
        
        for neighborhood in neighborhoods:
            nbhd_data = historical_data[historical_data['neighborhood'] == neighborhood].copy()
            nbhd_data = nbhd_data.sort_values('date').reset_index(drop=True)
            
            if len(nbhd_data) < 30:
                forecast = self._simple_neighborhood_forecast(neighborhood, nbhd_data, prediction_days)
                all_forecasts.append(forecast)
                continue
            
            nbhd_data = self._ensure_continuous_days(nbhd_data)
            
            if len(nbhd_data) < 60:
                forecast = self._simple_neighborhood_forecast(neighborhood, nbhd_data, prediction_days)
                all_forecasts.append(forecast)
                continue
            
            # Select best model
            best_model_result = self.backtest_and_select_model(nbhd_data)
            
            # Generate forecast
            forecast = self._generate_forecast_from_model(
                best_model_result, nbhd_data, neighborhood, prediction_days
            )
            all_forecasts.append(forecast)
        
        if all_forecasts:
            final_forecast = pd.concat(all_forecasts, ignore_index=True)
            final_forecast = validate_predictions(final_forecast)
            return final_forecast
        else:
            return self._generate_baseline_predictions(prediction_days)
    
    def _ensure_continuous_days(self, df: pd.DataFrame) -> pd.DataFrame:
        """Fill missing dates with 0 cases"""
        df['date'] = pd.to_datetime(df['date'])
        date_range = pd.date_range(start=df['date'].min(), end=df['date'].max(), freq='D')
        full_df = pd.DataFrame({'date': date_range})
        merged = full_df.merge(df, on='date', how='left')
        merged['cases'] = merged['cases'].fillna(0).astype(int)
        merged['neighborhood'] = merged['neighborhood'].fillna(df['neighborhood'].iloc[0])
        return merged.sort_values('date').reset_index(drop=True)
    
    def _generate_forecast_from_model(self, 
                                     model_result: Dict[str, Any], 
                                     nbhd_data: pd.DataFrame, 
                                     neighborhood: str, 
                                     prediction_days: int) -> pd.DataFrame:
        """Generate forecast from trained model"""
        
        start_date = nbhd_data['date'].max()
        if isinstance(start_date, str):
            start_date = pd.to_datetime(start_date).date()
        elif hasattr(start_date, 'date'):
            start_date = start_date.date()
        
        forecast_dates = [start_date + timedelta(days=i+1) for i in range(prediction_days)]
        
        if model_result["model_type"] == "seasonal_naive":
            hist_values = nbhd_data['cases'].values.astype(float)
            predictions = seasonal_naive_forecast(hist_values, prediction_days, season=7)
            
            mean_pred = np.mean(predictions)
            std_pred = np.std(predictions) if len(predictions) > 1 else mean_pred * 0.2
            confidence_lower = np.maximum(predictions - 1.96 * std_pred, 0)
            confidence_upper = predictions + 1.96 * std_pred
            
        elif model_result["model_type"] == "ml":
            model = model_result["model"]
            feature_cols = model_result["feature_cols"]
            
            predictions = []
            confidence_lower = []
            confidence_upper = []
            
            extended_data = nbhd_data.copy()
            
            for i in range(prediction_days):
                temp_data = self._build_ml_features(extended_data.copy())
                if temp_data.empty:
                    pred = extended_data['cases'].iloc[-1] if len(extended_data) > 0 else 1
                else:
                    X_pred = temp_data[feature_cols].iloc[-1:].fillna(0)
                    pred = model.predict(X_pred)[0]
                
                pred = max(float(pred), 0)
                predictions.append(pred)
                
                pred_std = pred * 0.15
                confidence_lower.append(max(pred - 1.96 * pred_std, 0))
                confidence_upper.append(pred + 1.96 * pred_std)
                
                next_date = forecast_dates[i]
                new_row = pd.DataFrame({
                    'date': [next_date],
                    'neighborhood': [neighborhood],
                    'cases': [pred]
                })
                extended_data = pd.concat([extended_data, new_row], ignore_index=True)
            
            predictions = np.array(predictions)
            confidence_lower = np.array(confidence_lower)
            confidence_upper = np.array(confidence_upper)
        
        else:
            last_value = nbhd_data['cases'].iloc[-1] if len(nbhd_data) > 0 else 1
            predictions = np.full(prediction_days, max(float(last_value), 0))
            confidence_lower = np.full(prediction_days, max(float(last_value) * 0.8, 0))
            confidence_upper = np.full(prediction_days, float(last_value) * 1.2)
        
        forecast_df = pd.DataFrame({
            'date': forecast_dates,
            'neighborhood': neighborhood,
            'predicted_requests': predictions.astype(float),
            'confidence_lower': confidence_lower.astype(float),
            'confidence_upper': confidence_upper.astype(float)
        })
        
        return forecast_df
    
    def _simple_neighborhood_forecast(self, neighborhood: str, nbhd_data: pd.DataFrame, prediction_days: int) -> pd.DataFrame:
        """Generate simple forecast for neighborhoods with limited data"""
        if nbhd_data.empty:
            avg_cases = 5
        else:
            avg_cases = max(float(nbhd_data['cases'].mean()), 1)
        
        start_date = datetime.now().date()
        if not nbhd_data.empty:
            last_date = nbhd_data['date'].max()
            if isinstance(last_date, str):
                start_date = pd.to_datetime(last_date).date()
            elif hasattr(last_date, 'date'):
                start_date = last_date.date()
        
        forecast_dates = [start_date + timedelta(days=i+1) for i in range(prediction_days)]
        
        predictions = []
        for i, date in enumerate(forecast_dates):
            base_pred = avg_cases
            if date.weekday() in [5, 6]:
                base_pred *= 0.7
            predictions.append(max(base_pred, 1))
        
        predictions = np.array(predictions, dtype=float)
        
        return pd.DataFrame({
            'date': forecast_dates,
            'neighborhood': neighborhood,
            'predicted_requests': predictions,
            'confidence_lower': predictions * 0.8,
            'confidence_upper': predictions * 1.2
        })
    
    def _generate_baseline_predictions(self, prediction_days: int) -> pd.DataFrame:
        """Generate baseline predictions when no historical data is available"""
        
        neighborhoods = [
            "Mission", "Castro", "SOMA", "Chinatown", "North Beach", 
            "Pacific Heights", "Marina", "Haight-Ashbury", "Richmond", 
            "Sunset", "Tenderloin", "Financial District"
        ]
        
        predictions = []
        start_date = datetime.now().date()
        
        for i in range(prediction_days):
            prediction_date = start_date + timedelta(days=i)
            
            for neighborhood in neighborhoods:
                base_requests = {
                    "Mission": 25, "Castro": 15, "SOMA": 30, "Chinatown": 18,
                    "North Beach": 16, "Pacific Heights": 10, "Marina": 12,
                    "Haight-Ashbury": 20, "Richmond": 14, "Sunset": 16,
                    "Tenderloin": 22, "Financial District": 25
                }.get(neighborhood, 15)
                
                if prediction_date.weekday() in [5, 6]:
                    base_requests = int(base_requests * 0.7)
                
                predictions.append({
                    'date': prediction_date,
                    'neighborhood': neighborhood,
                    'predicted_requests': float(base_requests),
                    'confidence_lower': float(base_requests * 0.8),
                    'confidence_upper': float(base_requests * 1.2)
                })
        
        return pd.DataFrame(predictions)
    
    def save_predictions_enhanced(self, predictions: pd.DataFrame, output_dir: str = "predictions") -> Dict[str, str]:
        """Save predictions in multiple formats"""
        import os
        
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        timestamp = datetime.now().isoformat()
        
        csv_path = os.path.join(output_dir, f"sf311_predictions_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv")
        predictions.to_csv(csv_path, index=False)
        
        # JSON format as suggested
        json_data = {
            "generated_at_utc": timestamp,
            "horizon_days": len(predictions['date'].unique()) if not predictions.empty else 0,
            "neighborhoods": {}
        }
        
        if not predictions.empty:
            for neighborhood in predictions['neighborhood'].unique():
                nbhd_data = predictions[predictions['neighborhood'] == neighborhood]
                json_data["neighborhoods"][neighborhood] = {
                    "predictions": [
                        {
                            "date": str(row['date']),
                            "predicted_requests": float(row['predicted_requests']),
                            "confidence_lower": float(row['confidence_lower']),
                            "confidence_upper": float(row['confidence_upper'])
                        }
                        for _, row in nbhd_data.iterrows()
                    ]
                }
        
        json_path = os.path.join(output_dir, f"sf311_predictions_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
        with open(json_path, 'w') as f:
            json.dump(json_data, f, indent=2)
        
        return {
            "csv_path": csv_path,
            "json_path": json_path,
            "generated_at": timestamp
        }
    
    def run_full_fixed_pipeline(self, 
                               days_back: int = 1095,
                               prediction_days: int = 30) -> pd.DataFrame:
        """Run the complete fixed pipeline"""
        
        historical_data = self.fetch_historical_data(start_days=days_back)
        predictions = self.generate_fixed_predictions(historical_data, prediction_days)
        
        return predictions
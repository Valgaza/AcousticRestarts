"""
Traffic Congestion Forecasting Model using Prophet

This module implements a Prophet-based forecasting model for predicting
traffic congestion levels using multivariable inputs including road features
and time series data.

Author: Arnav Waghdhare
"""

import pandas as pd
import numpy as np
from prophet import Prophet
from sklearn.metrics import mean_absolute_error, mean_squared_error
from typing import Dict, List, Optional, Tuple
import warnings
warnings.filterwarnings('ignore')


class TrafficCongestionForecaster:
    """
    Prophet-based forecasting model for traffic congestion prediction.
    
    This model handles multivariable inputs including:
    - Road features: edge_id, road_length_meters, road_type, lane_count, 
                     speed_limit_kph, free_flow_speed_kph, road_capacity
    - Time series features: hour_of_day, is_weekend, is_holiday, 
                           weather_condition, temperature_celsius, 
                           visibility, event_impact_score
    
    Output: congestion_level (0.0 to 1.0)
    """
    
    def __init__(
        self,
        seasonality_mode: str = 'multiplicative',
        changepoint_prior_scale: float = 0.05,
        seasonality_prior_scale: float = 10.0,
        yearly_seasonality: bool = True,
        weekly_seasonality: bool = True,
        daily_seasonality: bool = True
    ):
        """
        Initialize the Traffic Congestion Forecaster.
        
        Args:
            seasonality_mode: 'additive' or 'multiplicative' seasonality
            changepoint_prior_scale: Flexibility of trend (higher = more flexible)
            seasonality_prior_scale: Flexibility of seasonality (higher = more flexible)
            yearly_seasonality: Include yearly seasonality patterns
            weekly_seasonality: Include weekly seasonality patterns
            daily_seasonality: Include daily seasonality patterns
        """
        self.seasonality_mode = seasonality_mode
        self.changepoint_prior_scale = changepoint_prior_scale
        self.seasonality_prior_scale = seasonality_prior_scale
        self.yearly_seasonality = yearly_seasonality
        self.weekly_seasonality = weekly_seasonality
        self.daily_seasonality = daily_seasonality
        
        self.model = None
        self.feature_columns = []
        self.is_fitted = False
        
    def _prepare_data(
        self,
        road_data: pd.DataFrame,
        time_series_data: pd.DataFrame,
        merge_on: str = 'edge_id'
    ) -> pd.DataFrame:
        """
        Prepare and merge road data with time series data for Prophet.
        
        Args:
            road_data: DataFrame with road features (edge_id, road_length_meters, 
                      road_type, lane_count, speed_limit_kph, free_flow_speed_kph, 
                      road_capacity)
            time_series_data: DataFrame with time series features and target
                             (timestamp, hour_of_day, is_weekend, is_holiday,
                              weather_condition, temperature_celsius, visibility,
                              event_impact_score, congestion_level)
            merge_on: Column to merge on (typically 'edge_id')
        
        Returns:
            Merged DataFrame ready for Prophet
        """
        # Merge road features with time series data
        if merge_on in road_data.columns and merge_on in time_series_data.columns:
            merged_data = time_series_data.merge(road_data, on=merge_on, how='left')
        else:
            # If no merge column, assume single road or already merged
            merged_data = time_series_data.copy()
            
        # Ensure timestamp is datetime
        if 'timestamp' in merged_data.columns:
            merged_data['timestamp'] = pd.to_datetime(merged_data['timestamp'])
        
        return merged_data
    
    def _encode_categorical_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Encode categorical features for Prophet.
        
        Args:
            data: DataFrame with categorical features
            
        Returns:
            DataFrame with encoded categorical features
        """
        data = data.copy()
        
        # One-hot encode road_type if present
        if 'road_type' in data.columns:
            road_type_dummies = pd.get_dummies(data['road_type'], prefix='road_type')
            data = pd.concat([data, road_type_dummies], axis=1)
            data.drop('road_type', axis=1, inplace=True)
        
        # One-hot encode weather_condition if present
        if 'weather_condition' in data.columns:
            weather_dummies = pd.get_dummies(data['weather_condition'], prefix='weather')
            data = pd.concat([data, weather_dummies], axis=1)
            data.drop('weather_condition', axis=1, inplace=True)
        
        return data
    
    def _create_prophet_dataframe(
        self,
        data: pd.DataFrame,
        target_column: str = 'congestion_level'
    ) -> pd.DataFrame:
        """
        Create Prophet-compatible DataFrame with 'ds' and 'y' columns.
        
        Args:
            data: Input DataFrame with timestamp and target
            target_column: Name of the target variable column
            
        Returns:
            Prophet-compatible DataFrame
        """
        prophet_df = pd.DataFrame()
        prophet_df['ds'] = data['timestamp']
        prophet_df['y'] = data[target_column]
        
        # Add all regressor columns
        regressor_columns = [
            col for col in data.columns 
            if col not in ['timestamp', target_column, 'edge_id', 
                          'start_node_id', 'end_node_id']
        ]
        
        for col in regressor_columns:
            prophet_df[col] = data[col].values
        
        self.feature_columns = regressor_columns
        
        return prophet_df
    
    def fit(
        self,
        road_data: pd.DataFrame,
        time_series_data: pd.DataFrame,
        target_column: str = 'congestion_level',
        merge_on: Optional[str] = 'edge_id'
    ) -> 'TrafficCongestionForecaster':
        """
        Fit the Prophet model on historical data.
        
        Args:
            road_data: DataFrame with road features
            time_series_data: DataFrame with time series features and target
            target_column: Name of the target variable column
            merge_on: Column to merge on (None if already merged)
            
        Returns:
            Self for method chaining
        """
        # Prepare data
        if merge_on and merge_on in road_data.columns:
            data = self._prepare_data(road_data, time_series_data, merge_on)
        else:
            data = time_series_data.copy()
            data['timestamp'] = pd.to_datetime(data['timestamp'])
        
        # Encode categorical features
        data = self._encode_categorical_features(data)
        
        # Create Prophet DataFrame
        prophet_df = self._create_prophet_dataframe(data, target_column)
        
        # Initialize Prophet model
        self.model = Prophet(
            seasonality_mode=self.seasonality_mode,
            changepoint_prior_scale=self.changepoint_prior_scale,
            seasonality_prior_scale=self.seasonality_prior_scale,
            yearly_seasonality=self.yearly_seasonality,
            weekly_seasonality=self.weekly_seasonality,
            daily_seasonality=self.daily_seasonality
        )
        
        # Add regressors
        for feature in self.feature_columns:
            self.model.add_regressor(feature)
        
        # Fit the model
        self.model.fit(prophet_df)
        self.is_fitted = True
        
        return self
    
    def predict(
        self,
        future_data: pd.DataFrame,
        include_history: bool = False
    ) -> pd.DataFrame:
        """
        Generate forecasts for future time periods.
        
        Args:
            future_data: DataFrame with future timestamps and regressor values
                        Must include 'timestamp' and all feature columns used in training
            include_history: Whether to include historical predictions
            
        Returns:
            DataFrame with predictions including confidence intervals
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making predictions")
        
        # Ensure timestamp is datetime
        future_data = future_data.copy()
        future_data['timestamp'] = pd.to_datetime(future_data['timestamp'])
        
        # Encode categorical features
        future_data = self._encode_categorical_features(future_data)
        
        # Create Prophet DataFrame
        prophet_future = pd.DataFrame()
        prophet_future['ds'] = future_data['timestamp']
        
        # Add all regressor columns
        for feature in self.feature_columns:
            if feature not in future_data.columns:
                raise ValueError(f"Missing feature: {feature}")
            prophet_future[feature] = future_data[feature].values
        
        # Make predictions
        forecast = self.model.predict(prophet_future)
        
        # Prepare output
        predictions = pd.DataFrame()
        predictions['timestamp'] = forecast['ds']
        predictions['congestion_level_predicted'] = forecast['yhat']
        predictions['congestion_level_lower'] = forecast['yhat_lower']
        predictions['congestion_level_upper'] = forecast['yhat_upper']
        
        # Clip predictions to valid range [0, 1]
        predictions['congestion_level_predicted'] = predictions['congestion_level_predicted'].clip(0, 1)
        predictions['congestion_level_lower'] = predictions['congestion_level_lower'].clip(0, 1)
        predictions['congestion_level_upper'] = predictions['congestion_level_upper'].clip(0, 1)
        
        return predictions
    
    def evaluate(
        self,
        road_data: pd.DataFrame,
        time_series_data: pd.DataFrame,
        target_column: str = 'congestion_level',
        merge_on: Optional[str] = 'edge_id'
    ) -> Dict[str, float]:
        """
        Evaluate model performance on test data.
        
        Args:
            road_data: DataFrame with road features
            time_series_data: DataFrame with time series features and actual target values
            target_column: Name of the target variable column
            merge_on: Column to merge on
            
        Returns:
            Dictionary with evaluation metrics (MAE, RMSE, MAPE)
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before evaluation")
        
        # Prepare data
        if merge_on and merge_on in road_data.columns:
            data = self._prepare_data(road_data, time_series_data, merge_on)
        else:
            data = time_series_data.copy()
            data['timestamp'] = pd.to_datetime(data['timestamp'])
        
        # Get predictions
        predictions = self.predict(data)
        
        # Merge with actual values
        results = predictions.merge(
            data[['timestamp', target_column]],
            on='timestamp',
            how='inner'
        )
        
        # Calculate metrics
        y_true = results[target_column].values
        y_pred = results['congestion_level_predicted'].values
        
        mae = mean_absolute_error(y_true, y_pred)
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        mape = np.mean(np.abs((y_true - y_pred) / (y_true + 1e-10))) * 100
        
        return {
            'MAE': mae,
            'RMSE': rmse,
            'MAPE': mape
        }
    
    def create_future_dataframe(
        self,
        periods: int,
        freq: str = 'H',
        include_history: bool = False
    ) -> pd.DataFrame:
        """
        Create a future dataframe for making predictions.
        
        Note: You'll need to populate this with regressor values manually.
        
        Args:
            periods: Number of periods to forecast
            freq: Frequency of predictions ('H' for hourly, 'D' for daily, etc.)
            include_history: Whether to include historical dates
            
        Returns:
            DataFrame with future timestamps
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before creating future dataframe")
        
        future = self.model.make_future_dataframe(
            periods=periods,
            freq=freq,
            include_history=include_history
        )
        
        return future
    
    def get_feature_importance(self) -> pd.DataFrame:
        """
        Get the importance of regressors in the model.
        
        Returns:
            DataFrame with regressor names and their coefficients
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before getting feature importance")
        
        # Extract regressor coefficients from the model
        regressor_coefficients = {}
        
        for feature in self.feature_columns:
            if feature in self.model.train_component_cols:
                coef_name = feature
                if coef_name in self.model.params:
                    regressor_coefficients[feature] = self.model.params[coef_name].mean()
        
        importance_df = pd.DataFrame.from_dict(
            regressor_coefficients,
            orient='index',
            columns=['coefficient']
        ).sort_values(by='coefficient', ascending=False)
        
        return importance_df
    
    def plot_forecast(self, forecast: pd.DataFrame):
        """
        Plot the forecast using Prophet's built-in plotting.
        
        Args:
            forecast: Forecast DataFrame from predict()
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before plotting")
        
        from prophet.plot import plot_plotly
        import plotly.offline as py
        
        fig = self.model.plot(forecast)
        return fig
    
    def plot_components(self, forecast: pd.DataFrame):
        """
        Plot the forecast components (trend, seasonality, etc.).
        
        Args:
            forecast: Forecast DataFrame from predict()
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before plotting")
        
        fig = self.model.plot_components(forecast)
        return fig


def create_sample_data(n_samples: int = 1000) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Create sample data for testing the forecasting model.
    
    Args:
        n_samples: Number of samples to generate
        
    Returns:
        Tuple of (road_data, time_series_data)
    """
    # Create sample road data
    road_data = pd.DataFrame({
        'edge_id': [1, 2, 3],
        'start_node_id': [100, 101, 102],
        'end_node_id': [200, 201, 202],
        'road_length_meters': [1500.0, 2000.0, 1200.0],
        'road_type': ['Highway', 'Arterial', 'Local'],
        'lane_count': [4, 3, 2],
        'speed_limit_kph': [100, 80, 50],
        'free_flow_speed_kph': [95, 75, 48],
        'road_capacity': [2000, 1500, 800]
    })
    
    # Create sample time series data
    np.random.seed(42)
    dates = pd.date_range(start='2024-01-01', periods=n_samples, freq='h')
    
    time_series_data = []
    for edge_id in [1, 2, 3]:
        for i, date in enumerate(dates[:n_samples // 3]):
            hour = date.hour
            is_weekend = date.dayofweek >= 5
            
            # Simulate congestion patterns
            base_congestion = 0.3
            hour_effect = 0.3 * np.sin((hour - 8) * np.pi / 12)  # Peak during rush hours
            weekend_effect = -0.1 if is_weekend else 0.0
            random_noise = np.random.normal(0, 0.05)
            
            congestion = np.clip(
                base_congestion + hour_effect + weekend_effect + random_noise,
                0.0, 1.0
            )
            
            time_series_data.append({
                'edge_id': edge_id,
                'timestamp': date,
                'hour_of_day': hour,
                'is_weekend': is_weekend,
                'is_holiday': False,
                'weather_condition': np.random.choice(['Clear', 'Rainy', 'Cloudy']),
                'temperature_celsius': np.random.uniform(10, 35),
                'visibility': np.random.uniform(5, 10),
                'event_impact_score': np.random.uniform(0, 0.2),
                'congestion_level': congestion
            })
    
    time_series_df = pd.DataFrame(time_series_data)
    
    return road_data, time_series_df

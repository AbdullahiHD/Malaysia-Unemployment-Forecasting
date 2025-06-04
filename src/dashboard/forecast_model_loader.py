"""
Model loader and prediction utilities for forecasting hub.
Handles loading pre-trained ARIMA/SARIMA/LSTM models and generating predictions.
Updated to work with the actual trained models from the notebook.
"""

import pickle
import joblib
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
import warnings

warnings.filterwarnings("ignore")

# Try to import required packages for models
try:
    from skforecast.recursive import ForecasterSarimax
    from skforecast.sarimax import Sarimax

    SKFORECAST_AVAILABLE = True
except ImportError:
    SKFORECAST_AVAILABLE = False
    print("⚠️ skforecast not available")

try:
    from statsmodels.tsa.arima.model import ARIMA

    STATSMODELS_AVAILABLE = True
except ImportError:
    STATSMODELS_AVAILABLE = False
    print("⚠️ statsmodels not available")

try:
    import tensorflow as tf
    from tensorflow.keras.models import load_model
    from sklearn.preprocessing import MinMaxScaler

    TENSORFLOW_AVAILABLE = True
except ImportError:
    TENSORFLOW_AVAILABLE = False
    print("⚠️ TensorFlow not available")


class LSTMForecaster:
    """Recreate the LSTM Forecaster class from notebook"""

    def __init__(self, sequence_length=12, n_features=1):
        self.sequence_length = sequence_length
        self.n_features = n_features
        self.model = None
        self.scaler = MinMaxScaler()
        self.history = None

    def predict(self, last_sequence, steps=1):
        """Generate multi-step forecasts using recursive prediction"""
        predictions = []
        current_sequence = last_sequence.copy()

        for _ in range(steps):
            # Reshape for prediction
            X_pred = current_sequence.reshape(
                (1, self.sequence_length, self.n_features)
            )

            # Make prediction
            pred_scaled = self.model.predict(X_pred, verbose=0)
            pred_original = self.scaler.inverse_transform(pred_scaled)[0, 0]
            predictions.append(pred_original)

            # Update sequence for next prediction
            pred_scaled_flat = pred_scaled[0, 0]
            current_sequence = np.append(current_sequence[1:], pred_scaled_flat)

        return np.array(predictions)


class ForecastModelLoader:
    """Loads and manages pre-trained forecasting models"""

    def __init__(self, models_dir="models/saved"):
        self.models_dir = Path(models_dir)
        self.loaded_models = {}
        self.model_metadata = {}

        # Model file patterns based on your saved files
        self.model_patterns = {
            "arima_general": "arima_general_*.pkl",
            "arima_sa": "arima_sa_*.pkl",
            "sarima_general": "sarima_general_*.pkl",
            "sarima_sa": "sarima_sa_*.pkl",
            "lstm_general_model": "lstm_general_model_*.h5",
            "lstm_general_metadata": "lstm_general_metadata_*.pkl",
            "lstm_sa_model": "lstm_sa_model_*.h5",
            "lstm_sa_metadata": "lstm_sa_metadata_*.pkl",
        }

    def get_available_models(self):
        """Get list of available models"""
        available = {}
        for model_key, pattern in self.model_patterns.items():
            model_files = list(self.models_dir.glob(pattern))
            if model_files:
                # Get the most recent model file
                latest_file = max(model_files, key=lambda x: x.stat().st_mtime)
                available[model_key] = {
                    "file_path": latest_file,
                    "last_modified": datetime.fromtimestamp(
                        latest_file.stat().st_mtime
                    ),
                    "size_kb": latest_file.stat().st_size / 1024,
                }
        return available

    def load_sarima_model(self, dataset_type):
        """Load SARIMA model (skforecast ForecasterSarimax)"""
        model_key = f"sarima_{dataset_type}"

        if not SKFORECAST_AVAILABLE:
            raise ImportError("skforecast is required for SARIMA models")

        available_models = self.get_available_models()
        if model_key not in available_models:
            raise FileNotFoundError(f"SARIMA model for {dataset_type} not found")

        model_path = available_models[model_key]["file_path"]

        try:
            # Load using joblib as saved in notebook
            model = joblib.load(model_path)
            self.loaded_models[model_key] = model

            # Set metadata based on notebook results
            if dataset_type == "general":
                order = (0, 1, 1)
                seasonal_order = (0, 1, 1, 12)
                mape = 0.92
            else:  # sa
                order = (0, 1, 2)
                seasonal_order = (0, 1, 1, 12)
                mape = 1.66

            self.model_metadata[model_key] = {
                "model_type": "SARIMA",
                "dataset": (
                    "General Labor Force"
                    if dataset_type == "general"
                    else "Seasonally Adjusted"
                ),
                "order": order,
                "seasonal_order": seasonal_order,
                "file_path": str(model_path),
                "last_updated": available_models[model_key]["last_modified"].strftime(
                    "%Y-%m-%d"
                ),
                "training_period": "2010-2022",
                "validation_period": "2023-2024",
                "mape": mape,
                "accuracy": 100 - mape,
            }

            return model

        except Exception as e:
            raise RuntimeError(f"Failed to load SARIMA model {model_key}: {str(e)}")

    def load_arima_model(self, dataset_type):
        """Load ARIMA model"""
        model_key = f"arima_{dataset_type}"

        if not STATSMODELS_AVAILABLE:
            raise ImportError("statsmodels is required for ARIMA models")

        available_models = self.get_available_models()
        if model_key not in available_models:
            raise FileNotFoundError(f"ARIMA model for {dataset_type} not found")

        model_path = available_models[model_key]["file_path"]

        try:
            # Load ARIMA model data
            with open(model_path, "rb") as f:
                model_data = joblib.load(f)

            self.loaded_models[model_key] = model_data

            # Set metadata based on notebook results
            self.model_metadata[model_key] = {
                "model_type": "ARIMA",
                "dataset": (
                    "General Labor Force"
                    if dataset_type == "general"
                    else "Seasonally Adjusted"
                ),
                "order": model_data.get("order", (0, 1, 0)),
                "file_path": str(model_path),
                "last_updated": available_models[model_key]["last_modified"].strftime(
                    "%Y-%m-%d"
                ),
                "training_period": "2010-2022",
                "mape": 16.13 if dataset_type == "general" else 21.34,
                "accuracy": 83.87 if dataset_type == "general" else 78.66,
            }

            return model_data

        except Exception as e:
            raise RuntimeError(f"Failed to load ARIMA model {model_key}: {str(e)}")

    def load_lstm_model(self, dataset_type):
        """Load LSTM model with metadata"""
        if not TENSORFLOW_AVAILABLE:
            raise ImportError("TensorFlow is required for LSTM models")

        available_models = self.get_available_models()
        model_key = f"lstm_{dataset_type}_model"
        metadata_key = f"lstm_{dataset_type}_metadata"

        if model_key not in available_models or metadata_key not in available_models:
            raise FileNotFoundError(f"LSTM model files for {dataset_type} not found")

        model_path = available_models[model_key]["file_path"]
        metadata_path = available_models[metadata_key]["file_path"]

        try:
            # Load Keras model
            keras_model = load_model(model_path)

            # Load metadata
            with open(metadata_path, "rb") as f:
                metadata = joblib.load(f)

            # Reconstruct LSTM forecaster
            lstm_forecaster = LSTMForecaster(
                sequence_length=metadata["sequence_length"],
                n_features=metadata["n_features"],
            )
            lstm_forecaster.model = keras_model
            lstm_forecaster.scaler = metadata["scaler"]

            self.loaded_models[f"lstm_{dataset_type}"] = lstm_forecaster

            # Set metadata based on notebook results
            self.model_metadata[f"lstm_{dataset_type}"] = {
                "model_type": "LSTM",
                "dataset": (
                    "General Labor Force"
                    if dataset_type == "general"
                    else "Seasonally Adjusted"
                ),
                "sequence_length": metadata["sequence_length"],
                "file_path": str(model_path),
                "last_updated": available_models[model_key]["last_modified"].strftime(
                    "%Y-%m-%d"
                ),
                "training_period": "2010-2022",
                "architecture": "64x32 units with dropout",
                "mape": 2.67 if dataset_type == "general" else 4.54,
                "accuracy": 97.33 if dataset_type == "general" else 95.46,
            }

            return lstm_forecaster

        except Exception as e:
            raise RuntimeError(
                f"Failed to load LSTM model for {dataset_type}: {str(e)}"
            )

    def load_model(self, model_type, dataset_type):
        """Load specific model based on type and dataset"""
        model_key = f"{model_type}_{dataset_type}"

        if model_key in self.loaded_models:
            return self.loaded_models[model_key]

        if model_type == "sarima":
            return self.load_sarima_model(dataset_type)
        elif model_type == "arima":
            return self.load_arima_model(dataset_type)
        elif model_type == "lstm":
            return self.load_lstm_model(dataset_type)
        else:
            raise ValueError(f"Unknown model type: {model_type}")

    def generate_forecast(self, model_type, dataset_type, periods, historical_data):
        """Generate forecast using loaded model with the same logic as notebook"""
        try:
            # Load model
            model = self.load_model(model_type, dataset_type)
            model_key = f"{model_type}_{dataset_type}"

            # Get current unemployment rate
            current_rate = historical_data["u_rate"].iloc[-1]

            # Generate forecast based on model type
            if model_type == "sarima":
                # SARIMA prediction similar to notebook
                forecast_result = model.predict(steps=periods)
                forecast_values = forecast_result.values

                # Generate confidence intervals (approximate)
                std_dev = historical_data["u_rate"].diff().dropna().std()
                confidence_lower = forecast_values - 1.96 * std_dev
                confidence_upper = forecast_values + 1.96 * std_dev

            elif model_type == "arima":
                # ARIMA prediction
                model_fit = model["model_fit"]
                forecast_result = model_fit.forecast(steps=periods)

                if hasattr(forecast_result, "values"):
                    forecast_values = forecast_result.values
                else:
                    forecast_values = np.array(
                        [forecast_result]
                        if np.isscalar(forecast_result)
                        else forecast_result
                    )

                # Generate confidence intervals
                std_dev = historical_data["u_rate"].diff().dropna().std()
                confidence_lower = forecast_values - 1.96 * std_dev
                confidence_upper = forecast_values + 1.96 * std_dev

            elif model_type == "lstm":
                # LSTM prediction similar to notebook
                # Get last 12 months of data for sequence
                last_sequence = historical_data["u_rate"].tail(12).values
                last_seq_scaled = model.scaler.transform(
                    last_sequence.reshape(-1, 1)
                ).flatten()

                forecast_values = model.predict(last_seq_scaled, steps=periods)

                # Generate confidence intervals
                std_dev = historical_data["u_rate"].diff().dropna().std()
                confidence_lower = forecast_values - 1.96 * std_dev
                confidence_upper = forecast_values + 1.96 * std_dev

            else:
                raise ValueError(f"Unknown model type: {model_type}")

            # Create forecast dates
            last_date = historical_data.index[-1]
            forecast_dates = pd.date_range(
                start=last_date + pd.DateOffset(months=1), periods=periods, freq="MS"
            )

            # Calculate trend analysis
            trend_direction, trend_magnitude = self._analyze_trend(
                current_rate, forecast_values
            )

            # Package results
            forecast_data = {
                "dates": forecast_dates,
                "forecast_values": forecast_values,
                "confidence_upper": confidence_upper,
                "confidence_lower": confidence_lower,
                "current_rate": current_rate,
                "trend_direction": trend_direction,
                "trend_magnitude": trend_magnitude,
                "confidence_level": 95,
                "model_info": self.model_metadata[model_key],
            }

            return forecast_data

        except Exception as e:
            raise RuntimeError(f"Forecast generation failed: {str(e)}")

    def _analyze_trend(self, current_rate, forecast_values):
        """Analyze forecast trend direction and magnitude"""
        final_rate = forecast_values[-1]
        change = final_rate - current_rate

        if abs(change) < 0.1:
            trend_direction = "Stable"
        elif change > 0:
            trend_direction = "Rising"
        else:
            trend_direction = "Falling"

        trend_magnitude = abs(change)

        return trend_direction, trend_magnitude

    def get_model_info(self, model_type, dataset_type):
        """Get metadata for specific model"""
        model_key = f"{model_type}_{dataset_type}"
        if model_key in self.model_metadata:
            return self.model_metadata[model_key]

        # Try to load model to get metadata
        try:
            self.load_model(model_type, dataset_type)
            return self.model_metadata[model_key]
        except Exception as e:
            return {
                "model_type": model_type.upper(),
                "dataset": (
                    "General Labor Force"
                    if dataset_type == "general"
                    else "Seasonally Adjusted"
                ),
                "error": f"Model not available: {str(e)}",
            }


class ForecastDataPreparer:
    """Prepares historical data for forecasting and visualization"""

    def __init__(self, data_manager):
        self.data_manager = data_manager

    def prepare_historical_data(self, dataset_type="general", lookback_months=24):
        """Prepare historical unemployment data for forecasting"""
        try:
            # Get appropriate dataset - using the same data as notebook training
            if dataset_type == "general":
                df = self.data_manager.get_dataset("Overall Unemployment")
            else:  # seasonally adjusted
                df = self.data_manager.get_dataset("Seasonally Adjusted")

            # Ensure proper datetime index with monthly frequency
            df = df.asfreq("MS")

            # Handle any missing values using same method as notebook
            if df["u_rate"].isna().sum() > 0:
                df["u_rate"] = df["u_rate"].interpolate(
                    method="linear", limit_direction="both"
                )
                df["u_rate"] = df["u_rate"].ffill().bfill()

            # Get recent data for visualization
            recent_data = df.tail(lookback_months)

            # Prepare historical data structure
            historical_data = {
                "dates": recent_data.index.tolist(),
                "values": recent_data["u_rate"].tolist(),
                "full_series": recent_data["u_rate"],
            }

            return historical_data, df

        except Exception as e:
            raise RuntimeError(f"Failed to prepare historical data: {str(e)}")

    def validate_forecast_inputs(self, model_type, dataset_type, periods):
        """Validate forecast input parameters"""
        valid_models = ["arima", "sarima", "lstm"]
        valid_datasets = ["general", "sa"]
        valid_periods = [1, 3, 6, 12]

        if model_type not in valid_models:
            raise ValueError(
                f"Invalid model type: {model_type}. Must be one of {valid_models}"
            )

        if dataset_type not in valid_datasets:
            raise ValueError(
                f"Invalid dataset type: {dataset_type}. Must be one of {valid_datasets}"
            )

        if periods not in valid_periods:
            raise ValueError(
                f"Invalid forecast period: {periods}. Must be one of {valid_periods}"
            )

        return True


class ForecastManager:
    """Main interface for forecast generation and management"""

    def __init__(self, data_manager, models_dir="models/saved"):
        self.data_manager = data_manager
        self.model_loader = ForecastModelLoader(models_dir)
        self.data_preparer = ForecastDataPreparer(data_manager)

    def generate_complete_forecast(self, model_type, dataset_type, periods):
        """Generate complete forecast with all components"""
        try:
            # Validate inputs
            self.data_preparer.validate_forecast_inputs(
                model_type, dataset_type, periods
            )

            # Prepare historical data using same preprocessing as notebook
            historical_data, full_dataset = self.data_preparer.prepare_historical_data(
                dataset_type
            )

            # Generate forecast using trained models
            forecast_data = self.model_loader.generate_forecast(
                model_type, dataset_type, periods, full_dataset
            )

            # Package complete results
            complete_forecast = {
                "historical_data": historical_data,
                "forecast_data": forecast_data,
                "model_info": forecast_data["model_info"],
                "generation_time": datetime.now(),
                "parameters": {
                    "model_type": model_type,
                    "dataset_type": dataset_type,
                    "forecast_periods": periods,
                },
            }

            return complete_forecast

        except Exception as e:
            raise RuntimeError(f"Complete forecast generation failed: {str(e)}")

    def get_model_status(self):
        """Get status of available models"""
        try:
            available_models = self.model_loader.get_available_models()

            status = {
                "total_models": len(available_models),
                "models": {},
                "last_check": datetime.now(),
            }

            # Group models by type
            for model_key, model_info in available_models.items():
                if "metadata" not in model_key:  # Skip metadata files
                    model_parts = model_key.split("_")
                    if len(model_parts) >= 2:
                        model_type = model_parts[0]
                        dataset_type = (
                            model_parts[1] if model_parts[1] != "model" else "general"
                        )

                        status["models"][model_key] = {
                            "type": model_type.upper(),
                            "dataset": (
                                "General"
                                if dataset_type == "general"
                                else "Seasonally Adjusted"
                            ),
                            "last_modified": model_info["last_modified"],
                            "size_kb": model_info["size_kb"],
                            "status": "Available",
                        }

            return status

        except Exception as e:
            return {
                "total_models": 0,
                "models": {},
                "error": str(e),
                "last_check": datetime.now(),
            }

    def validate_model_availability(self, model_type, dataset_type):
        """Check if specific model is available"""
        try:
            available_models = self.model_loader.get_available_models()

            if model_type == "lstm":
                model_key = f"lstm_{dataset_type}_model"
                metadata_key = f"lstm_{dataset_type}_metadata"
                return (
                    model_key in available_models and metadata_key in available_models
                )
            else:
                model_key = f"{model_type}_{dataset_type}"
                return model_key in available_models

        except Exception:
            return False


# Utility functions
def format_forecast_period(periods):
    """Format forecast period for display"""
    period_labels = {1: "1 Month", 3: "3 Months", 6: "6 Months", 12: "1 Year"}
    return period_labels.get(periods, f"{periods} Months")


def calculate_forecast_accuracy_estimate(model_type, periods):
    """Estimate forecast accuracy based on model type and validation results from notebook"""
    base_accuracy = {
        "arima": 83.87,  # Based on notebook MAPE results
        "sarima": 99.08,  # Best performing model
        "lstm": 97.33,  # Good performance
    }

    # Accuracy decreases with longer forecast horizons
    horizon_penalty = {1: 0, 3: -1, 6: -2, 12: -5}

    estimated_accuracy = base_accuracy.get(model_type, 85) + horizon_penalty.get(
        periods, -10
    )
    return max(60, min(99, estimated_accuracy))


def generate_forecast_summary(forecast_data):
    """Generate human-readable forecast summary"""
    current_rate = forecast_data["current_rate"]
    final_rate = forecast_data["forecast_values"][-1]
    trend_direction = forecast_data["trend_direction"]

    change = final_rate - current_rate
    change_desc = f"{abs(change):.1f} percentage points"

    if trend_direction == "Rising":
        summary = f"Unemployment is expected to rise by {change_desc} over the forecast period."
    elif trend_direction == "Falling":
        summary = f"Unemployment is expected to fall by {change_desc} over the forecast period."
    else:
        summary = (
            f"Unemployment is expected to remain relatively stable with minimal change."
        )

    # Add context based on magnitude
    if abs(change) > 0.5:
        urgency = "significant"
    elif abs(change) > 0.2:
        urgency = "moderate"
    else:
        urgency = "minimal"

    full_summary = f"{summary} This represents a {urgency} change from current levels."

    return full_summary


# Example usage and testing
if __name__ == "__main__":
    print("Testing Forecast Model Loader...")

    # Test model availability check
    loader = ForecastModelLoader()
    available = loader.get_available_models()

    print(f"Available models: {len(available)}")
    for model_key, model_info in available.items():
        print(f"  - {model_key}: {model_info['last_modified']}")

"""
Dashboard data management with fallback capabilities.
Handles real DOSM data and fallback synthetic data.
"""

import pandas as pd
import numpy as np
from datetime import datetime
from pathlib import Path
import sys

# Import path setup
current_dir = Path(__file__).parent
project_root = current_dir.parent.parent.parent
sys.path.insert(0, str(project_root / "src"))

try:
    from data.data_loader import DataManager
    from analysis.statistical_tests import (
        quick_stationarity_test,
        quick_normality_test,
        full_series_analysis,
    )

    REAL_DATA_AVAILABLE = True
except ImportError:
    REAL_DATA_AVAILABLE = False


class DashboardDataManager:
    """Enhanced data manager with real data integration and fallback"""

    def __init__(self):
        self.datasets = {}
        self.initialized = False
        self.data_source = "fallback"

        if REAL_DATA_AVAILABLE:
            self.real_data_manager = DataManager()

    def initialize(self, force_refresh=False):
        """Initialize data with real DOSM data or fallback"""
        if REAL_DATA_AVAILABLE:
            try:
                success = self.real_data_manager.initialize(force_refresh)
                if success:
                    self.datasets = self.real_data_manager.datasets
                    self.initialized = True
                    self.data_source = "dosm_api"
                    print("✅ Real DOSM data loaded successfully")
                    return True
            except Exception as e:
                print(f"⚠️ Real data failed, using fallback: {e}")

        # Fallback to synthetic data
        return self._initialize_fallback_data()

    def _initialize_fallback_data(self):
        """Generate professional fallback data"""
        try:
            dates = pd.date_range("2010-01-01", "2025-02-01", freq="MS")
            np.random.seed(42)

            # Overall unemployment with realistic patterns
            n = len(dates)
            base_rate = 3.4
            trend = 0.1 * np.sin(np.arange(n) * 2 * np.pi / 120)
            seasonal = 0.3 * np.sin(np.arange(n) * 2 * np.pi / 12)
            noise = 0.2 * np.random.randn(n)
            u_rate = base_rate + trend + seasonal + noise

            lf = 15000 + 50 * np.arange(n) + 100 * np.random.randn(n)
            lf_unemployed = lf * u_rate / 100
            lf_employed = lf - lf_unemployed
            p_rate = (
                67.8
                + 0.2 * np.sin(np.arange(n) * 2 * np.pi / 12)
                + 0.1 * np.random.randn(n)
            )

            self.datasets["Overall Unemployment"] = pd.DataFrame(
                {
                    "u_rate": u_rate,
                    "lf": lf,
                    "lf_employed": lf_employed,
                    "lf_unemployed": lf_unemployed,
                    "p_rate": p_rate,
                    "ep_ratio": p_rate - u_rate,
                },
                index=dates,
            )

            # Youth unemployment
            youth_dates = pd.date_range("2016-01-01", "2025-02-01", freq="MS")
            n_youth = len(youth_dates)
            u_rate_15_24 = (
                11.5
                + 0.8 * np.sin(np.arange(n_youth) * 2 * np.pi / 12)
                + 0.3 * np.random.randn(n_youth)
            )
            u_rate_15_30 = (
                7.2
                + 0.6 * np.sin(np.arange(n_youth) * 2 * np.pi / 12)
                + 0.2 * np.random.randn(n_youth)
            )

            self.datasets["Youth Unemployment"] = pd.DataFrame(
                {
                    "u_rate_15_24": u_rate_15_24,
                    "u_rate_15_30": u_rate_15_30,
                    "unemployed_15_24": 300 + 20 * np.random.randn(n_youth),
                    "unemployed_15_30": 450 + 30 * np.random.randn(n_youth),
                },
                index=youth_dates,
            )

            # Seasonally adjusted
            u_rate_sa = base_rate + trend + 0.15 * np.random.randn(n)
            lf_sa = 15000 + 50 * np.arange(n) + 80 * np.random.randn(n)

            self.datasets["Seasonally Adjusted"] = pd.DataFrame(
                {
                    "u_rate": u_rate_sa,
                    "lf": lf_sa,
                    "lf_employed": lf_sa * (1 - u_rate_sa / 100),
                    "lf_unemployed": lf_sa * u_rate_sa / 100,
                    "p_rate": 67.8 + 0.1 * np.random.randn(n),
                },
                index=dates,
            )

            self.initialized = True
            self.data_source = "synthetic"
            return True

        except Exception as e:
            print(f"❌ Fallback data generation failed: {e}")
            return False

    def get_dataset(self, name):
        """Get dataset by name"""
        if name not in self.datasets:
            raise ValueError(f"Dataset {name} not found")
        return self.datasets[name]

    def get_available_datasets(self):
        """Get list of available dataset names"""
        return list(self.datasets.keys())

    def get_numeric_columns(self, dataset_name):
        """Get numeric columns from dataset"""
        df = self.get_dataset(dataset_name)
        return df.select_dtypes(include=[np.number]).columns.tolist()

    def get_latest_metrics(self):
        """Get latest key metrics for dashboard"""
        if not self.initialized:
            return {}

        metrics = {}
        try:
            if "Overall Unemployment" in self.datasets:
                df = self.datasets["Overall Unemployment"]
                metrics.update(
                    {
                        "unemployment_rate": df["u_rate"].iloc[-1],
                        "labor_force": df["lf"].iloc[-1],
                        "participation_rate": df["p_rate"].iloc[-1],
                        "employment_ratio": df["ep_ratio"].iloc[-1],
                    }
                )

            if "Youth Unemployment" in self.datasets:
                youth_df = self.datasets["Youth Unemployment"]
                metrics["youth_unemployment"] = youth_df["u_rate_15_24"].iloc[-1]

            metrics["data_source"] = self.data_source
            metrics["last_updated"] = datetime.now().strftime("%H:%M:%S")

        except Exception as e:
            print(f"Error getting metrics: {e}")

        return metrics

    def run_statistical_analysis(self, dataset_name, variable_name, test_type="full"):
        """Run statistical analysis on dataset variable"""
        if not REAL_DATA_AVAILABLE:
            return self._fallback_statistical_analysis(
                dataset_name, variable_name, test_type
            )

        try:
            df = self.get_dataset(dataset_name)
            series = df[variable_name]

            if test_type == "stationarity":
                return quick_stationarity_test(series)
            elif test_type == "normality":
                return quick_normality_test(series)
            else:
                return full_series_analysis(series, f"{dataset_name} - {variable_name}")

        except Exception as e:
            return {"error": f"Analysis failed: {str(e)}"}

    def _fallback_statistical_analysis(self, dataset_name, variable_name, test_type):
        """Fallback statistical analysis without external dependencies"""
        try:
            df = self.get_dataset(dataset_name)
            series = df[variable_name].dropna()

            if test_type == "stationarity":
                # Simple trend test
                x = np.arange(len(series))
                slope = np.polyfit(x, series, 1)[0]
                has_trend = abs(slope) > 0.01

                return {
                    "combined_analysis": {
                        "conclusion": "Non-stationary" if has_trend else "Stationary",
                        "confidence": "Medium",
                        "recommendation": (
                            "Apply differencing"
                            if has_trend
                            else "Series suitable for modeling"
                        ),
                    }
                }

            elif test_type == "normality":
                # Simple normality check using skewness and kurtosis
                skewness = series.skew()
                kurtosis = series.kurtosis()
                is_normal = abs(skewness) < 0.5 and abs(kurtosis) < 3

                return {
                    "consensus": "Normal" if is_normal else "Non-normal",
                    "confidence": "Medium",
                    "recommendation": (
                        "Parametric methods suitable"
                        if is_normal
                        else "Consider transformation"
                    ),
                }

            else:
                # Full analysis fallback
                return {
                    "series_info": {
                        "name": f"{dataset_name} - {variable_name}",
                        "valid_observations": len(series),
                        "missing_values": df[variable_name].isna().sum(),
                        "date_range": {
                            "start": series.index.min(),
                            "end": series.index.max(),
                        },
                    },
                    "descriptive_statistics": {
                        "mean": series.mean(),
                        "std": series.std(),
                        "min": series.min(),
                        "max": series.max(),
                        "count": len(series),
                    },
                    "stationarity_analysis": self._fallback_statistical_analysis(
                        dataset_name, variable_name, "stationarity"
                    ),
                    "normality_analysis": self._fallback_statistical_analysis(
                        dataset_name, variable_name, "normality"
                    ),
                    "recommendations": ["Data suitable for basic time series analysis"],
                }

        except Exception as e:
            return {"error": f"Fallback analysis failed: {str(e)}"}

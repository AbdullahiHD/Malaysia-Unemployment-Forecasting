# import pandas as pd
# import numpy as np
# from datetime import datetime, timedelta
# from pathlib import Path
# import sys

# # Import path setup
# current_dir = Path(__file__).parent
# project_root = current_dir.parent.parent.parent
# sys.path.insert(0, str(project_root / "src"))

# try:
#     from data.data_loader import DataManager
#     from analysis.statistical_tests import (
#         quick_stationarity_test,
#         quick_normality_test,
#         full_series_analysis,
#     )

#     REAL_DATA_AVAILABLE = True
# except ImportError:
#     REAL_DATA_AVAILABLE = False


# class DashboardDataManager:
#     def __init__(self, update_interval_days: int = 28, auto_update: bool = True):
#         """
#         Initialize dashboard data manager with enhanced update capabilities.

#         Args:
#             update_interval_days: Days between automatic data updates (default: 28)
#             auto_update: Enable automatic background updates (default: True)
#         """
#         self.datasets = {}
#         self.initialized = False
#         self.data_source = "fallback"
#         self.update_interval_days = update_interval_days
#         self.auto_update = auto_update

#         if REAL_DATA_AVAILABLE:
#             # Pass update parameters to the real data manager
#             self.real_data_manager = DataManager(
#                 update_interval_days=update_interval_days, auto_update=auto_update
#             )

#     def initialize(self, force_refresh=False):
#         """
#         Initialize data with real DOSM data or fallback.
#         ENHANCED: Now supports forced refresh and automatic update checking.

#         Args:
#             force_refresh: Force fresh download ignoring cache and time checks

#         Returns:
#             True if initialization successful
#         """
#         if REAL_DATA_AVAILABLE:
#             try:
#                 success = self.real_data_manager.initialize(force_refresh)
#                 if success:
#                     self.datasets = self.real_data_manager.datasets
#                     self.initialized = True
#                     self.data_source = "dosm_api"

#                     # Check if we have April 2025 data
#                     self._check_data_freshness()
#                     return True
#             except Exception as e:
#                 print(f"Real data failed, using fallback: {e}")

#         # Fallback to synthetic data
#         return self._initialize_fallback_data()

#     def _check_data_freshness(self):
#         """
#         NEW: Check if loaded data includes the latest expected months.
#         """
#         try:
#             if "Overall Unemployment" in self.datasets:
#                 df = self.datasets["Overall Unemployment"]
#                 latest_date = df.index.max()
#                 current_date = datetime.now()

#                 print(f"Latest data available: {latest_date.strftime('%B %Y')}")

#                 # Check if we have current month's data
#                 if (
#                     latest_date.year == current_date.year
#                     and latest_date.month >= current_date.month - 1
#                 ):
#                     print("🎉 Data is up-to-date with latest available period")
#                 elif latest_date.year == 2025 and latest_date.month >= 4:
#                     print("🎉 April 2025 data successfully loaded!")
#                 else:
#                     expected = current_date.replace(day=1) - timedelta(
#                         days=1
#                     )  # Previous month
#                     print(
#                         f"⚠️ Data may not be current. Expected: {expected.strftime('%B %Y')}, Got: {latest_date.strftime('%B %Y')}"
#                     )

#         except Exception as e:
#             print(f"Could not check data freshness: {e}")

#     def force_update(self):
#         """
#         NEW: Manually force an immediate update of all datasets.

#         Returns:
#             True if update successful
#         """
#         print("Forcing immediate data update...")

#         if REAL_DATA_AVAILABLE and hasattr(self.real_data_manager, "force_update"):
#             try:
#                 success = self.real_data_manager.force_update()
#                 if success:
#                     # Reload datasets
#                     self.datasets = self.real_data_manager.datasets
#                     self._check_data_freshness()
#                     print("Force update completed successfully")
#                     return True
#                 else:
#                     print("Force update failed")
#                     return False
#             except Exception as e:
#                 print(f"Force update error: {e}")
#                 return False
#         else:
#             # Fallback: re-initialize with force refresh
#             return self.initialize(force_refresh=True)

#     def get_update_status(self):
#         """
#         NEW: Get comprehensive update status for dashboard display.

#         Returns:
#             Dictionary with update status information
#         """
#         status = {
#             "data_source": self.data_source,
#             "auto_update_enabled": self.auto_update,
#             "update_interval_days": self.update_interval_days,
#             "initialized": self.initialized,
#             "last_check": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
#         }

#         if REAL_DATA_AVAILABLE and hasattr(self.real_data_manager, "get_update_status"):
#             try:
#                 real_status = self.real_data_manager.get_update_status()
#                 status.update(real_status)
#             except Exception as e:
#                 status["error"] = f"Could not get real data status: {e}"

#         # Add dataset freshness info
#         if self.initialized and "Overall Unemployment" in self.datasets:
#             try:
#                 df = self.datasets["Overall Unemployment"]
#                 latest_date = df.index.max()
#                 current_date = datetime.now()

#                 days_behind = (current_date.replace(day=1) - latest_date).days

#                 status["data_freshness"] = {
#                     "latest_period": latest_date.strftime("%B %Y"),
#                     "days_behind_current": max(0, days_behind),
#                     "is_current": days_behind
#                     <= 45,  # Within 1.5 months is considered current
#                     "has_april_2025": latest_date.year == 2025
#                     and latest_date.month >= 4,
#                 }
#             except Exception as e:
#                 status["data_freshness"] = {"error": str(e)}

#         return status

#     def get_dataset(self, name):
#         """Get dataset by name. UNCHANGED."""
#         if name not in self.datasets:
#             raise ValueError(f"Dataset {name} not found")
#         return self.datasets[name]

#     def get_available_datasets(self):
#         """Get list of available dataset names. UNCHANGED."""
#         return list(self.datasets.keys())

#     def get_numeric_columns(self, dataset_name):
#         """Get numeric columns from dataset. UNCHANGED."""
#         df = self.get_dataset(dataset_name)
#         return df.select_dtypes(include=[np.number]).columns.tolist()

#     def get_latest_metrics(self):
#         """
#         Get latest key metrics for dashboard.
#         ENHANCED: Now includes update status and data freshness information.
#         """
#         if not self.initialized:
#             return {}

#         metrics = {}
#         try:
#             if "Overall Unemployment" in self.datasets:
#                 df = self.datasets["Overall Unemployment"]
#                 metrics.update(
#                     {
#                         "unemployment_rate": df["u_rate"].iloc[-1],
#                         "labor_force": df["lf"].iloc[-1],
#                         "participation_rate": df["p_rate"].iloc[-1],
#                         "employment_ratio": df["ep_ratio"].iloc[-1],
#                     }
#                 )

#             if "Youth Unemployment" in self.datasets:
#                 youth_df = self.datasets["Youth Unemployment"]
#                 metrics["youth_unemployment"] = youth_df["u_rate_15_24"].iloc[-1]

#             # Enhanced metadata
#             metrics["data_source"] = self.data_source
#             metrics["last_updated"] = datetime.now().strftime("%H:%M:%S")

#             # NEW: Add update status summary
#             update_status = self.get_update_status()
#             metrics["auto_update_enabled"] = update_status.get(
#                 "auto_update_enabled", False
#             )
#             metrics["update_interval_days"] = update_status.get(
#                 "update_interval_days", 28
#             )

#             # NEW: Add data freshness info
#             if "data_freshness" in update_status:
#                 freshness = update_status["data_freshness"]
#                 metrics["latest_period"] = freshness.get("latest_period", "Unknown")
#                 metrics["data_is_current"] = freshness.get("is_current", False)
#                 metrics["has_april_2025"] = freshness.get("has_april_2025", False)

#         except Exception as e:
#             print(f"Error getting metrics: {e}")

#         return metrics

#     def run_statistical_analysis(self, dataset_name, variable_name, test_type="full"):
#         """
#         Run statistical analysis on dataset variable.
#         UNCHANGED: Maintains existing functionality.
#         """
#         if not REAL_DATA_AVAILABLE:
#             return self._fallback_statistical_analysis(
#                 dataset_name, variable_name, test_type
#             )

#         try:
#             df = self.get_dataset(dataset_name)
#             series = df[variable_name]

#             if test_type == "stationarity":
#                 return quick_stationarity_test(series)
#             elif test_type == "normality":
#                 return quick_normality_test(series)
#             else:
#                 return full_series_analysis(series, f"{dataset_name} - {variable_name}")

#         except Exception as e:
#             return {"error": f"Analysis failed: {str(e)}"}

#     def _fallback_statistical_analysis(self, dataset_name, variable_name, test_type):
#         """
#         Fallback statistical analysis without external dependencies.
#         UNCHANGED: Maintains existing functionality.
#         """
#         try:
#             df = self.get_dataset(dataset_name)
#             series = df[variable_name].dropna()

#             if test_type == "stationarity":
#                 # Simple trend test
#                 x = np.arange(len(series))
#                 slope = np.polyfit(x, series, 1)[0]
#                 has_trend = abs(slope) > 0.01

#                 return {
#                     "combined_analysis": {
#                         "conclusion": "Non-stationary" if has_trend else "Stationary",
#                         "confidence": "Medium",
#                         "recommendation": (
#                             "Apply differencing"
#                             if has_trend
#                             else "Series suitable for modeling"
#                         ),
#                     }
#                 }

#             elif test_type == "normality":
#                 # Simple normality check using skewness and kurtosis
#                 skewness = series.skew()
#                 kurtosis = series.kurtosis()
#                 is_normal = abs(skewness) < 0.5 and abs(kurtosis) < 3

#                 return {
#                     "consensus": "Normal" if is_normal else "Non-normal",
#                     "confidence": "Medium",
#                     "recommendation": (
#                         "Parametric methods suitable"
#                         if is_normal
#                         else "Consider transformation"
#                     ),
#                 }

#             else:
#                 # Full analysis fallback
#                 return {
#                     "series_info": {
#                         "name": f"{dataset_name} - {variable_name}",
#                         "valid_observations": len(series),
#                         "missing_values": df[variable_name].isna().sum(),
#                         "date_range": {
#                             "start": series.index.min(),
#                             "end": series.index.max(),
#                         },
#                     },
#                     "descriptive_statistics": {
#                         "mean": series.mean(),
#                         "std": series.std(),
#                         "min": series.min(),
#                         "max": series.max(),
#                         "count": len(series),
#                     },
#                     "stationarity_analysis": self._fallback_statistical_analysis(
#                         dataset_name, variable_name, "stationarity"
#                     ),
#                     "normality_analysis": self._fallback_statistical_analysis(
#                         dataset_name, variable_name, "normality"
#                     ),
#                     "recommendations": ["Data suitable for basic time series analysis"],
#                 }

#         except Exception as e:
#             return {"error": f"Fallback analysis failed: {str(e)}"}

#     def stop_auto_updates(self):
#         """
#         NEW: Stop automatic background updates.
#         """
#         if REAL_DATA_AVAILABLE and hasattr(self.real_data_manager, "stop_auto_updates"):
#             self.real_data_manager.stop_auto_updates()
#         self.auto_update = False
#         print("Auto-updates disabled")

#     def start_auto_updates(self):
#         """
#         NEW: Start automatic background updates.
#         """
#         if REAL_DATA_AVAILABLE and hasattr(
#             self.real_data_manager, "start_auto_updates"
#         ):
#             self.real_data_manager.start_auto_updates()
#         self.auto_update = True
#         print("▶Auto-updates enabled")

#     def get_system_info(self):
#         """
#         NEW: Get comprehensive system information for debugging and monitoring.

#         Returns:
#             Dictionary with system status information
#         """
#         info = {
#             "real_data_available": REAL_DATA_AVAILABLE,
#             "initialized": self.initialized,
#             "data_source": self.data_source,
#             "auto_update_enabled": self.auto_update,
#             "update_interval_days": self.update_interval_days,
#             "datasets_loaded": len(self.datasets),
#             "available_datasets": list(self.datasets.keys()),
#         }

#         # Add dataset shapes and date ranges
#         if self.initialized:
#             info["dataset_details"] = {}
#             for name, df in self.datasets.items():
#                 info["dataset_details"][name] = {
#                     "shape": df.shape,
#                     "date_range": f"{df.index.min()} to {df.index.max()}",
#                     "latest_value": df.iloc[-1].to_dict() if len(df) > 0 else None,
#                 }

#         # Add update status if available
#         try:
#             update_status = self.get_update_status()
#             info["update_status"] = update_status
#         except Exception as e:
#             info["update_status_error"] = str(e)

#         return info


# def test_enhanced_system():

#     # Test with 1-day update interval for rapid testing
#     manager = DashboardDataManager(update_interval_days=1, auto_update=True)

#     print("1. Testing initialization...")
#     success = manager.initialize(force_refresh=True)
#     print(f"   Initialization: {'Success' if success else 'Failed'}")

#     print("\n2. Testing data availability...")
#     datasets = manager.get_available_datasets()
#     print(f"   Available datasets: {len(datasets)}")
#     for dataset in datasets:
#         df = manager.get_dataset(dataset)
#         print(f"   - {dataset}: {df.shape}")

#     print("\n3. Testing latest metrics...")
#     metrics = manager.get_latest_metrics()
#     print(f"   Data source: {metrics.get('data_source')}")
#     print(f"   Latest period: {metrics.get('latest_period', 'Unknown')}")
#     print(f"   Has April 2025: {metrics.get('has_april_2025', False)}")
#     print(f"   Unemployment rate: {metrics.get('unemployment_rate', 'N/A'):.1f}%")

#     print("\n4. Testing update status...")
#     status = manager.get_update_status()
#     print(f"   Auto-update enabled: {status.get('auto_update_enabled')}")
#     print(f"   Update interval: {status.get('update_interval_days')} days")

#     if "data_freshness" in status:
#         freshness = status["data_freshness"]
#         print(f"   Latest period: {freshness.get('latest_period')}")
#         print(f"   Data is current: {freshness.get('is_current')}")

#     print("\n5. Testing force update...")
#     update_success = manager.force_update()
#     print(f"   Force update: {'Success' if update_success else 'Failed'}")

#     print("\n6. Testing system info...")
#     system_info = manager.get_system_info()
#     print(f"   Real data available: {system_info.get('real_data_available')}")
#     print(f"   Datasets loaded: {system_info.get('datasets_loaded')}")

#     print("\n🎉 Enhanced system test completed!")
#     return manager


# if __name__ == "__main__":
#     # Run the enhanced system test
#     test_manager = test_enhanced_system()

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
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


class ColumnDisplayManager:
    """
    Professional column name mapping for unemployment data display.
    """

    def __init__(self):
        self.column_mappings = {
            # Overall unemployment metrics
            "u_rate": "Unemployment Rate (%)",
            "lf": "Total Labor Force (thousands)",
            "lf_employed": "Employed Persons (thousands)",
            "lf_unemployed": "Unemployed Persons (thousands)",
            "lf_outside": "Persons Outside Labor Force (thousands)",
            "p_rate": "Labor Force Participation Rate (%)",
            "ep_ratio": "Employment-to-Population Ratio (%)",
            "employment_rate": "Employment Rate (%)",
            # Youth unemployment metrics
            "u_rate_15_24": "Youth Unemployment Rate 15-24 (%)",
            "u_rate_15_30": "Youth Unemployment Rate 15-30 (%)",
            "unemployed_15_24": "Unemployed Youth 15-24 (thousands)",
            "unemployed_15_30": "Unemployed Youth 15-30 (thousands)",
            # Duration metrics
            "unemployed": "Total Unemployed (thousands)",
            "unemployed_active": "Actively Job Searching (thousands)",
            "unemployed_active_3mo": "Active Job Search Under 3 Months (thousands)",
            "unemployed_active_6mo": "Active Job Search Under 6 Months (thousands)",
            "unemployed_passive": "Passive Job Seekers (thousands)",
            # Employment status metrics
            "employed": "Total Employed (thousands)",
            "employed_employer": "Employers (thousands)",
            "employed_employee": "Employees (thousands)",
            "employed_own_account": "Own Account Workers (thousands)",
            "employed_unpaid_family": "Unpaid Family Workers (thousands)",
            # Seasonally adjusted variants
            "u_rate_sa": "Unemployment Rate - Seasonally Adjusted (%)",
            "lf_sa": "Labor Force - Seasonally Adjusted (thousands)",
            "lf_employed_sa": "Employed - Seasonally Adjusted (thousands)",
            "lf_unemployed_sa": "Unemployed - Seasonally Adjusted (thousands)",
            "p_rate_sa": "Participation Rate - Seasonally Adjusted (%)",
        }

        self.column_categories = {
            "Overall Labor Market": [
                "u_rate",
                "lf",
                "lf_employed",
                "lf_unemployed",
                "p_rate",
                "ep_ratio",
            ],
            "Youth Employment": [
                "u_rate_15_24",
                "u_rate_15_30",
                "unemployed_15_24",
                "unemployed_15_30",
            ],
            "Unemployment Duration": [
                "unemployed",
                "unemployed_active",
                "unemployed_active_3mo",
                "unemployed_active_6mo",
                "unemployed_passive",
            ],
            "Employment Status": [
                "employed",
                "employed_employer",
                "employed_employee",
                "employed_own_account",
                "employed_unpaid_family",
            ],
            "Seasonally Adjusted": [
                "u_rate_sa",
                "lf_sa",
                "lf_employed_sa",
                "lf_unemployed_sa",
                "p_rate_sa",
            ],
        }

    def get_display_name(self, column_name):
        """Get user-friendly display name for a column."""
        return self.column_mappings.get(
            column_name, column_name.replace("_", " ").title()
        )

    def create_dropdown_options(self, columns, group_by_category=False):
        """Create formatted options for dropdown components."""
        if group_by_category:
            categorized_options = []

            for category, category_columns in self.column_categories.items():
                available_in_category = [
                    col for col in category_columns if col in columns
                ]

                if available_in_category:
                    categorized_options.append(
                        {
                            "label": category,
                            "options": [
                                {"label": self.get_display_name(col), "value": col}
                                for col in available_in_category
                            ],
                        }
                    )

            # Add uncategorized columns
            categorized_columns = [
                col for cols in self.column_categories.values() for col in cols
            ]
            uncategorized = [col for col in columns if col not in categorized_columns]

            if uncategorized:
                categorized_options.append(
                    {
                        "label": "Other Metrics",
                        "options": [
                            {"label": self.get_display_name(col), "value": col}
                            for col in uncategorized
                        ],
                    }
                )

            return categorized_options
        else:
            return [
                {"label": self.get_display_name(col), "value": col} for col in columns
            ]


class DashboardDataManager:
    def __init__(self, update_interval_days: int = 28, auto_update: bool = True):
        """
        Initialize dashboard data manager with enhanced update capabilities.

        Args:
            update_interval_days: Days between automatic data updates (default: 28)
            auto_update: Enable automatic background updates (default: True)
        """
        self.datasets = {}
        self.initialized = False
        self.data_source = "fallback"
        self.update_interval_days = update_interval_days
        self.auto_update = auto_update

        # NEW: Initialize column display manager
        self.column_manager = ColumnDisplayManager()

        if REAL_DATA_AVAILABLE:
            # Pass update parameters to the real data manager
            self.real_data_manager = DataManager(
                update_interval_days=update_interval_days, auto_update=auto_update
            )

    def initialize(self, force_refresh=False):
        """
        Initialize data with real DOSM data or fallback.
        ENHANCED: Now supports forced refresh and automatic update checking.

        Args:
            force_refresh: Force fresh download ignoring cache and time checks

        Returns:
            True if initialization successful
        """
        if REAL_DATA_AVAILABLE:
            try:
                success = self.real_data_manager.initialize(force_refresh)
                if success:
                    self.datasets = self.real_data_manager.datasets
                    self.initialized = True
                    self.data_source = "dosm_api"

                    # Check if we have April 2025 data
                    self._check_data_freshness()
                    return True
            except Exception as e:
                print(f"Real data failed, using fallback: {e}")

        # Fallback to synthetic data
        return self._initialize_fallback_data()

    def _check_data_freshness(self):
        """
        NEW: Check if loaded data includes the latest expected months.
        """
        try:
            if "Overall Unemployment" in self.datasets:
                df = self.datasets["Overall Unemployment"]
                latest_date = df.index.max()
                current_date = datetime.now()

                print(f"Latest data available: {latest_date.strftime('%B %Y')}")

                # Check if we have current month's data
                if (
                    latest_date.year == current_date.year
                    and latest_date.month >= current_date.month - 1
                ):
                    print("Data is up-to-date with latest available period")
                elif latest_date.year == 2025 and latest_date.month >= 4:
                    print("April 2025 data successfully loaded!")
                else:
                    expected = current_date.replace(day=1) - timedelta(
                        days=1
                    )  # Previous month
                    print(
                        f"Data may not be current. Expected: {expected.strftime('%B %Y')}, Got: {latest_date.strftime('%B %Y')}"
                    )

        except Exception as e:
            print(f"Could not check data freshness: {e}")

    def force_update(self):
        """
        NEW: Manually force an immediate update of all datasets.

        Returns:
            True if update successful
        """
        print("Forcing immediate data update...")

        if REAL_DATA_AVAILABLE and hasattr(self.real_data_manager, "force_update"):
            try:
                success = self.real_data_manager.force_update()
                if success:
                    # Reload datasets
                    self.datasets = self.real_data_manager.datasets
                    self._check_data_freshness()
                    print("Force update completed successfully")
                    return True
                else:
                    print("Force update failed")
                    return False
            except Exception as e:
                print(f"Force update error: {e}")
                return False
        else:
            # Fallback: re-initialize with force refresh
            return self.initialize(force_refresh=True)

    def get_update_status(self):
        """
        NEW: Get comprehensive update status for dashboard display.

        Returns:
            Dictionary with update status information
        """
        status = {
            "data_source": self.data_source,
            "auto_update_enabled": self.auto_update,
            "update_interval_days": self.update_interval_days,
            "initialized": self.initialized,
            "last_check": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        }

        if REAL_DATA_AVAILABLE and hasattr(self.real_data_manager, "get_update_status"):
            try:
                real_status = self.real_data_manager.get_update_status()
                status.update(real_status)
            except Exception as e:
                status["error"] = f"Could not get real data status: {e}"

        # Add dataset freshness info
        if self.initialized and "Overall Unemployment" in self.datasets:
            try:
                df = self.datasets["Overall Unemployment"]
                latest_date = df.index.max()
                current_date = datetime.now()

                days_behind = (current_date.replace(day=1) - latest_date).days

                status["data_freshness"] = {
                    "latest_period": latest_date.strftime("%B %Y"),
                    "days_behind_current": max(0, days_behind),
                    "is_current": days_behind
                    <= 45,  # Within 1.5 months is considered current
                    "has_april_2025": latest_date.year == 2025
                    and latest_date.month >= 4,
                }
            except Exception as e:
                status["data_freshness"] = {"error": str(e)}

        return status

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
            print(f"Fallback data generation failed: {e}")
            return False

    def get_dataset(self, name):
        """Get dataset by name. UNCHANGED."""
        if name not in self.datasets:
            raise ValueError(f"Dataset {name} not found")
        return self.datasets[name]

    def get_available_datasets(self):
        """Get list of available dataset names. UNCHANGED."""
        return list(self.datasets.keys())

    def get_numeric_columns(self, dataset_name):
        """Get numeric columns from dataset. UNCHANGED."""
        df = self.get_dataset(dataset_name)
        return df.select_dtypes(include=[np.number]).columns.tolist()

    # NEW: Column mapping methods
    def get_column_display_options(self, dataset_name, group_by_category=False):
        """
        Get formatted column options for UI dropdowns with professional display names.

        Args:
            dataset_name: Dataset name
            group_by_category: Whether to group options by category

        Returns:
            List of formatted options for dropdown components
        """
        numeric_columns = self.get_numeric_columns(dataset_name)
        return self.column_manager.create_dropdown_options(
            numeric_columns, group_by_category=group_by_category
        )

    def get_column_display_name(self, column_name):
        """Get user-friendly display name for a column."""
        return self.column_manager.get_display_name(column_name)

    def get_latest_metrics(self):
        """
        Get latest key metrics for dashboard.
        ENHANCED: Now includes update status and data freshness information.
        """
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

            # Enhanced metadata
            metrics["data_source"] = self.data_source
            metrics["last_updated"] = datetime.now().strftime("%H:%M:%S")

            # NEW: Add update status summary
            update_status = self.get_update_status()
            metrics["auto_update_enabled"] = update_status.get(
                "auto_update_enabled", False
            )
            metrics["update_interval_days"] = update_status.get(
                "update_interval_days", 28
            )

            # NEW: Add data freshness info
            if "data_freshness" in update_status:
                freshness = update_status["data_freshness"]
                metrics["latest_period"] = freshness.get("latest_period", "Unknown")
                metrics["data_is_current"] = freshness.get("is_current", False)
                metrics["has_april_2025"] = freshness.get("has_april_2025", False)

        except Exception as e:
            print(f"Error getting metrics: {e}")

        return metrics

    def run_statistical_analysis(self, dataset_name, variable_name, test_type="full"):
        """
        Run statistical analysis on dataset variable.
        UNCHANGED: Maintains existing functionality.
        """
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
        """
        Fallback statistical analysis without external dependencies.
        UNCHANGED: Maintains existing functionality.
        """
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

    def stop_auto_updates(self):
        """
        NEW: Stop automatic background updates.
        """
        if REAL_DATA_AVAILABLE and hasattr(self.real_data_manager, "stop_auto_updates"):
            self.real_data_manager.stop_auto_updates()
        self.auto_update = False
        print("Auto-updates disabled")

    def start_auto_updates(self):
        """
        NEW: Start automatic background updates.
        """
        if REAL_DATA_AVAILABLE and hasattr(
            self.real_data_manager, "start_auto_updates"
        ):
            self.real_data_manager.start_auto_updates()
        self.auto_update = True
        print("Auto-updates enabled")

    def get_system_info(self):
        """
        NEW: Get comprehensive system information for debugging and monitoring.

        Returns:
            Dictionary with system status information
        """
        info = {
            "real_data_available": REAL_DATA_AVAILABLE,
            "initialized": self.initialized,
            "data_source": self.data_source,
            "auto_update_enabled": self.auto_update,
            "update_interval_days": self.update_interval_days,
            "datasets_loaded": len(self.datasets),
            "available_datasets": list(self.datasets.keys()),
        }

        # Add dataset shapes and date ranges
        if self.initialized:
            info["dataset_details"] = {}
            for name, df in self.datasets.items():
                info["dataset_details"][name] = {
                    "shape": df.shape,
                    "date_range": f"{df.index.min()} to {df.index.max()}",
                    "latest_value": df.iloc[-1].to_dict() if len(df) > 0 else None,
                }

        # Add update status if available
        try:
            update_status = self.get_update_status()
            info["update_status"] = update_status
        except Exception as e:
            info["update_status_error"] = str(e)

        return info


def test_enhanced_system():

    # Test with 1-day update interval for rapid testing
    manager = DashboardDataManager(update_interval_days=1, auto_update=True)

    print("1. Testing initialization...")
    success = manager.initialize(force_refresh=True)
    print(f"   Initialization: {'Success' if success else 'Failed'}")

    print("\n2. Testing data availability...")
    datasets = manager.get_available_datasets()
    print(f"   Available datasets: {len(datasets)}")
    for dataset in datasets:
        df = manager.get_dataset(dataset)
        print(f"   - {dataset}: {df.shape}")

    print("\n3. Testing latest metrics...")
    metrics = manager.get_latest_metrics()
    print(f"   Data source: {metrics.get('data_source')}")
    print(f"   Latest period: {metrics.get('latest_period', 'Unknown')}")
    print(f"   Has April 2025: {metrics.get('has_april_2025', False)}")
    print(f"   Unemployment rate: {metrics.get('unemployment_rate', 'N/A'):.1f}%")

    print("\n4. Testing update status...")
    status = manager.get_update_status()
    print(f"   Auto-update enabled: {status.get('auto_update_enabled')}")
    print(f"   Update interval: {status.get('update_interval_days')} days")

    if "data_freshness" in status:
        freshness = status["data_freshness"]
        print(f"   Latest period: {freshness.get('latest_period')}")
        print(f"   Data is current: {freshness.get('is_current')}")

    print("\n5. Testing force update...")
    update_success = manager.force_update()
    print(f"   Force update: {'Success' if update_success else 'Failed'}")

    print("\n6. Testing system info...")
    system_info = manager.get_system_info()
    print(f"   Real data available: {system_info.get('real_data_available')}")
    print(f"   Datasets loaded: {system_info.get('datasets_loaded')}")

    # NEW: Test column mapping functionality
    print("\n7. Testing column mapping...")
    for dataset_name in manager.get_available_datasets():
        print(f"   Testing {dataset_name}:")

        # Test regular dropdown options
        regular_options = manager.get_column_display_options(
            dataset_name, group_by_category=False
        )
        print(f"   - Regular options: {len(regular_options)}")

        # Test grouped dropdown options
        grouped_options = manager.get_column_display_options(
            dataset_name, group_by_category=True
        )
        print(f"   - Grouped options: {len(grouped_options)} categories")

        # Show sample mappings
        numeric_cols = manager.get_numeric_columns(dataset_name)
        for col in numeric_cols[:3]:  # Show first 3 columns
            display_name = manager.get_column_display_name(col)
            print(f"     {col} -> {display_name}")

        break  # Just test first dataset

    print("\nColumn mapping test completed!")
    return manager


if __name__ == "__main__":
    # Run the enhanced system test
    test_manager = test_enhanced_system()

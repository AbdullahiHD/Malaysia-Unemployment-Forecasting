# """
# Data loader module for Malaysia unemployment forecasting for real-time DOSM data access.
# """

# import pandas as pd
# import numpy as np
# import requests
# import json
# import threading
# import time
# from pathlib import Path
# from typing import Dict, Optional, List, Union
# from datetime import datetime, timedelta
# import warnings

# warnings.filterwarnings("ignore")


# class DOSMDataLoader:
#     def __init__(
#         self,
#         cache_dir: str = "data/raw",
#         enable_cache: bool = True,
#         update_interval_days: int = 28,
#         auto_update: bool = True,
#     ):
#         self.cache_dir = Path(cache_dir)
#         self.cache_dir.mkdir(parents=True, exist_ok=True)
#         self.enable_cache = enable_cache
#         self.update_interval_days = update_interval_days
#         self.auto_update = auto_update

#         self.update_tracker_file = self.cache_dir / ".update_tracker.json"
#         self._update_thread = None
#         self._stop_updates = False

#         # Official DOSM API endpoints
#         self.api_endpoints = {
#             "unemployment_general": "https://storage.dosm.gov.my/labour/lfs_month.parquet",
#             "unemployment_sa": "https://storage.dosm.gov.my/labour/lfs_month_sa.parquet",
#             "youth_unemployment": "https://storage.dosm.gov.my/labour/lfs_month_youth.parquet",
#             "unemployment_duration": "https://storage.dosm.gov.my/labour/lfs_month_duration.parquet",
#             "employment_status": "https://storage.dosm.gov.my/labour/lfs_month_status.parquet",
#         }

#         # Dataset metadata for user-friendly display (unchanged)
#         self.dataset_metadata = {
#             "unemployment_general": {
#                 "name": "Overall Unemployment",
#                 "description": "General labor force statistics including unemployment rate, labor force participation",
#                 "key_columns": [
#                     "u_rate",
#                     "lf",
#                     "lf_employed",
#                     "lf_unemployed",
#                     "p_rate",
#                     "ep_ratio",
#                 ],
#                 "frequency": "Monthly",
#                 "start_year": 2010,
#             },
#             "unemployment_sa": {
#                 "name": "Seasonally Adjusted",
#                 "description": "Seasonally adjusted unemployment and labor force data",
#                 "key_columns": [
#                     "u_rate",
#                     "lf",
#                     "lf_employed",
#                     "lf_unemployed",
#                     "p_rate",
#                 ],
#                 "frequency": "Monthly",
#                 "start_year": 2010,
#             },
#             "youth_unemployment": {
#                 "name": "Youth Unemployment",
#                 "description": "Unemployment statistics for age groups 15-24 and 15-30",
#                 "key_columns": [
#                     "u_rate_15_24",
#                     "u_rate_15_30",
#                     "unemployed_15_24",
#                     "unemployed_15_30",
#                 ],
#                 "frequency": "Monthly",
#                 "start_year": 2016,
#             },
#             "unemployment_duration": {
#                 "name": "Unemployment Duration",
#                 "description": "Analysis of unemployment duration patterns and job search activity",
#                 "key_columns": [
#                     "unemployed",
#                     "unemployed_active",
#                     "unemployed_active_3mo",
#                     "unemployed_active_6mo",
#                 ],
#                 "frequency": "Monthly",
#                 "start_year": 2016,
#             },
#             "employment_status": {
#                 "name": "Employment Status",
#                 "description": "Employment breakdown by status: employers, employees, self-employed",
#                 "key_columns": [
#                     "employed",
#                     "employed_employer",
#                     "employed_employee",
#                     "employed_own_account",
#                 ],
#                 "frequency": "Monthly",
#                 "start_year": 2016,
#             },
#         }

#         self._datasets = {}
#         self._last_updated = {}

#         # NEW: Start background updater if enabled
#         if self.auto_update:
#             self._start_background_updater()

#     def _should_update_dataset(self, dataset_key: str) -> bool:
#         """
#         NEW: Check if a dataset needs updating based on time interval.

#         Args:
#             dataset_key: Dataset to check

#         Returns:
#             True if update is needed
#         """
#         if not self.update_tracker_file.exists():
#             return True

#         try:
#             with open(self.update_tracker_file, "r") as f:
#                 update_tracker = json.load(f)

#             if dataset_key not in update_tracker:
#                 return True

#             last_update_str = update_tracker[dataset_key].get("last_update")
#             if not last_update_str:
#                 return True

#             last_update = datetime.fromisoformat(last_update_str)
#             days_since_update = (datetime.now() - last_update).days

#             return days_since_update >= self.update_interval_days

#         except Exception as e:
#             print(f"Warning: Error checking update status for {dataset_key}: {e}")
#             return True

#     def _save_update_timestamp(self, dataset_key: str, success: bool = True):
#         """
#         NEW: Save timestamp of last update attempt.

#         Args:
#             dataset_key: Dataset that was updated
#             success: Whether the update was successful
#         """
#         try:
#             # Load existing tracker or create new
#             if self.update_tracker_file.exists():
#                 with open(self.update_tracker_file, "r") as f:
#                     update_tracker = json.load(f)
#             else:
#                 update_tracker = {}

#             # Update entry
#             update_tracker[dataset_key] = {
#                 "last_update": datetime.now().isoformat(),
#                 "success": success,
#                 "update_interval_days": self.update_interval_days,
#             }

#             # Save tracker
#             with open(self.update_tracker_file, "w") as f:
#                 json.dump(update_tracker, f, indent=2)

#         except Exception as e:
#             print(f"Warning: Could not save update timestamp for {dataset_key}: {e}")

#     def _start_background_updater(self):
#         """
#         NEW: Start background thread for automatic updates.
#         """
#         if self._update_thread and self._update_thread.is_alive():
#             return

#         def update_loop():
#             while not self._stop_updates:
#                 try:
#                     # Sleep for 6 hours, then check all datasets
#                     time.sleep(6 * 60 * 60)  # 6 hours

#                     if self._stop_updates:
#                         break

#                     # Check each dataset for updates needed
#                     for dataset_key in self.api_endpoints.keys():
#                         if self._should_update_dataset(dataset_key):
#                             try:
#                                 print(f"🔄 Background update: {dataset_key}")
#                                 self.load_dataset(dataset_key, force_refresh=True)
#                                 print(f"✅ Background update completed: {dataset_key}")
#                             except Exception as e:
#                                 print(
#                                     f"⚠️ Background update failed for {dataset_key}: {e}"
#                                 )

#                 except Exception as e:
#                     print(f"Warning: Background updater error: {e}")

#         self._update_thread = threading.Thread(target=update_loop, daemon=True)
#         self._update_thread.start()
#         print("🔄 Background data updater started")

#     def load_dataset(
#         self, dataset_key: str, force_refresh: bool = False
#     ) -> pd.DataFrame:
#         """
#         Load a specific dataset from DOSM API with intelligent caching and auto-updates.
#         ENHANCED: Now checks for automatic updates based on time interval.

#         Args:
#             dataset_key: Key for the dataset to load
#             force_refresh: Force fresh download ignoring cache and time checks

#         Returns:
#             Preprocessed DataFrame

#         Raises:
#             ValueError: If dataset_key is invalid
#             ConnectionError: If API is unreachable
#         """
#         if dataset_key not in self.api_endpoints:
#             available_keys = list(self.api_endpoints.keys())
#             raise ValueError(
#                 f"Invalid dataset key '{dataset_key}'. Available: {available_keys}"
#             )

#         cache_file = self.cache_dir / f"{dataset_key}.parquet"

#         # NEW: Check if update is needed (unless force_refresh is True)
#         needs_update = force_refresh or self._should_update_dataset(dataset_key)

#         # Try cache first (if enabled, not forcing refresh, and no update needed)
#         if self.enable_cache and not needs_update and cache_file.exists():
#             try:
#                 df = pd.read_parquet(cache_file)
#                 print(f"📂 Loaded {dataset_key} from cache ({len(df)} records)")
#                 return self._preprocess_dataframe(df, dataset_key)
#             except Exception as e:
#                 print(
#                     f"⚠️ Cache read failed for {dataset_key}: {e}. Downloading fresh data."
#                 )
#                 needs_update = True

#         # Download from DOSM API if update is needed
#         if needs_update:
#             try:
#                 print(f"📡 Downloading {dataset_key} from DOSM API...")

#                 # Download with timeout and retries
#                 for attempt in range(3):
#                     try:
#                         response = requests.get(
#                             self.api_endpoints[dataset_key], timeout=30
#                         )
#                         response.raise_for_status()

#                         # Read parquet data from response
#                         df = pd.read_parquet(pd.io.common.BytesIO(response.content))
#                         break

#                     except requests.RequestException as e:
#                         print(
#                             f"⚠️ Download attempt {attempt + 1} failed for {dataset_key}: {e}"
#                         )
#                         if attempt == 2:
#                             # Try cache as fallback
#                             if self.enable_cache and cache_file.exists():
#                                 print(
#                                     f"📂 Using cached version of {dataset_key} as fallback"
#                                 )
#                                 df = pd.read_parquet(cache_file)
#                                 self._save_update_timestamp(dataset_key, success=False)
#                                 return self._preprocess_dataframe(df, dataset_key)
#                             else:
#                                 raise ConnectionError(
#                                     f"Unable to access DOSM API for {dataset_key}"
#                                 )
#                         time.sleep(5)  # Wait before retry
#                 else:
#                     # This shouldn't execute, but just in case
#                     raise ConnectionError(
#                         f"Unable to access DOSM API for {dataset_key}"
#                     )

#                 # Cache the raw data
#                 if self.enable_cache:
#                     df.to_parquet(cache_file)
#                     print(f"💾 Cached {dataset_key} locally")

#                 # Store in memory with metadata
#                 self._datasets[dataset_key] = df
#                 self._last_updated[dataset_key] = datetime.now()

#                 # NEW: Save successful update timestamp
#                 self._save_update_timestamp(dataset_key, success=True)

#                 print(f"✅ Successfully loaded {dataset_key} ({len(df)} records)")
#                 return self._preprocess_dataframe(df, dataset_key)

#             except Exception as e:
#                 print(f"❌ Failed to download {dataset_key}: {e}")
#                 # NEW: Save failed update timestamp
#                 self._save_update_timestamp(dataset_key, success=False)
#                 raise ConnectionError(f"Unable to access DOSM API for {dataset_key}")

#         # This shouldn't be reached, but just in case
#         if cache_file.exists():
#             df = pd.read_parquet(cache_file)
#             return self._preprocess_dataframe(df, dataset_key)
#         else:
#             raise ConnectionError(f"No data available for {dataset_key}")

#     def force_update_all(self) -> Dict[str, bool]:
#         """
#         NEW: Force update all datasets immediately.

#         Returns:
#             Dictionary showing success/failure for each dataset
#         """
#         results = {}
#         print("🔄 Forcing update for all datasets...")

#         for dataset_key in self.api_endpoints.keys():
#             try:
#                 self.load_dataset(dataset_key, force_refresh=True)
#                 results[dataset_key] = True
#                 print(f"✅ {dataset_key} updated successfully")
#             except Exception as e:
#                 results[dataset_key] = False
#                 print(f"❌ {dataset_key} update failed: {e}")

#         successful_updates = sum(results.values())
#         total_datasets = len(results)
#         print(
#             f"📊 Update summary: {successful_updates}/{total_datasets} datasets updated"
#         )

#         return results

#     def get_update_status(self) -> Dict:
#         """
#         NEW: Get detailed update status for all datasets.

#         Returns:
#             Dictionary with update information
#         """
#         if not self.update_tracker_file.exists():
#             return {
#                 "auto_update_enabled": self.auto_update,
#                 "update_interval_days": self.update_interval_days,
#                 "datasets": {
#                     key: {"status": "never_updated"}
#                     for key in self.api_endpoints.keys()
#                 },
#             }

#         try:
#             with open(self.update_tracker_file, "r") as f:
#                 update_tracker = json.load(f)

#             status = {
#                 "auto_update_enabled": self.auto_update,
#                 "update_interval_days": self.update_interval_days,
#                 "datasets": {},
#             }

#             for dataset_key in self.api_endpoints.keys():
#                 if dataset_key in update_tracker:
#                     last_update_str = update_tracker[dataset_key].get("last_update")
#                     if last_update_str:
#                         last_update = datetime.fromisoformat(last_update_str)
#                         days_since = (datetime.now() - last_update).days
#                         needs_update = days_since >= self.update_interval_days

#                         status["datasets"][dataset_key] = {
#                             "last_update": last_update.strftime("%Y-%m-%d %H:%M"),
#                             "days_since_update": days_since,
#                             "needs_update": needs_update,
#                             "last_success": update_tracker[dataset_key].get(
#                                 "success", True
#                             ),
#                             "next_update_in_days": max(
#                                 0, self.update_interval_days - days_since
#                             ),
#                         }
#                     else:
#                         status["datasets"][dataset_key] = {
#                             "status": "invalid_timestamp"
#                         }
#                 else:
#                     status["datasets"][dataset_key] = {"status": "never_updated"}

#             return status

#         except Exception as e:
#             return {"error": f"Could not read update status: {e}"}

#     def stop_auto_updates(self):
#         """NEW: Stop automatic background updates."""
#         self._stop_updates = True
#         if self._update_thread:
#             self._update_thread.join(timeout=5)
#         print("🛑 Automatic updates stopped")

#     def start_auto_updates(self):
#         """NEW: Start automatic background updates."""
#         if not self.auto_update:
#             self.auto_update = True
#             self._stop_updates = False
#             self._start_background_updater()
#             print("▶️ Automatic updates started")

#     def _preprocess_dataframe(self, df: pd.DataFrame, dataset_key: str) -> pd.DataFrame:
#         """
#         Apply standard preprocessing to raw DOSM data.
#         UNCHANGED: Maintains existing preprocessing logic.

#         Args:
#             df: Raw dataframe from DOSM
#             dataset_key: Dataset identifier for specific processing

#         Returns:
#             Preprocessed dataframe
#         """
#         df = df.copy()

#         # Standardize date handling
#         if "date" in df.columns:
#             df["date"] = pd.to_datetime(df["date"])
#             df = df.set_index("date")

#         # Sort chronologically
#         df = df.sort_index()

#         # Remove duplicates (keep most recent)
#         df = df[~df.index.duplicated(keep="last")]

#         # Handle missing values with forward fill (appropriate for time series)
#         df = df.fillna(method="ffill").fillna(method="bfill")

#         # Ensure monthly frequency
#         if hasattr(df.index, "freq"):
#             df = df.asfreq("MS")

#         return df

#     def load_all_datasets(self, force_refresh: bool = False) -> Dict[str, pd.DataFrame]:
#         """
#         Load all available DOSM datasets.
#         ENHANCED: Now respects automatic update checking.

#         Args:
#             force_refresh: Force fresh download for all datasets

#         Returns:
#             Dictionary mapping dataset names to DataFrames
#         """
#         datasets = {}
#         successful_loads = 0

#         print("Loading Malaysia Labor Force Data from DOSM...")
#         print("-" * 50)

#         for dataset_key in self.api_endpoints.keys():
#             try:
#                 datasets[dataset_key] = self.load_dataset(dataset_key, force_refresh)
#                 successful_loads += 1
#             except Exception as e:
#                 print(f"Warning: Failed to load {dataset_key} - {e}")
#                 continue

#         print("-" * 50)
#         print(
#             f"Successfully loaded {successful_loads}/{len(self.api_endpoints)} datasets"
#         )

#         return datasets

#     def get_dataset_info(self, dataset_key: str = None) -> Dict:
#         """
#         Get comprehensive information about datasets.
#         ENHANCED: Now includes update status information.

#         Args:
#             dataset_key: Specific dataset key (None for all)

#         Returns:
#             Dataset information dictionary
#         """
#         if dataset_key:
#             if dataset_key not in self.dataset_metadata:
#                 raise ValueError(f"Unknown dataset key: {dataset_key}")

#             metadata = self.dataset_metadata[dataset_key].copy()

#             # Add runtime information if dataset is loaded
#             if dataset_key in self._datasets:
#                 df = self._datasets[dataset_key]
#                 metadata.update(
#                     {
#                         "loaded": True,
#                         "shape": df.shape,
#                         "date_range": f"{df.index.min()} to {df.index.max()}",
#                         "last_updated": self._last_updated.get(dataset_key),
#                         "memory_usage_mb": round(
#                             df.memory_usage(deep=True).sum() / 1024**2, 2
#                         ),
#                     }
#                 )
#             else:
#                 metadata["loaded"] = False

#             # NEW: Add update status
#             update_status = self.get_update_status()
#             if "datasets" in update_status and dataset_key in update_status["datasets"]:
#                 metadata["update_status"] = update_status["datasets"][dataset_key]

#             return metadata

#         # Return info for all datasets
#         return {key: self.get_dataset_info(key) for key in self.dataset_metadata.keys()}

#     def get_latest_values(self, dataset_key: str) -> Dict[str, float]:
#         """
#         Get the most recent values from a dataset.
#         UNCHANGED: Maintains existing functionality.

#         Args:
#             dataset_key: Dataset to query

#         Returns:
#             Dictionary of latest values
#         """
#         if dataset_key not in self._datasets:
#             raise ValueError(
#                 f"Dataset {dataset_key} not loaded. Call load_dataset() first."
#             )

#         df = self._datasets[dataset_key]
#         latest_row = df.iloc[-1]

#         return {"date": df.index[-1], "values": latest_row.to_dict()}

#     def __del__(self):
#         """Cleanup when object is destroyed."""
#         self.stop_auto_updates()


# class DataManager:
#     """
#     High-level data management interface for the dashboard and analysis modules.
#     Provides user-friendly access to Malaysia labor force data.
#     ENHANCED: Now supports automatic updates and real-time data refresh.
#     """

#     def __init__(self, update_interval_days: int = 28, auto_update: bool = True):
#         """
#         Initialize DataManager with enhanced update capabilities.

#         Args:
#             update_interval_days: Days between automatic updates
#             auto_update: Enable automatic background updates
#         """
#         self.loader = DOSMDataLoader(
#             update_interval_days=update_interval_days, auto_update=auto_update
#         )
#         self.datasets = {}
#         self.user_friendly_names = {
#             "unemployment_general": "Overall Unemployment",
#             "unemployment_sa": "Seasonally Adjusted",
#             "youth_unemployment": "Youth Unemployment",
#             "unemployment_duration": "Unemployment Duration",
#             "employment_status": "Employment Status",
#         }

#     def initialize(self, force_refresh: bool = False) -> bool:
#         """
#         Initialize all datasets for the application.
#         ENHANCED: Now supports forced refresh and automatic update checking.

#         Args:
#             force_refresh: Force fresh download of all data

#         Returns:
#             True if initialization successful
#         """
#         try:
#             print("Initializing Malaysia Labor Force Analytics Platform...")
#             raw_datasets = self.loader.load_all_datasets(force_refresh)

#             # Map to user-friendly names
#             self.datasets = {
#                 self.user_friendly_names.get(key, key): df
#                 for key, df in raw_datasets.items()
#             }

#             print(f"Platform ready with {len(self.datasets)} datasets")
#             return True

#         except Exception as e:
#             print(f"Initialization failed: {e}")
#             return False

#     def force_update(self) -> bool:
#         """
#         NEW: Force immediate update of all datasets.

#         Returns:
#             True if at least one dataset updated successfully
#         """
#         try:
#             results = self.loader.force_update_all()

#             # Reload datasets with fresh data
#             if any(results.values()):
#                 raw_datasets = {}
#                 for key, success in results.items():
#                     if success:
#                         try:
#                             raw_datasets[key] = self.loader._datasets[key]
#                         except KeyError:
#                             # Re-load if not in memory
#                             raw_datasets[key] = self.loader.load_dataset(key)

#                 # Update user-friendly dataset mapping
#                 updated_datasets = {
#                     self.user_friendly_names.get(key, key): df
#                     for key, df in raw_datasets.items()
#                 }
#                 self.datasets.update(updated_datasets)

#                 return True
#             return False

#         except Exception as e:
#             print(f"Force update failed: {e}")
#             return False

#     def get_update_status(self) -> Dict:
#         """
#         NEW: Get update status for dashboard display.

#         Returns:
#             Dictionary with update status information
#         """
#         return self.loader.get_update_status()

#     def get_dataset(self, name: str) -> pd.DataFrame:
#         """Get dataset by user-friendly name. UNCHANGED."""
#         if name not in self.datasets:
#             available = list(self.datasets.keys())
#             raise ValueError(f"Dataset '{name}' not available. Options: {available}")
#         return self.datasets[name]

#     def get_available_datasets(self) -> List[str]:
#         """Get list of available dataset names. UNCHANGED."""
#         return list(self.datasets.keys())

#     def get_numeric_columns(self, dataset_name: str) -> List[str]:
#         """Get numeric columns from a specific dataset. UNCHANGED."""
#         df = self.get_dataset(dataset_name)
#         return df.select_dtypes(include=[np.number]).columns.tolist()

#     def get_dashboard_summary(self) -> Dict:
#         """
#         Get summary information for dashboard display.
#         ENHANCED: Now includes update status information.

#         Returns:
#             Dictionary with key metrics and metadata
#         """
#         summary = {
#             "total_datasets": len(self.datasets),
#             "data_status": "Active" if self.datasets else "No Data",
#             "latest_update": datetime.now().strftime("%Y-%m-%d %H:%M"),
#             "key_metrics": {},
#         }

#         # NEW: Add update status
#         update_status = self.get_update_status()
#         summary["update_status"] = update_status

#         # Get key unemployment metrics if available (unchanged logic)
#         if "Overall Unemployment" in self.datasets:
#             df = self.datasets["Overall Unemployment"]
#             if "u_rate" in df.columns:
#                 latest_rate = df["u_rate"].iloc[-1]
#                 avg_rate = df["u_rate"].mean()
#                 summary["key_metrics"]["unemployment_rate"] = {
#                     "current": round(latest_rate, 1),
#                     "average": round(avg_rate, 1),
#                     "trend": "increasing" if latest_rate > avg_rate else "decreasing",
#                 }

#         # Get labor force size if available (unchanged logic)
#         if "Overall Unemployment" in self.datasets:
#             df = self.datasets["Overall Unemployment"]
#             if "lf" in df.columns:
#                 latest_lf = df["lf"].iloc[-1]
#                 summary["key_metrics"]["labor_force"] = {
#                     "current": round(latest_lf, 0),
#                     "unit": "thousands",
#                 }

#         return summary

#     def export_dataset(
#         self, dataset_name: str, format: str = "csv", output_dir: str = "data/processed"
#     ) -> str:
#         """
#         Export dataset to file. UNCHANGED.

#         Args:
#             dataset_name: Name of dataset to export
#             format: Export format ('csv', 'excel', 'parquet')
#             output_dir: Output directory

#         Returns:
#             Path to exported file
#         """
#         df = self.get_dataset(dataset_name)
#         output_path = Path(output_dir)
#         output_path.mkdir(parents=True, exist_ok=True)

#         timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
#         safe_name = dataset_name.lower().replace(" ", "_")

#         if format == "csv":
#             filepath = output_path / f"{safe_name}_{timestamp}.csv"
#             df.to_csv(filepath)
#         elif format == "excel":
#             filepath = output_path / f"{safe_name}_{timestamp}.xlsx"
#             df.to_excel(filepath)
#         elif format == "parquet":
#             filepath = output_path / f"{safe_name}_{timestamp}.parquet"
#             df.to_parquet(filepath)
#         else:
#             raise ValueError("Format must be 'csv', 'excel', or 'parquet'")

#         print(f"Exported {dataset_name} to {filepath}")
#         return str(filepath)

#     def stop_auto_updates(self):
#         """NEW: Stop automatic updates."""
#         self.loader.stop_auto_updates()

#     def start_auto_updates(self):
#         """NEW: Start automatic updates."""
#         self.loader.start_auto_updates()


# # Convenience function for quick access (ENHANCED)
# def load_malaysia_data(
#     force_refresh: bool = False, auto_update: bool = True
# ) -> Dict[str, pd.DataFrame]:
#     """
#     Quick function to load Malaysia unemployment data.
#     Ideal for Jupyter notebooks and rapid analysis.
#     ENHANCED: Now supports automatic updates.

#     Args:
#         force_refresh: Force fresh download from DOSM
#         auto_update: Enable automatic background updates

#     Returns:
#         Dictionary of all available datasets
#     """
#     manager = DataManager(auto_update=auto_update)
#     if manager.initialize(force_refresh):
#         return manager.datasets
#     else:
#         return {}


# if __name__ == "__main__":
#     # Test the data management system with REAL DOSM data
#     print("Testing Malaysia Labor Force Data System with Real DOSM Data + Auto Updates")
#     print("=" * 70)

#     manager = DataManager(update_interval_days=1, auto_update=True)  # 1 day for testing

#     print("Attempting to connect to DOSM APIs and load real data...")
#     if manager.initialize(force_refresh=True):  # Force fresh download for testing
#         print("\nReal Data Loading Successful!")
#         print("Available Datasets:")
#         for name in manager.get_available_datasets():
#             df = manager.get_dataset(name)
#             print(f"  - {name}: {len(df)} records, {len(df.columns)} columns")
#             print(
#                 f"    Date range: {df.index.min().strftime('%Y-%m-%d')} to {df.index.max().strftime('%Y-%m-%d')}"
#             )

#             # Show sample of actual data
#             if "u_rate" in df.columns:
#                 latest_rate = df["u_rate"].iloc[-1]
#                 print(f"    Latest unemployment rate: {latest_rate:.1f}%")
#             print()

#         # NEW: Display update status
#         update_status = manager.get_update_status()
#         print("Update Status:")
#         print(f"  - Auto-update enabled: {update_status.get('auto_update_enabled')}")
#         print(f"  - Update interval: {update_status.get('update_interval_days')} days")

#         for dataset, status in update_status.get("datasets", {}).items():
#             if isinstance(status, dict) and "last_update" in status:
#                 print(f"  - {dataset}: Last updated {status['last_update']}")

#         # Display dashboard summary with real metrics
#         summary = manager.get_dashboard_summary()
#         print(f"\nDashboard Summary (Real Data):")
#         print(f"  - Status: {summary['data_status']}")
#         print(f"  - Datasets: {summary['total_datasets']}")

#         if "unemployment_rate" in summary["key_metrics"]:
#             ur = summary["key_metrics"]["unemployment_rate"]
#             print(f"  - Current Unemployment Rate: {ur['current']}%")
#             print(f"  - Average Rate: {ur['average']}%")
#             print(f"  - Trend: {ur['trend']}")

#         if "labor_force" in summary["key_metrics"]:
#             lf = summary["key_metrics"]["labor_force"]
#             print(f"  - Labor Force Size: {lf['current']:,.0f} thousand")

#     else:
#         print("Failed to initialize data system with real DOSM data")
#         print("This could be due to:")
#         print("  - Internet connectivity issues")
#         print("  - DOSM API temporary unavailability")
#         print("  - Missing dependencies (fastparquet, pyarrow)")
#         print("\nPlease check your internet connection and try again.")


"""
Data loader module for Malaysia unemployment forecasting for real-time DOSM data access.
Enhanced with professional column mapping and display utilities.
"""

import pandas as pd
import numpy as np
import requests
import json
import threading
import time
from pathlib import Path
from typing import Dict, Optional, List, Union, Tuple
from datetime import datetime, timedelta
import warnings
import logging

warnings.filterwarnings("ignore")

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class ColumnDisplayManager:
    """
    Manages column name mapping and display formatting for unemployment data.
    Provides professional, user-friendly column names and descriptions.
    """

    def __init__(self):
        self._column_mappings = self._initialize_column_mappings()
        self._column_descriptions = self._initialize_column_descriptions()
        self._column_units = self._initialize_column_units()
        self._column_categories = self._initialize_column_categories()

    def _initialize_column_mappings(self) -> Dict[str, str]:
        """Initialize mapping from technical column names to user-friendly display names."""
        return {
            # Overall unemployment metrics
            "u_rate": "Unemployment Rate",
            "lf": "Total Labor Force",
            "lf_employed": "Employed Persons",
            "lf_unemployed": "Unemployed Persons",
            "lf_outside": "Persons Outside Labor Force",
            "p_rate": "Labor Force Participation Rate",
            "ep_ratio": "Employment-to-Population Ratio",
            "employment_rate": "Employment Rate",
            # Youth unemployment metrics
            "u_rate_15_24": "Youth Unemployment Rate (15-24)",
            "u_rate_15_30": "Youth Unemployment Rate (15-30)",
            "unemployed_15_24": "Unemployed Youth (15-24)",
            "unemployed_15_30": "Unemployed Youth (15-30)",
            # Duration metrics
            "unemployed": "Total Unemployed",
            "unemployed_active": "Actively Job Searching",
            "unemployed_active_3mo": "Active Job Search (Under 3 Months)",
            "unemployed_active_6mo": "Active Job Search (Under 6 Months)",
            "unemployed_passive": "Passive Job Seekers",
            # Employment status metrics
            "employed": "Total Employed",
            "employed_employer": "Employers",
            "employed_employee": "Employees",
            "employed_own_account": "Own Account Workers",
            "employed_unpaid_family": "Unpaid Family Workers",
            # Seasonally adjusted variants
            "u_rate_sa": "Unemployment Rate (Seasonally Adjusted)",
            "lf_sa": "Labor Force (Seasonally Adjusted)",
            "lf_employed_sa": "Employed Persons (Seasonally Adjusted)",
            "lf_unemployed_sa": "Unemployed Persons (Seasonally Adjusted)",
            "p_rate_sa": "Participation Rate (Seasonally Adjusted)",
        }

    def _initialize_column_descriptions(self) -> Dict[str, str]:
        """Initialize detailed descriptions for each column."""
        return {
            "u_rate": "Percentage of labor force that is unemployed and actively seeking work",
            "lf": "Total number of persons in the labor force (employed + unemployed)",
            "lf_employed": "Number of persons currently employed in any capacity",
            "lf_unemployed": "Number of persons unemployed but actively seeking work",
            "lf_outside": "Persons not in labor force (students, retirees, homemakers, etc.)",
            "p_rate": "Labor force as percentage of working-age population",
            "ep_ratio": "Employed persons as percentage of working-age population",
            "employment_rate": "Employment rate as percentage of working-age population",
            "u_rate_15_24": "Unemployment rate for youth aged 15-24 years",
            "u_rate_15_30": "Unemployment rate for youth aged 15-30 years",
            "unemployed_15_24": "Number of unemployed persons aged 15-24 years",
            "unemployed_15_30": "Number of unemployed persons aged 15-30 years",
            "unemployed": "Total number of unemployed persons across all categories",
            "unemployed_active": "Unemployed persons actively searching for employment",
            "unemployed_active_3mo": "Active job seekers unemployed for less than 3 months",
            "unemployed_active_6mo": "Active job seekers unemployed for less than 6 months",
            "unemployed_passive": "Unemployed persons not actively seeking work",
            "employed": "Total number of employed persons across all employment types",
            "employed_employer": "Persons who employ others in their business or profession",
            "employed_employee": "Persons working for wages, salary, or commission",
            "employed_own_account": "Self-employed persons without employees",
            "employed_unpaid_family": "Persons working without pay in family business",
        }

    def _initialize_column_units(self) -> Dict[str, str]:
        """Initialize units of measurement for each column."""
        return {
            # Rates and ratios (percentages)
            "u_rate": "%",
            "p_rate": "%",
            "ep_ratio": "%",
            "employment_rate": "%",
            "u_rate_15_24": "%",
            "u_rate_15_30": "%",
            "u_rate_sa": "%",
            "p_rate_sa": "%",
            # Counts (thousands of persons)
            "lf": "thousands",
            "lf_employed": "thousands",
            "lf_unemployed": "thousands",
            "lf_outside": "thousands",
            "unemployed_15_24": "thousands",
            "unemployed_15_30": "thousands",
            "unemployed": "thousands",
            "unemployed_active": "thousands",
            "unemployed_active_3mo": "thousands",
            "unemployed_active_6mo": "thousands",
            "unemployed_passive": "thousands",
            "employed": "thousands",
            "employed_employer": "thousands",
            "employed_employee": "thousands",
            "employed_own_account": "thousands",
            "employed_unpaid_family": "thousands",
            "lf_sa": "thousands",
            "lf_employed_sa": "thousands",
            "lf_unemployed_sa": "thousands",
        }

    def _initialize_column_categories(self) -> Dict[str, List[str]]:
        """Initialize categorical groupings of columns."""
        return {
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

    def get_display_name(self, column_name: str) -> str:
        """Get user-friendly display name for a column."""
        return self._column_mappings.get(
            column_name, self._format_fallback_name(column_name)
        )

    def get_full_display_name(self, column_name: str, include_unit: bool = True) -> str:
        """Get full display name including units."""
        display_name = self.get_display_name(column_name)
        if include_unit:
            unit = self._column_units.get(column_name)
            if unit:
                return f"{display_name} ({unit})"
        return display_name

    def get_description(self, column_name: str) -> str:
        """Get detailed description for a column."""
        return self._column_descriptions.get(
            column_name, f"Data series for {self.get_display_name(column_name)}"
        )

    def get_unit(self, column_name: str) -> Optional[str]:
        """Get unit of measurement for a column."""
        return self._column_units.get(column_name)

    def get_category(self, column_name: str) -> Optional[str]:
        """Get category for a column."""
        for category, columns in self._column_categories.items():
            if column_name in columns:
                return category
        return "Other Metrics"

    def get_columns_by_category(
        self, available_columns: List[str]
    ) -> Dict[str, List[Dict[str, str]]]:
        """Group available columns by category with display information."""
        categorized = {}

        for category, category_columns in self._column_categories.items():
            available_in_category = [
                col for col in category_columns if col in available_columns
            ]

            if available_in_category:
                categorized[category] = [
                    {
                        "value": col,
                        "label": self.get_display_name(col),
                        "full_label": self.get_full_display_name(col),
                        "description": self.get_description(col),
                        "unit": self.get_unit(col) or "",
                        "category": category,
                    }
                    for col in available_in_category
                ]

        # Add uncategorized columns
        categorized_columns = [
            col for cols in self._column_categories.values() for col in cols
        ]
        uncategorized = [
            col for col in available_columns if col not in categorized_columns
        ]

        if uncategorized:
            categorized["Other Metrics"] = [
                {
                    "value": col,
                    "label": self.get_display_name(col),
                    "full_label": self.get_full_display_name(col),
                    "description": self.get_description(col),
                    "unit": self.get_unit(col) or "",
                    "category": "Other Metrics",
                }
                for col in uncategorized
            ]

        return categorized

    def create_dropdown_options(
        self, available_columns: List[str], group_by_category: bool = False
    ) -> List[Dict]:
        """Create formatted options for dropdown components."""
        if group_by_category:
            categorized = self.get_columns_by_category(available_columns)
            options = []

            for category, columns in categorized.items():
                options.append(
                    {
                        "label": category,
                        "options": [
                            {
                                "label": col["full_label"],
                                "value": col["value"],
                                "title": col["description"],  # Tooltip text
                            }
                            for col in columns
                        ],
                    }
                )
            return options
        else:
            return [
                {
                    "label": self.get_full_display_name(col),
                    "value": col,
                    "title": self.get_description(col),
                }
                for col in available_columns
            ]

    def _format_fallback_name(self, column_name: str) -> str:
        """Create a readable name for unmapped columns."""
        # Replace underscores with spaces and title case
        formatted = column_name.replace("_", " ").title()

        # Handle common abbreviations
        replacements = {
            "Lf": "Labor Force",
            "U Rate": "Unemployment Rate",
            "P Rate": "Participation Rate",
            "Ep Ratio": "Employment Population Ratio",
            "Sa": "Seasonally Adjusted",
        }

        for old, new in replacements.items():
            formatted = formatted.replace(old, new)

        return formatted


class DOSMDataLoader:
    """
    Professional data loader for Department of Statistics Malaysia (DOSM) labor force data.
    Handles real-time API access, local caching, data preprocessing, and automatic updates.
    """

    def __init__(
        self,
        cache_dir: str = "data/raw",
        enable_cache: bool = True,
        update_interval_days: int = 28,
        auto_update: bool = True,
    ):
        """
        Initialize the DOSM data loader.

        Args:
            cache_dir: Directory for local data caching
            enable_cache: Whether to enable local file caching
            update_interval_days: Days between automatic data updates
            auto_update: Enable automatic background updates
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.enable_cache = enable_cache
        self.update_interval_days = update_interval_days
        self.auto_update = auto_update

        # Initialize column display manager
        self.column_manager = ColumnDisplayManager()

        self.update_tracker_file = self.cache_dir / ".update_tracker.json"
        self._update_thread = None
        self._stop_updates = False

        # Official DOSM API endpoints
        self.api_endpoints = {
            "unemployment_general": "https://storage.dosm.gov.my/labour/lfs_month.parquet",
            "unemployment_sa": "https://storage.dosm.gov.my/labour/lfs_month_sa.parquet",
            "youth_unemployment": "https://storage.dosm.gov.my/labour/lfs_month_youth.parquet",
            "unemployment_duration": "https://storage.dosm.gov.my/labour/lfs_month_duration.parquet",
            "employment_status": "https://storage.dosm.gov.my/labour/lfs_month_status.parquet",
        }

        # Dataset metadata for user-friendly display
        self.dataset_metadata = {
            "unemployment_general": {
                "name": "Overall Unemployment",
                "description": "General labor force statistics including unemployment rate, labor force participation",
                "key_columns": [
                    "u_rate",
                    "lf",
                    "lf_employed",
                    "lf_unemployed",
                    "p_rate",
                    "ep_ratio",
                ],
                "frequency": "Monthly",
                "start_year": 2010,
            },
            "unemployment_sa": {
                "name": "Seasonally Adjusted",
                "description": "Seasonally adjusted unemployment and labor force data",
                "key_columns": [
                    "u_rate",
                    "lf",
                    "lf_employed",
                    "lf_unemployed",
                    "p_rate",
                ],
                "frequency": "Monthly",
                "start_year": 2010,
            },
            "youth_unemployment": {
                "name": "Youth Unemployment",
                "description": "Unemployment statistics for age groups 15-24 and 15-30",
                "key_columns": [
                    "u_rate_15_24",
                    "u_rate_15_30",
                    "unemployed_15_24",
                    "unemployed_15_30",
                ],
                "frequency": "Monthly",
                "start_year": 2016,
            },
            "unemployment_duration": {
                "name": "Unemployment Duration",
                "description": "Analysis of unemployment duration patterns and job search activity",
                "key_columns": [
                    "unemployed",
                    "unemployed_active",
                    "unemployed_active_3mo",
                    "unemployed_active_6mo",
                ],
                "frequency": "Monthly",
                "start_year": 2016,
            },
            "employment_status": {
                "name": "Employment Status",
                "description": "Employment breakdown by status: employers, employees, self-employed",
                "key_columns": [
                    "employed",
                    "employed_employer",
                    "employed_employee",
                    "employed_own_account",
                ],
                "frequency": "Monthly",
                "start_year": 2016,
            },
        }

        self._datasets = {}
        self._last_updated = {}

        # Start background updater if enabled
        if self.auto_update:
            self._start_background_updater()

    def _should_update_dataset(self, dataset_key: str) -> bool:
        """Check if a dataset needs updating based on time interval."""
        if not self.update_tracker_file.exists():
            return True

        try:
            with open(self.update_tracker_file, "r") as f:
                update_tracker = json.load(f)

            if dataset_key not in update_tracker:
                return True

            last_update_str = update_tracker[dataset_key].get("last_update")
            if not last_update_str:
                return True

            last_update = datetime.fromisoformat(last_update_str)
            days_since_update = (datetime.now() - last_update).days

            return days_since_update >= self.update_interval_days

        except Exception as e:
            logger.warning(f"Error checking update status for {dataset_key}: {e}")
            return True

    def _save_update_timestamp(self, dataset_key: str, success: bool = True):
        """Save timestamp of last update attempt."""
        try:
            # Load existing tracker or create new
            if self.update_tracker_file.exists():
                with open(self.update_tracker_file, "r") as f:
                    update_tracker = json.load(f)
            else:
                update_tracker = {}

            # Update entry
            update_tracker[dataset_key] = {
                "last_update": datetime.now().isoformat(),
                "success": success,
                "update_interval_days": self.update_interval_days,
            }

            # Save tracker
            with open(self.update_tracker_file, "w") as f:
                json.dump(update_tracker, f, indent=2)

        except Exception as e:
            logger.warning(f"Could not save update timestamp for {dataset_key}: {e}")

    def _start_background_updater(self):
        """Start background thread for automatic updates."""
        if self._update_thread and self._update_thread.is_alive():
            return

        def update_loop():
            while not self._stop_updates:
                try:
                    # Sleep for 6 hours, then check all datasets
                    time.sleep(6 * 60 * 60)  # 6 hours

                    if self._stop_updates:
                        break

                    # Check each dataset for updates needed
                    for dataset_key in self.api_endpoints.keys():
                        if self._should_update_dataset(dataset_key):
                            try:
                                logger.info(f"Background update: {dataset_key}")
                                self.load_dataset(dataset_key, force_refresh=True)
                                logger.info(
                                    f"Background update completed: {dataset_key}"
                                )
                            except Exception as e:
                                logger.warning(
                                    f"Background update failed for {dataset_key}: {e}"
                                )

                except Exception as e:
                    logger.warning(f"Background updater error: {e}")

        self._update_thread = threading.Thread(target=update_loop, daemon=True)
        self._update_thread.start()
        logger.info("Background data updater started")

    def load_dataset(
        self, dataset_key: str, force_refresh: bool = False
    ) -> pd.DataFrame:
        """
        Load a specific dataset from DOSM API with intelligent caching and auto-updates.

        Args:
            dataset_key: Key for the dataset to load
            force_refresh: Force fresh download ignoring cache and time checks

        Returns:
            Preprocessed DataFrame

        Raises:
            ValueError: If dataset_key is invalid
            ConnectionError: If API is unreachable
        """
        if dataset_key not in self.api_endpoints:
            available_keys = list(self.api_endpoints.keys())
            raise ValueError(
                f"Invalid dataset key '{dataset_key}'. Available: {available_keys}"
            )

        cache_file = self.cache_dir / f"{dataset_key}.parquet"

        # Check if update is needed (unless force_refresh is True)
        needs_update = force_refresh or self._should_update_dataset(dataset_key)

        # Try cache first (if enabled, not forcing refresh, and no update needed)
        if self.enable_cache and not needs_update and cache_file.exists():
            try:
                df = pd.read_parquet(cache_file)
                logger.info(f"Loaded {dataset_key} from cache ({len(df)} records)")
                return self._preprocess_dataframe(df, dataset_key)
            except Exception as e:
                logger.warning(
                    f"Cache read failed for {dataset_key}: {e}. Downloading fresh data."
                )
                needs_update = True

        # Download from DOSM API if update is needed
        if needs_update:
            try:
                logger.info(f"Downloading {dataset_key} from DOSM API...")

                # Download with timeout and retries
                for attempt in range(3):
                    try:
                        response = requests.get(
                            self.api_endpoints[dataset_key], timeout=30
                        )
                        response.raise_for_status()

                        # Read parquet data from response
                        df = pd.read_parquet(pd.io.common.BytesIO(response.content))
                        break

                    except requests.RequestException as e:
                        logger.warning(
                            f"Download attempt {attempt + 1} failed for {dataset_key}: {e}"
                        )
                        if attempt == 2:
                            # Try cache as fallback
                            if self.enable_cache and cache_file.exists():
                                logger.info(
                                    f"Using cached version of {dataset_key} as fallback"
                                )
                                df = pd.read_parquet(cache_file)
                                self._save_update_timestamp(dataset_key, success=False)
                                return self._preprocess_dataframe(df, dataset_key)
                            else:
                                raise ConnectionError(
                                    f"Unable to access DOSM API for {dataset_key}"
                                )
                        time.sleep(5)  # Wait before retry
                else:
                    # This shouldn't execute, but just in case
                    raise ConnectionError(
                        f"Unable to access DOSM API for {dataset_key}"
                    )

                # Cache the raw data
                if self.enable_cache:
                    df.to_parquet(cache_file)
                    logger.info(f"Cached {dataset_key} locally")

                # Store in memory with metadata
                self._datasets[dataset_key] = df
                self._last_updated[dataset_key] = datetime.now()

                # Save successful update timestamp
                self._save_update_timestamp(dataset_key, success=True)

                logger.info(f"Successfully loaded {dataset_key} ({len(df)} records)")
                return self._preprocess_dataframe(df, dataset_key)

            except Exception as e:
                logger.error(f"Failed to download {dataset_key}: {e}")
                # Save failed update timestamp
                self._save_update_timestamp(dataset_key, success=False)
                raise ConnectionError(f"Unable to access DOSM API for {dataset_key}")

        # This shouldn't be reached, but just in case
        if cache_file.exists():
            df = pd.read_parquet(cache_file)
            return self._preprocess_dataframe(df, dataset_key)
        else:
            raise ConnectionError(f"No data available for {dataset_key}")

    def force_update_all(self) -> Dict[str, bool]:
        """Force update all datasets immediately."""
        results = {}
        logger.info("Forcing update for all datasets...")

        for dataset_key in self.api_endpoints.keys():
            try:
                self.load_dataset(dataset_key, force_refresh=True)
                results[dataset_key] = True
                logger.info(f"{dataset_key} updated successfully")
            except Exception as e:
                results[dataset_key] = False
                logger.error(f"{dataset_key} update failed: {e}")

        successful_updates = sum(results.values())
        total_datasets = len(results)
        logger.info(
            f"Update summary: {successful_updates}/{total_datasets} datasets updated"
        )

        return results

    def get_update_status(self) -> Dict:
        """Get detailed update status for all datasets."""
        if not self.update_tracker_file.exists():
            return {
                "auto_update_enabled": self.auto_update,
                "update_interval_days": self.update_interval_days,
                "datasets": {
                    key: {"status": "never_updated"}
                    for key in self.api_endpoints.keys()
                },
            }

        try:
            with open(self.update_tracker_file, "r") as f:
                update_tracker = json.load(f)

            status = {
                "auto_update_enabled": self.auto_update,
                "update_interval_days": self.update_interval_days,
                "datasets": {},
            }

            for dataset_key in self.api_endpoints.keys():
                if dataset_key in update_tracker:
                    last_update_str = update_tracker[dataset_key].get("last_update")
                    if last_update_str:
                        last_update = datetime.fromisoformat(last_update_str)
                        days_since = (datetime.now() - last_update).days
                        needs_update = days_since >= self.update_interval_days

                        status["datasets"][dataset_key] = {
                            "last_update": last_update.strftime("%Y-%m-%d %H:%M"),
                            "days_since_update": days_since,
                            "needs_update": needs_update,
                            "last_success": update_tracker[dataset_key].get(
                                "success", True
                            ),
                            "next_update_in_days": max(
                                0, self.update_interval_days - days_since
                            ),
                        }
                    else:
                        status["datasets"][dataset_key] = {
                            "status": "invalid_timestamp"
                        }
                else:
                    status["datasets"][dataset_key] = {"status": "never_updated"}

            return status

        except Exception as e:
            return {"error": f"Could not read update status: {e}"}

    def stop_auto_updates(self):
        """Stop automatic background updates."""
        self._stop_updates = True
        if self._update_thread:
            self._update_thread.join(timeout=5)
        logger.info("Automatic updates stopped")

    def start_auto_updates(self):
        """Start automatic background updates."""
        if not self.auto_update:
            self.auto_update = True
            self._stop_updates = False
            self._start_background_updater()
            logger.info("Automatic updates started")

    def get_column_display_options(
        self, dataset_key: str, group_by_category: bool = False
    ) -> List[Dict]:
        """Get formatted column options for UI components."""
        if dataset_key not in self._datasets:
            raise ValueError(
                f"Dataset {dataset_key} not loaded. Call load_dataset() first."
            )

        df = self._datasets[dataset_key]
        numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()

        return self.column_manager.create_dropdown_options(
            numeric_columns, group_by_category=group_by_category
        )

    def get_column_info(self, column_name: str) -> Dict[str, str]:
        """Get comprehensive information about a column."""
        return {
            "technical_name": column_name,
            "display_name": self.column_manager.get_display_name(column_name),
            "full_display_name": self.column_manager.get_full_display_name(column_name),
            "description": self.column_manager.get_description(column_name),
            "unit": self.column_manager.get_unit(column_name) or "",
            "category": self.column_manager.get_category(column_name),
        }

    def _preprocess_dataframe(self, df: pd.DataFrame, dataset_key: str) -> pd.DataFrame:
        """Apply standard preprocessing to raw DOSM data."""
        df = df.copy()

        # Standardize date handling
        if "date" in df.columns:
            df["date"] = pd.to_datetime(df["date"])
            df = df.set_index("date")

        # Sort chronologically
        df = df.sort_index()

        # Remove duplicates (keep most recent)
        df = df[~df.index.duplicated(keep="last")]

        # Handle missing values with forward fill (appropriate for time series)
        df = df.fillna(method="ffill").fillna(method="bfill")

        # Ensure monthly frequency
        if hasattr(df.index, "freq"):
            df = df.asfreq("MS")

        return df

    def load_all_datasets(self, force_refresh: bool = False) -> Dict[str, pd.DataFrame]:
        """Load all available DOSM datasets."""
        datasets = {}
        successful_loads = 0

        logger.info("Loading Malaysia Labor Force Data from DOSM...")

        for dataset_key in self.api_endpoints.keys():
            try:
                datasets[dataset_key] = self.load_dataset(dataset_key, force_refresh)
                successful_loads += 1
            except Exception as e:
                logger.warning(f"Failed to load {dataset_key} - {e}")
                continue

        logger.info(
            f"Successfully loaded {successful_loads}/{len(self.api_endpoints)} datasets"
        )
        return datasets

    def get_dataset_info(self, dataset_key: str = None) -> Dict:
        """Get comprehensive information about datasets."""
        if dataset_key:
            if dataset_key not in self.dataset_metadata:
                raise ValueError(f"Unknown dataset key: {dataset_key}")

            metadata = self.dataset_metadata[dataset_key].copy()

            # Add runtime information if dataset is loaded
            if dataset_key in self._datasets:
                df = self._datasets[dataset_key]
                metadata.update(
                    {
                        "loaded": True,
                        "shape": df.shape,
                        "date_range": f"{df.index.min()} to {df.index.max()}",
                        "last_updated": self._last_updated.get(dataset_key),
                        "memory_usage_mb": round(
                            df.memory_usage(deep=True).sum() / 1024**2, 2
                        ),
                    }
                )
            else:
                metadata["loaded"] = False

            # Add update status
            update_status = self.get_update_status()
            if "datasets" in update_status and dataset_key in update_status["datasets"]:
                metadata["update_status"] = update_status["datasets"][dataset_key]

            return metadata

        # Return info for all datasets
        return {key: self.get_dataset_info(key) for key in self.dataset_metadata.keys()}

    def get_latest_values(self, dataset_key: str) -> Dict[str, float]:
        """Get the most recent values from a dataset."""
        if dataset_key not in self._datasets:
            raise ValueError(
                f"Dataset {dataset_key} not loaded. Call load_dataset() first."
            )

        df = self._datasets[dataset_key]
        latest_row = df.iloc[-1]

        return {"date": df.index[-1], "values": latest_row.to_dict()}

    def __del__(self):
        """Cleanup when object is destroyed."""
        self.stop_auto_updates()


class DataManager:
    """
    High-level data management interface for the dashboard and analysis modules.
    Provides user-friendly access to Malaysia labor force data with professional column mapping.
    """

    def __init__(self, update_interval_days: int = 28, auto_update: bool = True):
        """
        Initialize DataManager with enhanced update capabilities.

        Args:
            update_interval_days: Days between automatic updates
            auto_update: Enable automatic background updates
        """
        self.loader = DOSMDataLoader(
            update_interval_days=update_interval_days, auto_update=auto_update
        )
        self.datasets = {}
        self.user_friendly_names = {
            "unemployment_general": "Overall Unemployment",
            "unemployment_sa": "Seasonally Adjusted",
            "youth_unemployment": "Youth Unemployment",
            "unemployment_duration": "Unemployment Duration",
            "employment_status": "Employment Status",
        }

    def initialize(self, force_refresh: bool = False) -> bool:
        """
        Initialize all datasets for the application.

        Args:
            force_refresh: Force fresh download of all data

        Returns:
            True if initialization successful
        """
        try:
            logger.info("Initializing Malaysia Labor Force Analytics Platform...")
            raw_datasets = self.loader.load_all_datasets(force_refresh)

            # Map to user-friendly names
            self.datasets = {
                self.user_friendly_names.get(key, key): df
                for key, df in raw_datasets.items()
            }

            logger.info(f"Platform ready with {len(self.datasets)} datasets")
            return True

        except Exception as e:
            logger.error(f"Initialization failed: {e}")
            return False

    def force_update(self) -> bool:
        """Force immediate update of all datasets."""
        try:
            results = self.loader.force_update_all()

            # Reload datasets with fresh data
            if any(results.values()):
                raw_datasets = {}
                for key, success in results.items():
                    if success:
                        try:
                            raw_datasets[key] = self.loader._datasets[key]
                        except KeyError:
                            # Re-load if not in memory
                            raw_datasets[key] = self.loader.load_dataset(key)

                # Update user-friendly dataset mapping
                updated_datasets = {
                    self.user_friendly_names.get(key, key): df
                    for key, df in raw_datasets.items()
                }
                self.datasets.update(updated_datasets)

                return True
            return False

        except Exception as e:
            logger.error(f"Force update failed: {e}")
            return False

    def get_update_status(self) -> Dict:
        """Get update status for dashboard display."""
        return self.loader.get_update_status()

    def get_dataset(self, name: str) -> pd.DataFrame:
        """Get dataset by user-friendly name."""
        if name not in self.datasets:
            available = list(self.datasets.keys())
            raise ValueError(f"Dataset '{name}' not available. Options: {available}")
        return self.datasets[name]

    def get_available_datasets(self) -> List[str]:
        """Get list of available dataset names."""
        return list(self.datasets.keys())

    def get_numeric_columns(self, dataset_name: str) -> List[str]:
        """Get numeric columns from a specific dataset."""
        df = self.get_dataset(dataset_name)
        return df.select_dtypes(include=[np.number]).columns.tolist()

    def get_column_display_options(
        self, dataset_name: str, group_by_category: bool = False
    ) -> List[Dict]:
        """
        Get formatted column options for UI dropdowns with professional display names.

        Args:
            dataset_name: User-friendly dataset name
            group_by_category: Whether to group options by category

        Returns:
            List of formatted options for dropdown components
        """
        # Map user-friendly name back to technical key
        reverse_mapping = {v: k for k, v in self.user_friendly_names.items()}
        dataset_key = reverse_mapping.get(dataset_name)

        if not dataset_key:
            raise ValueError(f"Unknown dataset: {dataset_name}")

        # Get numeric columns
        numeric_columns = self.get_numeric_columns(dataset_name)

        # Use column manager to create options
        return self.loader.column_manager.create_dropdown_options(
            numeric_columns, group_by_category=group_by_category
        )

    def get_column_info(self, column_name: str) -> Dict[str, str]:
        """Get comprehensive display information about a column."""
        return self.loader.get_column_info(column_name)

    def get_categorized_columns(self, dataset_name: str) -> Dict[str, List[Dict]]:
        """Get columns organized by category for advanced UI components."""
        numeric_columns = self.get_numeric_columns(dataset_name)
        return self.loader.column_manager.get_columns_by_category(numeric_columns)

    def format_column_value(
        self, column_name: str, value: float, precision: int = 1
    ) -> str:
        """
        Format a column value with appropriate units and precision.

        Args:
            column_name: Technical column name
            value: Numeric value to format
            precision: Number of decimal places

        Returns:
            Formatted string with value and units
        """
        if pd.isna(value):
            return "N/A"

        unit = self.loader.column_manager.get_unit(column_name)

        if unit == "%":
            return f"{value:.{precision}f}%"
        elif unit == "thousands":
            return f"{value:,.{precision}f}K"
        else:
            return f"{value:,.{precision}f}"

    def get_latest_metrics(self) -> Dict:
        """Get latest key metrics with professional formatting."""
        if not self.datasets:
            return {}

        metrics = {}
        try:
            if "Overall Unemployment" in self.datasets:
                df = self.datasets["Overall Unemployment"]

                # Get latest values with professional formatting
                latest_data = {}
                for col in [
                    "u_rate",
                    "lf",
                    "lf_employed",
                    "lf_unemployed",
                    "p_rate",
                    "ep_ratio",
                ]:
                    if col in df.columns:
                        raw_value = df[col].iloc[-1]
                        latest_data[col] = {
                            "raw_value": raw_value,
                            "formatted_value": self.format_column_value(col, raw_value),
                            "display_name": self.loader.column_manager.get_display_name(
                                col
                            ),
                            "description": self.loader.column_manager.get_description(
                                col
                            ),
                        }

                metrics["unemployment_data"] = latest_data

            if "Youth Unemployment" in self.datasets:
                youth_df = self.datasets["Youth Unemployment"]
                youth_data = {}

                for col in [
                    "u_rate_15_24",
                    "u_rate_15_30",
                    "unemployed_15_24",
                    "unemployed_15_30",
                ]:
                    if col in youth_df.columns:
                        raw_value = youth_df[col].iloc[-1]
                        youth_data[col] = {
                            "raw_value": raw_value,
                            "formatted_value": self.format_column_value(col, raw_value),
                            "display_name": self.loader.column_manager.get_display_name(
                                col
                            ),
                            "description": self.loader.column_manager.get_description(
                                col
                            ),
                        }

                metrics["youth_data"] = youth_data

            # Add metadata
            metrics["metadata"] = {
                "total_datasets": len(self.datasets),
                "data_status": "Active" if self.datasets else "No Data",
                "last_updated": datetime.now().strftime("%Y-%m-%d %H:%M"),
                "update_status": self.get_update_status(),
            }

        except Exception as e:
            logger.error(f"Error getting latest metrics: {e}")
            metrics["error"] = str(e)

        return metrics

    def get_dashboard_summary(self) -> Dict:
        """Get summary information for dashboard display with professional formatting."""
        summary = {
            "total_datasets": len(self.datasets),
            "data_status": "Active" if self.datasets else "No Data",
            "latest_update": datetime.now().strftime("%Y-%m-%d %H:%M"),
            "key_metrics": {},
        }

        # Add update status
        update_status = self.get_update_status()
        summary["update_status"] = update_status

        # Get key unemployment metrics with professional display
        if "Overall Unemployment" in self.datasets:
            df = self.datasets["Overall Unemployment"]
            if "u_rate" in df.columns:
                latest_rate = df["u_rate"].iloc[-1]
                avg_rate = df["u_rate"].mean()

                summary["key_metrics"]["unemployment_rate"] = {
                    "current": round(latest_rate, 1),
                    "current_formatted": self.format_column_value(
                        "u_rate", latest_rate
                    ),
                    "average": round(avg_rate, 1),
                    "average_formatted": self.format_column_value("u_rate", avg_rate),
                    "trend": "increasing" if latest_rate > avg_rate else "decreasing",
                    "display_name": self.loader.column_manager.get_display_name(
                        "u_rate"
                    ),
                    "description": self.loader.column_manager.get_description("u_rate"),
                }

        # Get labor force size with professional display
        if "Overall Unemployment" in self.datasets:
            df = self.datasets["Overall Unemployment"]
            if "lf" in df.columns:
                latest_lf = df["lf"].iloc[-1]

                summary["key_metrics"]["labor_force"] = {
                    "current": round(latest_lf, 0),
                    "current_formatted": self.format_column_value("lf", latest_lf),
                    "unit": "thousands",
                    "display_name": self.loader.column_manager.get_display_name("lf"),
                    "description": self.loader.column_manager.get_description("lf"),
                }

        return summary

    def export_dataset(
        self, dataset_name: str, format: str = "csv", output_dir: str = "data/processed"
    ) -> str:
        """Export dataset to file with metadata."""
        df = self.get_dataset(dataset_name)
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        safe_name = dataset_name.lower().replace(" ", "_")

        if format == "csv":
            filepath = output_path / f"{safe_name}_{timestamp}.csv"
            df.to_csv(filepath)

            # Export column mapping as separate file
            mapping_file = output_path / f"{safe_name}_{timestamp}_column_mapping.json"
            column_mapping = {
                col: self.get_column_info(col)
                for col in df.select_dtypes(include=[np.number]).columns
            }
            with open(mapping_file, "w") as f:
                json.dump(column_mapping, f, indent=2, default=str)

        elif format == "excel":
            filepath = output_path / f"{safe_name}_{timestamp}.xlsx"
            with pd.ExcelWriter(filepath, engine="openpyxl") as writer:
                # Write data
                df.to_excel(writer, sheet_name="Data")

                # Write column mapping
                mapping_df = pd.DataFrame(
                    [
                        {
                            "Technical_Name": col,
                            "Display_Name": info["display_name"],
                            "Full_Display_Name": info["full_display_name"],
                            "Description": info["description"],
                            "Unit": info["unit"],
                            "Category": info["category"],
                        }
                        for col, info in {
                            col: self.get_column_info(col)
                            for col in df.select_dtypes(include=[np.number]).columns
                        }.items()
                    ]
                )
                mapping_df.to_excel(writer, sheet_name="Column_Mapping", index=False)

        elif format == "parquet":
            filepath = output_path / f"{safe_name}_{timestamp}.parquet"
            df.to_parquet(filepath)
        else:
            raise ValueError("Format must be 'csv', 'excel', or 'parquet'")

        logger.info(f"Exported {dataset_name} to {filepath}")
        return str(filepath)

    def stop_auto_updates(self):
        """Stop automatic updates."""
        self.loader.stop_auto_updates()

    def start_auto_updates(self):
        """Start automatic updates."""
        self.loader.start_auto_updates()


# Convenience function for quick access
def load_malaysia_data(
    force_refresh: bool = False, auto_update: bool = True
) -> Dict[str, pd.DataFrame]:
    """
    Quick function to load Malaysia unemployment data with professional column mapping.

    Args:
        force_refresh: Force fresh download from DOSM
        auto_update: Enable automatic background updates

    Returns:
        Dictionary of all available datasets
    """
    manager = DataManager(auto_update=auto_update)
    if manager.initialize(force_refresh):
        return manager.datasets
    else:
        return {}


# Professional testing and validation
def validate_system_functionality():
    """Validate that the enhanced system works correctly."""
    logger.info("Validating Malaysia Labor Force Data System")

    try:
        # Test data manager initialization
        manager = DataManager(update_interval_days=28, auto_update=False)
        success = manager.initialize(force_refresh=False)

        if not success:
            logger.warning("System validation failed during initialization")
            return False

        # Test column mapping functionality
        for dataset_name in manager.get_available_datasets():
            logger.info(f"Validating dataset: {dataset_name}")

            # Test column display options
            options = manager.get_column_display_options(
                dataset_name, group_by_category=True
            )
            if not options:
                logger.warning(f"No column options found for {dataset_name}")
                continue

            # Test column info retrieval
            numeric_columns = manager.get_numeric_columns(dataset_name)
            for col in numeric_columns[:3]:  # Test first 3 columns
                col_info = manager.get_column_info(col)
                if not col_info.get("display_name"):
                    logger.warning(f"Missing display name for column {col}")

            # Test data formatting
            df = manager.get_dataset(dataset_name)
            for col in numeric_columns[:2]:  # Test first 2 columns
                if len(df) > 0:
                    latest_value = df[col].iloc[-1]
                    formatted = manager.format_column_value(col, latest_value)
                    if not formatted or formatted == "N/A":
                        logger.warning(f"Formatting issue for column {col}")

        # Test summary generation
        summary = manager.get_dashboard_summary()
        if not summary.get("key_metrics"):
            logger.warning("Dashboard summary generation failed")

        logger.info("System validation completed successfully")
        return True

    except Exception as e:
        logger.error(f"System validation failed: {e}")
        return False


if __name__ == "__main__":
    # Test the enhanced data management system
    logger.info("Testing Enhanced Malaysia Labor Force Data System")

    # Run validation
    if validate_system_functionality():
        logger.info("All systems operational")
    else:
        logger.error("System validation failed")

    # Example usage demonstration
    try:
        manager = DataManager(update_interval_days=28, auto_update=True)

        if manager.initialize(force_refresh=False):
            logger.info("System initialized successfully")

            # Demonstrate professional column mapping
            for dataset_name in manager.get_available_datasets():
                logger.info(f"Dataset: {dataset_name}")

                # Show categorized columns
                categorized = manager.get_categorized_columns(dataset_name)
                for category, columns in categorized.items():
                    logger.info(f"  Category: {category}")
                    for col_info in columns[:2]:  # Show first 2 in each category
                        logger.info(
                            f"    {col_info['value']} -> {col_info['full_label']}"
                        )

                break  # Just show first dataset for demo

            # Show formatted metrics
            summary = manager.get_dashboard_summary()
            if "unemployment_rate" in summary.get("key_metrics", {}):
                ur_info = summary["key_metrics"]["unemployment_rate"]
                logger.info(
                    f"Current unemployment: {ur_info.get('current_formatted', 'N/A')}"
                )

    except Exception as e:
        logger.error(f"Demo failed: {e}")

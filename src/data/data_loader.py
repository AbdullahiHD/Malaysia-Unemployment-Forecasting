"""
Data loader module for Malaysia unemployment forecasting project.
Professional implementation for real-time DOSM data access.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Optional, List, Union
from datetime import datetime
import warnings

warnings.filterwarnings("ignore")


class DOSMDataLoader:
    """
    Professional data loader for Department of Statistics Malaysia (DOSM) labor force data.
    Handles real-time API access, local caching, and data preprocessing.
    """

    def __init__(self, cache_dir: str = "data/raw", enable_cache: bool = True):
        """
        Initialize the DOSM data loader.

        Args:
            cache_dir: Directory for local data caching
            enable_cache: Whether to enable local file caching
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.enable_cache = enable_cache

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

    def load_dataset(
        self, dataset_key: str, force_refresh: bool = False
    ) -> pd.DataFrame:
        """
        Load a specific dataset from DOSM API with intelligent caching.

        Args:
            dataset_key: Key for the dataset to load
            force_refresh: Force fresh download ignoring cache

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

        # Try cache first (if enabled and not forcing refresh)
        if self.enable_cache and not force_refresh and cache_file.exists():
            try:
                df = pd.read_parquet(cache_file)
                print(f"Loaded {dataset_key} from cache ({len(df)} records)")
                return self._preprocess_dataframe(df, dataset_key)
            except Exception as e:
                print(
                    f"Cache read failed for {dataset_key}: {e}. Downloading fresh data."
                )

        # Download from DOSM API
        try:
            print(f"Downloading {dataset_key} from DOSM API...")
            df = pd.read_parquet(self.api_endpoints[dataset_key])

            # Cache the raw data
            if self.enable_cache:
                df.to_parquet(cache_file)
                print(f"Cached {dataset_key} locally")

            # Store in memory with metadata
            self._datasets[dataset_key] = df
            self._last_updated[dataset_key] = datetime.now()

            print(f"Successfully loaded {dataset_key} ({len(df)} records)")
            return self._preprocess_dataframe(df, dataset_key)

        except Exception as e:
            print(f"Failed to download {dataset_key}: {e}")
            raise ConnectionError(f"Unable to access DOSM API for {dataset_key}")

    def _preprocess_dataframe(self, df: pd.DataFrame, dataset_key: str) -> pd.DataFrame:
        """
        Apply standard preprocessing to raw DOSM data.

        Args:
            df: Raw dataframe from DOSM
            dataset_key: Dataset identifier for specific processing

        Returns:
            Preprocessed dataframe
        """
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
        """
        Load all available DOSM datasets.

        Args:
            force_refresh: Force fresh download for all datasets

        Returns:
            Dictionary mapping dataset names to DataFrames
        """
        datasets = {}
        successful_loads = 0

        print("Loading Malaysia Labor Force Data from DOSM...")
        print("-" * 50)

        for dataset_key in self.api_endpoints.keys():
            try:
                datasets[dataset_key] = self.load_dataset(dataset_key, force_refresh)
                successful_loads += 1
            except Exception as e:
                print(f"Warning: Failed to load {dataset_key} - {e}")
                continue

        print("-" * 50)
        print(
            f"Successfully loaded {successful_loads}/{len(self.api_endpoints)} datasets"
        )

        return datasets

    def get_dataset_info(self, dataset_key: str = None) -> Dict:
        """
        Get comprehensive information about datasets.

        Args:
            dataset_key: Specific dataset key (None for all)

        Returns:
            Dataset information dictionary
        """
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

            return metadata

        # Return info for all datasets
        return {key: self.get_dataset_info(key) for key in self.dataset_metadata.keys()}

    def get_latest_values(self, dataset_key: str) -> Dict[str, float]:
        """
        Get the most recent values from a dataset.

        Args:
            dataset_key: Dataset to query

        Returns:
            Dictionary of latest values
        """
        if dataset_key not in self._datasets:
            raise ValueError(
                f"Dataset {dataset_key} not loaded. Call load_dataset() first."
            )

        df = self._datasets[dataset_key]
        latest_row = df.iloc[-1]

        return {"date": df.index[-1], "values": latest_row.to_dict()}


class DataManager:
    """
    High-level data management interface for the dashboard and analysis modules.
    Provides user-friendly access to Malaysia labor force data.
    """

    def __init__(self):
        self.loader = DOSMDataLoader()
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
            print("Initializing Malaysia Labor Force Analytics Platform...")
            raw_datasets = self.loader.load_all_datasets(force_refresh)

            # Map to user-friendly names
            self.datasets = {
                self.user_friendly_names.get(key, key): df
                for key, df in raw_datasets.items()
            }

            print(f"Platform ready with {len(self.datasets)} datasets")
            return True

        except Exception as e:
            print(f"Initialization failed: {e}")
            return False

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

    def get_dashboard_summary(self) -> Dict:
        """
        Get summary information for dashboard display.

        Returns:
            Dictionary with key metrics and metadata
        """
        summary = {
            "total_datasets": len(self.datasets),
            "data_status": "Active" if self.datasets else "No Data",
            "latest_update": datetime.now().strftime("%Y-%m-%d %H:%M"),
            "key_metrics": {},
        }

        # Get key unemployment metrics if available
        if "Overall Unemployment" in self.datasets:
            df = self.datasets["Overall Unemployment"]
            if "u_rate" in df.columns:
                latest_rate = df["u_rate"].iloc[-1]
                avg_rate = df["u_rate"].mean()
                summary["key_metrics"]["unemployment_rate"] = {
                    "current": round(latest_rate, 1),
                    "average": round(avg_rate, 1),
                    "trend": "increasing" if latest_rate > avg_rate else "decreasing",
                }

        # Get labor force size if available
        if "Overall Unemployment" in self.datasets:
            df = self.datasets["Overall Unemployment"]
            if "lf" in df.columns:
                latest_lf = df["lf"].iloc[-1]
                summary["key_metrics"]["labor_force"] = {
                    "current": round(latest_lf, 0),
                    "unit": "thousands",
                }

        return summary

    def export_dataset(
        self, dataset_name: str, format: str = "csv", output_dir: str = "data/processed"
    ) -> str:
        """
        Export dataset to file.

        Args:
            dataset_name: Name of dataset to export
            format: Export format ('csv', 'excel', 'parquet')
            output_dir: Output directory

        Returns:
            Path to exported file
        """
        df = self.get_dataset(dataset_name)
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        safe_name = dataset_name.lower().replace(" ", "_")

        if format == "csv":
            filepath = output_path / f"{safe_name}_{timestamp}.csv"
            df.to_csv(filepath)
        elif format == "excel":
            filepath = output_path / f"{safe_name}_{timestamp}.xlsx"
            df.to_excel(filepath)
        elif format == "parquet":
            filepath = output_path / f"{safe_name}_{timestamp}.parquet"
            df.to_parquet(filepath)
        else:
            raise ValueError("Format must be 'csv', 'excel', or 'parquet'")

        print(f"Exported {dataset_name} to {filepath}")
        return str(filepath)


# Convenience function for quick access
def load_malaysia_data(force_refresh: bool = False) -> Dict[str, pd.DataFrame]:
    """
    Quick function to load Malaysia unemployment data.
    Ideal for Jupyter notebooks and rapid analysis.

    Args:
        force_refresh: Force fresh download from DOSM

    Returns:
        Dictionary of all available datasets
    """
    manager = DataManager()
    if manager.initialize(force_refresh):
        return manager.datasets
    else:
        return {}


if __name__ == "__main__":
    # Test the data management system with REAL DOSM data
    print("Testing Malaysia Labor Force Data System with Real DOSM Data")
    print("=" * 60)

    manager = DataManager()

    print("Attempting to connect to DOSM APIs and load real data...")
    if manager.initialize(force_refresh=True):  # Force fresh download for testing
        print("\nReal Data Loading Successful!")
        print("Available Datasets:")
        for name in manager.get_available_datasets():
            df = manager.get_dataset(name)
            print(f"  - {name}: {len(df)} records, {len(df.columns)} columns")
            print(
                f"    Date range: {df.index.min().strftime('%Y-%m-%d')} to {df.index.max().strftime('%Y-%m-%d')}"
            )

            # Show sample of actual data
            if "u_rate" in df.columns:
                latest_rate = df["u_rate"].iloc[-1]
                print(f"    Latest unemployment rate: {latest_rate:.1f}%")
            print()

        # Display dashboard summary with real metrics
        summary = manager.get_dashboard_summary()
        print(f"Dashboard Summary (Real Data):")
        print(f"  - Status: {summary['data_status']}")
        print(f"  - Datasets: {summary['total_datasets']}")

        if "unemployment_rate" in summary["key_metrics"]:
            ur = summary["key_metrics"]["unemployment_rate"]
            print(f"  - Current Unemployment Rate: {ur['current']}%")
            print(f"  - Average Rate: {ur['average']}%")
            print(f"  - Trend: {ur['trend']}")

        if "labor_force" in summary["key_metrics"]:
            lf = summary["key_metrics"]["labor_force"]
            print(f"  - Labor Force Size: {lf['current']:,.0f} thousand")

    else:
        print("Failed to initialize data system with real DOSM data")
        print("This could be due to:")
        print("  - Internet connectivity issues")
        print("  - DOSM API temporary unavailability")
        print("  - Missing dependencies (fastparquet, pyarrow)")
        print("\nPlease check your internet connection and try again.")

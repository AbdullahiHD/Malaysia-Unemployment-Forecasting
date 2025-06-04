"""
Comprehensive statistical analysis module for Malaysia unemployment forecasting.
Professional implementation of time series statistical tests and analysis.
"""

import pandas as pd
import numpy as np
import scipy.stats as stats
from statsmodels.tsa.stattools import adfuller, kpss
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.tsa.seasonal import seasonal_decompose
from typing import Dict, Any, Tuple, Optional, List, Union
import warnings

warnings.filterwarnings("ignore")


class StationarityAnalyzer:
    """
    Professional stationarity testing and analysis for time series data.
    Implements multiple statistical tests with proper interpretation.
    """

    @staticmethod
    def augmented_dickey_fuller_test(
        series: pd.Series,
        maxlag: Optional[int] = None,
        regression: str = "c",
        autolag: str = "AIC",
    ) -> Dict[str, Any]:
        """
        Perform Augmented Dickey-Fuller test for unit root (stationarity).

        Args:
            series: Time series data
            maxlag: Maximum lag order for test
            regression: Regression type ('c', 'ct', 'ctt', 'n')
            autolag: Method for automatic lag selection

        Returns:
            Comprehensive test results dictionary
        """
        try:
            clean_series = series.dropna()
            if len(clean_series) < 10:
                return {
                    "error": "Insufficient data points for ADF test (minimum 10 required)"
                }

            adf_result = adfuller(
                clean_series, maxlag=maxlag, regression=regression, autolag=autolag
            )

            # Extract results
            adf_statistic = adf_result[0]
            p_value = adf_result[1]
            used_lag = adf_result[2]
            n_observations = adf_result[3]
            critical_values = adf_result[4]

            # Determine stationarity
            is_stationary = p_value < 0.05

            # Provide detailed interpretation
            interpretation = {
                "null_hypothesis": "Series has a unit root (non-stationary)",
                "alternative_hypothesis": "Series does not have a unit root (stationary)",
                "decision": (
                    "Reject H0 (stationary)"
                    if is_stationary
                    else "Fail to reject H0 (non-stationary)"
                ),
                "confidence_level": "95%",
                "recommendation": (
                    "Series is suitable for modeling"
                    if is_stationary
                    else "Apply differencing or transformation"
                ),
            }

            return {
                "test_name": "Augmented Dickey-Fuller Test",
                "adf_statistic": adf_statistic,
                "p_value": p_value,
                "used_lag": used_lag,
                "n_observations": n_observations,
                "critical_values": critical_values,
                "is_stationary": is_stationary,
                "conclusion": "Stationary" if is_stationary else "Non-stationary",
                "interpretation": interpretation,
                "test_parameters": {
                    "regression": regression,
                    "autolag": autolag,
                    "maxlag": maxlag,
                },
            }

        except Exception as e:
            return {
                "error": f"ADF test failed: {str(e)}",
                "test_name": "Augmented Dickey-Fuller Test",
            }

    @staticmethod
    def kpss_test(
        series: pd.Series, regression: str = "c", nlags: Union[str, int] = "auto"
    ) -> Dict[str, Any]:
        """
        Perform KPSS test for stationarity.

        Args:
            series: Time series data
            regression: Type of regression ('c' for level, 'ct' for trend)
            nlags: Number of lags ('auto' or integer)

        Returns:
            Comprehensive test results dictionary
        """
        try:
            clean_series = series.dropna()
            if len(clean_series) < 10:
                return {
                    "error": "Insufficient data points for KPSS test (minimum 10 required)"
                }

            kpss_result = kpss(clean_series, regression=regression, nlags=nlags)

            # Extract results
            kpss_statistic = kpss_result[0]
            p_value = kpss_result[1]
            used_lags = kpss_result[2]
            critical_values = kpss_result[3]

            # Determine stationarity (opposite logic to ADF)
            is_stationary = p_value > 0.05

            # Provide detailed interpretation
            interpretation = {
                "null_hypothesis": "Series is stationary around a deterministic trend",
                "alternative_hypothesis": "Series has a unit root (non-stationary)",
                "decision": (
                    "Fail to reject H0 (stationary)"
                    if is_stationary
                    else "Reject H0 (non-stationary)"
                ),
                "confidence_level": "95%",
                "recommendation": (
                    "Series is suitable for modeling"
                    if is_stationary
                    else "Apply differencing or detrending"
                ),
            }

            return {
                "test_name": "KPSS Test",
                "kpss_statistic": kpss_statistic,
                "p_value": p_value,
                "used_lags": used_lags,
                "critical_values": critical_values,
                "is_stationary": is_stationary,
                "conclusion": "Stationary" if is_stationary else "Non-stationary",
                "interpretation": interpretation,
                "test_parameters": {"regression": regression, "nlags": nlags},
            }

        except Exception as e:
            return {"error": f"KPSS test failed: {str(e)}", "test_name": "KPSS Test"}

    @staticmethod
    def comprehensive_stationarity_analysis(series: pd.Series) -> Dict[str, Any]:
        """
        Perform comprehensive stationarity analysis using multiple tests.

        Args:
            series: Time series data

        Returns:
            Combined analysis results with recommendations
        """
        # Perform both tests
        adf_result = StationarityAnalyzer.augmented_dickey_fuller_test(series)
        kpss_result = StationarityAnalyzer.kpss_test(series)

        # Extract stationarity conclusions (handle errors)
        adf_stationary = adf_result.get("is_stationary", None)
        kpss_stationary = kpss_result.get("is_stationary", None)

        # Combined interpretation logic
        if adf_stationary is None or kpss_stationary is None:
            combined_conclusion = "Test Error"
            confidence = "Unknown"
            recommendation = "Unable to determine stationarity due to test errors"
        elif adf_stationary and kpss_stationary:
            combined_conclusion = "Stationary"
            confidence = "High"
            recommendation = (
                "Series is stationary. Proceed with level data for ARIMA modeling."
            )
        elif not adf_stationary and not kpss_stationary:
            combined_conclusion = "Non-stationary"
            confidence = "High"
            recommendation = (
                "Series is non-stationary. Apply first differencing and retest."
            )
        elif adf_stationary and not kpss_stationary:
            combined_conclusion = "Trend-stationary"
            confidence = "Medium"
            recommendation = (
                "Series may be trend-stationary. Consider detrending or differencing."
            )
        else:  # not adf_stationary and kpss_stationary
            combined_conclusion = "Difference-stationary"
            confidence = "Medium"
            recommendation = (
                "Series appears difference-stationary. Apply first differencing."
            )

        return {
            "individual_tests": {"adf": adf_result, "kpss": kpss_result},
            "combined_analysis": {
                "conclusion": combined_conclusion,
                "confidence": confidence,
                "recommendation": recommendation,
            },
            "suggested_transformations": StationarityAnalyzer._get_transformation_suggestions(
                adf_stationary, kpss_stationary
            ),
        }

    @staticmethod
    def _get_transformation_suggestions(
        adf_stationary: Optional[bool], kpss_stationary: Optional[bool]
    ) -> List[str]:
        """Generate transformation suggestions based on test results."""
        suggestions = []

        if adf_stationary is None or kpss_stationary is None:
            return ["Unable to provide suggestions due to test errors"]

        if not adf_stationary or not kpss_stationary:
            suggestions.append("First differencing: y_t - y_{t-1}")
            suggestions.append("Seasonal differencing (if seasonal): y_t - y_{t-12}")
            suggestions.append(
                "Log transformation (if multiplicative trends): log(y_t)"
            )

        if not suggestions:
            suggestions.append("No transformation needed - series appears stationary")

        return suggestions


class NormalityAnalyzer:
    """
    Professional normality testing for time series residuals and data.
    """

    @staticmethod
    def shapiro_wilk_test(series: pd.Series) -> Dict[str, Any]:
        """
        Perform Shapiro-Wilk test for normality.

        Args:
            series: Time series data

        Returns:
            Test results dictionary
        """
        try:
            clean_series = series.dropna()

            if len(clean_series) > 5000:
                return {
                    "test_name": "Shapiro-Wilk Test",
                    "error": "Sample size too large for Shapiro-Wilk test (max 5000)",
                    "recommendation": "Use Jarque-Bera or Anderson-Darling test instead",
                }

            if len(clean_series) < 3:
                return {
                    "test_name": "Shapiro-Wilk Test",
                    "error": "Insufficient data points (minimum 3 required)",
                }

            statistic, p_value = stats.shapiro(clean_series)
            is_normal = p_value > 0.05

            return {
                "test_name": "Shapiro-Wilk Test",
                "statistic": statistic,
                "p_value": p_value,
                "is_normal": is_normal,
                "conclusion": "Normal" if is_normal else "Non-normal",
                "sample_size": len(clean_series),
                "interpretation": {
                    "null_hypothesis": "Data is normally distributed",
                    "alternative_hypothesis": "Data is not normally distributed",
                    "decision": "Fail to reject H0" if is_normal else "Reject H0",
                },
            }

        except Exception as e:
            return {
                "error": f"Shapiro-Wilk test failed: {str(e)}",
                "test_name": "Shapiro-Wilk Test",
            }

    @staticmethod
    def jarque_bera_test(series: pd.Series) -> Dict[str, Any]:
        """
        Perform Jarque-Bera test for normality.

        Args:
            series: Time series data

        Returns:
            Test results dictionary
        """
        try:
            clean_series = series.dropna()

            if len(clean_series) < 5:
                return {
                    "test_name": "Jarque-Bera Test",
                    "error": "Insufficient data points (minimum 5 required)",
                }

            jb_statistic, p_value, skewness, kurtosis = stats.jarque_bera(clean_series)
            is_normal = p_value > 0.05

            return {
                "test_name": "Jarque-Bera Test",
                "jb_statistic": jb_statistic,
                "p_value": p_value,
                "skewness": skewness,
                "kurtosis": kurtosis,
                "is_normal": is_normal,
                "conclusion": "Normal" if is_normal else "Non-normal",
                "sample_size": len(clean_series),
                "interpretation": {
                    "null_hypothesis": "Data is normally distributed",
                    "alternative_hypothesis": "Data is not normally distributed",
                    "decision": "Fail to reject H0" if is_normal else "Reject H0",
                    "skewness_interpretation": NormalityAnalyzer._interpret_skewness(
                        skewness
                    ),
                    "kurtosis_interpretation": NormalityAnalyzer._interpret_kurtosis(
                        kurtosis
                    ),
                },
            }

        except Exception as e:
            return {
                "error": f"Jarque-Bera test failed: {str(e)}",
                "test_name": "Jarque-Bera Test",
            }

    @staticmethod
    def anderson_darling_test(series: pd.Series) -> Dict[str, Any]:
        """
        Perform Anderson-Darling test for normality.

        Args:
            series: Time series data

        Returns:
            Test results dictionary
        """
        try:
            clean_series = series.dropna()

            if len(clean_series) < 8:
                return {
                    "test_name": "Anderson-Darling Test",
                    "error": "Insufficient data points (minimum 8 required)",
                }

            result = stats.anderson(clean_series, dist="norm")

            # Determine significance level
            significance_levels = [15.0, 10.0, 5.0, 2.5, 1.0]
            significance_level = None

            for i, critical_value in enumerate(result.critical_values):
                if result.statistic < critical_value:
                    significance_level = significance_levels[i]
                    break

            if significance_level is None:
                significance_level = "< 1.0"

            is_normal = result.statistic < result.critical_values[2]  # 5% level

            return {
                "test_name": "Anderson-Darling Test",
                "ad_statistic": result.statistic,
                "critical_values": dict(
                    zip(significance_levels, result.critical_values)
                ),
                "significance_level": significance_level,
                "is_normal": is_normal,
                "conclusion": "Normal" if is_normal else "Non-normal",
                "sample_size": len(clean_series),
                "interpretation": {
                    "null_hypothesis": "Data follows normal distribution",
                    "alternative_hypothesis": "Data does not follow normal distribution",
                    "decision": "Fail to reject H0" if is_normal else "Reject H0",
                },
            }

        except Exception as e:
            return {
                "error": f"Anderson-Darling test failed: {str(e)}",
                "test_name": "Anderson-Darling Test",
            }

    @staticmethod
    def comprehensive_normality_analysis(series: pd.Series) -> Dict[str, Any]:
        """
        Perform comprehensive normality analysis using multiple tests.

        Args:
            series: Time series data

        Returns:
            Combined analysis results
        """
        results = {}

        # Always perform Jarque-Bera and Anderson-Darling
        results["jarque_bera"] = NormalityAnalyzer.jarque_bera_test(series)
        results["anderson_darling"] = NormalityAnalyzer.anderson_darling_test(series)

        # Add Shapiro-Wilk for smaller samples
        if len(series.dropna()) <= 5000:
            results["shapiro_wilk"] = NormalityAnalyzer.shapiro_wilk_test(series)

        # Determine consensus
        valid_tests = [test for test in results.values() if "error" not in test]
        normal_results = [test.get("is_normal", False) for test in valid_tests]

        if not valid_tests:
            consensus = "Unknown"
            confidence = "No valid tests"
        elif all(normal_results):
            consensus = "Normal"
            confidence = "High" if len(normal_results) >= 2 else "Medium"
        elif not any(normal_results):
            consensus = "Non-normal"
            confidence = "High" if len(normal_results) >= 2 else "Medium"
        else:
            consensus = "Mixed"
            confidence = "Low"

        return {
            "individual_tests": results,
            "consensus": consensus,
            "confidence": confidence,
            "recommendation": NormalityAnalyzer._get_normality_recommendation(
                consensus
            ),
            "summary": {
                "total_tests": len(results),
                "valid_tests": len(valid_tests),
                "normal_count": sum(normal_results),
                "non_normal_count": len(normal_results) - sum(normal_results),
            },
        }

    @staticmethod
    def _interpret_skewness(skewness: float) -> str:
        """Interpret skewness value."""
        if abs(skewness) < 0.5:
            return "Approximately symmetric"
        elif skewness > 0.5:
            return "Right-skewed (long right tail)"
        else:
            return "Left-skewed (long left tail)"

    @staticmethod
    def _interpret_kurtosis(kurtosis: float) -> str:
        """Interpret excess kurtosis value."""
        if abs(kurtosis) < 0.5:
            return "Normal kurtosis (mesokurtic)"
        elif kurtosis > 0.5:
            return "Heavy-tailed (leptokurtic)"
        else:
            return "Light-tailed (platykurtic)"

    @staticmethod
    def _get_normality_recommendation(consensus: str) -> str:
        """Generate recommendation based on normality consensus."""
        recommendations = {
            "Normal": "Data appears normally distributed. Parametric methods and Gaussian assumptions are appropriate.",
            "Non-normal": "Data is not normally distributed. Consider data transformation or non-parametric methods.",
            "Mixed": "Normality tests show mixed results. Examine data visually and consider robust methods.",
            "Unknown": "Unable to determine normality due to test errors. Visual inspection recommended.",
        }
        return recommendations.get(consensus, "No recommendation available")


class TimeSeriesAnalyzer:
    """
    Comprehensive time series analysis combining stationarity, normality, and autocorrelation tests.
    """

    def __init__(self):
        self.stationarity_analyzer = StationarityAnalyzer()
        self.normality_analyzer = NormalityAnalyzer()

    def full_statistical_analysis(
        self, series: pd.Series, series_name: str = "Time Series"
    ) -> Dict[str, Any]:
        """
        Perform comprehensive statistical analysis of a time series.

        Args:
            series: Time series data
            series_name: Descriptive name for the series

        Returns:
            Complete analysis results dictionary
        """
        print(f"Analyzing {series_name}...")

        analysis_results = {
            "series_info": {
                "name": series_name,
                "length": len(series),
                "valid_observations": len(series.dropna()),
                "missing_values": series.isna().sum(),
                "date_range": {
                    "start": (
                        series.index.min() if hasattr(series.index, "min") else "N/A"
                    ),
                    "end": (
                        series.index.max() if hasattr(series.index, "max") else "N/A"
                    ),
                },
            },
            "descriptive_statistics": self._calculate_descriptive_stats(series),
            "stationarity_analysis": self.stationarity_analyzer.comprehensive_stationarity_analysis(
                series
            ),
            "normality_analysis": self.normality_analyzer.comprehensive_normality_analysis(
                series
            ),
            "autocorrelation_analysis": self._analyze_autocorrelation(series),
            "seasonal_analysis": self._analyze_seasonality(series),
        }

        # Generate comprehensive recommendations
        analysis_results["recommendations"] = (
            self._generate_comprehensive_recommendations(analysis_results)
        )

        return analysis_results

    def _calculate_descriptive_stats(self, series: pd.Series) -> Dict[str, Any]:
        """Calculate comprehensive descriptive statistics."""
        clean_series = series.dropna()

        if len(clean_series) == 0:
            return {"error": "No valid data points for descriptive statistics"}

        return {
            "central_tendency": {
                "mean": clean_series.mean(),
                "median": clean_series.median(),
                "mode": (
                    clean_series.mode().iloc[0]
                    if len(clean_series.mode()) > 0
                    else None
                ),
            },
            "dispersion": {
                "std": clean_series.std(),
                "variance": clean_series.var(),
                "range": clean_series.max() - clean_series.min(),
                "iqr": clean_series.quantile(0.75) - clean_series.quantile(0.25),
            },
            "shape": {
                "skewness": clean_series.skew(),
                "kurtosis": clean_series.kurtosis(),
            },
            "extremes": {
                "min": clean_series.min(),
                "max": clean_series.max(),
                "q25": clean_series.quantile(0.25),
                "q75": clean_series.quantile(0.75),
            },
        }

    def _analyze_autocorrelation(
        self, series: pd.Series, max_lags: int = 20
    ) -> Dict[str, Any]:
        """Analyze autocorrelation structure."""
        try:
            clean_series = series.dropna()

            if len(clean_series) < max_lags + 5:
                return {
                    "error": f"Insufficient data for autocorrelation analysis (need {max_lags + 5}, have {len(clean_series)})"
                }

            # Ljung-Box test for autocorrelation
            lb_result = acorr_ljungbox(clean_series, lags=max_lags, return_df=True)

            # Find significant autocorrelations
            significant_lags = lb_result[lb_result["lb_pvalue"] < 0.05].index.tolist()

            return {
                "ljung_box_test": {
                    "test_statistics": lb_result["lb_stat"].to_dict(),
                    "p_values": lb_result["lb_pvalue"].to_dict(),
                    "significant_lags": significant_lags,
                    "has_autocorrelation": len(significant_lags) > 0,
                },
                "interpretation": {
                    "autocorrelation_present": len(significant_lags) > 0,
                    "strongest_lags": significant_lags[:5] if significant_lags else [],
                    "recommendation": (
                        "ARIMA modeling appropriate"
                        if significant_lags
                        else "White noise or random walk"
                    ),
                },
            }

        except Exception as e:
            return {"error": f"Autocorrelation analysis failed: {str(e)}"}

    def _analyze_seasonality(
        self, series: pd.Series, period: int = 12
    ) -> Dict[str, Any]:
        """Analyze seasonal patterns in the data."""
        try:
            clean_series = series.dropna()

            if len(clean_series) < 2 * period:
                return {
                    "error": f"Insufficient data for seasonal analysis (need {2 * period}, have {len(clean_series)})"
                }

            # Perform seasonal decomposition
            decomposition = seasonal_decompose(
                clean_series, model="additive", period=period
            )

            # Calculate variance explained by components
            total_var = clean_series.var()
            trend_var = decomposition.trend.dropna().var()
            seasonal_var = decomposition.seasonal.var()
            residual_var = decomposition.resid.dropna().var()

            # Seasonal strength indicators
            seasonal_strength = (
                seasonal_var / (seasonal_var + residual_var)
                if (seasonal_var + residual_var) > 0
                else 0
            )
            trend_strength = (
                trend_var / (trend_var + residual_var)
                if (trend_var + residual_var) > 0
                else 0
            )

            return {
                "decomposition_results": {
                    "has_trend": trend_strength > 0.3,
                    "has_seasonality": seasonal_strength > 0.3,
                    "trend_strength": trend_strength,
                    "seasonal_strength": seasonal_strength,
                },
                "variance_explained": {
                    "trend_pct": (trend_var / total_var) * 100 if total_var > 0 else 0,
                    "seasonal_pct": (
                        (seasonal_var / total_var) * 100 if total_var > 0 else 0
                    ),
                    "residual_pct": (
                        (residual_var / total_var) * 100 if total_var > 0 else 0
                    ),
                },
                "modeling_implications": {
                    "recommended_model": self._recommend_model_type(
                        trend_strength, seasonal_strength
                    ),
                    "seasonal_period": period,
                    "preprocessing_needed": trend_strength > 0.6
                    or seasonal_strength > 0.6,
                },
            }

        except Exception as e:
            return {"error": f"Seasonal analysis failed: {str(e)}"}

    def _recommend_model_type(
        self, trend_strength: float, seasonal_strength: float
    ) -> str:
        """Recommend model type based on trend and seasonal strength."""
        if trend_strength > 0.6 and seasonal_strength > 0.6:
            return "SARIMA (Seasonal ARIMA)"
        elif trend_strength > 0.6:
            return "ARIMA with trend"
        elif seasonal_strength > 0.6:
            return "Seasonal ARIMA"
        else:
            return "Simple ARIMA"

    def _generate_comprehensive_recommendations(
        self, analysis: Dict[str, Any]
    ) -> List[str]:
        """Generate comprehensive modeling recommendations."""
        recommendations = []

        # Stationarity recommendations
        stationarity = analysis.get("stationarity_analysis", {}).get(
            "combined_analysis", {}
        )
        if stationarity.get("conclusion") == "Non-stationary":
            recommendations.append(
                "Apply first differencing to achieve stationarity before modeling"
            )
        elif stationarity.get("conclusion") == "Trend-stationary":
            recommendations.append("Consider detrending or first differencing")

        # Normality recommendations
        normality = analysis.get("normality_analysis", {})
        if normality.get("consensus") == "Non-normal":
            recommendations.append(
                "Data is non-normal; consider transformation or robust methods"
            )

        # Autocorrelation recommendations
        autocorr = analysis.get("autocorrelation_analysis", {}).get(
            "interpretation", {}
        )
        if autocorr.get("autocorrelation_present"):
            recommendations.append(
                "Significant autocorrelation detected; ARIMA modeling is appropriate"
            )

        # Seasonal recommendations
        seasonal = analysis.get("seasonal_analysis", {}).get(
            "modeling_implications", {}
        )
        if seasonal.get("recommended_model"):
            recommendations.append(
                f"Recommended model type: {seasonal['recommended_model']}"
            )

        if not recommendations:
            recommendations.append(
                "Series appears suitable for basic time series modeling"
            )

        return recommendations


# Convenience functions for quick analysis
def quick_stationarity_test(series: pd.Series) -> Dict[str, Any]:
    """Quick stationarity test for dashboard use."""
    analyzer = StationarityAnalyzer()
    return analyzer.comprehensive_stationarity_analysis(series)


def quick_normality_test(series: pd.Series) -> Dict[str, Any]:
    """Quick normality test for dashboard use."""
    analyzer = NormalityAnalyzer()
    return analyzer.comprehensive_normality_analysis(series)


def full_series_analysis(series: pd.Series, name: str = "Series") -> Dict[str, Any]:
    """Complete statistical analysis for any time series."""
    analyzer = TimeSeriesAnalyzer()
    return analyzer.full_statistical_analysis(series, name)


if __name__ == "__main__":
    # Test the statistical analysis system with REAL Malaysia unemployment data
    print("Testing Statistical Analysis System with Real Malaysia Data")
    print("=" * 60)

    # Import data loader to get real data
    import sys
    from pathlib import Path

    # Add parent directory to path for imports
    parent_dir = Path(__file__).parent.parent
    sys.path.insert(0, str(parent_dir))

    try:
        from data.data_loader import DataManager

        print("Loading real Malaysia unemployment data from DOSM...")
        manager = DataManager()

        if manager.initialize():
            print("Successfully loaded real data from DOSM APIs!")

            # Test with real Overall Unemployment data
            if "Overall Unemployment" in manager.get_available_datasets():
                df = manager.get_dataset("Overall Unemployment")

                # Test multiple series from real data
                test_series = {
                    "Unemployment Rate": df["u_rate"],
                    "Labor Force": df["lf"],
                    "Employed Population": df["lf_employed"],
                }

                for series_name, series in test_series.items():
                    if series_name in [
                        "Unemployment Rate"
                    ]:  # Focus on main series for demo
                        print(f"\n" + "=" * 50)
                        print(f"ANALYZING REAL DATA: {series_name}")
                        print(
                            f"Data range: {series.index.min()} to {series.index.max()}"
                        )
                        print(f"Data points: {len(series)}")
                        print(f"Latest value: {series.iloc[-1]:.2f}")
                        print("=" * 50)

                        # Run comprehensive analysis on real data
                        results = full_series_analysis(
                            series, f"Malaysia {series_name}"
                        )

                        print(f"\nSTATISTICAL ANALYSIS RESULTS:")
                        print(f"  Series: {results['series_info']['name']}")
                        print(
                            f"  Valid observations: {results['series_info']['valid_observations']:,}"
                        )
                        print(
                            f"  Date range: {results['series_info']['date_range']['start']} to {results['series_info']['date_range']['end']}"
                        )

                        # Stationarity results
                        stationarity = results["stationarity_analysis"][
                            "combined_analysis"
                        ]
                        print(f"\nSTATIONARITY ANALYSIS:")
                        print(f"  Conclusion: {stationarity['conclusion']}")
                        print(f"  Confidence: {stationarity['confidence']}")
                        print(f"  Recommendation: {stationarity['recommendation']}")

                        # Show individual test results
                        adf_result = results["stationarity_analysis"][
                            "individual_tests"
                        ]["adf"]
                        if "error" not in adf_result:
                            print(
                                f"  ADF Test: p-value = {adf_result['p_value']:.4f} ({adf_result['conclusion']})"
                            )

                        kpss_result = results["stationarity_analysis"][
                            "individual_tests"
                        ]["kpss"]
                        if "error" not in kpss_result:
                            print(
                                f"  KPSS Test: p-value = {kpss_result['p_value']:.4f} ({kpss_result['conclusion']})"
                            )

                        # Normality results
                        normality = results["normality_analysis"]
                        print(f"\nNORMALITY ANALYSIS:")
                        print(f"  Consensus: {normality['consensus']}")
                        print(f"  Confidence: {normality['confidence']}")

                        # Show individual normality tests
                        jb_result = normality["individual_tests"]["jarque_bera"]
                        if "error" not in jb_result:
                            print(
                                f"  Jarque-Bera: p-value = {jb_result['p_value']:.4f} ({jb_result['conclusion']})"
                            )
                            print(
                                f"    Skewness: {jb_result['skewness']:.3f} ({jb_result['interpretation']['skewness_interpretation']})"
                            )
                            print(
                                f"    Kurtosis: {jb_result['kurtosis']:.3f} ({jb_result['interpretation']['kurtosis_interpretation']})"
                            )

                        # Descriptive statistics from real data
                        desc_stats = results["descriptive_statistics"]
                        if "error" not in desc_stats:
                            print(f"\nDESCRIPTIVE STATISTICS:")
                            print(
                                f"  Mean: {desc_stats['central_tendency']['mean']:.3f}"
                            )
                            print(
                                f"  Median: {desc_stats['central_tendency']['median']:.3f}"
                            )
                            print(f"  Std Dev: {desc_stats['dispersion']['std']:.3f}")
                            print(f"  Min: {desc_stats['extremes']['min']:.3f}")
                            print(f"  Max: {desc_stats['extremes']['max']:.3f}")

                        # Recommendations based on real data
                        print(f"\nMODELING RECOMMENDATIONS:")
                        for i, rec in enumerate(results["recommendations"], 1):
                            print(f"  {i}. {rec}")

                        # Test quick functions with real data
                        print(f"\nQUICK TEST RESULTS:")
                        quick_stat = quick_stationarity_test(series)
                        quick_norm = quick_normality_test(series)
                        print(
                            f"  Quick Stationarity: {quick_stat['combined_analysis']['conclusion']}"
                        )
                        print(f"  Quick Normality: {quick_norm['consensus']}")

            # Test with Youth Unemployment if available
            if "Youth Unemployment" in manager.get_available_datasets():
                youth_df = manager.get_dataset("Youth Unemployment")
                if "u_rate_15_24" in youth_df.columns:
                    youth_series = youth_df["u_rate_15_24"]
                    print(f"\n" + "=" * 50)
                    print(f"QUICK ANALYSIS: Youth Unemployment Rate (15-24)")
                    print(f"Latest rate: {youth_series.iloc[-1]:.1f}%")

                    # Quick analysis
                    quick_results = full_series_analysis(
                        youth_series, "Malaysia Youth Unemployment Rate"
                    )
                    stationarity = quick_results["stationarity_analysis"][
                        "combined_analysis"
                    ]
                    print(
                        f"Stationarity: {stationarity['conclusion']} (Confidence: {stationarity['confidence']})"
                    )
                    print("=" * 50)

        else:
            print("Failed to load real data from DOSM. Testing with sample data...")
            # Fallback to sample data if real data unavailable
            dates = pd.date_range("2010-01-01", periods=120, freq="MS")
            np.random.seed(42)
            trend = np.linspace(3.5, 4.2, 120)
            seasonal = 0.3 * np.sin(2 * np.pi * np.arange(120) / 12)
            noise = np.random.normal(0, 0.15, 120)
            sample_data = trend + seasonal + noise
            sample_series = pd.Series(sample_data, index=dates)

            results = full_series_analysis(
                sample_series, "Sample Malaysia Unemployment Rate"
            )
            print(f"Sample Analysis Results:")
            print(
                f"  Stationarity: {results['stationarity_analysis']['combined_analysis']['conclusion']}"
            )
            print(f"  Normality: {results['normality_analysis']['consensus']}")

    except ImportError as e:
        print(f"Could not import data loader: {e}")
        print("Testing with sample data instead...")

        # Generate sample data for testing
        dates = pd.date_range("2010-01-01", periods=120, freq="MS")
        np.random.seed(42)
        trend = np.linspace(3.5, 4.2, 120)
        seasonal = 0.3 * np.sin(2 * np.pi * np.arange(120) / 12)
        noise = np.random.normal(0, 0.15, 120)
        sample_data = trend + seasonal + noise
        sample_series = pd.Series(
            sample_data, index=dates, name="Sample Unemployment Rate"
        )

        print("Running analysis on sample unemployment data...")
        results = full_series_analysis(
            sample_series, "Sample Malaysia Unemployment Rate"
        )

        print(f"Sample Analysis Results:")
        print(f"  Data points: {results['series_info']['valid_observations']}")
        print(
            f"  Stationarity: {results['stationarity_analysis']['combined_analysis']['conclusion']}"
        )
        print(f"  Normality: {results['normality_analysis']['consensus']}")

        print("\nRecommendations:")
        for i, rec in enumerate(results["recommendations"], 1):
            print(f"  {i}. {rec}")

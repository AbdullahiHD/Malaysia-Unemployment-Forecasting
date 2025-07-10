# Malaysia Unemployment Rate Forecasting

A comprehensive time series forecasting project to predict Malaysia's unemployment rate using traditional statistical models (ARIMA/SARIMA) and modern machine learning approaches (LSTM).

## 🎯 Project Overview

This project addresses the critical need for accurate employment forecasting in Malaysia by developing and comparing multiple time series models. As final-year students, we aim to create a practical tool for workforce planning and policy decision-making.

### Key Features
- **Comprehensive EDA**: Visual and statistical analysis of Malaysia's labor market data
- **Statistical Testing**: Stationarity, normality, and multivariate relationship analysis
- **Multiple Models**: ARIMA/SARIMA, LSTM/GRU, and ensemble approaches
- **Interactive Dashboard**: Real-time visualization of forecasts and historical trends
- **Model Comparison**: Performance evaluation using MAPE, RMSE, and other metrics

## 📊 Data Sources

- **Primary Data**: Department of Statistics Malaysia (DOSM) - OpenDOSM
- **Time Range**: 2010-2025 (monthly data)
- **Coverage**: Youth unemployment, employment status, duration analysis, and general labor force indicators

## 🚀 Quick Start

### Prerequisites
```bash
Python 3.8+
Git
```

### Installation
```bash
# Clone the repository
git clone https://github.com/AbdullahiHD/Malaysia-Unemployment-Forecasting.git
cd malaysia-unemployment-forecasting

# Create virtual environment
python -m venv venv
source venv/bin/activate  

# Install dependencies
pip install -r requirements.txt

```

### Running the Project
```bash
# Run exploratory data analysis
jupyter notebook notebooks/01_exploratory_data_analysis.ipynb

# Train models
python scripts/train_models.py

# Generate forecasts
python scripts/generate_forecasts.py

# Launch dashboard
python run_dashboard.py
```

## 📁 Project Structure

```
├── data/           # Raw and processed datasets
├── src/            # Source code modules
├── notebooks/      # Jupyter notebooks for analysis
├── models/         # Trained model artifacts
├── results/        # Outputs, figures, and reports
├── app/           # Dashboard application
└── docs/          # Documentation
```

## 🔬 Methodology

### Statistical Models
- **ARIMA/SARIMA**: Traditional time series forecasting with seasonal components

### Machine Learning Models
- **LSTM/**: Deep learning for capturing long-term dependencies
- **Ensemble Methods**: Combining multiple models for improved accuracy

### Evaluation Metrics
- Mean Absolute Percentage Error (MAPE)
- Root Mean Square Error (RMSE)
- Mean Absolute Error (MAE)

## 📈 Key Findings

- **Non-stationarity**: All unemployment series require differencing
- **Strong Seasonality**: Clear 12-month seasonal patterns
- **Youth vs Overall**: Youth unemployment shows higher volatility
- **Model Performance**: LSTM Models achieved the highest performance

## 🎪 Interactive Dashboard

Our dashboard provides:
- Real-time forecast visualization
- Historical trend analysis
- Model performance comparison
- Seasonal decomposition views
- Statistical test results

## 📝 Research Questions

1. What are the most effective models for forecasting Malaysia's unemployment trends?
2. How do different methods compare in terms of forecast accuracy?
3. Can we produce reliable short-term forecasts (3-6 months) for workforce planning?
4. What patterns in Malaysia's labor data reveal economic shifts?


## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 👥 Team

- Shifaz Ahamed - Data Analysis & 
- Yassir - Statistical Modeling
- Abdullahi Hussein - Machine Learning & Neural Networks
- Rafid - Dashboard Development & Visualization
- Terrence - Documentation & Testing

## 📚 References

- Aziz, A. A., & Foo, Y. C. (2024). Forecasting unemployment rate in Malaysia using ARIMA model.
- Ismail, N., et al. (2022). Forecasting the unemployment rate in Malaysia during COVID-19 pandemic using ARIMA and ARFIMA models.
- Tay, V. (2024). Comparing forecasting accuracy of time series models on Malaysian unemployment data.


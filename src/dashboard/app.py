# import dash
# from dash import dcc, html, Input, Output, State, callback_context, dash_table
# import dash_bootstrap_components as dbc
# import plotly.graph_objects as go
# import plotly.express as px
# from plotly.subplots import make_subplots
# import pandas as pd
# import numpy as np
# from datetime import datetime, date
# import warnings

# warnings.filterwarnings("ignore")

# # Import data management and statistical analysis modules
# import sys
# from pathlib import Path

# # Add parent directories to path for imports
# current_dir = Path(__file__).parent
# project_root = current_dir.parent.parent
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
#     # Fallback to built-in implementations if modules not available
#     REAL_DATA_AVAILABLE = False
#     print("Warning: Using fallback implementations for data and analysis modules")

# # Fallback DataManager if real one not available
# if not REAL_DATA_AVAILABLE:

#     class DataManager:
#         """Fallback DataManager with sample data"""

#         def __init__(self):
#             self.datasets = {}
#             self.initialized = False

#         def initialize(self, force_refresh=False):
#             """Initialize with sample data"""
#             try:
#                 dates = pd.date_range("2010-01-01", "2025-02-01", freq="MS")
#                 np.random.seed(42)

#                 # Overall unemployment data
#                 n = len(dates)
#                 base_rate = 3.4
#                 trend = 0.1 * np.sin(np.arange(n) * 2 * np.pi / 120)
#                 seasonal = 0.3 * np.sin(np.arange(n) * 2 * np.pi / 12)
#                 noise = 0.2 * np.random.randn(n)
#                 u_rate = base_rate + trend + seasonal + noise

#                 lf = 15000 + 50 * np.arange(n) + 100 * np.random.randn(n)
#                 lf_unemployed = lf * u_rate / 100
#                 lf_employed = lf - lf_unemployed
#                 p_rate = (
#                     67.8
#                     + 0.2 * np.sin(np.arange(n) * 2 * np.pi / 12)
#                     + 0.1 * np.random.randn(n)
#                 )
#                 ep_ratio = p_rate - u_rate

#                 self.datasets["Overall Unemployment"] = pd.DataFrame(
#                     {
#                         "u_rate": u_rate,
#                         "lf": lf,
#                         "lf_employed": lf_employed,
#                         "lf_unemployed": lf_unemployed,
#                         "p_rate": p_rate,
#                         "ep_ratio": ep_ratio,
#                     },
#                     index=dates,
#                 )

#                 # Youth unemployment (starts 2016)
#                 youth_dates = pd.date_range("2016-01-01", "2025-02-01", freq="MS")
#                 n_youth = len(youth_dates)
#                 u_rate_15_24 = (
#                     11.5
#                     + 0.8 * np.sin(np.arange(n_youth) * 2 * np.pi / 12)
#                     + 0.3 * np.random.randn(n_youth)
#                 )
#                 u_rate_15_30 = (
#                     7.2
#                     + 0.6 * np.sin(np.arange(n_youth) * 2 * np.pi / 12)
#                     + 0.2 * np.random.randn(n_youth)
#                 )

#                 self.datasets["Youth Unemployment"] = pd.DataFrame(
#                     {
#                         "u_rate_15_24": u_rate_15_24,
#                         "u_rate_15_30": u_rate_15_30,
#                         "unemployed_15_24": 300 + 20 * np.random.randn(n_youth),
#                         "unemployed_15_30": 450 + 30 * np.random.randn(n_youth),
#                     },
#                     index=youth_dates,
#                 )

#                 # Seasonally adjusted data
#                 u_rate_sa = base_rate + trend + 0.15 * np.random.randn(n)
#                 lf_sa = 15000 + 50 * np.arange(n) + 80 * np.random.randn(n)

#                 self.datasets["Seasonally Adjusted"] = pd.DataFrame(
#                     {
#                         "u_rate": u_rate_sa,
#                         "lf": lf_sa,
#                         "lf_employed": lf_sa * (1 - u_rate_sa / 100),
#                         "lf_unemployed": lf_sa * u_rate_sa / 100,
#                         "p_rate": 67.8 + 0.1 * np.random.randn(n),
#                     },
#                     index=dates,
#                 )

#                 self.initialized = True
#                 return True
#             except Exception as e:
#                 print(f"Initialization error: {e}")
#                 return False

#         def get_dataset(self, name):
#             if name not in self.datasets:
#                 raise ValueError(f"Dataset {name} not found")
#             return self.datasets[name]

#         def get_available_datasets(self):
#             return list(self.datasets.keys())

#         def get_numeric_columns(self, dataset_name):
#             df = self.get_dataset(dataset_name)
#             return df.select_dtypes(include=[np.number]).columns.tolist()


# # Fallback statistical analysis functions
# if not REAL_DATA_AVAILABLE:

#     def quick_stationarity_test(series):
#         """Simple stationarity test"""
#         from scipy import stats

#         # Simple trend test
#         x = np.arange(len(series))
#         slope, _, _, p_value, _ = stats.linregress(x, series)
#         has_trend = p_value < 0.05

#         if has_trend:
#             conclusion = "Non-stationary"
#             confidence = "Medium"
#             recommendation = "Consider differencing to remove trend"
#         else:
#             conclusion = "Stationary"
#             confidence = "Medium"
#             recommendation = "Series appears stationary, suitable for modeling"

#         return {
#             "combined_analysis": {
#                 "conclusion": conclusion,
#                 "confidence": confidence,
#                 "recommendation": recommendation,
#             }
#         }

#     def quick_normality_test(series):
#         """Simple normality test"""
#         from scipy import stats

#         try:
#             jb_stat, jb_p = stats.jarque_bera(series.dropna())
#             is_normal = jb_p > 0.05

#             consensus = "Normal" if is_normal else "Non-normal"
#             confidence = "High"
#             recommendation = (
#                 "Parametric methods suitable"
#                 if is_normal
#                 else "Consider transformation"
#             )

#         except:
#             consensus = "Unknown"
#             confidence = "Low"
#             recommendation = "Unable to determine normality"

#         return {
#             "consensus": consensus,
#             "confidence": confidence,
#             "recommendation": recommendation,
#         }

#     def full_series_analysis(series, name="Series"):
#         """Simple full analysis"""
#         stationarity = quick_stationarity_test(series)
#         normality = quick_normality_test(series)

#         desc_stats = {
#             "mean": series.mean(),
#             "std": series.std(),
#             "min": series.min(),
#             "max": series.max(),
#             "count": len(series.dropna()),
#         }

#         recommendations = []
#         if stationarity["combined_analysis"]["conclusion"] == "Non-stationary":
#             recommendations.append("Apply differencing for stationarity")
#         if normality["consensus"] == "Non-normal":
#             recommendations.append("Consider data transformation")
#         if not recommendations:
#             recommendations.append("Data suitable for standard time series modeling")

#         return {
#             "series_info": {
#                 "name": name,
#                 "valid_observations": len(series.dropna()),
#                 "missing_values": series.isna().sum(),
#                 "date_range": {
#                     "start": (
#                         series.index.min() if hasattr(series.index, "min") else "N/A"
#                     ),
#                     "end": (
#                         series.index.max() if hasattr(series.index, "max") else "N/A"
#                     ),
#                 },
#             },
#             "descriptive_statistics": desc_stats,
#             "stationarity_analysis": stationarity,
#             "normality_analysis": normality,
#             "recommendations": recommendations,
#         }


# class MalaysiaUnemploymentDashboard:
#     """Main dashboard application class"""

#     def __init__(self):
#         self.app = dash.Dash(
#             __name__,
#             external_stylesheets=[dbc.themes.BOOTSTRAP],
#             suppress_callback_exceptions=True,
#         )
#         self.app.title = "Malaysia Unemployment Analytics"

#         # Color scheme
#         self.colors = {
#             "primary": "#2E86C1",
#             "secondary": "#F39C12",
#             "success": "#28B463",
#             "danger": "#E74C3C",
#             "warning": "#F7DC6F",
#             "info": "#5DADE2",
#             "light": "#F8F9FA",
#             "dark": "#2C3E50",
#         }

#         # Initialize data manager
#         self.data_manager = DataManager()
#         self.data_initialized = False

#         self.setup_layout()
#         self.setup_callbacks()

#     def setup_layout(self):
#         """Setup main dashboard layout"""
#         self.app.layout = dbc.Container(
#             [
#                 # Header
#                 dbc.Row(
#                     [
#                         dbc.Col(
#                             [
#                                 html.H1(
#                                     "Malaysia Labor Force Analytics Platform",
#                                     className="text-primary mb-2",
#                                     style={"fontWeight": "bold"},
#                                 ),
#                                 html.P(
#                                     "Professional unemployment forecasting and analysis powered by DOSM data",
#                                     className="text-muted mb-4",
#                                 ),
#                                 dbc.Button(
#                                     "Initialize Data",
#                                     id="init-button",
#                                     color="primary",
#                                     className="mb-3",
#                                     n_clicks=0,
#                                 ),
#                             ]
#                         )
#                     ]
#                 ),
#                 # Status indicator
#                 html.Div(id="status-indicator", className="mb-4"),
#                 # Navigation tabs
#                 dcc.Tabs(
#                     id="main-tabs",
#                     value="overview",
#                     children=[
#                         dcc.Tab(
#                             label="📊 Market Overview",
#                             value="overview",
#                             className="tab-style",
#                         ),
#                         dcc.Tab(
#                             label="🔍 Data Explorer",
#                             value="explorer",
#                             className="tab-style",
#                         ),
#                         dcc.Tab(
#                             label="📈 Statistical Analysis",
#                             value="statistics",
#                             className="tab-style",
#                         ),
#                         dcc.Tab(
#                             label="🔮 Forecasting Hub",
#                             value="forecasting",
#                             className="tab-style",
#                         ),
#                     ],
#                     className="mb-4",
#                 ),
#                 # Main content area
#                 html.Div(id="tab-content"),
#                 # Loading indicator
#                 dcc.Loading(id="loading", children=[html.Div(id="loading-output")]),
#             ],
#             fluid=True,
#             style={"fontFamily": "Arial, sans-serif"},
#         )

#     def setup_callbacks(self):
#         """Setup all dashboard callbacks"""

#         @self.app.callback(
#             [
#                 Output("status-indicator", "children"),
#                 Output("loading-output", "children"),
#             ],
#             [Input("init-button", "n_clicks")],
#         )
#         def initialize_data(n_clicks):
#             if n_clicks > 0:
#                 success = self.data_manager.initialize()
#                 if success:
#                     self.data_initialized = True
#                     datasets = self.data_manager.get_available_datasets()
#                     status = dbc.Alert(
#                         [
#                             html.H5(
#                                 "✅ Data Initialized Successfully",
#                                 className="alert-heading",
#                             ),
#                             html.P(f"Loaded {len(datasets)} datasets from DOSM"),
#                             html.Hr(),
#                             html.P(
#                                 f"Datasets: {', '.join(datasets)}", className="mb-0"
#                             ),
#                         ],
#                         color="success",
#                     )
#                     return status, ""
#                 else:
#                     status = dbc.Alert(
#                         [
#                             html.H5(
#                                 "❌ Data Initialization Failed",
#                                 className="alert-heading",
#                             ),
#                             html.P("Unable to load DOSM data. Please try again."),
#                         ],
#                         color="danger",
#                     )
#                     return status, ""

#             return (
#                 dbc.Alert(
#                     "Click 'Initialize Data' to load Malaysia unemployment data",
#                     color="info",
#                 ),
#                 "",
#             )

#         @self.app.callback(
#             Output("tab-content", "children"), [Input("main-tabs", "value")]
#         )
#         def render_tab_content(active_tab):
#             if not self.data_initialized:
#                 return dbc.Alert("Please initialize data first", color="warning")

#             if active_tab == "overview":
#                 return self.create_overview_tab()
#             elif active_tab == "explorer":
#                 return self.create_explorer_tab()
#             elif active_tab == "statistics":
#                 return self.create_statistics_tab()
#             elif active_tab == "forecasting":
#                 return self.create_forecasting_tab()

#             return html.Div("Select a tab to view content")

#         # Data explorer callbacks
#         @self.app.callback(
#             [Output("explorer-chart", "figure"), Output("data-summary", "children")],
#             [
#                 Input("dataset-dropdown", "value"),
#                 Input("variable-dropdown", "value"),
#                 Input("chart-type-dropdown", "value"),
#             ],
#         )
#         def update_explorer_display(dataset, variables, chart_type):
#             if not dataset or not variables:
#                 empty_fig = go.Figure()
#                 empty_fig.add_annotation(
#                     text="Select dataset and variables to display",
#                     xref="paper",
#                     yref="paper",
#                     x=0.5,
#                     y=0.5,
#                 )
#                 return empty_fig, "No data selected"

#             try:
#                 df = self.data_manager.get_dataset(dataset)
#                 fig = self.create_time_series_chart(df, variables, chart_type)

#                 # Create summary table
#                 summary_stats = df[variables].describe().round(3)
#                 table = dash_table.DataTable(
#                     data=summary_stats.reset_index().to_dict("records"),
#                     columns=[
#                         {"name": i, "id": i}
#                         for i in summary_stats.reset_index().columns
#                     ],
#                     style_cell={"textAlign": "left"},
#                     style_header={
#                         "backgroundColor": self.colors["light"],
#                         "fontWeight": "bold",
#                     },
#                 )

#                 return fig, table
#             except Exception as e:
#                 error_fig = go.Figure()
#                 error_fig.add_annotation(
#                     text=f"Error: {str(e)}", xref="paper", yref="paper", x=0.5, y=0.5
#                 )
#                 return error_fig, f"Error: {str(e)}"

#         @self.app.callback(
#             Output("variable-dropdown", "options"), [Input("dataset-dropdown", "value")]
#         )
#         def update_variable_options(dataset):
#             if not dataset:
#                 return []
#             try:
#                 variables = self.data_manager.get_numeric_columns(dataset)
#                 return [{"label": var, "value": var} for var in variables]
#             except:
#                 return []

#         # Statistical analysis callbacks
#         @self.app.callback(
#             Output("stat-results", "children"),
#             [
#                 Input("run-stats-button", "n_clicks"),
#                 Input("quick-stationarity-btn", "n_clicks"),
#                 Input("quick-normality-btn", "n_clicks"),
#             ],
#             [
#                 State("stat-dataset-dropdown", "value"),
#                 State("stat-variable-dropdown", "value"),
#             ],
#         )
#         def run_statistical_tests(
#             full_clicks, stat_clicks, norm_clicks, dataset, variable
#         ):
#             ctx = callback_context
#             if not ctx.triggered or not dataset or not variable:
#                 return dbc.Alert(
#                     "Select dataset and variable, then choose a test", color="info"
#                 )

#             button_id = ctx.triggered[0]["prop_id"].split(".")[0]

#             try:
#                 df = self.data_manager.get_dataset(dataset)
#                 series = df[variable]

#                 if button_id == "run-stats-button" and full_clicks > 0:
#                     results = full_series_analysis(series, f"{dataset} - {variable}")
#                     return self.create_full_stats_display(results)
#                 elif button_id == "quick-stationarity-btn" and stat_clicks > 0:
#                     results = quick_stationarity_test(series)
#                     return self.create_quick_stat_display(results, "Stationarity")
#                 elif button_id == "quick-normality-btn" and norm_clicks > 0:
#                     results = quick_normality_test(series)
#                     return self.create_quick_norm_display(results, "Normality")

#                 return dbc.Alert("Click a test button to run analysis", color="info")
#             except Exception as e:
#                 return dbc.Alert(f"Analysis error: {str(e)}", color="danger")

#         @self.app.callback(
#             Output("stat-variable-dropdown", "options"),
#             [Input("stat-dataset-dropdown", "value")],
#         )
#         def update_stat_variable_options(dataset):
#             if not dataset:
#                 return []
#             try:
#                 variables = self.data_manager.get_numeric_columns(dataset)
#                 return [{"label": var, "value": var} for var in variables]
#             except:
#                 return []

#     def create_overview_tab(self):
#         """Create overview tab with key metrics"""
#         try:
#             df = self.data_manager.get_dataset("Overall Unemployment")
#             youth_df = self.data_manager.get_dataset("Youth Unemployment")

#             # Key metrics cards
#             metrics_row = dbc.Row(
#                 [
#                     dbc.Col(
#                         [
#                             dbc.Card(
#                                 [
#                                     dbc.CardBody(
#                                         [
#                                             html.H3(
#                                                 f"{df['u_rate'].iloc[-1]:.1f}%",
#                                                 className="text-primary",
#                                             ),
#                                             html.P(
#                                                 "Current Unemployment Rate",
#                                                 className="text-muted",
#                                             ),
#                                         ]
#                                     )
#                                 ]
#                             )
#                         ],
#                         width=3,
#                     ),
#                     dbc.Col(
#                         [
#                             dbc.Card(
#                                 [
#                                     dbc.CardBody(
#                                         [
#                                             html.H3(
#                                                 f"{df['lf'].iloc[-1]:,.0f}K",
#                                                 className="text-success",
#                                             ),
#                                             html.P(
#                                                 "Labor Force Size",
#                                                 className="text-muted",
#                                             ),
#                                         ]
#                                     )
#                                 ]
#                             )
#                         ],
#                         width=3,
#                     ),
#                     dbc.Col(
#                         [
#                             dbc.Card(
#                                 [
#                                     dbc.CardBody(
#                                         [
#                                             html.H3(
#                                                 f"{youth_df['u_rate_15_24'].iloc[-1]:.1f}%",
#                                                 className="text-warning",
#                                             ),
#                                             html.P(
#                                                 "Youth Unemployment",
#                                                 className="text-muted",
#                                             ),
#                                         ]
#                                     )
#                                 ]
#                             )
#                         ],
#                         width=3,
#                     ),
#                     dbc.Col(
#                         [
#                             dbc.Card(
#                                 [
#                                     dbc.CardBody(
#                                         [
#                                             html.H3(
#                                                 f"{df['p_rate'].iloc[-1]:.1f}%",
#                                                 className="text-info",
#                                             ),
#                                             html.P(
#                                                 "Participation Rate",
#                                                 className="text-muted",
#                                             ),
#                                         ]
#                                     )
#                                 ]
#                             )
#                         ],
#                         width=3,
#                     ),
#                 ],
#                 className="mb-4",
#             )

#             # Main trend chart
#             fig = self.create_overview_chart(df, youth_df)
#             chart_card = dbc.Card(
#                 [
#                     dbc.CardHeader("Malaysia Labor Market Trends"),
#                     dbc.CardBody([dcc.Graph(figure=fig, style={"height": "500px"})]),
#                 ]
#             )

#             return html.Div(
#                 [html.H2("Market Overview", className="mb-4"), metrics_row, chart_card]
#             )

#         except Exception as e:
#             return dbc.Alert(f"Error creating overview: {str(e)}", color="danger")

#     def create_explorer_tab(self):
#         """Create data explorer tab"""
#         datasets = self.data_manager.get_available_datasets()

#         controls = dbc.Card(
#             [
#                 dbc.CardHeader("Data Explorer Controls"),
#                 dbc.CardBody(
#                     [
#                         dbc.Row(
#                             [
#                                 dbc.Col(
#                                     [
#                                         dbc.Label("Dataset:"),
#                                         dcc.Dropdown(
#                                             id="dataset-dropdown",
#                                             options=[
#                                                 {"label": ds, "value": ds}
#                                                 for ds in datasets
#                                             ],
#                                             value=datasets[0] if datasets else None,
#                                         ),
#                                     ],
#                                     width=4,
#                                 ),
#                                 dbc.Col(
#                                     [
#                                         dbc.Label("Variables:"),
#                                         dcc.Dropdown(
#                                             id="variable-dropdown", multi=True
#                                         ),
#                                     ],
#                                     width=4,
#                                 ),
#                                 dbc.Col(
#                                     [
#                                         dbc.Label("Chart Type:"),
#                                         dcc.Dropdown(
#                                             id="chart-type-dropdown",
#                                             options=[
#                                                 {
#                                                     "label": "Line Chart",
#                                                     "value": "line",
#                                                 },
#                                                 {
#                                                     "label": "Area Chart",
#                                                     "value": "area",
#                                                 },
#                                             ],
#                                             value="line",
#                                         ),
#                                     ],
#                                     width=4,
#                                 ),
#                             ]
#                         )
#                     ]
#                 ),
#             ],
#             className="mb-4",
#         )

#         content = html.Div(
#             [
#                 dbc.Row(
#                     [
#                         dbc.Col(
#                             [
#                                 dbc.Card(
#                                     [
#                                         dbc.CardHeader("Time Series Visualization"),
#                                         dbc.CardBody(
#                                             [
#                                                 dcc.Graph(
#                                                     id="explorer-chart",
#                                                     style={"height": "400px"},
#                                                 )
#                                             ]
#                                         ),
#                                     ]
#                                 )
#                             ],
#                             width=12,
#                         )
#                     ],
#                     className="mb-4",
#                 ),
#                 dbc.Row(
#                     [
#                         dbc.Col(
#                             [
#                                 dbc.Card(
#                                     [
#                                         dbc.CardHeader("Statistical Summary"),
#                                         dbc.CardBody([html.Div(id="data-summary")]),
#                                     ]
#                                 )
#                             ],
#                             width=12,
#                         )
#                     ]
#                 ),
#             ]
#         )

#         return html.Div([html.H2("Data Explorer", className="mb-4"), controls, content])

#     def create_statistics_tab(self):
#         """Create statistical analysis tab"""
#         datasets = self.data_manager.get_available_datasets()

#         controls = dbc.Card(
#             [
#                 dbc.CardHeader("Statistical Analysis Controls"),
#                 dbc.CardBody(
#                     [
#                         dbc.Row(
#                             [
#                                 dbc.Col(
#                                     [
#                                         dbc.Label("Dataset:"),
#                                         dcc.Dropdown(
#                                             id="stat-dataset-dropdown",
#                                             options=[
#                                                 {"label": ds, "value": ds}
#                                                 for ds in datasets
#                                             ],
#                                             value=datasets[0] if datasets else None,
#                                         ),
#                                     ],
#                                     width=6,
#                                 ),
#                                 dbc.Col(
#                                     [
#                                         dbc.Label("Variable:"),
#                                         dcc.Dropdown(id="stat-variable-dropdown"),
#                                     ],
#                                     width=6,
#                                 ),
#                             ],
#                             className="mb-3",
#                         ),
#                         dbc.Row(
#                             [
#                                 dbc.Col(
#                                     [
#                                         dbc.Button(
#                                             "Full Statistical Analysis",
#                                             id="run-stats-button",
#                                             color="primary",
#                                             className="me-2",
#                                         )
#                                     ],
#                                     width=4,
#                                 ),
#                                 dbc.Col(
#                                     [
#                                         dbc.Button(
#                                             "Quick Stationarity Test",
#                                             id="quick-stationarity-btn",
#                                             color="secondary",
#                                             className="me-2",
#                                         )
#                                     ],
#                                     width=4,
#                                 ),
#                                 dbc.Col(
#                                     [
#                                         dbc.Button(
#                                             "Quick Normality Test",
#                                             id="quick-normality-btn",
#                                             color="info",
#                                         )
#                                     ],
#                                     width=4,
#                                 ),
#                             ]
#                         ),
#                     ]
#                 ),
#             ],
#             className="mb-4",
#         )

#         results = html.Div([html.Div(id="stat-results")])

#         return html.Div(
#             [html.H2("Statistical Analysis", className="mb-4"), controls, results]
#         )

#     def create_forecasting_tab(self):
#         """Create forecasting tab"""
#         return dbc.Card(
#             [
#                 dbc.CardHeader("Advanced Forecasting Module"),
#                 dbc.CardBody(
#                     [
#                         html.H4(
#                             "🔮 Forecasting Capabilities", className="text-primary mb-3"
#                         ),
#                         html.P(
#                             "This module integrates with the comprehensive EDA findings to provide:"
#                         ),
#                         dbc.Row(
#                             [
#                                 dbc.Col(
#                                     [
#                                         dbc.ListGroup(
#                                             [
#                                                 dbc.ListGroupItem(
#                                                     [
#                                                         html.Strong("SARIMA Modeling"),
#                                                         html.P(
#                                                             "Based on ACF/PACF analysis showing seasonal patterns",
#                                                             className="mb-0 text-muted",
#                                                         ),
#                                                     ]
#                                                 ),
#                                                 dbc.ListGroupItem(
#                                                     [
#                                                         html.Strong(
#                                                             "LSTM Neural Networks"
#                                                         ),
#                                                         html.P(
#                                                             "Leveraging engineered features from EDA",
#                                                             className="mb-0 text-muted",
#                                                         ),
#                                                     ]
#                                                 ),
#                                                 dbc.ListGroupItem(
#                                                     [
#                                                         html.Strong(
#                                                             "Statistical Validation"
#                                                         ),
#                                                         html.P(
#                                                             "Using stationarity and normality tests",
#                                                             className="mb-0 text-muted",
#                                                         ),
#                                                     ]
#                                                 ),
#                                             ]
#                                         )
#                                     ],
#                                     width=8,
#                                 ),
#                                 dbc.Col(
#                                     [
#                                         dbc.Card(
#                                             [
#                                                 dbc.CardBody(
#                                                     [
#                                                         html.H6("EDA Insights Applied"),
#                                                         html.Ul(
#                                                             [
#                                                                 html.Li(
#                                                                     "Non-stationary series → Differencing required"
#                                                                 ),
#                                                                 html.Li(
#                                                                     "Strong seasonality → SARIMA appropriate"
#                                                                 ),
#                                                                 html.Li(
#                                                                     "High autocorrelation → ARIMA suitable"
#                                                                 ),
#                                                                 html.Li(
#                                                                     "Non-normal distribution → Robust methods"
#                                                                 ),
#                                                             ]
#                                                         ),
#                                                     ]
#                                                 )
#                                             ]
#                                         )
#                                     ],
#                                     width=4,
#                                 ),
#                             ]
#                         ),
#                     ]
#                 ),
#             ]
#         )

#     def create_overview_chart(self, df, youth_df):
#         """Create overview chart with unemployment trends"""
#         fig = make_subplots(
#             rows=2,
#             cols=1,
#             subplot_titles=("Unemployment Rate Trends", "Labor Force Dynamics"),
#             vertical_spacing=0.1,
#         )

#         # Unemployment rates
#         fig.add_trace(
#             go.Scatter(
#                 x=df.index,
#                 y=df["u_rate"],
#                 name="Overall Rate",
#                 line=dict(color=self.colors["primary"], width=2),
#             ),
#             row=1,
#             col=1,
#         )

#         # Add youth rate (reindex to match overall data dates)
#         youth_reindexed = youth_df["u_rate_15_24"].reindex(df.index)
#         fig.add_trace(
#             go.Scatter(
#                 x=df.index,
#                 y=youth_reindexed,
#                 name="Youth Rate (15-24)",
#                 line=dict(color=self.colors["warning"], width=2),
#             ),
#             row=1,
#             col=1,
#         )

#         # Labor force
#         fig.add_trace(
#             go.Scatter(
#                 x=df.index,
#                 y=df["lf"],
#                 name="Labor Force",
#                 line=dict(color=self.colors["success"]),
#             ),
#             row=2,
#             col=1,
#         )

#         fig.add_trace(
#             go.Scatter(
#                 x=df.index,
#                 y=df["lf_employed"],
#                 name="Employed",
#                 line=dict(color=self.colors["info"]),
#             ),
#             row=2,
#             col=1,
#         )

#         fig.update_layout(height=500, showlegend=True, hovermode="x unified")

#         return fig

#     def create_time_series_chart(self, df, variables, chart_type):
#         """Create time series chart for data explorer"""
#         fig = go.Figure()

#         for var in variables:
#             if var in df.columns:
#                 if chart_type == "line":
#                     fig.add_trace(
#                         go.Scatter(x=df.index, y=df[var], name=var, mode="lines")
#                     )
#                 elif chart_type == "area":
#                     fig.add_trace(
#                         go.Scatter(
#                             x=df.index,
#                             y=df[var],
#                             name=var,
#                             fill="tonexty" if len(fig.data) > 0 else "tozeroy",
#                         )
#                     )

#         fig.update_layout(
#             title="Time Series Analysis",
#             xaxis_title="Date",
#             yaxis_title="Value",
#             hovermode="x unified",
#             height=400,
#         )

#         return fig

#     def create_full_stats_display(self, results):
#         """Create display for full statistical analysis"""
#         stationarity = results["stationarity_analysis"]["combined_analysis"]
#         normality = results["normality_analysis"]
#         desc_stats = results["descriptive_statistics"]

#         return dbc.Row(
#             [
#                 # Series Information
#                 dbc.Col(
#                     [
#                         dbc.Card(
#                             [
#                                 dbc.CardHeader("📊 Series Information"),
#                                 dbc.CardBody(
#                                     [
#                                         html.P(
#                                             f"Series: {results['series_info']['name']}"
#                                         ),
#                                         html.P(
#                                             f"Valid observations: {results['series_info']['valid_observations']:,}"
#                                         ),
#                                         html.P(
#                                             f"Missing values: {results['series_info']['missing_values']}"
#                                         ),
#                                         html.P(
#                                             f"Date range: {results['series_info']['date_range']['start']} to {results['series_info']['date_range']['end']}"
#                                         ),
#                                     ]
#                                 ),
#                             ]
#                         )
#                     ],
#                     width=6,
#                     className="mb-3",
#                 ),
#                 # Descriptive Statistics
#                 dbc.Col(
#                     [
#                         dbc.Card(
#                             [
#                                 dbc.CardHeader("📈 Descriptive Statistics"),
#                                 dbc.CardBody(
#                                     [
#                                         html.P(f"Mean: {desc_stats['mean']:.3f}"),
#                                         html.P(f"Std Dev: {desc_stats['std']:.3f}"),
#                                         html.P(f"Min: {desc_stats['min']:.3f}"),
#                                         html.P(f"Max: {desc_stats['max']:.3f}"),
#                                     ]
#                                 ),
#                             ]
#                         )
#                     ],
#                     width=6,
#                     className="mb-3",
#                 ),
#                 # Stationarity Analysis
#                 dbc.Col(
#                     [
#                         dbc.Card(
#                             [
#                                 dbc.CardHeader("🔄 Stationarity Analysis"),
#                                 dbc.CardBody(
#                                     [
#                                         dbc.Alert(
#                                             [
#                                                 html.H6(
#                                                     f"Result: {stationarity['conclusion']}",
#                                                     className="alert-heading",
#                                                 ),
#                                                 html.P(
#                                                     f"Confidence: {stationarity['confidence']}"
#                                                 ),
#                                                 html.Hr(),
#                                                 html.P(
#                                                     stationarity["recommendation"],
#                                                     className="mb-0",
#                                                 ),
#                                             ],
#                                             color=(
#                                                 "success"
#                                                 if stationarity["conclusion"]
#                                                 == "Stationary"
#                                                 else "warning"
#                                             ),
#                                         )
#                                     ]
#                                 ),
#                             ]
#                         )
#                     ],
#                     width=6,
#                     className="mb-3",
#                 ),
#                 # Normality Analysis
#                 dbc.Col(
#                     [
#                         dbc.Card(
#                             [
#                                 dbc.CardHeader("📊 Normality Analysis"),
#                                 dbc.CardBody(
#                                     [
#                                         dbc.Alert(
#                                             [
#                                                 html.H6(
#                                                     f"Result: {normality['consensus']}",
#                                                     className="alert-heading",
#                                                 ),
#                                                 html.P(
#                                                     f"Confidence: {normality['confidence']}"
#                                                 ),
#                                                 html.Hr(),
#                                                 html.P(
#                                                     normality["recommendation"],
#                                                     className="mb-0",
#                                                 ),
#                                             ],
#                                             color=(
#                                                 "info"
#                                                 if normality["consensus"] == "Normal"
#                                                 else "secondary"
#                                             ),
#                                         )
#                                     ]
#                                 ),
#                             ]
#                         )
#                     ],
#                     width=6,
#                     className="mb-3",
#                 ),
#                 # Recommendations
#                 dbc.Col(
#                     [
#                         dbc.Card(
#                             [
#                                 dbc.CardHeader("💡 Modeling Recommendations"),
#                                 dbc.CardBody(
#                                     [
#                                         html.Ol(
#                                             [
#                                                 html.Li(rec)
#                                                 for rec in results["recommendations"]
#                                             ]
#                                         )
#                                     ]
#                                 ),
#                             ]
#                         )
#                     ],
#                     width=12,
#                 ),
#             ]
#         )

#     def create_quick_stat_display(self, results, test_type):
#         """Create display for quick stationarity test"""
#         conclusion = results["combined_analysis"]["conclusion"]
#         confidence = results["combined_analysis"]["confidence"]
#         recommendation = results["combined_analysis"]["recommendation"]

#         return dbc.Card(
#             [
#                 dbc.CardHeader(f"⚡ Quick {test_type} Test"),
#                 dbc.CardBody(
#                     [
#                         dbc.Alert(
#                             [
#                                 html.H5(
#                                     f"Result: {conclusion}", className="alert-heading"
#                                 ),
#                                 html.P(f"Confidence: {confidence}"),
#                                 html.Hr(),
#                                 html.P(recommendation, className="mb-0"),
#                             ],
#                             color=(
#                                 "primary" if conclusion == "Stationary" else "warning"
#                             ),
#                         )
#                     ]
#                 ),
#             ]
#         )

#     def create_quick_norm_display(self, results, test_type):
#         """Create display for quick normality test"""
#         consensus = results["consensus"]
#         confidence = results["confidence"]
#         recommendation = results["recommendation"]

#         return dbc.Card(
#             [
#                 dbc.CardHeader(f"⚡ Quick {test_type} Test"),
#                 dbc.CardBody(
#                     [
#                         dbc.Alert(
#                             [
#                                 html.H5(
#                                     f"Result: {consensus}", className="alert-heading"
#                                 ),
#                                 html.P(f"Confidence: {confidence}"),
#                                 html.Hr(),
#                                 html.P(recommendation, className="mb-0"),
#                             ],
#                             color="info" if consensus == "Normal" else "secondary",
#                         )
#                     ]
#                 ),
#             ]
#         )


# # Create the application instance that run_dashboard.py expects
# app = MalaysiaUnemploymentDashboard()

# # Expose the server for WSGI deployment
# server = app.app.server

# if __name__ == "__main__":
#     # This allows running the dashboard directly
#     print("=" * 60)
#     print("MALAYSIA UNEMPLOYMENT ANALYTICS DASHBOARD")
#     print("=" * 60)
#     print("🏗️  Starting dashboard server...")
#     print("📊 Dashboard available at: http://127.0.0.1:8050")
#     print("💡 Features:")
#     print("   - Market Overview with key metrics")
#     print("   - Interactive Data Explorer")
#     print("   - Statistical Analysis (Stationarity & Normality)")
#     print("   - Forecasting Framework")
#     print("=" * 60)

#     # Run with Dash's app.run method (newer versions)
#     try:
#         app.app.run(host="127.0.0.1", port=8050, debug=True)
#     except AttributeError:
#         # Fallback to older method if needed
#         try:
#             app.app.run_server(host="127.0.0.1", port=8050, debug=True)
#         except Exception as e:
#             print(f"Error starting dashboard: {e}")
#             print("Please check your Dash version or use run_dashboard.py")

"""
Malaysia Unemployment Analytics Dashboard - Main Application
Professional entry point with modular architecture.
"""

import dash
from dash import html, dcc
import dash_bootstrap_components as dbc
from datetime import datetime
import sys
from pathlib import Path

# Add project root to path for imports
current_dir = Path(__file__).parent
project_root = current_dir.parent.parent
sys.path.insert(0, str(project_root / "src"))

from dashboard.utils.data_manager import DashboardDataManager
from dashboard.utils.theme import ThemeManager
from dashboard.components.layout import create_main_layout
from dashboard.components.callbacks import register_all_callbacks


class MalaysiaUnemploymentDashboard:
    """Professional Malaysia Labor Force Analytics Dashboard"""

    def __init__(self):
        self.app = dash.Dash(
            __name__,
            external_stylesheets=[
                dbc.themes.BOOTSTRAP,
                "https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0/css/all.min.css",
            ],
            suppress_callback_exceptions=True,
        )

        self.app.title = "Malaysia Labor Force Analytics"

        # Initialize managers
        self.theme = ThemeManager()
        self.data_manager = DashboardDataManager()

        # Setup application
        self._setup_layout()
        self._register_callbacks()
        self._initialize_data()

    def _setup_layout(self):
        """Setup main application layout with custom CSS"""
        self.app.index_string = self.theme.get_custom_css()
        self.app.layout = create_main_layout(self.theme.colors)

    def _register_callbacks(self):
        """Register all dashboard callbacks"""
        register_all_callbacks(self.app, self.data_manager, self.theme.colors)

    def _initialize_data(self):
        """Auto-initialize data on startup"""
        success = self.data_manager.initialize()
        status = (
            "✅ Data auto-initialized successfully"
            if success
            else "⚠️ Using fallback data"
        )
        print(status)

    def run(self, host="127.0.0.1", port=8050, debug=True):
        """Run the dashboard application"""
        print("=" * 70)
        print("🇲🇾 MALAYSIA UNEMPLOYMENT ANALYTICS DASHBOARD")
        print("=" * 70)
        print(f"🚀 Starting server on http://{host}:{port}")
        print("📊 Professional modular architecture")
        print("🎨 Enhanced styling and user experience")
        print("=" * 70)

        try:
            self.app.run(host=host, port=port, debug=debug)
        except AttributeError:
            self.app.run_server(host=host, port=port, debug=debug)


# Create application instance
app = MalaysiaUnemploymentDashboard()
server = app.app.server


def main():
    """Main function to run the dashboard"""
    app.run()


if __name__ == "__main__":
    main()

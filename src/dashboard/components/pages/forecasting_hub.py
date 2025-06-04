# # """
# # Forecasting Hub page components for advanced modeling.
# # Displays SARIMA and LSTM capabilities with EDA integration.
# # """

# # from dash import html
# # import dash_bootstrap_components as dbc
# # from dashboard.components.layout import create_page_header


# # def create_forecasting_page(data_manager, colors):
# #     """Create forecasting hub page"""
# #     header = create_page_header(
# #         "Forecasting Hub",
# #         "Advanced unemployment forecasting using SARIMA and LSTM models",
# #         "fas fa-crystal-ball",
# #         colors,
# #     )

# #     main_content = create_forecasting_content(colors)

# #     return html.Div([header, main_content])


# # def create_forecasting_content(colors):
# #     """Create main forecasting content"""
# #     return dbc.Card(
# #         [
# #             dbc.CardHeader(
# #                 [
# #                     html.I(className="fas fa-magic", style={"marginRight": "8px"}),
# #                     "Advanced Forecasting Module",
# #                 ],
# #                 style={
# #                     "background": f'linear-gradient(90deg, {colors["primary"]} 0%, {colors["secondary"]} 100%)',
# #                     "color": "white",
# #                     "fontWeight": "bold",
# #                 },
# #             ),
# #             dbc.CardBody(
# #                 [
# #                     html.H4(
# #                         [
# #                             html.I(
# #                                 className="fas fa-brain", style={"marginRight": "10px"}
# #                             ),
# #                             "Forecasting Capabilities",
# #                         ],
# #                         style={"color": colors["primary"], "marginBottom": "20px"},
# #                     ),
# #                     html.P(
# #                         "This module integrates with comprehensive EDA findings and statistical analysis:",
# #                         style={"color": colors["text"], "fontSize": "16px"},
# #                     ),
# #                     create_capabilities_section(colors),
# #                     create_eda_insights_section(colors),
# #                 ],
# #                 style={"backgroundColor": "rgba(255,255,255,0.5)"},
# #             ),
# #         ],
# #         className="status-card",
# #     )


# # def create_capabilities_section(colors):
# #     """Create forecasting capabilities section"""
# #     capabilities = [
# #         {
# #             "icon": "fas fa-chart-line",
# #             "title": "SARIMA Modeling",
# #             "description": "Based on ACF/PACF analysis showing seasonal patterns",
# #         },
# #         {
# #             "icon": "fas fa-brain",
# #             "title": "LSTM Neural Networks",
# #             "description": "Leveraging engineered features from EDA",
# #         },
# #         {
# #             "icon": "fas fa-check-circle",
# #             "title": "Statistical Validation",
# #             "description": "Using stationarity and normality tests",
# #         },
# #         {
# #             "icon": "fas fa-sync-alt",
# #             "title": "Real-time Updates",
# #             "description": "Auto-refresh capabilities with live DOSM data",
# #         },
# #     ]

# #     capability_items = []
# #     for cap in capabilities:
# #         item = dbc.ListGroupItem(
# #             [
# #                 html.Strong(
# #                     [
# #                         html.I(className=cap["icon"], style={"marginRight": "8px"}),
# #                         cap["title"],
# #                     ]
# #                 ),
# #                 html.P(
# #                     cap["description"],
# #                     className="mb-0",
# #                     style={"color": colors["text"], "opacity": 0.8},
# #                 ),
# #             ],
# #             style={"backgroundColor": "rgba(255,255,255,0.7)"},
# #         )
# #         capability_items.append(item)

# #     return dbc.Row(
# #         [
# #             dbc.Col([dbc.ListGroup(capability_items)], width=8),
# #             dbc.Col([create_eda_insights_card(colors)], width=4),
# #         ]
# #     )


# # def create_eda_insights_card(colors):
# #     """Create EDA insights card"""
# #     insights = [
# #         "Non-stationary series → Differencing required",
# #         "Strong seasonality → SARIMA appropriate",
# #         "High autocorrelation → ARIMA suitable",
# #         "Non-normal distribution → Robust methods",
# #         "Youth unemployment correlation patterns",
# #     ]

# #     return dbc.Card(
# #         [
# #             dbc.CardBody(
# #                 [
# #                     html.H6(
# #                         [
# #                             html.I(
# #                                 className="fas fa-lightbulb",
# #                                 style={"marginRight": "8px"},
# #                             ),
# #                             "EDA Insights Applied",
# #                         ],
# #                         style={"color": colors["primary"]},
# #                     ),
# #                     html.Ul(
# #                         [html.Li(insight) for insight in insights],
# #                         style={"color": colors["text"]},
# #                     ),
# #                 ]
# #             )
# #         ],
# #         style={"background": colors["light"]},
# #     )


# # def create_eda_insights_section(colors):
# #     """Create detailed EDA insights section"""
# #     return html.Div(
# #         [
# #             html.Hr(style={"margin": "30px 0"}),
# #             html.H5(
# #                 [
# #                     html.I(className="fas fa-analytics", style={"marginRight": "10px"}),
# #                     "Integration with Statistical Analysis",
# #                 ],
# #                 style={"color": colors["primary"], "marginBottom": "20px"},
# #             ),
# #             create_modeling_workflow(colors),
# #             create_feature_engineering_section(colors),
# #         ]
# #     )


# # def create_modeling_workflow(colors):
# #     """Create modeling workflow visualization"""
# #     workflow_steps = [
# #         {
# #             "step": "1",
# #             "title": "Data Preparation",
# #             "items": [
# #                 "Stationarity testing",
# #                 "Seasonal decomposition",
# #                 "Outlier detection",
# #             ],
# #             "color": colors["info"],
# #         },
# #         {
# #             "step": "2",
# #             "title": "Model Selection",
# #             "items": ["ACF/PACF analysis", "Information criteria", "Cross-validation"],
# #             "color": colors["warning"],
# #         },
# #         {
# #             "step": "3",
# #             "title": "Forecasting",
# #             "items": ["SARIMA modeling", "LSTM networks", "Ensemble methods"],
# #             "color": colors["success"],
# #         },
# #         {
# #             "step": "4",
# #             "title": "Validation",
# #             "items": ["Residual analysis", "Forecast accuracy", "Model diagnostics"],
# #             "color": colors["danger"],
# #         },
# #     ]

# #     workflow_cards = []
# #     for step in workflow_steps:
# #         card = dbc.Col(
# #             [
# #                 dbc.Card(
# #                     [
# #                         dbc.CardHeader(
# #                             [
# #                                 html.H5(
# #                                     f"Step {step['step']}",
# #                                     style={"color": "white", "margin": 0},
# #                                 )
# #                             ],
# #                             style={"background": step["color"], "textAlign": "center"},
# #                         ),
# #                         dbc.CardBody(
# #                             [
# #                                 html.H6(
# #                                     step["title"],
# #                                     style={
# #                                         "color": colors["text"],
# #                                         "fontWeight": "bold",
# #                                     },
# #                                 ),
# #                                 html.Ul(
# #                                     [
# #                                         html.Li(item, style={"fontSize": "14px"})
# #                                         for item in step["items"]
# #                                     ],
# #                                     style={"color": colors["text"], "marginBottom": 0},
# #                                 ),
# #                             ],
# #                             style={"backgroundColor": "rgba(255,255,255,0.8)"},
# #                         ),
# #                     ],
# #                     className="h-100",
# #                 )
# #             ],
# #             width=3,
# #         )
# #         workflow_cards.append(card)

# #     return dbc.Row(workflow_cards, className="mb-4")


# # def create_feature_engineering_section(colors):
# #     """Create feature engineering section"""
# #     return html.Div(
# #         [
# #             html.H6(
# #                 [
# #                     html.I(className="fas fa-tools", style={"marginRight": "8px"}),
# #                     "Feature Engineering Strategy",
# #                 ],
# #                 style={"color": colors["primary"], "marginBottom": "15px"},
# #             ),
# #             dbc.Row(
# #                 [
# #                     dbc.Col(
# #                         [
# #                             create_feature_category_card(
# #                                 "Time-based Features",
# #                                 [
# #                                     "Month indicators",
# #                                     "Seasonal dummies",
# #                                     "Trend components",
# #                                     "Cyclical patterns",
# #                                 ],
# #                                 "fas fa-calendar",
# #                                 colors["info"],
# #                                 colors,
# #                             )
# #                         ],
# #                         width=4,
# #                     ),
# #                     dbc.Col(
# #                         [
# #                             create_feature_category_card(
# #                                 "Lagged Variables",
# #                                 [
# #                                     "Previous unemployment rates",
# #                                     "Lagged labor force",
# #                                     "Moving averages",
# #                                     "Seasonal lags",
# #                                 ],
# #                                 "fas fa-history",
# #                                 colors["warning"],
# #                                 colors,
# #                             )
# #                         ],
# #                         width=4,
# #                     ),
# #                     dbc.Col(
# #                         [
# #                             create_feature_category_card(
# #                                 "External Indicators",
# #                                 [
# #                                     "Economic indicators",
# #                                     "Policy changes",
# #                                     "Global factors",
# #                                     "Sectoral data",
# #                                 ],
# #                                 "fas fa-globe",
# #                                 colors["success"],
# #                                 colors,
# #                             )
# #                         ],
# #                         width=4,
# #                     ),
# #                 ],
# #                 className="mb-4",
# #             ),
# #             create_model_comparison_section(colors),
# #         ]
# #     )


# # def create_feature_category_card(title, features, icon, header_color, colors):
# #     """Create feature category card"""
# #     return dbc.Card(
# #         [
# #             dbc.CardHeader(
# #                 [html.I(className=icon, style={"marginRight": "8px"}), title],
# #                 style={
# #                     "background": header_color,
# #                     "color": "white",
# #                     "fontWeight": "bold",
# #                 },
# #             ),
# #             dbc.CardBody(
# #                 [
# #                     html.Ul(
# #                         [
# #                             html.Li(feature, style={"fontSize": "13px"})
# #                             for feature in features
# #                         ],
# #                         style={"color": colors["text"], "marginBottom": 0},
# #                     )
# #                 ],
# #                 style={"backgroundColor": "rgba(255,255,255,0.8)"},
# #             ),
# #         ],
# #         className="h-100",
# #     )


# # def create_model_comparison_section(colors):
# #     """Create model comparison section"""
# #     return html.Div(
# #         [
# #             html.H6(
# #                 [
# #                     html.I(
# #                         className="fas fa-balance-scale", style={"marginRight": "8px"}
# #                     ),
# #                     "Model Comparison Framework",
# #                 ],
# #                 style={"color": colors["primary"], "marginBottom": "15px"},
# #             ),
# #             dbc.Row(
# #                 [
# #                     dbc.Col(
# #                         [
# #                             create_model_card(
# #                                 "SARIMA",
# #                                 {
# #                                     "Strengths": [
# #                                         "Interpretable",
# #                                         "Statistical foundation",
# #                                         "Seasonal handling",
# #                                         "Confidence intervals",
# #                                     ],
# #                                     "Best for": [
# #                                         "Medium-term forecasts",
# #                                         "Policy analysis",
# #                                         "Trend interpretation",
# #                                     ],
# #                                 },
# #                                 colors["primary"],
# #                                 colors,
# #                             )
# #                         ],
# #                         width=6,
# #                     ),
# #                     dbc.Col(
# #                         [
# #                             create_model_card(
# #                                 "LSTM",
# #                                 {
# #                                     "Strengths": [
# #                                         "Non-linear patterns",
# #                                         "Complex dependencies",
# #                                         "Multi-variate",
# #                                         "Long sequences",
# #                                     ],
# #                                     "Best for": [
# #                                         "Short-term forecasts",
# #                                         "Pattern recognition",
# #                                         "High-frequency data",
# #                                     ],
# #                                 },
# #                                 colors["secondary"],
# #                                 colors,
# #                             )
# #                         ],
# #                         width=6,
# #                     ),
# #                 ],
# #                 className="mb-4",
# #             ),
# #             create_ensemble_strategy_section(colors),
# #         ]
# #     )


# # def create_model_card(model_name, details, header_color, colors):
# #     """Create individual model comparison card"""
# #     return dbc.Card(
# #         [
# #             dbc.CardHeader(
# #                 [
# #                     html.H6(
# #                         model_name,
# #                         style={"color": "white", "margin": 0, "fontWeight": "bold"},
# #                     )
# #                 ],
# #                 style={"background": header_color, "textAlign": "center"},
# #             ),
# #             dbc.CardBody(
# #                 [
# #                     html.Div(
# #                         [
# #                             html.Strong("Strengths:", style={"color": colors["text"]}),
# #                             html.Ul(
# #                                 [
# #                                     html.Li(strength, style={"fontSize": "12px"})
# #                                     for strength in details["Strengths"]
# #                                 ],
# #                                 style={"color": colors["text"], "marginBottom": "10px"},
# #                             ),
# #                         ]
# #                     ),
# #                     html.Div(
# #                         [
# #                             html.Strong("Best for:", style={"color": colors["text"]}),
# #                             html.Ul(
# #                                 [
# #                                     html.Li(use_case, style={"fontSize": "12px"})
# #                                     for use_case in details["Best for"]
# #                                 ],
# #                                 style={"color": colors["text"], "marginBottom": 0},
# #                             ),
# #                         ]
# #                     ),
# #                 ],
# #                 style={"backgroundColor": "rgba(255,255,255,0.8)"},
# #             ),
# #         ],
# #         className="h-100",
# #     )


# # def create_ensemble_strategy_section(colors):
# #     """Create ensemble strategy section"""
# #     return html.Div(
# #         [
# #             html.H6(
# #                 [
# #                     html.I(
# #                         className="fas fa-layer-group", style={"marginRight": "8px"}
# #                     ),
# #                     "Ensemble Strategy",
# #                 ],
# #                 style={"color": colors["primary"], "marginBottom": "15px"},
# #             ),
# #             dbc.Alert(
# #                 [
# #                     html.H6(
# #                         [
# #                             html.I(
# #                                 className="fas fa-rocket", style={"marginRight": "8px"}
# #                             ),
# #                             "Hybrid Approach",
# #                         ],
# #                         className="alert-heading",
# #                     ),
# #                     html.P(
# #                         [
# #                             "Combine SARIMA for trend and seasonality with LSTM for complex patterns. ",
# #                             "Weight models based on forecast horizon and historical performance.",
# #                         ],
# #                         className="mb-0",
# #                     ),
# #                 ],
# #                 color="info",
# #                 style={
# #                     "backgroundColor": "rgba(123,154,168,0.2)",
# #                     "border": f'1px solid {colors["info"]}',
# #                 },
# #             ),
# #             create_performance_metrics_section(colors),
# #         ]
# #     )


# # def create_performance_metrics_section(colors):
# #     """Create performance metrics section"""
# #     return html.Div(
# #         [
# #             html.H6(
# #                 [
# #                     html.I(className="fas fa-chart-bar", style={"marginRight": "8px"}),
# #                     "Performance Evaluation",
# #                 ],
# #                 style={
# #                     "color": colors["primary"],
# #                     "marginTop": "20px",
# #                     "marginBottom": "15px",
# #                 },
# #             ),
# #             dbc.Row(
# #                 [
# #                     dbc.Col(
# #                         [
# #                             create_metrics_card(
# #                                 "Accuracy Metrics",
# #                                 ["MAPE", "RMSE", "MAE", "SMAPE"],
# #                                 "fas fa-bullseye",
# #                                 colors,
# #                             )
# #                         ],
# #                         width=3,
# #                     ),
# #                     dbc.Col(
# #                         [
# #                             create_metrics_card(
# #                                 "Statistical Tests",
# #                                 [
# #                                     "Ljung-Box",
# #                                     "Jarque-Bera",
# #                                     "ADF residuals",
# #                                     "Heteroscedasticity",
# #                                 ],
# #                                 "fas fa-flask",
# #                                 colors,
# #                             )
# #                         ],
# #                         width=3,
# #                     ),
# #                     dbc.Col(
# #                         [
# #                             create_metrics_card(
# #                                 "Business Metrics",
# #                                 [
# #                                     "Directional accuracy",
# #                                     "Peak detection",
# #                                     "Policy relevance",
# #                                     "Stakeholder feedback",
# #                                 ],
# #                                 "fas fa-briefcase",
# #                                 colors,
# #                             )
# #                         ],
# #                         width=3,
# #                     ),
# #                     dbc.Col(
# #                         [
# #                             create_metrics_card(
# #                                 "Robustness",
# #                                 [
# #                                     "Cross-validation",
# #                                     "Out-of-sample",
# #                                     "Stress testing",
# #                                     "Scenario analysis",
# #                                 ],
# #                                 "fas fa-shield-alt",
# #                                 colors,
# #                             )
# #                         ],
# #                         width=3,
# #                     ),
# #                 ]
# #             ),
# #         ]
# #     )


# # def create_metrics_card(title, metrics, icon, colors):
# #     """Create metrics category card"""
# #     return dbc.Card(
# #         [
# #             dbc.CardHeader(
# #                 [html.I(className=icon, style={"marginRight": "5px"}), title],
# #                 style={
# #                     "background": colors["light"],
# #                     "color": colors["text"],
# #                     "fontSize": "14px",
# #                     "fontWeight": "bold",
# #                     "textAlign": "center",
# #                 },
# #             ),
# #             dbc.CardBody(
# #                 [
# #                     html.Ul(
# #                         [
# #                             html.Li(metric, style={"fontSize": "11px"})
# #                             for metric in metrics
# #                         ],
# #                         style={"color": colors["text"], "marginBottom": 0},
# #                     )
# #                 ],
# #                 style={"backgroundColor": "rgba(255,255,255,0.9)", "padding": "10px"},
# #             ),
# #         ],
# #         className="h-100",
# #     )

# """
# Enhanced Forecasting Hub page for unemployment rate predictions.
# User-friendly interface for general users with pre-trained models.
# """

# from dash import html, dcc
# import dash_bootstrap_components as dbc
# import plotly.graph_objects as go
# from plotly.subplots import make_subplots
# import pandas as pd
# import numpy as np
# from datetime import datetime, timedelta
# from dashboard.components.layout import (
#     create_page_header,
#     create_control_card,
#     create_chart_container,
# )


# def create_forecasting_page(data_manager, colors):
#     """Create enhanced forecasting hub page for general users"""
#     header = create_page_header(
#         "Forecasting Hub",
#         "Generate unemployment rate forecasts using advanced AI models",
#         "fas fa-crystal-ball",
#         colors,
#     )

#     controls = create_forecasting_controls(colors)
#     forecast_section = create_forecast_section(colors)

#     return html.Div([header, controls, forecast_section])


# def create_forecasting_controls(colors):
#     """Create user-friendly forecasting controls"""
#     controls_content = [
#         # Dataset Selection
#         dbc.Row(
#             [
#                 dbc.Col(
#                     [
#                         dbc.Label(
#                             "📊 Select Dataset:",
#                             style={
#                                 "fontWeight": "bold",
#                                 "color": colors["text"],
#                                 "fontSize": "16px",
#                             },
#                         ),
#                         dcc.Dropdown(
#                             id="forecast-dataset-dropdown",
#                             options=[
#                                 {
#                                     "label": "🔹 General Labor Force (Raw Data)",
#                                     "value": "general",
#                                 },
#                                 {
#                                     "label": "🔹 Seasonally Adjusted Labor Force",
#                                     "value": "sa",
#                                 },
#                             ],
#                             value="general",
#                             style={"marginBottom": "15px"},
#                         ),
#                         html.P(
#                             "Choose between raw data or seasonally adjusted unemployment data",
#                             style={
#                                 "color": colors["text"],
#                                 "fontSize": "12px",
#                                 "opacity": 0.8,
#                             },
#                         ),
#                     ],
#                     width=6,
#                 ),
#                 # Model Selection
#                 dbc.Col(
#                     [
#                         dbc.Label(
#                             "🤖 Select AI Model:",
#                             style={
#                                 "fontWeight": "bold",
#                                 "color": colors["text"],
#                                 "fontSize": "16px",
#                             },
#                         ),
#                         dcc.Dropdown(
#                             id="forecast-model-dropdown",
#                             options=[
#                                 {
#                                     "label": "📈 ARIMA - Statistical Time Series",
#                                     "value": "arima",
#                                 },
#                                 {
#                                     "label": "📊 SARIMA - Seasonal Time Series",
#                                     "value": "sarima",
#                                 },
#                             ],
#                             value="sarima",
#                             style={"marginBottom": "15px"},
#                         ),
#                         html.P(
#                             "SARIMA handles seasonal patterns better for unemployment data",
#                             style={
#                                 "color": colors["text"],
#                                 "fontSize": "12px",
#                                 "opacity": 0.8,
#                             },
#                         ),
#                     ],
#                     width=6,
#                 ),
#             ],
#             className="mb-4",
#         ),
#         # Forecast Period and Generate Button
#         dbc.Row(
#             [
#                 dbc.Col(
#                     [
#                         dbc.Label(
#                             "⏱️ Forecast Period:",
#                             style={
#                                 "fontWeight": "bold",
#                                 "color": colors["text"],
#                                 "fontSize": "16px",
#                             },
#                         ),
#                         dcc.Dropdown(
#                             id="forecast-period-dropdown",
#                             options=[
#                                 {"label": "1 Month Ahead", "value": 1},
#                                 {"label": "3 Months Ahead", "value": 3},
#                                 {"label": "6 Months Ahead", "value": 6},
#                                 {"label": "12 Months Ahead", "value": 12},
#                             ],
#                             value=6,
#                             style={"marginBottom": "15px"},
#                         ),
#                         html.P(
#                             "Shorter periods are generally more accurate",
#                             style={
#                                 "color": colors["text"],
#                                 "fontSize": "12px",
#                                 "opacity": 0.8,
#                             },
#                         ),
#                     ],
#                     width=4,
#                 ),
#                 dbc.Col(
#                     [
#                         dbc.Label(
#                             "🎯 Target Variable:",
#                             style={
#                                 "fontWeight": "bold",
#                                 "color": colors["text"],
#                                 "fontSize": "16px",
#                             },
#                         ),
#                         html.Div(
#                             [
#                                 dbc.Badge(
#                                     "Unemployment Rate (%)",
#                                     color="primary",
#                                     style={"fontSize": "14px", "padding": "10px 15px"},
#                                 )
#                             ],
#                             style={"marginBottom": "15px"},
#                         ),
#                         html.P(
#                             "Models are trained specifically for unemployment rate prediction",
#                             style={
#                                 "color": colors["text"],
#                                 "fontSize": "12px",
#                                 "opacity": 0.8,
#                             },
#                         ),
#                     ],
#                     width=4,
#                 ),
#                 dbc.Col(
#                     [
#                         dbc.Label(
#                             "🚀 Generate Forecast:",
#                             style={
#                                 "fontWeight": "bold",
#                                 "color": colors["text"],
#                                 "fontSize": "16px",
#                             },
#                         ),
#                         dbc.Button(
#                             [
#                                 html.I(
#                                     className="fas fa-magic",
#                                     style={"marginRight": "10px"},
#                                 ),
#                                 "Generate Prediction",
#                             ],
#                             id="generate-forecast-btn",
#                             color="success",
#                             size="lg",
#                             className="w-100",
#                             style={"marginTop": "5px", "fontWeight": "bold"},
#                         ),
#                     ],
#                     width=4,
#                 ),
#             ]
#         ),
#     ]

#     return create_control_card(
#         "🔮 Forecast Configuration", "fas fa-cogs", controls_content, colors
#     )


# def create_forecast_section(colors):
#     """Create forecast results and visualization section"""
#     return html.Div(
#         [
#             # Results Summary Cards
#             html.Div(id="forecast-summary-cards", className="mb-4"),
#             # Main Forecast Chart
#             dbc.Row(
#                 [
#                     dbc.Col(
#                         [
#                             html.Div(
#                                 [
#                                     html.H5(
#                                         [
#                                             html.I(
#                                                 className="fas fa-chart-line",
#                                                 style={"marginRight": "10px"},
#                                             ),
#                                             "Unemployment Rate Forecast",
#                                         ],
#                                         style={
#                                             "color": colors["primary"],
#                                             "fontWeight": "bold",
#                                         },
#                                     ),
#                                     dcc.Graph(
#                                         id="forecast-chart", style={"height": "500px"}
#                                     ),
#                                 ],
#                                 className="chart-container",
#                             )
#                         ],
#                         width=12,
#                     )
#                 ],
#                 className="mb-4",
#             ),
#             # Detailed Forecast Table
#             dbc.Row(
#                 [
#                     dbc.Col(
#                         [
#                             html.Div(
#                                 [
#                                     html.H5(
#                                         [
#                                             html.I(
#                                                 className="fas fa-table",
#                                                 style={"marginRight": "10px"},
#                                             ),
#                                             "Detailed Forecast Values",
#                                         ],
#                                         style={
#                                             "color": colors["primary"],
#                                             "fontWeight": "bold",
#                                         },
#                                     ),
#                                     html.Div(id="forecast-table"),
#                                 ],
#                                 className="chart-container",
#                             )
#                         ],
#                         width=8,
#                     ),
#                     # Model Information
#                     dbc.Col(
#                         [
#                             html.Div(
#                                 [
#                                     html.H5(
#                                         [
#                                             html.I(
#                                                 className="fas fa-info-circle",
#                                                 style={"marginRight": "10px"},
#                                             ),
#                                             "Model Information",
#                                         ],
#                                         style={
#                                             "color": colors["primary"],
#                                             "fontWeight": "bold",
#                                         },
#                                     ),
#                                     html.Div(id="model-info"),
#                                 ],
#                                 className="chart-container",
#                             )
#                         ],
#                         width=4,
#                     ),
#                 ]
#             ),
#         ]
#     )


# def create_forecast_summary_cards(forecast_data, colors):
#     """Create summary cards showing key forecast insights"""
#     try:
#         current_rate = forecast_data["current_rate"]
#         forecast_values = forecast_data["forecast_values"]
#         trend_direction = forecast_data["trend_direction"]
#         trend_magnitude = forecast_data["trend_magnitude"]
#         confidence_level = forecast_data["confidence_level"]

#         # Determine trend icon and color
#         if trend_direction == "Rising":
#             trend_icon = "fas fa-arrow-up"
#             trend_color = colors["danger"]
#             trend_desc = "Unemployment Expected to Rise"
#         elif trend_direction == "Falling":
#             trend_icon = "fas fa-arrow-down"
#             trend_color = colors["success"]
#             trend_desc = "Unemployment Expected to Fall"
#         else:
#             trend_icon = "fas fa-minus"
#             trend_color = colors["info"]
#             trend_desc = "Unemployment Expected to Remain Stable"

#         cards = [
#             # Current Rate
#             dbc.Col(
#                 [
#                     dbc.Card(
#                         [
#                             dbc.CardBody(
#                                 [
#                                     html.Div(
#                                         [
#                                             html.I(
#                                                 className="fas fa-chart-line",
#                                                 style={
#                                                     "fontSize": "28px",
#                                                     "color": colors["primary"],
#                                                     "float": "right",
#                                                 },
#                                             ),
#                                             html.H3(
#                                                 f"{current_rate:.1f}%",
#                                                 className="mb-1",
#                                                 style={
#                                                     "color": colors["primary"],
#                                                     "fontWeight": "bold",
#                                                 },
#                                             ),
#                                             html.P(
#                                                 "CURRENT RATE",
#                                                 className="text-muted mb-0",
#                                                 style={
#                                                     "fontSize": "12px",
#                                                     "fontWeight": "bold",
#                                                 },
#                                             ),
#                                         ]
#                                     )
#                                 ]
#                             )
#                         ],
#                         className="metric-card",
#                     )
#                 ],
#                 width=3,
#             ),
#             # Forecast End Rate
#             dbc.Col(
#                 [
#                     dbc.Card(
#                         [
#                             dbc.CardBody(
#                                 [
#                                     html.Div(
#                                         [
#                                             html.I(
#                                                 className="fas fa-target",
#                                                 style={
#                                                     "fontSize": "28px",
#                                                     "color": colors["secondary"],
#                                                     "float": "right",
#                                                 },
#                                             ),
#                                             html.H3(
#                                                 f"{forecast_values[-1]:.1f}%",
#                                                 className="mb-1",
#                                                 style={
#                                                     "color": colors["secondary"],
#                                                     "fontWeight": "bold",
#                                                 },
#                                             ),
#                                             html.P(
#                                                 "FORECAST END",
#                                                 className="text-muted mb-0",
#                                                 style={
#                                                     "fontSize": "12px",
#                                                     "fontWeight": "bold",
#                                                 },
#                                             ),
#                                         ]
#                                     )
#                                 ]
#                             )
#                         ],
#                         className="metric-card",
#                     )
#                 ],
#                 width=3,
#             ),
#             # Trend Direction
#             dbc.Col(
#                 [
#                     dbc.Card(
#                         [
#                             dbc.CardBody(
#                                 [
#                                     html.Div(
#                                         [
#                                             html.I(
#                                                 className=trend_icon,
#                                                 style={
#                                                     "fontSize": "28px",
#                                                     "color": trend_color,
#                                                     "float": "right",
#                                                 },
#                                             ),
#                                             html.H3(
#                                                 f"{trend_magnitude:.1f}%",
#                                                 className="mb-1",
#                                                 style={
#                                                     "color": trend_color,
#                                                     "fontWeight": "bold",
#                                                 },
#                                             ),
#                                             html.P(
#                                                 "CHANGE",
#                                                 className="text-muted mb-0",
#                                                 style={
#                                                     "fontSize": "12px",
#                                                     "fontWeight": "bold",
#                                                 },
#                                             ),
#                                         ]
#                                     )
#                                 ]
#                             )
#                         ],
#                         className="metric-card",
#                     )
#                 ],
#                 width=3,
#             ),
#             # Confidence
#             dbc.Col(
#                 [
#                     dbc.Card(
#                         [
#                             dbc.CardBody(
#                                 [
#                                     html.Div(
#                                         [
#                                             html.I(
#                                                 className="fas fa-shield-alt",
#                                                 style={
#                                                     "fontSize": "28px",
#                                                     "color": colors["success"],
#                                                     "float": "right",
#                                                 },
#                                             ),
#                                             html.H3(
#                                                 f"{confidence_level:.0f}%",
#                                                 className="mb-1",
#                                                 style={
#                                                     "color": colors["success"],
#                                                     "fontWeight": "bold",
#                                                 },
#                                             ),
#                                             html.P(
#                                                 "CONFIDENCE",
#                                                 className="text-muted mb-0",
#                                                 style={
#                                                     "fontSize": "12px",
#                                                     "fontWeight": "bold",
#                                                 },
#                                             ),
#                                         ]
#                                     )
#                                 ]
#                             )
#                         ],
#                         className="metric-card",
#                     )
#                 ],
#                 width=3,
#             ),
#         ]

#         return dbc.Row(cards, className="mb-4")

#     except Exception as e:
#         return dbc.Alert(f"Error creating summary cards: {str(e)}", color="warning")


# def create_forecast_chart(historical_data, forecast_data, colors):
#     """Create comprehensive forecast visualization"""
#     try:
#         # Extract data
#         hist_dates = historical_data["dates"]
#         hist_values = historical_data["values"]
#         forecast_dates = forecast_data["dates"]
#         forecast_values = forecast_data["forecast_values"]
#         confidence_upper = forecast_data.get("confidence_upper", [])
#         confidence_lower = forecast_data.get("confidence_lower", [])

#         # Create subplot
#         fig = go.Figure()

#         # Historical data
#         fig.add_trace(
#             go.Scatter(
#                 x=hist_dates,
#                 y=hist_values,
#                 mode="lines",
#                 name="Historical Data",
#                 line=dict(color=colors["primary"], width=3),
#                 hovertemplate="<b>%{x}</b><br>Rate: %{y:.2f}%<extra></extra>",
#             )
#         )

#         # Forecast line
#         fig.add_trace(
#             go.Scatter(
#                 x=forecast_dates,
#                 y=forecast_values,
#                 mode="lines+markers",
#                 name="Forecast",
#                 line=dict(color=colors["danger"], width=3, dash="dash"),
#                 marker=dict(size=8, color=colors["danger"]),
#                 hovertemplate="<b>%{x}</b><br>Forecast: %{y:.2f}%<extra></extra>",
#             )
#         )

#         # Confidence interval (if available)
#         if confidence_upper and confidence_lower:
#             # Upper bound
#             fig.add_trace(
#                 go.Scatter(
#                     x=forecast_dates,
#                     y=confidence_upper,
#                     mode="lines",
#                     name="Upper Confidence",
#                     line=dict(color=colors["danger"], width=1, dash="dot"),
#                     showlegend=False,
#                     hovertemplate="Upper CI: %{y:.2f}%<extra></extra>",
#                 )
#             )

#             # Lower bound
#             fig.add_trace(
#                 go.Scatter(
#                     x=forecast_dates,
#                     y=confidence_lower,
#                     mode="lines",
#                     name="Lower Confidence",
#                     line=dict(color=colors["danger"], width=1, dash="dot"),
#                     fill="tonexty",
#                     fillcolor=f'rgba({int(colors["danger"][1:3], 16)}, {int(colors["danger"][3:5], 16)}, {int(colors["danger"][5:7], 16)}, 0.2)',
#                     showlegend=False,
#                     hovertemplate="Lower CI: %{y:.2f}%<extra></extra>",
#                 )
#             )

#         # Add vertical line at forecast start
#         fig.add_vline(
#             x=forecast_dates[0],
#             line_dash="solid",
#             line_color=colors["warning"],
#             annotation_text="Forecast Starts",
#             annotation_position="top",
#         )

#         # Layout
#         fig.update_layout(
#             title={
#                 "text": "Malaysia Unemployment Rate Forecast",
#                 "x": 0.5,
#                 "font": {"size": 20, "color": colors["primary"]},
#             },
#             xaxis_title="Date",
#             yaxis_title="Unemployment Rate (%)",
#             hovermode="x unified",
#             height=500,
#             template="plotly_white",
#             plot_bgcolor="rgba(245,242,237,0.8)",
#             paper_bgcolor="rgba(245,242,237,0.8)",
#             legend=dict(
#                 orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1
#             ),
#         )

#         fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor="rgba(74,102,112,0.2)")
#         fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor="rgba(74,102,112,0.2)")

#         return fig

#     except Exception as e:
#         # Return empty chart with error message
#         fig = go.Figure()
#         fig.add_annotation(
#             text=f"Error creating forecast chart: {str(e)}",
#             xref="paper",
#             yref="paper",
#             x=0.5,
#             y=0.5,
#             font=dict(size=16, color=colors["danger"]),
#         )
#         fig.update_layout(
#             template="plotly_white",
#             height=500,
#             plot_bgcolor="rgba(245,242,237,0.8)",
#             paper_bgcolor="rgba(245,242,237,0.8)",
#         )
#         return fig


# def create_forecast_table(forecast_data, colors):
#     """Create detailed forecast table"""
#     try:
#         from dash import dash_table

#         # Prepare table data
#         dates = forecast_data["dates"]
#         values = forecast_data["forecast_values"]

#         table_data = []
#         for i, (date, value) in enumerate(zip(dates, values)):
#             # Calculate change from previous period
#             if i == 0:
#                 change = 0
#                 change_text = "Baseline"
#                 change_color = colors["text"]
#             else:
#                 change = value - values[i - 1]
#                 if abs(change) < 0.05:
#                     change_text = "Stable"
#                     change_color = colors["info"]
#                 elif change > 0:
#                     change_text = f"+{change:.2f}%"
#                     change_color = colors["danger"]
#                 else:
#                     change_text = f"{change:.2f}%"
#                     change_color = colors["success"]

#             table_data.append(
#                 {
#                     "Period": date.strftime("%b %Y"),
#                     "Forecast": f"{value:.2f}%",
#                     "Change": change_text,
#                     "Status": (
#                         "Improving"
#                         if change < -0.1
#                         else "Worsening" if change > 0.1 else "Stable"
#                     ),
#                 }
#             )

#         # Create table
#         table = dash_table.DataTable(
#             data=table_data,
#             columns=[
#                 {"name": "Period", "id": "Period"},
#                 {"name": "Forecast Rate", "id": "Forecast"},
#                 {"name": "Monthly Change", "id": "Change"},
#                 {"name": "Trend Status", "id": "Status"},
#             ],
#             style_cell={
#                 "textAlign": "center",
#                 "padding": "15px",
#                 "fontFamily": "Arial, sans-serif",
#                 "backgroundColor": "rgba(255,255,255,0.9)",
#                 "color": colors["text"],
#                 "fontSize": "14px",
#             },
#             style_header={
#                 "backgroundColor": colors["primary"],
#                 "fontWeight": "bold",
#                 "color": "white",
#                 "fontSize": "16px",
#             },
#             style_data_conditional=[
#                 {
#                     "if": {"filter_query": "{Status} = Improving"},
#                     "backgroundColor": "rgba(107,142,90,0.2)",
#                 },
#                 {
#                     "if": {"filter_query": "{Status} = Worsening"},
#                     "backgroundColor": "rgba(184,92,87,0.2)",
#                 },
#             ],
#             style_table={"overflowX": "auto"},
#         )

#         return table

#     except Exception as e:
#         return html.Div(
#             f"Error creating forecast table: {str(e)}",
#             style={"color": colors["danger"]},
#         )


# def create_model_info_card(model_info, colors):
#     """Create model information display"""
#     try:
#         return dbc.Card(
#             [
#                 dbc.CardBody(
#                     [
#                         html.H6(
#                             [
#                                 html.I(
#                                     className="fas fa-robot",
#                                     style={"marginRight": "8px"},
#                                 ),
#                                 "Model Details",
#                             ],
#                             style={"color": colors["primary"], "marginBottom": "15px"},
#                         ),
#                         html.Div(
#                             [
#                                 html.P(
#                                     [
#                                         html.Strong("Model Type: "),
#                                         model_info.get("model_type", "SARIMA"),
#                                     ],
#                                     style={"marginBottom": "8px"},
#                                 ),
#                                 html.P(
#                                     [
#                                         html.Strong("Dataset: "),
#                                         model_info.get(
#                                             "dataset", "General Labor Force"
#                                         ),
#                                     ],
#                                     style={"marginBottom": "8px"},
#                                 ),
#                                 html.P(
#                                     [
#                                         html.Strong("Training Period: "),
#                                         model_info.get("training_period", "2010-2025"),
#                                     ],
#                                     style={"marginBottom": "8px"},
#                                 ),
#                                 html.P(
#                                     [
#                                         html.Strong("Last Updated: "),
#                                         model_info.get("last_updated", "2025-05-23"),
#                                     ],
#                                     style={"marginBottom": "8px"},
#                                 ),
#                                 html.Hr(),
#                                 html.P(
#                                     [
#                                         html.Strong("Model Accuracy: "),
#                                         f"{model_info.get('accuracy', 85):.1f}%",
#                                     ],
#                                     style={"marginBottom": "8px"},
#                                 ),
#                                 html.P(
#                                     [
#                                         html.Strong("MAPE: "),
#                                         f"{model_info.get('mape', 12.5):.1f}%",
#                                     ],
#                                     style={"marginBottom": "8px"},
#                                 ),
#                             ],
#                             style={"color": colors["text"]},
#                         ),
#                     ]
#                 )
#             ],
#             style={"backgroundColor": "rgba(255,255,255,0.9)"},
#         )

#     except Exception as e:
#         return html.Div(
#             f"Error loading model info: {str(e)}", style={"color": colors["danger"]}
#         )


# def create_loading_state(colors):
#     """Create loading state for forecast generation"""
#     return dbc.Card(
#         [
#             dbc.CardBody(
#                 [
#                     html.Div(
#                         [
#                             dbc.Spinner(size="lg", color="primary"),
#                             html.H4(
#                                 "Generating Forecast...",
#                                 className="mt-3 text-center",
#                                 style={"color": colors["primary"]},
#                             ),
#                             html.P(
#                                 "Loading AI model and processing unemployment data",
#                                 className="text-center",
#                                 style={"color": colors["text"]},
#                             ),
#                         ],
#                         className="text-center",
#                         style={"padding": "40px"},
#                     )
#                 ]
#             )
#         ],
#         className="status-card",
#     )


# def create_error_state(error_message, colors):
#     """Create error state display"""
#     return dbc.Alert(
#         [
#             html.I(
#                 className="fas fa-exclamation-triangle", style={"marginRight": "8px"}
#             ),
#             f"Forecast Error: {error_message}",
#             html.Hr(),
#             html.P(
#                 "Please try again with different settings or contact support.",
#                 className="mb-0",
#             ),
#         ],
#         color="danger",
#         className="status-card",
#     )

"""
Enhanced Forecasting Hub page for unemployment rate predictions.
Updated to work with real trained models and improved visualizations.
"""

from dash import html, dcc
import dash_bootstrap_components as dbc
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from dashboard.components.layout import (
    create_page_header,
    create_control_card,
    create_chart_container,
)


def create_forecasting_page(data_manager, colors):
    """Create enhanced forecasting hub page for general users"""
    header = create_page_header(
        "Forecasting Hub",
        "Generate unemployment rate forecasts using advanced AI models",
        "fas fa-crystal-ball",
        colors,
    )

    controls = create_forecasting_controls(colors)
    forecast_section = create_forecast_section(colors)

    return html.Div([header, controls, forecast_section])


def create_forecasting_controls(colors):
    """Create user-friendly forecasting controls"""
    controls_content = [
        # Dataset Selection
        dbc.Row(
            [
                dbc.Col(
                    [
                        dbc.Label(
                            "📊 Select Dataset:",
                            style={
                                "fontWeight": "bold",
                                "color": colors["text"],
                                "fontSize": "16px",
                            },
                        ),
                        dcc.Dropdown(
                            id="forecast-dataset-dropdown",
                            options=[
                                {
                                    "label": "🔹 General Labor Force (Raw Data)",
                                    "value": "general",
                                },
                                {
                                    "label": "🔹 Seasonally Adjusted Labor Force",
                                    "value": "sa",
                                },
                            ],
                            value="general",
                            style={"marginBottom": "15px"},
                        ),
                        html.P(
                            "Choose between raw data or seasonally adjusted unemployment data",
                            style={
                                "color": colors["text"],
                                "fontSize": "12px",
                                "opacity": 0.8,
                            },
                        ),
                    ],
                    width=6,
                ),
                # Model Selection
                dbc.Col(
                    [
                        dbc.Label(
                            "🤖 Select AI Model:",
                            style={
                                "fontWeight": "bold",
                                "color": colors["text"],
                                "fontSize": "16px",
                            },
                        ),
                        dcc.Dropdown(
                            id="forecast-model-dropdown",
                            options=[
                                {
                                    "label": "📈 SARIMA - Seasonal Time Series",
                                    "value": "sarima",
                                },
                                {
                                    "label": "📊 ARIMA - Statistical Time Series",
                                    "value": "arima",
                                },
                                {
                                    "label": "🧠 LSTM - Deep Learning Neural Network",
                                    "value": "lstm",
                                },
                            ],
                            value="sarima",
                            style={"marginBottom": "15px"},
                        ),
                        html.P(
                            "SARIMA handles seasonal patterns better for unemployment data",
                            style={
                                "color": colors["text"],
                                "fontSize": "12px",
                                "opacity": 0.8,
                            },
                        ),
                    ],
                    width=6,
                ),
            ],
            className="mb-4",
        ),
        # Forecast Period and Generate Button
        dbc.Row(
            [
                dbc.Col(
                    [
                        dbc.Label(
                            "⏱️ Forecast Period:",
                            style={
                                "fontWeight": "bold",
                                "color": colors["text"],
                                "fontSize": "16px",
                            },
                        ),
                        dcc.Dropdown(
                            id="forecast-period-dropdown",
                            options=[
                                {"label": "1 Month Ahead", "value": 1},
                                {"label": "3 Months Ahead", "value": 3},
                                {"label": "6 Months Ahead", "value": 6},
                                {"label": "12 Months Ahead", "value": 12},
                            ],
                            value=6,
                            style={"marginBottom": "15px"},
                        ),
                        html.P(
                            "Shorter periods are generally more accurate",
                            style={
                                "color": colors["text"],
                                "fontSize": "12px",
                                "opacity": 0.8,
                            },
                        ),
                    ],
                    width=4,
                ),
                dbc.Col(
                    [
                        dbc.Label(
                            "🎯 Target Variable:",
                            style={
                                "fontWeight": "bold",
                                "color": colors["text"],
                                "fontSize": "16px",
                            },
                        ),
                        html.Div(
                            [
                                dbc.Badge(
                                    "Unemployment Rate (%)",
                                    color="primary",
                                    style={"fontSize": "14px", "padding": "10px 15px"},
                                )
                            ],
                            style={"marginBottom": "15px"},
                        ),
                        html.P(
                            "Models are trained specifically for unemployment rate prediction",
                            style={
                                "color": colors["text"],
                                "fontSize": "12px",
                                "opacity": 0.8,
                            },
                        ),
                    ],
                    width=4,
                ),
                dbc.Col(
                    [
                        dbc.Label(
                            "🚀 Generate Forecast:",
                            style={
                                "fontWeight": "bold",
                                "color": colors["text"],
                                "fontSize": "16px",
                            },
                        ),
                        dbc.Button(
                            [
                                html.I(
                                    className="fas fa-magic",
                                    style={"marginRight": "10px"},
                                ),
                                "Generate Prediction",
                            ],
                            id="generate-forecast-btn",
                            color="success",
                            size="lg",
                            className="w-100",
                            style={"marginTop": "5px", "fontWeight": "bold"},
                        ),
                    ],
                    width=4,
                ),
            ]
        ),
    ]

    return create_control_card(
        "🔮 Forecast Configuration", "fas fa-cogs", controls_content, colors
    )


def create_forecast_section(colors):
    """Create forecast results and visualization section"""
    return html.Div(
        [
            # Results Summary Cards
            html.Div(id="forecast-summary-cards", className="mb-4"),
            # Main Forecast Chart
            dbc.Row(
                [
                    dbc.Col(
                        [
                            html.Div(
                                [
                                    html.H5(
                                        [
                                            html.I(
                                                className="fas fa-chart-line",
                                                style={"marginRight": "10px"},
                                            ),
                                            "Unemployment Rate Forecast",
                                        ],
                                        style={
                                            "color": colors["primary"],
                                            "fontWeight": "bold",
                                        },
                                    ),
                                    dcc.Graph(
                                        id="forecast-chart", style={"height": "500px"}
                                    ),
                                ],
                                className="chart-container",
                            )
                        ],
                        width=12,
                    )
                ],
                className="mb-4",
            ),
            # Detailed Forecast Table and Model Info
            dbc.Row(
                [
                    dbc.Col(
                        [
                            html.Div(
                                [
                                    html.H5(
                                        [
                                            html.I(
                                                className="fas fa-table",
                                                style={"marginRight": "10px"},
                                            ),
                                            "Detailed Forecast Values",
                                        ],
                                        style={
                                            "color": colors["primary"],
                                            "fontWeight": "bold",
                                        },
                                    ),
                                    html.Div(id="forecast-table"),
                                ],
                                className="chart-container",
                            )
                        ],
                        width=8,
                    ),
                    # Model Information
                    dbc.Col(
                        [
                            html.Div(
                                [
                                    html.H5(
                                        [
                                            html.I(
                                                className="fas fa-info-circle",
                                                style={"marginRight": "10px"},
                                            ),
                                            "Model Information",
                                        ],
                                        style={
                                            "color": colors["primary"],
                                            "fontWeight": "bold",
                                        },
                                    ),
                                    html.Div(id="model-info"),
                                ],
                                className="chart-container",
                            )
                        ],
                        width=4,
                    ),
                ]
            ),
        ]
    )


def create_forecast_summary_cards(forecast_data, colors):
    """Create summary cards showing key forecast insights"""
    try:
        current_rate = forecast_data["current_rate"]
        forecast_values = forecast_data["forecast_values"]
        trend_direction = forecast_data["trend_direction"]
        trend_magnitude = forecast_data["trend_magnitude"]
        confidence_level = forecast_data["confidence_level"]

        # Determine trend icon and color
        if trend_direction == "Rising":
            trend_icon = "fas fa-arrow-up"
            trend_color = colors["danger"]
            trend_desc = "Unemployment Expected to Rise"
        elif trend_direction == "Falling":
            trend_icon = "fas fa-arrow-down"
            trend_color = colors["success"]
            trend_desc = "Unemployment Expected to Fall"
        else:
            trend_icon = "fas fa-minus"
            trend_color = colors["info"]
            trend_desc = "Unemployment Expected to Remain Stable"

        cards = [
            # Current Rate
            dbc.Col(
                [
                    dbc.Card(
                        [
                            dbc.CardBody(
                                [
                                    html.Div(
                                        [
                                            html.I(
                                                className="fas fa-chart-line",
                                                style={
                                                    "fontSize": "28px",
                                                    "color": colors["primary"],
                                                    "float": "right",
                                                },
                                            ),
                                            html.H3(
                                                f"{current_rate:.1f}%",
                                                className="mb-1",
                                                style={
                                                    "color": colors["primary"],
                                                    "fontWeight": "bold",
                                                },
                                            ),
                                            html.P(
                                                "CURRENT RATE",
                                                className="text-muted mb-0",
                                                style={
                                                    "fontSize": "12px",
                                                    "fontWeight": "bold",
                                                },
                                            ),
                                        ]
                                    )
                                ]
                            )
                        ],
                        className="metric-card",
                    )
                ],
                width=3,
            ),
            # Forecast End Rate
            dbc.Col(
                [
                    dbc.Card(
                        [
                            dbc.CardBody(
                                [
                                    html.Div(
                                        [
                                            html.I(
                                                className="fas fa-target",
                                                style={
                                                    "fontSize": "28px",
                                                    "color": colors["secondary"],
                                                    "float": "right",
                                                },
                                            ),
                                            html.H3(
                                                f"{forecast_values[-1]:.1f}%",
                                                className="mb-1",
                                                style={
                                                    "color": colors["secondary"],
                                                    "fontWeight": "bold",
                                                },
                                            ),
                                            html.P(
                                                "FORECAST END",
                                                className="text-muted mb-0",
                                                style={
                                                    "fontSize": "12px",
                                                    "fontWeight": "bold",
                                                },
                                            ),
                                        ]
                                    )
                                ]
                            )
                        ],
                        className="metric-card",
                    )
                ],
                width=3,
            ),
            # Trend Direction
            dbc.Col(
                [
                    dbc.Card(
                        [
                            dbc.CardBody(
                                [
                                    html.Div(
                                        [
                                            html.I(
                                                className=trend_icon,
                                                style={
                                                    "fontSize": "28px",
                                                    "color": trend_color,
                                                    "float": "right",
                                                },
                                            ),
                                            html.H3(
                                                f"{trend_magnitude:.1f}%",
                                                className="mb-1",
                                                style={
                                                    "color": trend_color,
                                                    "fontWeight": "bold",
                                                },
                                            ),
                                            html.P(
                                                "CHANGE",
                                                className="text-muted mb-0",
                                                style={
                                                    "fontSize": "12px",
                                                    "fontWeight": "bold",
                                                },
                                            ),
                                        ]
                                    )
                                ]
                            )
                        ],
                        className="metric-card",
                    )
                ],
                width=3,
            ),
            # Confidence
            dbc.Col(
                [
                    dbc.Card(
                        [
                            dbc.CardBody(
                                [
                                    html.Div(
                                        [
                                            html.I(
                                                className="fas fa-shield-alt",
                                                style={
                                                    "fontSize": "28px",
                                                    "color": colors["success"],
                                                    "float": "right",
                                                },
                                            ),
                                            html.H3(
                                                f"{confidence_level:.0f}%",
                                                className="mb-1",
                                                style={
                                                    "color": colors["success"],
                                                    "fontWeight": "bold",
                                                },
                                            ),
                                            html.P(
                                                "CONFIDENCE",
                                                className="text-muted mb-0",
                                                style={
                                                    "fontSize": "12px",
                                                    "fontWeight": "bold",
                                                },
                                            ),
                                        ]
                                    )
                                ]
                            )
                        ],
                        className="metric-card",
                    )
                ],
                width=3,
            ),
        ]

        return dbc.Row(cards, className="mb-4")

    except Exception as e:
        return dbc.Alert(f"Error creating summary cards: {str(e)}", color="warning")


def create_forecast_chart(historical_data, forecast_data, colors):
    """Create comprehensive forecast visualization with enhanced features"""
    try:
        # Extract data
        hist_dates = historical_data["dates"]
        hist_values = historical_data["values"]
        forecast_dates = forecast_data["dates"]
        forecast_values = forecast_data["forecast_values"]
        confidence_upper = forecast_data.get("confidence_upper", [])
        confidence_lower = forecast_data.get("confidence_lower", [])

        # Create subplot
        fig = go.Figure()

        # Historical data with enhanced styling
        fig.add_trace(
            go.Scatter(
                x=hist_dates,
                y=hist_values,
                mode="lines",
                name="Historical Data",
                line=dict(color=colors["primary"], width=3),
                hovertemplate="<b>%{x}</b><br>Rate: %{y:.2f}%<extra></extra>",
            )
        )

        # Confidence interval (if available) - add first so it's behind forecast line
        if confidence_upper and confidence_lower:
            # Lower bound
            fig.add_trace(
                go.Scatter(
                    x=forecast_dates,
                    y=confidence_lower,
                    mode="lines",
                    name="Lower Confidence",
                    line=dict(color="rgba(0,0,0,0)", width=0),
                    showlegend=False,
                    hovertemplate="Lower CI: %{y:.2f}%<extra></extra>",
                )
            )

            # Upper bound with fill
            fig.add_trace(
                go.Scatter(
                    x=forecast_dates,
                    y=confidence_upper,
                    mode="lines",
                    name="95% Confidence Interval",
                    line=dict(color="rgba(0,0,0,0)", width=0),
                    fill="tonexty",
                    fillcolor=f"rgba(184,92,87,0.2)",
                    hovertemplate="Upper CI: %{y:.2f}%<extra></extra>",
                )
            )

        # Forecast line - enhanced with markers
        fig.add_trace(
            go.Scatter(
                x=forecast_dates,
                y=forecast_values,
                mode="lines+markers",
                name="Forecast",
                line=dict(color=colors["danger"], width=4, dash="dash"),
                marker=dict(size=8, color=colors["danger"], symbol="diamond"),
                hovertemplate="<b>%{x}</b><br>Forecast: %{y:.2f}%<extra></extra>",
            )
        )

        # Add vertical line at forecast start
        if len(hist_dates) > 0 and len(forecast_dates) > 0:
            fig.add_vline(
                x=forecast_dates[0],
                line_dash="solid",
                line_color=colors["warning"],
                line_width=2,
                annotation_text="Forecast Starts",
                annotation_position="top",
                annotation=dict(
                    font=dict(size=12, color=colors["warning"]),
                    bgcolor="rgba(255,255,255,0.8)",
                    bordercolor=colors["warning"],
                    borderwidth=1,
                ),
            )

        # Enhanced layout
        fig.update_layout(
            title={
                "text": "Malaysia Unemployment Rate Forecast",
                "x": 0.5,
                "font": {
                    "size": 22,
                    "color": colors["primary"],
                    "family": "Arial Black",
                },
            },
            xaxis_title="Date",
            yaxis_title="Unemployment Rate (%)",
            hovermode="x unified",
            height=500,
            template="plotly_white",
            plot_bgcolor="rgba(245,242,237,0.8)",
            paper_bgcolor="rgba(245,242,237,0.8)",
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1,
                bgcolor="rgba(255,255,255,0.8)",
                bordercolor="rgba(0,0,0,0.2)",
                borderwidth=1,
            ),
        )

        # Enhanced grid and axes
        fig.update_xaxes(
            showgrid=True,
            gridwidth=1,
            gridcolor="rgba(74,102,112,0.2)",
            tickformat="%b %Y",
        )
        fig.update_yaxes(
            showgrid=True,
            gridwidth=1,
            gridcolor="rgba(74,102,112,0.2)",
            tickformat=".1f",
        )

        return fig

    except Exception as e:
        # Return empty chart with error message
        fig = go.Figure()
        fig.add_annotation(
            text=f"Error creating forecast chart: {str(e)}",
            xref="paper",
            yref="paper",
            x=0.5,
            y=0.5,
            font=dict(size=16, color=colors["danger"]),
        )
        fig.update_layout(
            template="plotly_white",
            height=500,
            plot_bgcolor="rgba(245,242,237,0.8)",
            paper_bgcolor="rgba(245,242,237,0.8)",
        )
        return fig


def create_forecast_table(forecast_data, colors):
    """Create detailed forecast table with enhanced styling"""
    try:
        from dash import dash_table

        # Prepare table data
        dates = forecast_data["dates"]
        values = forecast_data["forecast_values"]
        confidence_upper = forecast_data.get("confidence_upper", [None] * len(values))
        confidence_lower = forecast_data.get("confidence_lower", [None] * len(values))

        table_data = []
        for i, (date, value) in enumerate(zip(dates, values)):
            # Calculate change from previous period
            if i == 0:
                change = 0
                change_text = "Baseline"
                change_color = colors["text"]
            else:
                change = value - values[i - 1]
                if abs(change) < 0.05:
                    change_text = "Stable"
                    change_color = colors["info"]
                elif change > 0:
                    change_text = f"+{change:.2f}%"
                    change_color = colors["danger"]
                else:
                    change_text = f"{change:.2f}%"
                    change_color = colors["success"]

            # Confidence interval
            if confidence_upper[i] is not None and confidence_lower[i] is not None:
                ci_text = f"{confidence_lower[i]:.2f}% - {confidence_upper[i]:.2f}%"
            else:
                ci_text = "N/A"

            table_data.append(
                {
                    "Period": date.strftime("%b %Y"),
                    "Forecast": f"{value:.2f}%",
                    "Change": change_text,
                    "Confidence_Interval": ci_text,
                    "Status": (
                        "Improving"
                        if change < -0.1
                        else "Worsening" if change > 0.1 else "Stable"
                    ),
                }
            )

        # Create table with enhanced styling
        table = dash_table.DataTable(
            data=table_data,
            columns=[
                {"name": "Period", "id": "Period"},
                {"name": "Forecast Rate", "id": "Forecast"},
                {"name": "Monthly Change", "id": "Change"},
                {"name": "95% CI", "id": "Confidence_Interval"},
                {"name": "Trend Status", "id": "Status"},
            ],
            style_cell={
                "textAlign": "center",
                "padding": "15px",
                "fontFamily": "Arial, sans-serif",
                "backgroundColor": "rgba(255,255,255,0.9)",
                "color": colors["text"],
                "fontSize": "14px",
                "border": "1px solid rgba(0,0,0,0.1)",
            },
            style_header={
                "backgroundColor": colors["primary"],
                "fontWeight": "bold",
                "color": "white",
                "fontSize": "16px",
                "border": "1px solid rgba(0,0,0,0.1)",
            },
            style_data_conditional=[
                {
                    "if": {"filter_query": "{Status} = Improving"},
                    "backgroundColor": "rgba(107,142,90,0.2)",
                },
                {
                    "if": {"filter_query": "{Status} = Worsening"},
                    "backgroundColor": "rgba(184,92,87,0.2)",
                },
                {
                    "if": {"filter_query": "{Status} = Stable"},
                    "backgroundColor": "rgba(74,102,112,0.2)",
                },
            ],
            style_table={"overflowX": "auto"},
            sort_action="native",
        )

        return table

    except Exception as e:
        return html.Div(
            f"Error creating forecast table: {str(e)}",
            style={"color": colors["danger"], "textAlign": "center", "padding": "20px"},
        )


def create_model_info_card(model_info, colors):
    """Create enhanced model information display"""
    try:
        return dbc.Card(
            [
                dbc.CardBody(
                    [
                        html.H6(
                            [
                                html.I(
                                    className="fas fa-robot",
                                    style={"marginRight": "8px"},
                                ),
                                "Model Details",
                            ],
                            style={
                                "color": colors["primary"],
                                "marginBottom": "15px",
                                "fontWeight": "bold",
                            },
                        ),
                        html.Div(
                            [
                                html.P(
                                    [
                                        html.Strong("Model Type: "),
                                        model_info.get("model_type", "SARIMA"),
                                    ],
                                    style={"marginBottom": "8px"},
                                ),
                                html.P(
                                    [
                                        html.Strong("Dataset: "),
                                        model_info.get(
                                            "dataset", "General Labor Force"
                                        ),
                                    ],
                                    style={"marginBottom": "8px"},
                                ),
                                html.P(
                                    [
                                        html.Strong("Training Period: "),
                                        model_info.get("training_period", "2010-2022"),
                                    ],
                                    style={"marginBottom": "8px"},
                                ),
                                html.P(
                                    [
                                        html.Strong("Last Updated: "),
                                        model_info.get("last_updated", "2025-05-23"),
                                    ],
                                    style={"marginBottom": "8px"},
                                ),
                                # Add model-specific parameters
                                html.Hr(),
                                # Model accuracy
                                html.P(
                                    [
                                        html.Strong("Model Accuracy: "),
                                        html.Span(
                                            f"{model_info.get('accuracy', 85):.1f}%",
                                            style={
                                                "color": (
                                                    colors["success"]
                                                    if model_info.get("accuracy", 85)
                                                    > 90
                                                    else colors["warning"]
                                                )
                                            },
                                        ),
                                    ],
                                    style={"marginBottom": "8px"},
                                ),
                                html.P(
                                    [
                                        html.Strong("MAPE: "),
                                        html.Span(
                                            f"{model_info.get('mape', 12.5):.1f}%",
                                            style={
                                                "color": (
                                                    colors["success"]
                                                    if model_info.get("mape", 12.5) < 5
                                                    else colors["warning"]
                                                )
                                            },
                                        ),
                                    ],
                                    style={"marginBottom": "8px"},
                                ),
                                # Model-specific info
                                html.Div(
                                    id="model-specific-info",
                                    children=create_model_specific_info(
                                        model_info, colors
                                    ),
                                ),
                                # Generation info if available
                                html.Hr() if "generation_time" in model_info else "",
                                (
                                    html.P(
                                        [
                                            html.Strong("Generated: "),
                                            model_info.get("generation_time", "N/A"),
                                        ],
                                        style={
                                            "marginBottom": "0px",
                                            "fontSize": "12px",
                                            "color": colors["text"],
                                        },
                                    )
                                    if "generation_time" in model_info
                                    else ""
                                ),
                            ],
                            style={"color": colors["text"]},
                        ),
                    ]
                )
            ],
            style={
                "backgroundColor": "rgba(255,255,255,0.9)",
                "border": f'1px solid {colors["primary"]}',
            },
        )

    except Exception as e:
        return html.Div(
            f"Error loading model info: {str(e)}",
            style={"color": colors["danger"], "textAlign": "center", "padding": "20px"},
        )


def create_model_specific_info(model_info, colors):
    """Create model-specific information based on model type"""
    model_type = model_info.get("model_type", "").upper()

    if "SARIMA" in model_type:
        order = model_info.get("order", (0, 1, 1))
        seasonal_order = model_info.get("seasonal_order", (0, 1, 1, 12))
        return html.Div(
            [
                html.P(
                    [html.Strong("Model Order: "), f"SARIMA{order}×{seasonal_order}"],
                    style={"marginBottom": "8px", "fontSize": "12px"},
                ),
                html.P(
                    [html.Strong("Seasonality: "), "Monthly (12 periods)"],
                    style={"marginBottom": "8px", "fontSize": "12px"},
                ),
            ]
        )
    elif "ARIMA" in model_type:
        order = model_info.get("order", (0, 1, 0))
        return html.Div(
            [
                html.P(
                    [html.Strong("Model Order: "), f"ARIMA{order}"],
                    style={"marginBottom": "8px", "fontSize": "12px"},
                ),
                html.P(
                    [html.Strong("Type: "), "Non-seasonal"],
                    style={"marginBottom": "8px", "fontSize": "12px"},
                ),
            ]
        )
    elif "LSTM" in model_type:
        architecture = model_info.get("architecture", "64x32 units")
        sequence_length = model_info.get("sequence_length", 12)
        return html.Div(
            [
                html.P(
                    [html.Strong("Architecture: "), architecture],
                    style={"marginBottom": "8px", "fontSize": "12px"},
                ),
                html.P(
                    [html.Strong("Sequence Length: "), f"{sequence_length} months"],
                    style={"marginBottom": "8px", "fontSize": "12px"},
                ),
            ]
        )
    else:
        return html.Div()


def create_loading_state(colors):
    """Create loading state for forecast generation"""
    return dbc.Card(
        [
            dbc.CardBody(
                [
                    html.Div(
                        [
                            dbc.Spinner(size="lg", color="primary"),
                            html.H4(
                                "Generating Forecast...",
                                className="mt-3 text-center",
                                style={"color": colors["primary"]},
                            ),
                            html.P(
                                "Loading AI model and processing unemployment data",
                                className="text-center",
                                style={"color": colors["text"]},
                            ),
                        ],
                        className="text-center",
                        style={"padding": "40px"},
                    )
                ]
            )
        ],
        className="status-card",
    )


def create_error_state(error_message, colors):
    """Create error state display"""
    return dbc.Alert(
        [
            html.I(
                className="fas fa-exclamation-triangle", style={"marginRight": "8px"}
            ),
            f"Forecast Error: {error_message}",
            html.Hr(),
            html.P(
                "Please try again with different settings or contact support.",
                className="mb-0",
            ),
        ],
        color="danger",
        className="status-card",
    )

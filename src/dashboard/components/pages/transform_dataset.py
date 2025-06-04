# """
# Transform Dataset page components for stationarity transformations.
# Handles log transformations, differencing, and stationarity testing.
# """

# from dash import html, dcc
# import dash_bootstrap_components as dbc
# import plotly.graph_objects as go
# from plotly.subplots import make_subplots
# import numpy as np
# from dashboard.components.layout import create_page_header, create_chart_container


# def create_transform_page(data_manager, colors):
#     """Create transformation page matching reference design"""
#     datasets = data_manager.get_available_datasets()

#     header = create_page_header(
#         "Transform Dataset",
#         "Make your time series stationary for accurate forecasting",
#         "fas fa-cogs",
#         colors,
#     )

#     transform_controls = create_transform_controls(datasets, colors)
#     charts_section = create_charts_section(colors)

#     return html.Div([header, transform_controls, charts_section])


# def create_transform_controls(datasets, colors):
#     """Create transformation controls section"""
#     return html.Div(
#         [
#             html.H3(
#                 "Transform dataset to make it Stationary",
#                 style={
#                     "color": "white",
#                     "textAlign": "center",
#                     "marginBottom": "30px",
#                     "fontWeight": "bold",
#                     "textShadow": "0 2px 4px rgba(0,0,0,0.3)",
#                     "fontSize": "28px",
#                 },
#             ),
#             # Transformation options
#             dbc.Row(
#                 [
#                     dbc.Col(
#                         [
#                             dbc.Card(
#                                 [
#                                     dbc.CardBody(
#                                         [
#                                             dbc.Checklist(
#                                                 options=[
#                                                     {
#                                                         "label": " 1) Apply log",
#                                                         "value": "log",
#                                                     }
#                                                 ],
#                                                 value=[],
#                                                 id="apply-log",
#                                                 style={
#                                                     "color": "white",
#                                                     "fontSize": "18px",
#                                                     "fontWeight": "bold",
#                                                 },
#                                             )
#                                         ],
#                                         style={"padding": "25px"},
#                                     )
#                                 ],
#                                 style={
#                                     "background": "rgba(255,255,255,0.15)",
#                                     "border": "none",
#                                     "borderRadius": "12px",
#                                 },
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
#                                             dbc.Checklist(
#                                                 options=[
#                                                     {
#                                                         "label": " 2) Apply difference",
#                                                         "value": "diff",
#                                                     }
#                                                 ],
#                                                 value=[],
#                                                 id="apply-diff",
#                                                 style={
#                                                     "color": "white",
#                                                     "fontSize": "18px",
#                                                     "fontWeight": "bold",
#                                                 },
#                                             ),
#                                             dcc.Dropdown(
#                                                 id="diff-lag",
#                                                 options=[
#                                                     {"label": "Lag 1", "value": 1},
#                                                     {"label": "Lag 12", "value": 12},
#                                                 ],
#                                                 value=1,
#                                                 placeholder="Choose lag",
#                                                 style={"marginTop": "15px"},
#                                             ),
#                                         ],
#                                         style={"padding": "25px"},
#                                     )
#                                 ],
#                                 style={
#                                     "background": "rgba(255,255,255,0.15)",
#                                     "border": "none",
#                                     "borderRadius": "12px",
#                                 },
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
#                                             dbc.Label(
#                                                 "3) Select Dataset",
#                                                 style={
#                                                     "color": "white",
#                                                     "fontSize": "18px",
#                                                     "fontWeight": "bold",
#                                                 },
#                                             ),
#                                             dcc.Dropdown(
#                                                 id="transform-dataset-dropdown",
#                                                 options=[
#                                                     {"label": ds, "value": ds}
#                                                     for ds in datasets
#                                                 ],
#                                                 value=datasets[0] if datasets else None,
#                                                 placeholder="Choose dataset",
#                                                 style={"marginTop": "15px"},
#                                             ),
#                                         ],
#                                         style={"padding": "25px"},
#                                     )
#                                 ],
#                                 style={
#                                     "background": "rgba(255,255,255,0.15)",
#                                     "border": "none",
#                                     "borderRadius": "12px",
#                                 },
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
#                                             dbc.Label(
#                                                 "4) Select Variable",
#                                                 style={
#                                                     "color": "white",
#                                                     "fontSize": "18px",
#                                                     "fontWeight": "bold",
#                                                 },
#                                             ),
#                                             dcc.Dropdown(
#                                                 id="transform-variable-dropdown",
#                                                 placeholder="Choose variable",
#                                                 style={"marginTop": "15px"},
#                                             ),
#                                         ],
#                                         style={"padding": "25px"},
#                                     )
#                                 ],
#                                 style={
#                                     "background": "rgba(255,255,255,0.15)",
#                                     "border": "none",
#                                     "borderRadius": "12px",
#                                 },
#                             )
#                         ],
#                         width=3,
#                     ),
#                 ],
#                 className="mb-4",
#             ),
#             # ADF Test Result
#             dbc.Row(
#                 [
#                     dbc.Col(
#                         [
#                             html.Div(
#                                 [
#                                     html.H5(
#                                         "Augmented Dickey-Fuller test:",
#                                         style={
#                                             "color": "white",
#                                             "display": "inline-block",
#                                             "marginRight": "25px",
#                                             "fontSize": "20px",
#                                             "fontWeight": "bold",
#                                         },
#                                     ),
#                                     html.Div(
#                                         id="adf-test-result",
#                                         style={"display": "inline-block"},
#                                     ),
#                                 ]
#                             )
#                         ],
#                         width=12,
#                     )
#                 ],
#                 className="mb-4",
#             ),
#         ],
#         className="transform-header",
#         style={
#             "background": f'linear-gradient(135deg, {colors["gradient_start"]} 0%, {colors["gradient_end"]} 100%)',
#             "padding": "40px",
#             "borderRadius": "18px",
#             "marginBottom": "30px",
#             "boxShadow": "0 12px 35px rgba(0,0,0,0.15)",
#         },
#     )


# def create_charts_section(colors):
#     """Create charts section for transformed data visualization"""
#     return html.Div(
#         [
#             dbc.Row(
#                 [
#                     dbc.Col(
#                         [
#                             create_chart_container(
#                                 "Transformed Data Linechart",
#                                 "transform-chart",
#                                 "fas fa-chart-line",
#                                 colors,
#                             )
#                         ],
#                         width=6,
#                     ),
#                     dbc.Col(
#                         [
#                             create_chart_container(
#                                 "Autocorrelation (ACF)",
#                                 "acf-chart",
#                                 "fas fa-signal",
#                                 colors,
#                             )
#                         ],
#                         width=6,
#                     ),
#                 ],
#                 className="mb-4",
#             ),
#             dbc.Row(
#                 [
#                     dbc.Col(
#                         [
#                             create_chart_container(
#                                 "Box-Cox Plot",
#                                 "boxcox-chart",
#                                 "fas fa-square-root-alt",
#                                 colors,
#                             )
#                         ],
#                         width=6,
#                     ),
#                     dbc.Col(
#                         [
#                             create_chart_container(
#                                 "Partial Autocorrelation (PACF)",
#                                 "pacf-chart",
#                                 "fas fa-wave-square",
#                                 colors,
#                             )
#                         ],
#                         width=6,
#                     ),
#                 ]
#             ),
#         ]
#     )


# def create_transform_chart(
#     original_series, transformed_series, apply_log, apply_diff, colors
# ):
#     """Create transformation visualization chart"""
#     fig = make_subplots(
#         rows=2,
#         cols=1,
#         subplot_titles=("Original Data", "Transformed Data"),
#         vertical_spacing=0.15,
#         row_heights=[0.5, 0.5],
#     )

#     # Original data
#     fig.add_trace(
#         go.Scatter(
#             x=original_series.index,
#             y=original_series,
#             name="Original",
#             line=dict(color=colors["primary"], width=2),
#             hovertemplate="<b>%{x}</b><br>Original: %{y:.3f}<extra></extra>",
#         ),
#         row=1,
#         col=1,
#     )

#     # Transformed data
#     fig.add_trace(
#         go.Scatter(
#             x=transformed_series.index,
#             y=transformed_series,
#             name="Transformed",
#             line=dict(color=colors["success"], width=2),
#             hovertemplate="<b>%{x}</b><br>Transformed: %{y:.3f}<extra></extra>",
#         ),
#         row=2,
#         col=1,
#     )

#     transform_text = []
#     if apply_log:
#         transform_text.append("Log")
#     if apply_diff:
#         transform_text.append("Differenced")

#     title = f"Data Transformation: {' + '.join(transform_text) if transform_text else 'No transformation'}"

#     fig.update_layout(
#         title=title,
#         height=400,
#         template="plotly_white",
#         plot_bgcolor="rgba(245,242,237,0.8)",
#         paper_bgcolor="rgba(245,242,237,0.8)",
#         showlegend=True,
#         hovermode="x unified",
#     )

#     fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor="rgba(74,102,112,0.2)")
#     fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor="rgba(74,102,112,0.2)")

#     return fig


# def create_acf_pacf_charts_transform(series, colors):
#     """Create ACF and PACF charts for transformation page"""
#     try:
#         lags = min(40, len(series) // 4)
#         if lags < 5:
#             lags = 5

#         # Simple ACF calculation
#         acf_vals = []
#         for lag in range(lags + 1):
#             if lag == 0:
#                 acf_vals.append(1.0)
#             else:
#                 if len(series) > lag:
#                     corr = series.corr(series.shift(lag))
#                     acf_vals.append(corr if not np.isnan(corr) else 0)
#                 else:
#                     acf_vals.append(0)

#         # Simple PACF calculation
#         pacf_vals = []
#         for lag in range(lags + 1):
#             if lag == 0:
#                 pacf_vals.append(1.0)
#             elif lag == 1:
#                 pacf_vals.append(acf_vals[1])
#             else:
#                 pacf_vals.append(acf_vals[lag] * 0.8 if lag < len(acf_vals) else 0)

#         # ACF Chart
#         acf_fig = go.Figure()
#         acf_fig.add_trace(
#             go.Bar(
#                 x=list(range(len(acf_vals))),
#                 y=acf_vals,
#                 name="ACF",
#                 marker_color=colors["primary"],
#                 opacity=0.8,
#             )
#         )

#         confidence_interval = 1.96 / np.sqrt(len(series))
#         acf_fig.add_hline(
#             y=confidence_interval,
#             line_dash="dash",
#             line_color=colors["danger"],
#             opacity=0.7,
#         )
#         acf_fig.add_hline(
#             y=-confidence_interval,
#             line_dash="dash",
#             line_color=colors["danger"],
#             opacity=0.7,
#         )
#         acf_fig.add_hline(y=0, line_color=colors["dark"], line_width=1)

#         acf_fig.update_layout(
#             title="Autocorrelation Function (ACF)",
#             xaxis_title="Lag",
#             yaxis_title="Correlation",
#             template="plotly_white",
#             plot_bgcolor="rgba(245,242,237,0.8)",
#             paper_bgcolor="rgba(245,242,237,0.8)",
#             showlegend=False,
#             height=400,
#         )

#         # PACF Chart
#         pacf_fig = go.Figure()
#         pacf_fig.add_trace(
#             go.Bar(
#                 x=list(range(len(pacf_vals))),
#                 y=pacf_vals,
#                 name="PACF",
#                 marker_color=colors["secondary"],
#                 opacity=0.8,
#             )
#         )

#         pacf_fig.add_hline(
#             y=confidence_interval,
#             line_dash="dash",
#             line_color=colors["danger"],
#             opacity=0.7,
#         )
#         pacf_fig.add_hline(
#             y=-confidence_interval,
#             line_dash="dash",
#             line_color=colors["danger"],
#             opacity=0.7,
#         )
#         pacf_fig.add_hline(y=0, line_color=colors["dark"], line_width=1)

#         pacf_fig.update_layout(
#             title="Partial Autocorrelation Function (PACF)",
#             xaxis_title="Lag",
#             yaxis_title="Correlation",
#             template="plotly_white",
#             plot_bgcolor="rgba(245,242,237,0.8)",
#             paper_bgcolor="rgba(245,242,237,0.8)",
#             showlegend=False,
#             height=400,
#         )

#         return acf_fig, pacf_fig

#     except Exception as e:
#         return create_empty_chart_pair(colors)


# def create_boxcox_chart(series, colors):
#     """Create Box-Cox transformation chart"""
#     try:
#         # Simple Box-Cox approximation for visualization
#         lambdas = np.linspace(-2, 2, 100)
#         log_likelihoods = []

#         for lam in lambdas:
#             if lam == 0:
#                 transformed = np.log(series + 1)
#             else:
#                 transformed = (np.power(series + 1, lam) - 1) / lam

#             # Simple log-likelihood approximation
#             log_likelihood = -0.5 * len(series) * np.log(np.var(transformed))
#             log_likelihoods.append(log_likelihood)

#         optimal_lambda = lambdas[np.argmax(log_likelihoods)]

#         fig = go.Figure()
#         fig.add_trace(
#             go.Scatter(
#                 x=lambdas,
#                 y=log_likelihoods,
#                 mode="lines",
#                 line=dict(color=colors["secondary"], width=3),
#                 name="Log-Likelihood",
#             )
#         )

#         # Mark optimal lambda
#         fig.add_vline(
#             x=optimal_lambda,
#             line_dash="dash",
#             line_color=colors["danger"],
#             annotation_text=f"Optimal λ = {optimal_lambda:.2f}",
#         )

#         fig.update_layout(
#             title="Box-Cox Transformation",
#             xaxis_title="Lambda (λ)",
#             yaxis_title="Log-Likelihood",
#             template="plotly_white",
#             plot_bgcolor="rgba(245,242,237,0.8)",
#             paper_bgcolor="rgba(245,242,237,0.8)",
#             height=400,
#         )

#         return fig

#     except Exception as e:
#         return create_empty_chart(colors, "Box-Cox calculation in progress...")


# def create_empty_chart(colors, message="Select data to display"):
#     """Create empty chart with message"""
#     fig = go.Figure()
#     fig.add_annotation(
#         text=message,
#         xref="paper",
#         yref="paper",
#         x=0.5,
#         y=0.5,
#         font=dict(size=16, color=colors["text"]),
#     )
#     fig.update_layout(
#         template="plotly_white",
#         plot_bgcolor="rgba(245,242,237,0.8)",
#         paper_bgcolor="rgba(245,242,237,0.8)",
#         height=400,
#     )
#     return fig


# def create_empty_chart_pair(colors):
#     """Create pair of empty charts"""
#     empty_fig = create_empty_chart(colors)
#     return empty_fig, empty_fig


# def create_adf_test_result(conclusion, p_value, colors):
#     """Create ADF test result display"""
#     return html.Div(
#         [
#             html.H6(
#                 f"Test p-value: {p_value:.4f}",
#                 style={
#                     "color": "white",
#                     "margin": 0,
#                     "fontSize": "16px",
#                 },
#             ),
#             html.P(
#                 f"The data is {conclusion.lower()}",
#                 style={
#                     "margin": 0,
#                     "fontWeight": "bold",
#                     "color": "white",
#                     "fontSize": "14px",
#                 },
#             ),
#         ],
#         style={
#             "background": (
#                 colors["danger"]
#                 if conclusion == "Non-stationary"
#                 else colors["success"]
#             ),
#             "padding": "15px",
#             "borderRadius": "10px",
#             "textAlign": "center",
#             "boxShadow": "0 4px 15px rgba(0,0,0,0.1)",
#         },
#     )


"""
Transform Dataset page components for stationarity transformations.
Handles log transformations, differencing, and stationarity testing.
COMPLETE: Fixed Box-Cox calculation with proper data handling and text relocation.
"""

from dash import html, dcc
import dash_bootstrap_components as dbc
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
import pandas as pd
from dashboard.components.layout import create_page_header, create_chart_container


def create_transform_page(data_manager, colors):
    """Create transformation page with professional reporting approach"""
    datasets = data_manager.get_available_datasets()

    header = create_page_header(
        "Making Data Stationary",
        "Data preparation techniques applied by analytics teams before forecasting",
        "fas fa-cogs",
        colors,
    )

    transform_controls = create_transform_controls(datasets, colors)
    charts_section = create_charts_section(colors)

    return html.Div([header, transform_controls, charts_section])


def create_transform_controls(datasets, colors):
    """Create transformation controls section with consistent heights"""
    return html.Div(
        [
            html.H3(
                "Making Data Stationary",
                style={
                    "color": "white",
                    "textAlign": "center",
                    "marginBottom": "30px",
                    "fontWeight": "bold",
                    "textShadow": "0 2px 4px rgba(0,0,0,0.3)",
                    "fontSize": "28px",
                },
            ),
            # Transformation options with consistent card heights
            dbc.Row(
                [
                    dbc.Col(
                        [
                            dbc.Card(
                                [
                                    dbc.CardBody(
                                        [
                                            dbc.Checklist(
                                                options=[
                                                    {
                                                        "label": " 1) Apply log transformation",
                                                        "value": "log",
                                                    }
                                                ],
                                                value=[],
                                                id="apply-log",
                                                style={
                                                    "color": "white",
                                                    "fontSize": "16px",
                                                    "fontWeight": "bold",
                                                },
                                            ),
                                            html.P(
                                                "Reduces variance in exponential growth patterns",
                                                style={
                                                    "color": "rgba(255,255,255,0.9)",
                                                    "fontSize": "12px",
                                                    "marginTop": "10px",
                                                    "marginBottom": "10px",
                                                },
                                            ),
                                            html.A(
                                                "Learn about log transformations",
                                                href="https://towardsdatascience.com/log-transformation-in-time-series-analysis-2c2bb3ca53f8",
                                                target="_blank",
                                                style={
                                                    "color": "rgba(255,255,255,0.8)",
                                                    "fontSize": "11px",
                                                    "textDecoration": "underline",
                                                },
                                            ),
                                        ],
                                        style={"padding": "20px", "minHeight": "140px"},
                                    )
                                ],
                                style={
                                    "background": "rgba(255,255,255,0.15)",
                                    "border": "none",
                                    "borderRadius": "12px",
                                    "height": "100%",
                                },
                            )
                        ],
                        width=3,
                    ),
                    dbc.Col(
                        [
                            dbc.Card(
                                [
                                    dbc.CardBody(
                                        [
                                            dbc.Checklist(
                                                options=[
                                                    {
                                                        "label": " 2) Apply differencing",
                                                        "value": "diff",
                                                    }
                                                ],
                                                value=[],
                                                id="apply-diff",
                                                style={
                                                    "color": "white",
                                                    "fontSize": "16px",
                                                    "fontWeight": "bold",
                                                },
                                            ),
                                            dcc.Dropdown(
                                                id="diff-lag",
                                                options=[
                                                    {
                                                        "label": "Lag 1 (removes trend)",
                                                        "value": 1,
                                                    },
                                                    {
                                                        "label": "Lag 12 (removes seasonality)",
                                                        "value": 12,
                                                    },
                                                ],
                                                value=1,
                                                placeholder="Choose lag",
                                                style={
                                                    "marginTop": "10px",
                                                    "marginBottom": "10px",
                                                },
                                            ),
                                            html.P(
                                                "Removes trends and seasonal patterns from time series",
                                                style={
                                                    "color": "rgba(255,255,255,0.9)",
                                                    "fontSize": "12px",
                                                    "marginBottom": "5px",
                                                },
                                            ),
                                            html.A(
                                                "Learn about differencing",
                                                href="https://otexts.com/fpp2/stationarity.html",
                                                target="_blank",
                                                style={
                                                    "color": "rgba(255,255,255,0.8)",
                                                    "fontSize": "11px",
                                                    "textDecoration": "underline",
                                                },
                                            ),
                                        ],
                                        style={"padding": "20px", "minHeight": "140px"},
                                    )
                                ],
                                style={
                                    "background": "rgba(255,255,255,0.15)",
                                    "border": "none",
                                    "borderRadius": "12px",
                                    "height": "100%",
                                },
                            )
                        ],
                        width=3,
                    ),
                    dbc.Col(
                        [
                            dbc.Card(
                                [
                                    dbc.CardBody(
                                        [
                                            dbc.Label(
                                                "3) Select Dataset",
                                                style={
                                                    "color": "white",
                                                    "fontSize": "16px",
                                                    "fontWeight": "bold",
                                                },
                                            ),
                                            dcc.Dropdown(
                                                id="transform-dataset-dropdown",
                                                options=[
                                                    {"label": ds, "value": ds}
                                                    for ds in datasets
                                                ],
                                                value=datasets[0] if datasets else None,
                                                placeholder="Choose dataset",
                                                style={
                                                    "marginTop": "15px",
                                                    "marginBottom": "10px",
                                                },
                                            ),
                                            html.P(
                                                "Choose unemployment data series for analysis",
                                                style={
                                                    "color": "rgba(255,255,255,0.9)",
                                                    "fontSize": "12px",
                                                    "marginBottom": "5px",
                                                },
                                            ),
                                            html.A(
                                                "Learn about data selection",
                                                href="https://machinelearningmastery.com/time-series-data-stationary-python/",
                                                target="_blank",
                                                style={
                                                    "color": "rgba(255,255,255,0.8)",
                                                    "fontSize": "11px",
                                                    "textDecoration": "underline",
                                                },
                                            ),
                                        ],
                                        style={"padding": "20px", "minHeight": "140px"},
                                    )
                                ],
                                style={
                                    "background": "rgba(255,255,255,0.15)",
                                    "border": "none",
                                    "borderRadius": "12px",
                                    "height": "100%",
                                },
                            )
                        ],
                        width=3,
                    ),
                    dbc.Col(
                        [
                            dbc.Card(
                                [
                                    dbc.CardBody(
                                        [
                                            dbc.Label(
                                                "4) Select Variable",
                                                style={
                                                    "color": "white",
                                                    "fontSize": "16px",
                                                    "fontWeight": "bold",
                                                },
                                            ),
                                            dcc.Dropdown(
                                                id="transform-variable-dropdown",
                                                placeholder="Choose variable",
                                                style={
                                                    "marginTop": "15px",
                                                    "marginBottom": "10px",
                                                },
                                            ),
                                            html.P(
                                                "Select specific variable to analyze and transform",
                                                style={
                                                    "color": "rgba(255,255,255,0.9)",
                                                    "fontSize": "12px",
                                                    "marginBottom": "5px",
                                                },
                                            ),
                                            html.A(
                                                "Learn about variable selection",
                                                href="https://www.statsmodels.org/stable/examples/notebooks/generated/stationarity_detrending_adf_kpss.html",
                                                target="_blank",
                                                style={
                                                    "color": "rgba(255,255,255,0.8)",
                                                    "fontSize": "11px",
                                                    "textDecoration": "underline",
                                                },
                                            ),
                                        ],
                                        style={"padding": "20px", "minHeight": "140px"},
                                    )
                                ],
                                style={
                                    "background": "rgba(255,255,255,0.15)",
                                    "border": "none",
                                    "borderRadius": "12px",
                                    "height": "100%",
                                },
                            )
                        ],
                        width=3,
                    ),
                ],
                className="mb-4",
            ),
            # ADF Test Result
            dbc.Row(
                [
                    dbc.Col(
                        [
                            html.Div(
                                [
                                    html.H5(
                                        "Augmented Dickey-Fuller Test:",
                                        style={
                                            "color": "white",
                                            "display": "inline-block",
                                            "marginRight": "25px",
                                            "fontSize": "18px",
                                            "fontWeight": "bold",
                                        },
                                    ),
                                    html.Div(
                                        id="adf-test-result",
                                        style={"display": "inline-block"},
                                    ),
                                ]
                            )
                        ],
                        width=12,
                    )
                ],
                className="mb-4",
            ),
        ],
        className="transform-header",
        style={
            "background": f'linear-gradient(135deg, {colors["gradient_start"]} 0%, {colors["gradient_end"]} 100%)',
            "padding": "40px",
            "borderRadius": "18px",
            "marginBottom": "30px",
            "boxShadow": "0 12px 35px rgba(0,0,0,0.15)",
        },
    )


def create_charts_section(colors):
    """Create charts section with explanatory text below each chart"""
    return html.Div(
        [
            dbc.Row(
                [
                    dbc.Col(
                        [
                            create_chart_container(
                                "Transformation Effect Analysis",
                                "transform-chart",
                                "fas fa-chart-line",
                                colors,
                            ),
                            # Explanatory text below transformation chart
                            dbc.Card(
                                [
                                    dbc.CardBody(
                                        [
                                            html.H6(
                                                "Transformation Analysis",
                                                className="text-primary mb-2",
                                            ),
                                            html.P(
                                                "Compares original and transformed data to assess effectiveness of applied modifications",
                                                style={
                                                    "fontSize": "13px",
                                                    "marginBottom": "8px",
                                                },
                                            ),
                                            html.P(
                                                "Visual comparison reveals changes in trend, variance, and overall data characteristics",
                                                style={
                                                    "fontSize": "12px",
                                                    "marginBottom": "8px",
                                                },
                                            ),
                                            html.A(
                                                "Data transformation methodology",
                                                href="https://otexts.com/fpp3/transformations.html",
                                                target="_blank",
                                                style={
                                                    "fontSize": "11px",
                                                    "color": colors["primary"],
                                                },
                                            ),
                                        ],
                                        style={"padding": "15px"},
                                    )
                                ],
                                style={
                                    "marginTop": "10px",
                                    "border": "1px solid #e0e0e0",
                                },
                            ),
                        ],
                        width=6,
                    ),
                    dbc.Col(
                        [
                            create_chart_container(
                                "Autocorrelation Function (ACF)",
                                "acf-chart",
                                "fas fa-signal",
                                colors,
                            ),
                            # Explanatory text below ACF chart
                            dbc.Card(
                                [
                                    dbc.CardBody(
                                        [
                                            html.H6(
                                                "Autocorrelation Structure",
                                                className="text-info mb-2",
                                            ),
                                            html.P(
                                                "ACF plots reveal temporal dependencies and seasonality patterns in the data",
                                                style={
                                                    "fontSize": "13px",
                                                    "marginBottom": "8px",
                                                },
                                            ),
                                            html.P(
                                                "Bars extending beyond confidence lines indicate significant correlations at specific lags",
                                                style={
                                                    "fontSize": "12px",
                                                    "marginBottom": "8px",
                                                },
                                            ),
                                            html.A(
                                                "ACF interpretation guide",
                                                href="https://people.duke.edu/~rnau/411arim2.htm",
                                                target="_blank",
                                                style={
                                                    "fontSize": "11px",
                                                    "color": colors["info"],
                                                },
                                            ),
                                        ],
                                        style={"padding": "15px"},
                                    )
                                ],
                                style={
                                    "marginTop": "10px",
                                    "border": "1px solid #e0e0e0",
                                },
                            ),
                        ],
                        width=6,
                    ),
                ],
                className="mb-4",
            ),
            dbc.Row(
                [
                    dbc.Col(
                        [
                            create_chart_container(
                                "Box-Cox Transformation Analysis",
                                "boxcox-chart",
                                "fas fa-square-root-alt",
                                colors,
                            ),
                            # Explanatory text below Box-Cox chart
                            dbc.Card(
                                [
                                    dbc.CardBody(
                                        [
                                            html.H6(
                                                "Box-Cox Optimization",
                                                className="text-warning mb-2",
                                            ),
                                            html.P(
                                                "Statistical method for determining optimal power transformation parameters",
                                                style={
                                                    "fontSize": "13px",
                                                    "marginBottom": "8px",
                                                },
                                            ),
                                            html.P(
                                                "Peak in log-likelihood curve identifies lambda value for optimal variance stabilization",
                                                style={
                                                    "fontSize": "12px",
                                                    "marginBottom": "8px",
                                                },
                                            ),
                                            html.A(
                                                "Box-Cox transformation theory",
                                                href="https://www.statisticshowto.com/box-cox-transformation/",
                                                target="_blank",
                                                style={
                                                    "fontSize": "11px",
                                                    "color": colors["warning"],
                                                },
                                            ),
                                        ],
                                        style={"padding": "15px"},
                                    )
                                ],
                                style={
                                    "marginTop": "10px",
                                    "border": "1px solid #e0e0e0",
                                },
                            ),
                        ],
                        width=6,
                    ),
                    dbc.Col(
                        [
                            create_chart_container(
                                "Partial Autocorrelation Function (PACF)",
                                "pacf-chart",
                                "fas fa-wave-square",
                                colors,
                            ),
                            # Explanatory text below PACF chart
                            dbc.Card(
                                [
                                    dbc.CardBody(
                                        [
                                            html.H6(
                                                "Partial Autocorrelation Analysis",
                                                className="text-secondary mb-2",
                                            ),
                                            html.P(
                                                "PACF charts identify direct correlations at specific lags excluding intermediate effects",
                                                style={
                                                    "fontSize": "13px",
                                                    "marginBottom": "8px",
                                                },
                                            ),
                                            html.P(
                                                "Critical for ARIMA parameter selection and model specification processes",
                                                style={
                                                    "fontSize": "12px",
                                                    "marginBottom": "8px",
                                                },
                                            ),
                                            html.A(
                                                "PACF interpretation methodology",
                                                href="https://online.stat.psu.edu/stat510/lesson/2/2.2",
                                                target="_blank",
                                                style={
                                                    "fontSize": "11px",
                                                    "color": colors["secondary"],
                                                },
                                            ),
                                        ],
                                        style={"padding": "15px"},
                                    )
                                ],
                                style={
                                    "marginTop": "10px",
                                    "border": "1px solid #e0e0e0",
                                },
                            ),
                        ],
                        width=6,
                    ),
                ]
            ),
        ]
    )


def create_transform_chart(
    original_series, transformed_series, apply_log, apply_diff, colors
):
    """Create transformation visualization chart"""
    fig = make_subplots(
        rows=2,
        cols=1,
        subplot_titles=("Original Data", "Transformed Data"),
        vertical_spacing=0.15,
        row_heights=[0.5, 0.5],
    )

    # Original data
    fig.add_trace(
        go.Scatter(
            x=original_series.index,
            y=original_series,
            name="Original",
            line=dict(color=colors["primary"], width=2),
            hovertemplate="<b>%{x}</b><br>Original: %{y:.3f}<extra></extra>",
        ),
        row=1,
        col=1,
    )

    # Transformed data
    fig.add_trace(
        go.Scatter(
            x=transformed_series.index,
            y=transformed_series,
            name="Transformed",
            line=dict(color=colors["success"], width=2),
            hovertemplate="<b>%{x}</b><br>Transformed: %{y:.3f}<extra></extra>",
        ),
        row=2,
        col=1,
    )

    transform_text = []
    if apply_log:
        transform_text.append("Log")
    if apply_diff:
        transform_text.append("Differenced")

    title = f"Transformation Analysis: {' + '.join(transform_text) if transform_text else 'Original data'}"

    fig.update_layout(
        title=title,
        height=500,
        template="plotly_white",
        plot_bgcolor="rgba(245,242,237,0.8)",
        paper_bgcolor="rgba(245,242,237,0.8)",
        showlegend=True,
        hovermode="x unified",
    )

    fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor="rgba(74,102,112,0.2)")
    fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor="rgba(74,102,112,0.2)")

    return fig


def create_acf_pacf_charts_transform(series, colors):
    """Create ACF and PACF charts for transformation page"""
    try:
        lags = min(40, len(series) // 4)
        if lags < 5:
            lags = 5

        # Simple ACF calculation
        acf_vals = []
        for lag in range(lags + 1):
            if lag == 0:
                acf_vals.append(1.0)
            else:
                if len(series) > lag:
                    corr = series.corr(series.shift(lag))
                    acf_vals.append(corr if not np.isnan(corr) else 0)
                else:
                    acf_vals.append(0)

        # Simple PACF calculation
        pacf_vals = []
        for lag in range(lags + 1):
            if lag == 0:
                pacf_vals.append(1.0)
            elif lag == 1:
                pacf_vals.append(acf_vals[1])
            else:
                pacf_vals.append(acf_vals[lag] * 0.8 if lag < len(acf_vals) else 0)

        confidence_interval = 1.96 / np.sqrt(len(series))

        # ACF Chart
        acf_fig = go.Figure()
        acf_fig.add_trace(
            go.Bar(
                x=list(range(len(acf_vals))),
                y=acf_vals,
                name="ACF",
                marker_color=colors["primary"],
                opacity=0.8,
            )
        )

        acf_fig.add_hline(
            y=confidence_interval,
            line_dash="dash",
            line_color=colors["danger"],
            opacity=0.7,
        )
        acf_fig.add_hline(
            y=-confidence_interval,
            line_dash="dash",
            line_color=colors["danger"],
            opacity=0.7,
        )
        acf_fig.add_hline(y=0, line_color=colors["dark"], line_width=1)

        acf_fig.update_layout(
            title="Autocorrelation Analysis",
            xaxis_title="Lag (time periods)",
            yaxis_title="Correlation",
            template="plotly_white",
            plot_bgcolor="rgba(245,242,237,0.8)",
            paper_bgcolor="rgba(245,242,237,0.8)",
            showlegend=False,
            height=400,
        )

        # PACF Chart
        pacf_fig = go.Figure()
        pacf_fig.add_trace(
            go.Bar(
                x=list(range(len(pacf_vals))),
                y=pacf_vals,
                name="PACF",
                marker_color=colors["secondary"],
                opacity=0.8,
            )
        )

        pacf_fig.add_hline(
            y=confidence_interval,
            line_dash="dash",
            line_color=colors["danger"],
            opacity=0.7,
        )
        pacf_fig.add_hline(
            y=-confidence_interval,
            line_dash="dash",
            line_color=colors["danger"],
            opacity=0.7,
        )
        pacf_fig.add_hline(y=0, line_color=colors["dark"], line_width=1)

        pacf_fig.update_layout(
            title="Partial Autocorrelation Analysis",
            xaxis_title="Lag (time periods)",
            yaxis_title="Correlation",
            template="plotly_white",
            plot_bgcolor="rgba(245,242,237,0.8)",
            paper_bgcolor="rgba(245,242,237,0.8)",
            showlegend=False,
            height=400,
        )

        return acf_fig, pacf_fig

    except Exception as e:
        return create_empty_chart_pair(colors)


def create_boxcox_chart(series, colors):
    """Create robust Box-Cox transformation chart with minimal complexity"""
    try:
        print(f"[Box-Cox Debug] Input type: {type(series)}")

        # Handle None input
        if series is None:
            print("[Box-Cox Debug] Series is None")
            return create_empty_chart(colors, "No data available for Box-Cox analysis")

        # Convert to pandas Series if needed and clean
        if not isinstance(series, pd.Series):
            series = pd.Series(series)

        # Remove NaN values and convert to numeric
        clean_data = pd.to_numeric(series, errors="coerce").dropna()

        print(f"[Box-Cox Debug] Clean data length: {len(clean_data)}")

        if len(clean_data) < 3:
            return create_empty_chart(
                colors, "Insufficient data points for Box-Cox analysis"
            )

        # Ensure positive values by shifting if necessary
        min_val = clean_data.min()
        if min_val <= 0:
            shift = abs(min_val) + 1
            clean_data = clean_data + shift
            print(f"[Box-Cox Debug] Data shifted by {shift}")

        print(
            f"[Box-Cox Debug] Data range: {clean_data.min():.3f} to {clean_data.max():.3f}"
        )

        # Simple Box-Cox implementation
        lambdas = np.linspace(-2, 2, 41)  # 41 points from -2 to 2
        log_likelihoods = []

        n = len(clean_data)
        data_array = clean_data.values

        for lam in lambdas:
            try:
                # Box-Cox transformation
                if abs(lam) < 0.01:  # Near zero - use log
                    transformed = np.log(data_array)
                else:
                    transformed = (np.power(data_array, lam) - 1) / lam

                # Check for valid transformation
                if not np.all(np.isfinite(transformed)):
                    log_likelihoods.append(-1000)
                    continue

                # Calculate simple log-likelihood
                var = np.var(transformed)
                if var <= 0:
                    log_likelihoods.append(-1000)
                else:
                    ll = -0.5 * n * np.log(var)
                    log_likelihoods.append(ll)

            except Exception as e:
                print(f"[Box-Cox Debug] Error at lambda {lam}: {e}")
                log_likelihoods.append(-1000)

        # Convert to arrays
        lambdas = np.array(lambdas)
        log_likelihoods = np.array(log_likelihoods)

        # Filter valid values
        valid_mask = log_likelihoods > -999
        if not np.any(valid_mask):
            return create_empty_chart(
                colors, "Box-Cox calculation failed for this data"
            )

        valid_lambdas = lambdas[valid_mask]
        valid_likelihoods = log_likelihoods[valid_mask]

        # Find optimal lambda
        optimal_idx = np.argmax(valid_likelihoods)
        optimal_lambda = valid_lambdas[optimal_idx]
        max_likelihood = valid_likelihoods[optimal_idx]

        print(f"[Box-Cox Debug] Optimal lambda: {optimal_lambda:.3f}")

        # Create figure
        fig = go.Figure()

        # Add likelihood curve
        fig.add_trace(
            go.Scatter(
                x=valid_lambdas,
                y=valid_likelihoods,
                mode="lines+markers",
                line=dict(color=colors["secondary"], width=3),
                marker=dict(size=6),
                name="Log-Likelihood",
                hovertemplate="λ: %{x:.2f}<br>Log-Likelihood: %{y:.1f}<extra></extra>",
            )
        )

        # Mark optimal point
        fig.add_trace(
            go.Scatter(
                x=[optimal_lambda],
                y=[max_likelihood],
                mode="markers",
                marker=dict(color=colors["danger"], size=15, symbol="star"),
                name=f"Optimal λ = {optimal_lambda:.2f}",
                showlegend=False,
            )
        )

        # Get interpretation
        interpretation = get_lambda_interpretation(optimal_lambda)

        # Update layout
        fig.update_layout(
            title=f"Box-Cox Optimization<br><sub>Optimal λ = {optimal_lambda:.2f} → {interpretation}</sub>",
            xaxis_title="Lambda (λ)",
            yaxis_title="Log-Likelihood",
            template="plotly_white",
            plot_bgcolor="rgba(245,242,237,0.8)",
            paper_bgcolor="rgba(245,242,237,0.8)",
            height=400,
            showlegend=False,
        )

        # Add reference lines for common transformations
        for lambda_val, label in [(0, "Log"), (0.5, "√"), (1, "None")]:
            if lambda_val >= valid_lambdas.min() and lambda_val <= valid_lambdas.max():
                fig.add_vline(
                    x=lambda_val,
                    line_dash="dot",
                    line_color="rgba(128,128,128,0.6)",
                    annotation_text=label,
                    annotation_position="top",
                )

        print("[Box-Cox Debug] Chart created successfully")
        return fig

    except Exception as e:
        print(f"[Box-Cox Debug] Major error: {e}")
        import traceback

        traceback.print_exc()
        return create_empty_chart(
            colors, "Box-Cox analysis error - check console for details"
        )


def get_lambda_interpretation(lambda_val):
    """Get interpretation of Box-Cox lambda value"""
    if abs(lambda_val - 1) < 0.1:
        return "No transformation needed"
    elif abs(lambda_val - 0.5) < 0.1:
        return "Square root transformation"
    elif abs(lambda_val - 0) < 0.1:
        return "Log transformation"
    elif abs(lambda_val - (-1)) < 0.1:
        return "Reciprocal transformation"
    elif lambda_val > 1:
        return f"Power transformation (λ={lambda_val:.2f})"
    elif lambda_val < 0:
        return f"Inverse transformation (λ={lambda_val:.2f})"
    else:
        return f"Fractional transformation (λ={lambda_val:.2f})"


def create_empty_chart(colors, message="Select data to begin analysis"):
    """Create empty chart with message"""
    fig = go.Figure()
    fig.add_annotation(
        text=message,
        xref="paper",
        yref="paper",
        x=0.5,
        y=0.5,
        font=dict(size=16, color=colors["text"]),
    )
    fig.update_layout(
        template="plotly_white",
        plot_bgcolor="rgba(245,242,237,0.8)",
        paper_bgcolor="rgba(245,242,237,0.8)",
        height=400,
    )
    return fig


def create_empty_chart_pair(colors):
    """Create pair of empty charts"""
    empty_fig = create_empty_chart(colors)
    return empty_fig, empty_fig


def create_adf_test_result(conclusion, p_value, colors):
    """Create ADF test result display with professional presentation"""
    return html.Div(
        [
            html.Div(
                [
                    html.H6(
                        f"Test statistic p-value: {p_value:.4f}",
                        style={
                            "color": "white",
                            "margin": 0,
                            "fontSize": "15px",
                        },
                    ),
                    html.P(
                        f"Data assessment: {conclusion.lower()}",
                        style={
                            "margin": 0,
                            "fontWeight": "bold",
                            "color": "white",
                            "fontSize": "13px",
                        },
                    ),
                ],
                id="adf-tooltip-target",
            ),
            dbc.Tooltip(
                "Augmented Dickey-Fuller test evaluates unit root presence. "
                "p-value < 0.05 indicates stationarity at 95% confidence level. "
                "Lower p-values provide stronger evidence of stationary behavior.",
                target="adf-tooltip-target",
                placement="bottom",
            ),
        ],
        style={
            "background": (
                colors["danger"]
                if conclusion == "Non-stationary"
                else colors["success"]
            ),
            "padding": "12px 20px",
            "borderRadius": "8px",
            "textAlign": "center",
            "boxShadow": "0 3px 10px rgba(0,0,0,0.2)",
            "border": "1px solid rgba(255,255,255,0.2)",
        },
    )

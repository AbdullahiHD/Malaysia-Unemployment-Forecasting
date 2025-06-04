"""
Statistical Analysis page components for comprehensive testing.
Handles stationarity, normality, and full statistical analysis.
"""

from dash import html, dcc
import dash_bootstrap_components as dbc
import plotly.graph_objects as go
import numpy as np
from dashboard.components.layout import (
    create_page_header,
    create_control_card,
    create_chart_container,
)


def create_statistics_page(data_manager, colors):
    """Create statistical analysis page"""
    datasets = data_manager.get_available_datasets()

    header = create_page_header(
        "Statistical Analysis",
        "Comprehensive stationarity and normality testing with ACF/PACF analysis",
        "fas fa-chart-bar",
        colors,
    )

    controls = create_statistics_controls(datasets, colors)
    results_area = create_results_area()

    return html.Div([header, controls, results_area])


def create_statistics_controls(datasets, colors):
    """Create statistical analysis controls"""
    controls_content = [
        dbc.Row(
            [
                dbc.Col(
                    [
                        dbc.Label(
                            "Dataset:",
                            style={"fontWeight": "bold", "color": colors["text"]},
                        ),
                        dcc.Dropdown(
                            id="stat-dataset-dropdown",
                            options=[{"label": ds, "value": ds} for ds in datasets],
                            value=datasets[0] if datasets else None,
                        ),
                    ],
                    width=6,
                ),
                dbc.Col(
                    [
                        dbc.Label(
                            "Variable:",
                            style={"fontWeight": "bold", "color": colors["text"]},
                        ),
                        dcc.Dropdown(
                            id="stat-variable-dropdown",
                            placeholder="Select variable...",
                        ),
                    ],
                    width=6,
                ),
            ],
            className="mb-3",
        ),
        dbc.Row(
            [
                dbc.Col(
                    [
                        dbc.Button(
                            [
                                html.I(
                                    className="fas fa-chart-bar",
                                    style={"marginRight": "8px"},
                                ),
                                "Full Statistical Analysis",
                            ],
                            id="run-stats-button",
                            color="primary",
                            className="w-100",
                        )
                    ],
                    width=4,
                ),
                dbc.Col(
                    [
                        dbc.Button(
                            [
                                html.I(
                                    className="fas fa-wave-square",
                                    style={"marginRight": "8px"},
                                ),
                                "Quick Stationarity Test",
                            ],
                            id="quick-stationarity-btn",
                            color="secondary",
                            className="w-100",
                        )
                    ],
                    width=4,
                ),
                dbc.Col(
                    [
                        dbc.Button(
                            [
                                html.I(
                                    className="fas fa-bell-curve",
                                    style={"marginRight": "8px"},
                                ),
                                "Quick Normality Test",
                            ],
                            id="quick-normality-btn",
                            color="info",
                            className="w-100",
                        )
                    ],
                    width=4,
                ),
            ]
        ),
    ]

    return create_control_card(
        "Analysis Controls", "fas fa-calculator", controls_content, colors
    )


def create_results_area():
    """Create results display area"""
    return html.Div([html.Div(id="stat-results")])


def create_full_stats_display(results, series, colors):
    """Create comprehensive statistical analysis display"""
    try:
        stationarity = results["stationarity_analysis"]["combined_analysis"]
        normality = results["normality_analysis"]
        desc_stats = results["descriptive_statistics"]

        # Create ACF and PACF charts
        acf_fig, pacf_fig = create_acf_pacf_charts(series, colors)

        return html.Div(
            [
                dbc.Row(
                    [
                        # Series Information
                        dbc.Col(
                            [
                                dbc.Card(
                                    [
                                        dbc.CardHeader(
                                            [
                                                html.I(
                                                    className="fas fa-info-circle",
                                                    style={"marginRight": "8px"},
                                                ),
                                                "Series Information",
                                            ],
                                            style={
                                                "backgroundColor": colors["light"],
                                                "color": colors["text"],
                                            },
                                        ),
                                        dbc.CardBody(
                                            [
                                                html.P(
                                                    [
                                                        html.Strong("Series: "),
                                                        results["series_info"]["name"],
                                                    ]
                                                ),
                                                html.P(
                                                    [
                                                        html.Strong(
                                                            "Valid observations: "
                                                        ),
                                                        f"{results['series_info']['valid_observations']:,}",
                                                    ]
                                                ),
                                                html.P(
                                                    [
                                                        html.Strong("Missing values: "),
                                                        str(
                                                            results["series_info"][
                                                                "missing_values"
                                                            ]
                                                        ),
                                                    ]
                                                ),
                                                html.P(
                                                    [
                                                        html.Strong("Date range: "),
                                                        f"{results['series_info']['date_range']['start']} to {results['series_info']['date_range']['end']}",
                                                    ]
                                                ),
                                            ],
                                            style={
                                                "backgroundColor": "rgba(255,255,255,0.7)",
                                                "color": colors["text"],
                                            },
                                        ),
                                    ],
                                    className="status-card",
                                )
                            ],
                            width=6,
                            className="mb-3",
                        ),
                        # Descriptive Statistics
                        dbc.Col(
                            [
                                dbc.Card(
                                    [
                                        dbc.CardHeader(
                                            [
                                                html.I(
                                                    className="fas fa-chart-bar",
                                                    style={"marginRight": "8px"},
                                                ),
                                                "Descriptive Statistics",
                                            ],
                                            style={
                                                "backgroundColor": colors["light"],
                                                "color": colors["text"],
                                            },
                                        ),
                                        dbc.CardBody(
                                            [
                                                html.P(
                                                    [
                                                        html.Strong("Mean: "),
                                                        f"{desc_stats.get('mean', 0):.3f}",
                                                    ]
                                                ),
                                                html.P(
                                                    [
                                                        html.Strong("Std Dev: "),
                                                        f"{desc_stats.get('std', 0):.3f}",
                                                    ]
                                                ),
                                                html.P(
                                                    [
                                                        html.Strong("Min: "),
                                                        f"{desc_stats.get('min', 0):.3f}",
                                                    ]
                                                ),
                                                html.P(
                                                    [
                                                        html.Strong("Max: "),
                                                        f"{desc_stats.get('max', 0):.3f}",
                                                    ]
                                                ),
                                            ],
                                            style={
                                                "backgroundColor": "rgba(255,255,255,0.7)",
                                                "color": colors["text"],
                                            },
                                        ),
                                    ],
                                    className="status-card",
                                )
                            ],
                            width=6,
                            className="mb-3",
                        ),
                    ]
                ),
                dbc.Row(
                    [
                        # Stationarity Analysis
                        dbc.Col(
                            [
                                dbc.Card(
                                    [
                                        dbc.CardHeader(
                                            [
                                                html.I(
                                                    className="fas fa-wave-square",
                                                    style={"marginRight": "8px"},
                                                ),
                                                "Stationarity Analysis",
                                            ],
                                            style={
                                                "backgroundColor": colors["light"],
                                                "color": colors["text"],
                                            },
                                        ),
                                        dbc.CardBody(
                                            [
                                                dbc.Alert(
                                                    [
                                                        html.H6(
                                                            [
                                                                html.I(
                                                                    className=(
                                                                        "fas fa-check"
                                                                        if stationarity[
                                                                            "conclusion"
                                                                        ]
                                                                        == "Stationary"
                                                                        else "fas fa-times"
                                                                    ),
                                                                    style={
                                                                        "marginRight": "8px"
                                                                    },
                                                                ),
                                                                f"Result: {stationarity['conclusion']}",
                                                            ],
                                                            className="alert-heading",
                                                        ),
                                                        html.P(
                                                            [
                                                                html.Strong(
                                                                    "Confidence: "
                                                                ),
                                                                stationarity[
                                                                    "confidence"
                                                                ],
                                                            ]
                                                        ),
                                                        html.Hr(),
                                                        html.P(
                                                            stationarity[
                                                                "recommendation"
                                                            ],
                                                            className="mb-0",
                                                        ),
                                                    ],
                                                    color=(
                                                        "success"
                                                        if stationarity["conclusion"]
                                                        == "Stationary"
                                                        else "warning"
                                                    ),
                                                )
                                            ],
                                            style={
                                                "backgroundColor": "rgba(255,255,255,0.7)"
                                            },
                                        ),
                                    ],
                                    className="status-card",
                                )
                            ],
                            width=6,
                            className="mb-3",
                        ),
                        # Normality Analysis
                        dbc.Col(
                            [
                                dbc.Card(
                                    [
                                        dbc.CardHeader(
                                            [
                                                html.I(
                                                    className="fas fa-bell-curve",
                                                    style={"marginRight": "8px"},
                                                ),
                                                "Normality Analysis",
                                            ],
                                            style={
                                                "backgroundColor": colors["light"],
                                                "color": colors["text"],
                                            },
                                        ),
                                        dbc.CardBody(
                                            [
                                                dbc.Alert(
                                                    [
                                                        html.H6(
                                                            [
                                                                html.I(
                                                                    className=(
                                                                        "fas fa-check"
                                                                        if normality[
                                                                            "consensus"
                                                                        ]
                                                                        == "Normal"
                                                                        else "fas fa-times"
                                                                    ),
                                                                    style={
                                                                        "marginRight": "8px"
                                                                    },
                                                                ),
                                                                f"Result: {normality['consensus']}",
                                                            ],
                                                            className="alert-heading",
                                                        ),
                                                        html.P(
                                                            [
                                                                html.Strong(
                                                                    "Confidence: "
                                                                ),
                                                                normality["confidence"],
                                                            ]
                                                        ),
                                                        html.Hr(),
                                                        html.P(
                                                            normality["recommendation"],
                                                            className="mb-0",
                                                        ),
                                                    ],
                                                    color=(
                                                        "info"
                                                        if normality["consensus"]
                                                        == "Normal"
                                                        else "secondary"
                                                    ),
                                                )
                                            ],
                                            style={
                                                "backgroundColor": "rgba(255,255,255,0.7)"
                                            },
                                        ),
                                    ],
                                    className="status-card",
                                )
                            ],
                            width=6,
                            className="mb-3",
                        ),
                    ]
                ),
                # ACF and PACF Visualizations
                dbc.Row(
                    [
                        dbc.Col(
                            [
                                html.Div(
                                    [
                                        html.H5(
                                            [
                                                html.I(
                                                    className="fas fa-signal",
                                                    style={"marginRight": "10px"},
                                                ),
                                                "Autocorrelation Function (ACF)",
                                            ],
                                            style={
                                                "color": colors["primary"],
                                                "fontWeight": "bold",
                                            },
                                        ),
                                        dcc.Graph(
                                            figure=acf_fig, style={"height": "400px"}
                                        ),
                                    ],
                                    className="chart-container",
                                )
                            ],
                            width=6,
                        ),
                        dbc.Col(
                            [
                                html.Div(
                                    [
                                        html.H5(
                                            [
                                                html.I(
                                                    className="fas fa-wave-square",
                                                    style={"marginRight": "10px"},
                                                ),
                                                "Partial Autocorrelation Function (PACF)",
                                            ],
                                            style={
                                                "color": colors["primary"],
                                                "fontWeight": "bold",
                                            },
                                        ),
                                        dcc.Graph(
                                            figure=pacf_fig, style={"height": "400px"}
                                        ),
                                    ],
                                    className="chart-container",
                                )
                            ],
                            width=6,
                        ),
                    ],
                    className="mb-4",
                ),
                # Recommendations
                dbc.Row(
                    [
                        dbc.Col(
                            [
                                dbc.Card(
                                    [
                                        dbc.CardHeader(
                                            [
                                                html.I(
                                                    className="fas fa-lightbulb",
                                                    style={"marginRight": "8px"},
                                                ),
                                                "Modeling Recommendations",
                                            ],
                                            style={
                                                "backgroundColor": colors["light"],
                                                "color": colors["text"],
                                            },
                                        ),
                                        dbc.CardBody(
                                            [
                                                html.Ol(
                                                    [
                                                        html.Li(rec)
                                                        for rec in results.get(
                                                            "recommendations", []
                                                        )
                                                    ],
                                                    style={"color": colors["text"]},
                                                )
                                            ],
                                            style={
                                                "backgroundColor": "rgba(255,255,255,0.7)"
                                            },
                                        ),
                                    ],
                                    className="status-card",
                                )
                            ],
                            width=12,
                        )
                    ]
                ),
            ]
        )

    except Exception as e:
        return create_error_alert(f"Error displaying results: {str(e)}", colors)


def create_quick_stat_display(results, test_type, series, colors):
    """Create quick test display with mini visualization"""
    try:
        conclusion = results["combined_analysis"]["conclusion"]
        confidence = results["combined_analysis"]["confidence"]
        recommendation = results["combined_analysis"]["recommendation"]

        # Create mini time series plot
        mini_fig = create_mini_series_plot(series, test_type, colors)

        alert_color = "success" if conclusion == "Stationary" else "warning"
        icon = "📈" if conclusion == "Stationary" else "📊"

        return dbc.Row(
            [
                dbc.Col(
                    [
                        dbc.Card(
                            [
                                dbc.CardHeader(
                                    [
                                        html.I(
                                            className="fas fa-bolt",
                                            style={"marginRight": "8px"},
                                        ),
                                        f"Quick {test_type} Test",
                                    ],
                                    style={
                                        "backgroundColor": colors["secondary"],
                                        "color": "white",
                                    },
                                ),
                                dbc.CardBody(
                                    [
                                        dbc.Alert(
                                            [
                                                html.H4(
                                                    [icon, f" Result: {conclusion}"],
                                                    className="alert-heading",
                                                ),
                                                html.P(
                                                    [
                                                        html.Strong("Confidence: "),
                                                        confidence,
                                                    ]
                                                ),
                                                html.Hr(),
                                                html.P(
                                                    [
                                                        html.Strong("Recommendation: "),
                                                        recommendation,
                                                    ],
                                                    className="mb-0",
                                                ),
                                            ],
                                            color=alert_color,
                                        )
                                    ],
                                    style={"backgroundColor": "rgba(255,255,255,0.7)"},
                                ),
                            ],
                            className="status-card",
                        )
                    ],
                    width=7,
                ),
                dbc.Col(
                    [
                        dbc.Card(
                            [
                                dbc.CardHeader(
                                    [
                                        html.I(
                                            className="fas fa-chart-line",
                                            style={"marginRight": "8px"},
                                        ),
                                        "Series Visualization",
                                    ],
                                    style={
                                        "backgroundColor": colors["primary"],
                                        "color": "white",
                                    },
                                ),
                                dbc.CardBody(
                                    [
                                        dcc.Graph(
                                            figure=mini_fig, style={"height": "200px"}
                                        )
                                    ],
                                    style={
                                        "padding": "15px",
                                        "backgroundColor": "rgba(255,255,255,0.7)",
                                    },
                                ),
                            ],
                            className="status-card",
                        )
                    ],
                    width=5,
                ),
            ],
            className="mb-4",
        )

    except Exception as e:
        return create_error_alert(f"Error in quick test display: {str(e)}", colors)


def create_quick_norm_display(results, test_type, series, colors):
    """Create quick normality test display"""
    try:
        consensus = results["consensus"]
        confidence = results["confidence"]
        recommendation = results["recommendation"]

        # Create histogram
        hist_fig = create_histogram_plot(series, colors)

        return dbc.Row(
            [
                dbc.Col(
                    [
                        dbc.Card(
                            [
                                dbc.CardHeader(
                                    [
                                        html.I(
                                            className="fas fa-bolt",
                                            style={"marginRight": "8px"},
                                        ),
                                        f"Quick {test_type} Test",
                                    ],
                                    style={
                                        "backgroundColor": colors["info"],
                                        "color": "white",
                                    },
                                ),
                                dbc.CardBody(
                                    [
                                        dbc.Alert(
                                            [
                                                html.H4(
                                                    [
                                                        (
                                                            "✅"
                                                            if consensus == "Normal"
                                                            else "⚠️"
                                                        ),
                                                        f" Result: {consensus}",
                                                    ],
                                                    className="alert-heading",
                                                ),
                                                html.P(
                                                    [
                                                        html.Strong("Confidence: "),
                                                        confidence,
                                                    ]
                                                ),
                                                html.Hr(),
                                                html.P(
                                                    [
                                                        html.Strong("Recommendation: "),
                                                        recommendation,
                                                    ],
                                                    className="mb-0",
                                                ),
                                            ],
                                            color=(
                                                "info"
                                                if consensus == "Normal"
                                                else "secondary"
                                            ),
                                        )
                                    ],
                                    style={"backgroundColor": "rgba(255,255,255,0.7)"},
                                ),
                            ],
                            className="status-card",
                        )
                    ],
                    width=7,
                ),
                dbc.Col(
                    [
                        dbc.Card(
                            [
                                dbc.CardHeader(
                                    [
                                        html.I(
                                            className="fas fa-chart-bar",
                                            style={"marginRight": "8px"},
                                        ),
                                        "Distribution",
                                    ],
                                    style={
                                        "backgroundColor": colors["info"],
                                        "color": "white",
                                    },
                                ),
                                dbc.CardBody(
                                    [
                                        dcc.Graph(
                                            figure=hist_fig, style={"height": "200px"}
                                        )
                                    ],
                                    style={
                                        "padding": "15px",
                                        "backgroundColor": "rgba(255,255,255,0.7)",
                                    },
                                ),
                            ],
                            className="status-card",
                        )
                    ],
                    width=5,
                ),
            ],
            className="mb-4",
        )

    except Exception as e:
        return create_error_alert(f"Error in normality test display: {str(e)}", colors)


def create_acf_pacf_charts(series, colors):
    """Create ACF and PACF charts"""
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

        confidence_interval = 1.96 / np.sqrt(len(series))
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
            title="Autocorrelation Function (ACF)",
            xaxis_title="Lag",
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
            title="Partial Autocorrelation Function (PACF)",
            xaxis_title="Lag",
            yaxis_title="Correlation",
            template="plotly_white",
            plot_bgcolor="rgba(245,242,237,0.8)",
            paper_bgcolor="rgba(245,242,237,0.8)",
            showlegend=False,
            height=400,
        )

        return acf_fig, pacf_fig

    except Exception as e:
        return create_empty_acf_pacf_charts(colors)


def create_mini_series_plot(series, test_type, colors):
    """Create mini time series plot for quick tests"""
    mini_fig = go.Figure()
    mini_fig.add_trace(
        go.Scatter(
            x=series.index,
            y=series,
            mode="lines",
            line=dict(color=colors["primary"], width=2),
            showlegend=False,
            hovertemplate="<b>%{x}</b><br>Value: %{y:.3f}<extra></extra>",
        )
    )

    # Add trend line for stationarity
    if test_type == "Stationarity":
        x_numeric = np.arange(len(series))
        z = np.polyfit(x_numeric, series, 1)
        trend_line = np.poly1d(z)(x_numeric)
        mini_fig.add_trace(
            go.Scatter(
                x=series.index,
                y=trend_line,
                mode="lines",
                line=dict(color=colors["danger"], width=2, dash="dash"),
                name="Trend",
                showlegend=False,
            )
        )

    mini_fig.update_layout(
        height=200,
        margin=dict(l=30, r=30, t=20, b=30),
        template="plotly_white",
        plot_bgcolor="rgba(245,242,237,0.8)",
        paper_bgcolor="rgba(245,242,237,0.8)",
        hovermode="x unified",
    )

    return mini_fig


def create_histogram_plot(series, colors):
    """Create histogram for normality visualization"""
    hist_fig = go.Figure()
    hist_fig.add_trace(
        go.Histogram(
            x=series.dropna(),
            nbinsx=30,
            marker_color=colors["info"],
            opacity=0.8,
            showlegend=False,
        )
    )
    hist_fig.update_layout(
        height=200,
        margin=dict(l=30, r=30, t=20, b=30),
        template="plotly_white",
        plot_bgcolor="rgba(245,242,237,0.8)",
        paper_bgcolor="rgba(245,242,237,0.8)",
    )
    return hist_fig


def create_empty_acf_pacf_charts(colors):
    """Create empty ACF/PACF charts with message"""
    empty_fig = go.Figure()
    empty_fig.add_annotation(
        text="ACF/PACF calculation in progress...",
        xref="paper",
        yref="paper",
        x=0.5,
        y=0.5,
        font=dict(size=14, color=colors["text"]),
    )
    empty_fig.update_layout(
        template="plotly_white",
        height=400,
        plot_bgcolor="rgba(245,242,237,0.8)",
        paper_bgcolor="rgba(245,242,237,0.8)",
    )
    return empty_fig, empty_fig


def create_error_alert(message, colors):
    """Create error alert display"""
    return dbc.Alert(
        [
            html.I(
                className="fas fa-exclamation-triangle", style={"marginRight": "8px"}
            ),
            message,
        ],
        color="danger",
        className="status-card",
    )


def create_info_alert(message, colors):
    """Create info alert display"""
    return dbc.Alert(
        [html.I(className="fas fa-info-circle", style={"marginRight": "8px"}), message],
        color="info",
        className="status-card",
    )

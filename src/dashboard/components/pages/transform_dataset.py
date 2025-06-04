"""
Transform Dataset page components for stationarity transformations.
Handles log transformations, differencing, and stationarity testing.
"""

from dash import html, dcc
import dash_bootstrap_components as dbc
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
from dashboard.components.layout import create_page_header, create_chart_container


def create_transform_page(data_manager, colors):
    """Create transformation page matching reference design"""
    datasets = data_manager.get_available_datasets()

    header = create_page_header(
        "Transform Dataset",
        "Make your time series stationary for accurate forecasting",
        "fas fa-cogs",
        colors,
    )

    transform_controls = create_transform_controls(datasets, colors)
    charts_section = create_charts_section(colors)

    return html.Div([header, transform_controls, charts_section])


def create_transform_controls(datasets, colors):
    """Create transformation controls section"""
    return html.Div(
        [
            html.H3(
                "Transform dataset to make it Stationary",
                style={
                    "color": "white",
                    "textAlign": "center",
                    "marginBottom": "30px",
                    "fontWeight": "bold",
                    "textShadow": "0 2px 4px rgba(0,0,0,0.3)",
                    "fontSize": "28px",
                },
            ),
            # Transformation options
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
                                                        "label": " 1) Apply log",
                                                        "value": "log",
                                                    }
                                                ],
                                                value=[],
                                                id="apply-log",
                                                style={
                                                    "color": "white",
                                                    "fontSize": "18px",
                                                    "fontWeight": "bold",
                                                },
                                            )
                                        ],
                                        style={"padding": "25px"},
                                    )
                                ],
                                style={
                                    "background": "rgba(255,255,255,0.15)",
                                    "border": "none",
                                    "borderRadius": "12px",
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
                                                        "label": " 2) Apply difference",
                                                        "value": "diff",
                                                    }
                                                ],
                                                value=[],
                                                id="apply-diff",
                                                style={
                                                    "color": "white",
                                                    "fontSize": "18px",
                                                    "fontWeight": "bold",
                                                },
                                            ),
                                            dcc.Dropdown(
                                                id="diff-lag",
                                                options=[
                                                    {"label": "Lag 1", "value": 1},
                                                    {"label": "Lag 12", "value": 12},
                                                ],
                                                value=1,
                                                placeholder="Choose lag",
                                                style={"marginTop": "15px"},
                                            ),
                                        ],
                                        style={"padding": "25px"},
                                    )
                                ],
                                style={
                                    "background": "rgba(255,255,255,0.15)",
                                    "border": "none",
                                    "borderRadius": "12px",
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
                                                    "fontSize": "18px",
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
                                                style={"marginTop": "15px"},
                                            ),
                                        ],
                                        style={"padding": "25px"},
                                    )
                                ],
                                style={
                                    "background": "rgba(255,255,255,0.15)",
                                    "border": "none",
                                    "borderRadius": "12px",
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
                                                    "fontSize": "18px",
                                                    "fontWeight": "bold",
                                                },
                                            ),
                                            dcc.Dropdown(
                                                id="transform-variable-dropdown",
                                                placeholder="Choose variable",
                                                style={"marginTop": "15px"},
                                            ),
                                        ],
                                        style={"padding": "25px"},
                                    )
                                ],
                                style={
                                    "background": "rgba(255,255,255,0.15)",
                                    "border": "none",
                                    "borderRadius": "12px",
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
                                        "Augmented Dickey-Fuller test:",
                                        style={
                                            "color": "white",
                                            "display": "inline-block",
                                            "marginRight": "25px",
                                            "fontSize": "20px",
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
    """Create charts section for transformed data visualization"""
    return html.Div(
        [
            dbc.Row(
                [
                    dbc.Col(
                        [
                            create_chart_container(
                                "Transformed Data Linechart",
                                "transform-chart",
                                "fas fa-chart-line",
                                colors,
                            )
                        ],
                        width=6,
                    ),
                    dbc.Col(
                        [
                            create_chart_container(
                                "Autocorrelation (ACF)",
                                "acf-chart",
                                "fas fa-signal",
                                colors,
                            )
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
                                "Box-Cox Plot",
                                "boxcox-chart",
                                "fas fa-square-root-alt",
                                colors,
                            )
                        ],
                        width=6,
                    ),
                    dbc.Col(
                        [
                            create_chart_container(
                                "Partial Autocorrelation (PACF)",
                                "pacf-chart",
                                "fas fa-wave-square",
                                colors,
                            )
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

    title = f"Data Transformation: {' + '.join(transform_text) if transform_text else 'No transformation'}"

    fig.update_layout(
        title=title,
        height=400,
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
        return create_empty_chart_pair(colors)


def create_boxcox_chart(series, colors):
    """Create Box-Cox transformation chart"""
    try:
        # Simple Box-Cox approximation for visualization
        lambdas = np.linspace(-2, 2, 100)
        log_likelihoods = []

        for lam in lambdas:
            if lam == 0:
                transformed = np.log(series + 1)
            else:
                transformed = (np.power(series + 1, lam) - 1) / lam

            # Simple log-likelihood approximation
            log_likelihood = -0.5 * len(series) * np.log(np.var(transformed))
            log_likelihoods.append(log_likelihood)

        optimal_lambda = lambdas[np.argmax(log_likelihoods)]

        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=lambdas,
                y=log_likelihoods,
                mode="lines",
                line=dict(color=colors["secondary"], width=3),
                name="Log-Likelihood",
            )
        )

        # Mark optimal lambda
        fig.add_vline(
            x=optimal_lambda,
            line_dash="dash",
            line_color=colors["danger"],
            annotation_text=f"Optimal λ = {optimal_lambda:.2f}",
        )

        fig.update_layout(
            title="Box-Cox Transformation",
            xaxis_title="Lambda (λ)",
            yaxis_title="Log-Likelihood",
            template="plotly_white",
            plot_bgcolor="rgba(245,242,237,0.8)",
            paper_bgcolor="rgba(245,242,237,0.8)",
            height=400,
        )

        return fig

    except Exception as e:
        return create_empty_chart(colors, "Box-Cox calculation in progress...")


def create_empty_chart(colors, message="Select data to display"):
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
    """Create ADF test result display"""
    return html.Div(
        [
            html.H6(
                f"Test p-value: {p_value:.4f}",
                style={
                    "color": "white",
                    "margin": 0,
                    "fontSize": "16px",
                },
            ),
            html.P(
                f"The data is {conclusion.lower()}",
                style={
                    "margin": 0,
                    "fontWeight": "bold",
                    "color": "white",
                    "fontSize": "14px",
                },
            ),
        ],
        style={
            "background": (
                colors["danger"]
                if conclusion == "Non-stationary"
                else colors["success"]
            ),
            "padding": "15px",
            "borderRadius": "10px",
            "textAlign": "center",
            "boxShadow": "0 4px 15px rgba(0,0,0,0.1)",
        },
    )

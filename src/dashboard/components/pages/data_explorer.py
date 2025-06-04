"""
Data Explorer page components for interactive analysis.
Provides dataset selection, visualization, and summary statistics.
"""

from dash import html, dcc, dash_table
import dash_bootstrap_components as dbc
import plotly.graph_objects as go
import numpy as np
from dashboard.components.layout import (
    create_page_header,
    create_control_card,
    create_chart_container,
)


def create_explorer_page(data_manager, colors):
    """Create data explorer page with controls and visualization"""
    datasets = data_manager.get_available_datasets()

    header = create_page_header(
        "Data Explorer",
        "Interactive time series analysis and visualization",
        "fas fa-search",
        colors,
    )

    controls = create_explorer_controls(datasets, colors)
    content = create_explorer_content(colors)

    return html.Div([header, controls, content])


def create_explorer_controls(datasets, colors):
    """Create explorer control panel"""
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
                            id="dataset-dropdown",
                            options=[{"label": ds, "value": ds} for ds in datasets],
                            value=datasets[0] if datasets else None,
                            style={"marginBottom": "10px"},
                        ),
                    ],
                    width=4,
                ),
                dbc.Col(
                    [
                        dbc.Label(
                            "Variables:",
                            style={"fontWeight": "bold", "color": colors["text"]},
                        ),
                        dcc.Dropdown(
                            id="variable-dropdown",
                            multi=True,
                            placeholder="Select variables...",
                            style={"marginBottom": "10px"},
                        ),
                    ],
                    width=4,
                ),
                dbc.Col(
                    [
                        dbc.Label(
                            "Chart Type:",
                            style={"fontWeight": "bold", "color": colors["text"]},
                        ),
                        dcc.Dropdown(
                            id="chart-type-dropdown",
                            options=[
                                {"label": "Line Chart", "value": "line"},
                                {"label": "Area Chart", "value": "area"},
                            ],
                            value="line",
                            style={"marginBottom": "10px"},
                        ),
                    ],
                    width=4,
                ),
            ]
        )
    ]

    return create_control_card(
        "Explorer Controls", "fas fa-sliders-h", controls_content, colors
    )


def create_explorer_content(colors):
    """Create explorer content area with chart and summary"""
    return html.Div(
        [
            dbc.Row(
                [
                    dbc.Col(
                        [
                            create_chart_container(
                                "Time Series Visualization",
                                "explorer-chart",
                                "fas fa-chart-line",
                                colors,
                            )
                        ],
                        width=12,
                    )
                ],
                className="mb-4",
            ),
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
                                            "Statistical Summary",
                                        ],
                                        style={"color": colors["primary"]},
                                    ),
                                    html.Div(id="data-summary"),
                                ],
                                className="chart-container",
                            )
                        ],
                        width=12,
                    )
                ]
            ),
        ]
    )


def create_time_series_chart(df, variables, chart_type, colors):
    """Create time series chart based on selected variables and type"""
    fig = go.Figure()

    chart_colors = [
        colors["primary"],
        colors["secondary"],
        colors["success"],
        colors["danger"],
        colors["warning"],
        colors["info"],
    ]

    for i, var in enumerate(variables):
        if var in df.columns:
            color = chart_colors[i % len(chart_colors)]

            if chart_type == "line":
                fig.add_trace(
                    go.Scatter(
                        x=df.index,
                        y=df[var],
                        name=var,
                        mode="lines",
                        line=dict(color=color, width=3),
                        hovertemplate=f"<b>%{{x}}</b><br>{var}: %{{y}}<extra></extra>",
                    )
                )
            elif chart_type == "area":
                fig.add_trace(
                    go.Scatter(
                        x=df.index,
                        y=df[var],
                        name=var,
                        fill="tonexty" if i > 0 else "tozeroy",
                        line=dict(color=color),
                        hovertemplate=f"<b>%{{x}}</b><br>{var}: %{{y}}<extra></extra>",
                    )
                )

    fig.update_layout(
        title="Time Series Analysis",
        xaxis_title="Date",
        yaxis_title="Value",
        hovermode="x unified",
        height=400,
        template="plotly_white",
        plot_bgcolor="rgba(245,242,237,0.8)",
        paper_bgcolor="rgba(245,242,237,0.8)",
    )

    fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor="rgba(74,102,112,0.2)")
    fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor="rgba(74,102,112,0.2)")

    return fig


def create_summary_table(df, variables, colors):
    """Create summary statistics table"""
    try:
        summary_stats = df[variables].describe().round(3)

        table = dash_table.DataTable(
            data=summary_stats.reset_index().to_dict("records"),
            columns=[{"name": i, "id": i} for i in summary_stats.reset_index().columns],
            style_cell={
                "textAlign": "left",
                "padding": "12px",
                "fontFamily": "Arial, sans-serif",
                "backgroundColor": "rgba(255,255,255,0.8)",
                "color": colors["text"],
            },
            style_header={
                "backgroundColor": colors["secondary"],
                "fontWeight": "bold",
                "color": "white",
            },
            style_data={
                "backgroundColor": "rgba(255,255,255,0.9)",
                "color": colors["text"],
            },
            style_table={"overflowX": "auto"},
        )

        return table

    except Exception as e:
        return html.Div(f"Error creating summary: {str(e)}")


def create_empty_chart(colors, message="Select dataset and variables to display"):
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
    )
    return fig


def create_error_chart(colors, error_message):
    """Create error chart display"""
    fig = go.Figure()
    fig.add_annotation(
        text=f"Error: {error_message}",
        xref="paper",
        yref="paper",
        x=0.5,
        y=0.5,
        font=dict(size=14, color=colors["danger"]),
    )
    fig.update_layout(
        template="plotly_white",
        plot_bgcolor="rgba(245,242,237,0.8)",
        paper_bgcolor="rgba(245,242,237,0.8)",
    )
    return fig

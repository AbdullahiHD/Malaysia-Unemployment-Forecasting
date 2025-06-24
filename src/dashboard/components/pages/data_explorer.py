"""
Data Explorer page components for interactive analysis.
Provides dataset selection, visualization, and summary statistics.
UPDATED: Enhanced with professional column mapping for better usability.
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

    controls = create_explorer_controls(datasets, data_manager, colors)
    content = create_explorer_content(colors)

    return html.Div([header, controls, content])


def create_explorer_controls(datasets, data_manager, colors):
    """Create explorer control panel with enhanced variable dropdown"""
    # Get initial variable options for first dataset
    initial_variables = []
    if datasets:
        try:
            grouped_variables = data_manager.get_column_display_options(
                datasets[0], group_by_category=True
            )
            # Flatten the grouped structure for the dropdown
            initial_variables = flatten_grouped_options(grouped_variables)
        except:
            initial_variables = []

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
                            options=initial_variables,
                            multi=True,
                            placeholder="Select unemployment metrics...",
                            style={"marginBottom": "10px", "minWidth": "300px"},
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


def flatten_grouped_options(grouped_options):
    if not grouped_options:
        return []

    # Check if options are already flat (not grouped)
    if all(
        isinstance(opt, dict) and "value" in opt and "options" not in opt
        for opt in grouped_options
    ):
        return grouped_options

    flattened = []

    for group in grouped_options:
        if isinstance(group, dict) and "options" in group:
            # Add section header as disabled option
            flattened.append(
                {
                    "label": f"── {group['label']} ──",
                    "value": f"header_{group['label'].lower().replace(' ', '_')}",
                    "disabled": True,
                }
            )

            # Add the actual options
            for option in group["options"]:
                flattened.append(option)
        else:
            # If it's already a flat option, add it directly
            flattened.append(group)

    return flattened


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

def create_time_series_chart(df, variables, chart_type, colors, data_manager=None):
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

            # Get professional display name for legend
            if data_manager and hasattr(data_manager, "get_column_display_name"):
                display_name = data_manager.get_column_display_name(var)
            else:
                display_name = var

            if chart_type == "line":
                fig.add_trace(
                    go.Scatter(
                        x=df.index,
                        y=df[var],
                        name=display_name,
                        mode="lines",
                        line=dict(color=color, width=3),
                        hovertemplate=f"<b>%{{x}}</b><br>{display_name}: %{{y}}<extra></extra>",
                    )
                )
            elif chart_type == "area":
                fig.add_trace(
                    go.Scatter(
                        x=df.index,
                        y=df[var],
                        name=display_name,
                        fill="tonexty" if i > 0 else "tozeroy",
                        line=dict(color=color),
                        hovertemplate=f"<b>%{{x}}</b><br>{display_name}: %{{y}}<extra></extra>",
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
        # Legend positioning from market overview
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.08,
            xanchor="right",
            x=1,
            bgcolor="rgba(255,255,255,0.8)",
            bordercolor="rgba(0,0,0,0.1)",
            borderwidth=1,
        ),
    )

    fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor="rgba(74,102,112,0.2)")
    fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor="rgba(74,102,112,0.2)")

    return fig


def create_summary_table(df, variables, colors, data_manager=None):
    """Create summary statistics table with professional column names and equal-width columns"""
    try:
        summary_stats = df[variables].describe().round(3)

        # Create enhanced table data with professional column names
        table_data = []
        display_names = []

        # Get all display names first
        for var in variables:
            if data_manager and hasattr(data_manager, "get_column_display_name"):
                display_name = data_manager.get_column_display_name(var)
            else:
                display_name = var
            display_names.append(display_name)

        # Build table data
        for index, row in summary_stats.iterrows():
            row_data = {"Statistic": index}
            for i, var in enumerate(variables):
                row_data[display_names[i]] = row[var]
            table_data.append(row_data)

        # Calculate equal column width
        num_columns = len(variables) + 1  # +1 for Statistic column
        column_width = f"{100/num_columns:.1f}%"

        # Create column definitions with equal widths
        columns = [{"name": "Statistic", "id": "Statistic", "type": "text"}]

        for display_name in display_names:
            columns.append(
                {
                    "name": display_name,
                    "id": display_name,
                    "type": "numeric",
                    "format": {"specifier": ".3f"},
                }
            )

        # Professional table styling
        table_style = {
            "backgroundColor": "rgba(255,255,255,0.9)",
            "border": "1px solid rgba(0,0,0,0.1)",
            "borderRadius": "8px",
            "overflow": "hidden",
        }

        header_style = {
            "backgroundColor": colors["primary"],
            "color": "white",
            "fontWeight": "bold",
            "textAlign": "center",
            "padding": "12px",
            "border": "none",
        }

        cell_style = {
            "textAlign": "center",
            "padding": "10px",
            "borderBottom": "1px solid rgba(0,0,0,0.1)",
            "fontFamily": "Arial, sans-serif",
            "width": column_width,  # Equal width for all columns
            "minWidth": column_width,
            "maxWidth": column_width,
        }

        table = dash_table.DataTable(
            data=table_data,
            columns=columns,
            style_table=table_style,
            style_header=header_style,
            style_cell=cell_style,
            style_data={
                "backgroundColor": "rgba(248,249,250,0.8)",
                "color": colors["text"],
            },
            style_data_conditional=[
                {
                    "if": {"column_id": "Statistic"},
                    "backgroundColor": "rgba(233,236,239,0.8)",
                    "fontWeight": "bold",
                    "textAlign": "left",
                    "paddingLeft": "15px",
                }
            ],
        )

        return html.Div(
            [
                html.H6(
                    "Statistical Summary",
                    style={
                        "color": colors["primary"],
                        "marginBottom": "15px",
                        "fontWeight": "bold",
                    },
                ),
                table,
            ],
            style={"marginTop": "20px"},
        )

    except Exception as e:
        return html.Div(
            f"Error creating summary: {str(e)}",
            style={"color": colors["danger"], "textAlign": "center", "padding": "20px"},
        )


def create_variable_info_card(selected_variables, data_manager, colors):
    """Create an information card showing details about selected variables"""
    if (
        not selected_variables
        or not data_manager
        or not hasattr(data_manager, "get_column_display_name")
    ):
        return html.Div()

    variable_info = []
    for var in selected_variables:
        display_name = data_manager.get_column_display_name(var)
        variable_info.append(
            dbc.ListGroupItem(
                [
                    html.Strong(display_name),
                    html.Br(),
                    html.Small(f"Technical name: {var}", className="text-muted"),
                ]
            )
        )

    if variable_info:
        return dbc.Card(
            [
                dbc.CardHeader(
                    [html.I(className="fas fa-info-circle me-2"), "Selected Variables"]
                ),
                dbc.CardBody([dbc.ListGroup(variable_info, flush=True)]),
            ],
            className="mb-3",
        )

    return html.Div()


def create_enhanced_explorer_content(colors):
    """Create enhanced explorer content area with variable info"""
    return html.Div(
        [
            # Variable information card
            html.Div(id="variable-info-card"),
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
        height=400,
        showlegend=False,
    )
    fig.update_xaxes(showgrid=False, showticklabels=False)
    fig.update_yaxes(showgrid=False, showticklabels=False)
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
        height=400,
        showlegend=False,
    )
    fig.update_xaxes(showgrid=False, showticklabels=False)
    fig.update_yaxes(showgrid=False, showticklabels=False)
    return fig

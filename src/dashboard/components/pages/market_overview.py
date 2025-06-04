# """
# Market Overview page components for Malaysia Unemployment Dashboard.
# Displays key metrics and main unemployment trends.
# """

# from dash import html, dcc
# import dash_bootstrap_components as dbc
# import plotly.graph_objects as go
# from plotly.subplots import make_subplots
# from dashboard.components.layout import create_page_header, create_chart_container


# def create_overview_page(data_manager, colors):
#     """Create market overview page with key metrics and charts"""
#     try:
#         # Get latest metrics
#         metrics = data_manager.get_latest_metrics()

#         # Get datasets for charts
#         df = data_manager.get_dataset("Overall Unemployment")
#         youth_df = data_manager.get_dataset("Youth Unemployment")

#         header = create_page_header(
#             "Market Overview",
#             "Real-time insights into Malaysian labor market performance",
#             "fas fa-chart-line",
#             colors,
#         )

#         metrics_row = create_metrics_cards(metrics, colors)
#         chart_section = create_main_chart_section(df, youth_df, data_manager, colors)

#         return html.Div([header, metrics_row, chart_section])

#     except Exception as e:
#         return create_error_display(f"Error creating overview: {str(e)}", colors)


# def create_metrics_cards(metrics, colors):
#     """Create key metrics cards"""
#     metric_configs = [
#         {
#             "value": f"{metrics.get('unemployment_rate', 3.4):.1f}%",
#             "label": "UNEMPLOYMENT RATE",
#             "icon": "fas fa-chart-line",
#             "color": colors["danger"],
#         },
#         {
#             "value": f"{metrics.get('labor_force', 15200):,.0f}K",
#             "label": "LABOR FORCE",
#             "icon": "fas fa-users",
#             "color": colors["info"],
#         },
#         {
#             "value": f"{metrics.get('participation_rate', 67.8):.1f}%",
#             "label": "PARTICIPATION RATE",
#             "icon": "fas fa-briefcase",
#             "color": colors["success"],
#         },
#         {
#             "value": f"{metrics.get('youth_unemployment', 11.5):.1f}%",
#             "label": "YOUTH UNEMPLOYMENT",
#             "icon": "fas fa-graduation-cap",
#             "color": colors["warning"],
#         },
#     ]

#     cards = []
#     for config in metric_configs:
#         card = dbc.Col(
#             [
#                 dbc.Card(
#                     [
#                         dbc.CardBody(
#                             [
#                                 html.Div(
#                                     [
#                                         html.I(
#                                             className=config["icon"],
#                                             style={
#                                                 "fontSize": "28px",
#                                                 "color": config["color"],
#                                                 "float": "right",
#                                             },
#                                         ),
#                                         html.H3(
#                                             config["value"],
#                                             className="mb-1",
#                                             style={
#                                                 "color": config["color"],
#                                                 "fontWeight": "bold",
#                                             },
#                                         ),
#                                         html.P(
#                                             config["label"],
#                                             className="text-muted mb-0",
#                                             style={
#                                                 "fontSize": "12px",
#                                                 "fontWeight": "bold",
#                                                 "color": colors["text"],
#                                             },
#                                         ),
#                                     ]
#                                 )
#                             ]
#                         )
#                     ],
#                     className="metric-card",
#                 )
#             ],
#             width=3,
#         )
#         cards.append(card)

#     return dbc.Row(cards, className="mb-4")


# def create_main_chart_section(df, youth_df, data_manager, colors):
#     """Create main chart with unemployment trends"""
#     fig = create_enhanced_overview_chart(df, youth_df, data_manager, colors)

#     return html.Div(
#         [
#             html.Div(
#                 [
#                     html.H5(
#                         [
#                             html.I(
#                                 className="fas fa-chart-area",
#                                 style={"marginRight": "10px"},
#                             ),
#                             "Unemployment Rate Trends (2010-2025)",
#                         ],
#                         style={"color": colors["primary"], "marginBottom": "5px"},
#                     ),
#                     dbc.ButtonGroup(
#                         [
#                             dbc.Button(
#                                 "1Y", size="sm", outline=True, color="secondary"
#                             ),
#                             dbc.Button(
#                                 "3Y", size="sm", outline=True, color="secondary"
#                             ),
#                             dbc.Button(
#                                 "5Y", size="sm", outline=True, color="secondary"
#                             ),
#                             dbc.Button("All", size="sm", color="primary"),
#                         ],
#                         size="sm",
#                         style={"float": "right"},
#                     ),
#                 ],
#                 style={
#                     "display": "flex",
#                     "justifyContent": "space-between",
#                     "alignItems": "center",
#                     "marginBottom": "20px",
#                 },
#             ),
#             dcc.Graph(figure=fig, style={"height": "500px"}),
#         ],
#         className="chart-container",
#     )


# def create_enhanced_overview_chart(df, youth_df, data_manager, colors):
#     """Create enhanced overview chart with multiple series"""
#     fig = make_subplots(
#         rows=2,
#         cols=1,
#         subplot_titles=("Unemployment Rate Trends", "Labor Force Dynamics"),
#         vertical_spacing=0.12,
#         row_heights=[0.6, 0.4],
#     )

#     # Main unemployment rate
#     fig.add_trace(
#         go.Scatter(
#             x=df.index,
#             y=df["u_rate"],
#             name="Unemployment Rate",
#             line=dict(color=colors["danger"], width=3),
#             hovertemplate="<b>%{x}</b><br>Rate: %{y:.1f}%<extra></extra>",
#         ),
#         row=1,
#         col=1,
#     )

#     # Youth unemployment (reindex to match)
#     youth_reindexed = youth_df["u_rate_15_24"].reindex(df.index)
#     fig.add_trace(
#         go.Scatter(
#             x=df.index,
#             y=youth_reindexed,
#             name="Youth Rate (15-24)",
#             line=dict(color=colors["warning"], width=2, dash="dot"),
#             hovertemplate="<b>%{x}</b><br>Youth Rate: %{y:.1f}%<extra></extra>",
#         ),
#         row=1,
#         col=1,
#     )

#     # Seasonally adjusted (if available)
#     try:
#         sa_df = data_manager.get_dataset("Seasonally Adjusted")
#         fig.add_trace(
#             go.Scatter(
#                 x=sa_df.index,
#                 y=sa_df["u_rate"],
#                 name="Seasonally Adjusted",
#                 line=dict(color=colors["info"], width=2, dash="dash"),
#                 hovertemplate="<b>%{x}</b><br>SA Rate: %{y:.1f}%<extra></extra>",
#             ),
#             row=1,
#             col=1,
#         )
#     except:
#         pass

#     # Labor force trends
#     fig.add_trace(
#         go.Scatter(
#             x=df.index,
#             y=df["lf"],
#             name="Total Labor Force",
#             line=dict(color=colors["primary"], width=2),
#             hovertemplate="<b>%{x}</b><br>Labor Force: %{y:,.0f}K<extra></extra>",
#         ),
#         row=2,
#         col=1,
#     )

#     fig.add_trace(
#         go.Scatter(
#             x=df.index,
#             y=df["lf_employed"],
#             name="Employed",
#             line=dict(color=colors["success"], width=2),
#             hovertemplate="<b>%{x}</b><br>Employed: %{y:,.0f}K<extra></extra>",
#         ),
#         row=2,
#         col=1,
#     )

#     fig.update_layout(
#         height=500,
#         showlegend=True,
#         hovermode="x unified",
#         template="plotly_white",
#         plot_bgcolor="rgba(245,242,237,0.8)",
#         paper_bgcolor="rgba(245,242,237,0.8)",
#         legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
#     )

#     fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor="rgba(74,102,112,0.2)")
#     fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor="rgba(74,102,112,0.2)")

#     return fig


# def create_error_display(error_message, colors):
#     """Create error display for overview page"""
#     return dbc.Alert(
#         [
#             html.I(
#                 className="fas fa-exclamation-triangle", style={"marginRight": "8px"}
#             ),
#             error_message,
#         ],
#         color="danger",
#         className="status-card",
#     )


"""
Market Overview page components for Malaysia Unemployment Dashboard.
Displays key metrics and main unemployment trends.
FIXED: Removed problematic dcc.Store and integrated with proper callback system.
"""

from dash import html, dcc
import dash_bootstrap_components as dbc
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import pandas as pd
from dashboard.components.layout import create_page_header, create_chart_container


def create_overview_page(data_manager, colors):
    """Create market overview page with key metrics and charts"""
    try:
        # Get latest metrics
        metrics = data_manager.get_latest_metrics()

        # Get datasets for charts
        df = data_manager.get_dataset("Overall Unemployment")
        youth_df = data_manager.get_dataset("Youth Unemployment")

        header = create_page_header(
            "Market Overview",
            "Real-time insights into Malaysian labor market performance",
            "fas fa-chart-line",
            colors,
        )

        metrics_row = create_metrics_cards(metrics, colors)
        chart_section = create_main_chart_section(df, youth_df, data_manager, colors)

        return html.Div([header, metrics_row, chart_section])

    except Exception as e:
        return create_error_display(f"Error creating overview: {str(e)}", colors)


def create_metrics_cards(metrics, colors):
    """Create key metrics cards"""
    metric_configs = [
        {
            "value": f"{metrics.get('unemployment_rate', 3.4):.1f}%",
            "label": "UNEMPLOYMENT RATE",
            "icon": "fas fa-chart-line",
            "color": colors["danger"],
        },
        {
            "value": f"{metrics.get('labor_force', 15200):,.0f}K",
            "label": "LABOR FORCE",
            "icon": "fas fa-users",
            "color": colors["info"],
        },
        {
            "value": f"{metrics.get('participation_rate', 67.8):.1f}%",
            "label": "PARTICIPATION RATE",
            "icon": "fas fa-briefcase",
            "color": colors["success"],
        },
        {
            "value": f"{metrics.get('youth_unemployment', 11.5):.1f}%",
            "label": "YOUTH UNEMPLOYMENT",
            "icon": "fas fa-graduation-cap",
            "color": colors["warning"],
        },
    ]

    cards = []
    for config in metric_configs:
        card = dbc.Col(
            [
                dbc.Card(
                    [
                        dbc.CardBody(
                            [
                                html.Div(
                                    [
                                        html.I(
                                            className=config["icon"],
                                            style={
                                                "fontSize": "28px",
                                                "color": config["color"],
                                                "float": "right",
                                            },
                                        ),
                                        html.H3(
                                            config["value"],
                                            className="mb-1",
                                            style={
                                                "color": config["color"],
                                                "fontWeight": "bold",
                                            },
                                        ),
                                        html.P(
                                            config["label"],
                                            className="text-muted mb-0",
                                            style={
                                                "fontSize": "12px",
                                                "fontWeight": "bold",
                                                "color": colors["text"],
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
        )
        cards.append(card)

    return dbc.Row(cards, className="mb-4")


def create_main_chart_section(df, youth_df, data_manager, colors):
    """Create main chart with unemployment trends - FIXED: No dcc.Store"""

    return html.Div(
        [
            html.Div(
                [
                    html.H5(
                        [
                            html.I(
                                className="fas fa-chart-area",
                                style={"marginRight": "10px"},
                            ),
                            "Unemployment Rate Trends (2010-2025)",
                        ],
                        style={"color": colors["primary"], "marginBottom": "5px"},
                    ),
                    dbc.ButtonGroup(
                        [
                            dbc.Button(
                                "1Y",
                                id="btn-1y",
                                size="sm",
                                outline=True,
                                color="secondary",
                                n_clicks=0,
                            ),
                            dbc.Button(
                                "3Y",
                                id="btn-3y",
                                size="sm",
                                outline=True,
                                color="secondary",
                                n_clicks=0,
                            ),
                            dbc.Button(
                                "5Y",
                                id="btn-5y",
                                size="sm",
                                outline=True,
                                color="secondary",
                                n_clicks=0,
                            ),
                            dbc.Button(
                                "All",
                                id="btn-all",
                                size="sm",
                                color="primary",
                                n_clicks=0,
                            ),
                        ],
                        size="sm",
                        style={"float": "right"},
                    ),
                ],
                style={
                    "display": "flex",
                    "justifyContent": "space-between",
                    "alignItems": "center",
                    "marginBottom": "20px",
                },
            ),
            # REMOVED: dcc.Store component that was causing serialization issues
            # The data will be managed through the main callbacks.py system instead
            dcc.Graph(
                id="overview-chart",
                figure=create_enhanced_overview_chart(
                    df, youth_df, data_manager, colors, period="all"
                ),
                style={
                    "height": "550px"
                },  # Increased height to accommodate legend spacing
            ),
        ],
        className="chart-container",
    )


def create_enhanced_overview_chart(df, youth_df, data_manager, colors, period="all"):
    """Create enhanced overview chart with multiple series"""

    # Filter data based on period
    if df is not None and not df.empty:
        filtered_df = filter_data_by_period(df, period)
    else:
        filtered_df = df

    if youth_df is not None and not youth_df.empty:
        filtered_youth_df = filter_data_by_period(youth_df, period)
    else:
        filtered_youth_df = youth_df

    fig = make_subplots(
        rows=2,
        cols=1,
        subplot_titles=("Unemployment Rate Trends", "Labor Force Dynamics"),
        vertical_spacing=0.15,  # Increased spacing
        row_heights=[0.6, 0.4],
    )

    if filtered_df is not None and not filtered_df.empty:
        # Main unemployment rate
        fig.add_trace(
            go.Scatter(
                x=filtered_df.index,
                y=filtered_df["u_rate"],
                name="Unemployment Rate",
                line=dict(color=colors["danger"], width=3),
                hovertemplate="<b>%{x}</b><br>Rate: %{y:.1f}%<extra></extra>",
            ),
            row=1,
            col=1,
        )

        # Labor force trends
        fig.add_trace(
            go.Scatter(
                x=filtered_df.index,
                y=filtered_df["lf"],
                name="Total Labor Force",
                line=dict(color=colors["primary"], width=2),
                hovertemplate="<b>%{x}</b><br>Labor Force: %{y:,.0f}K<extra></extra>",
            ),
            row=2,
            col=1,
        )

        fig.add_trace(
            go.Scatter(
                x=filtered_df.index,
                y=filtered_df["lf_employed"],
                name="Employed",
                line=dict(color=colors["success"], width=2),
                hovertemplate="<b>%{x}</b><br>Employed: %{y:,.0f}K<extra></extra>",
            ),
            row=2,
            col=1,
        )

    # Youth unemployment (reindex to match)
    if (
        filtered_youth_df is not None
        and not filtered_youth_df.empty
        and filtered_df is not None
    ):
        youth_reindexed = filtered_youth_df["u_rate_15_24"].reindex(filtered_df.index)
        fig.add_trace(
            go.Scatter(
                x=filtered_df.index,
                y=youth_reindexed,
                name="Youth Rate (15-24)",
                line=dict(color=colors["warning"], width=2, dash="dot"),
                hovertemplate="<b>%{x}</b><br>Youth Rate: %{y:.1f}%<extra></extra>",
            ),
            row=1,
            col=1,
        )

    # Seasonally adjusted (if available)
    try:
        sa_df = data_manager.get_dataset("Seasonally Adjusted")
        if sa_df is not None and not sa_df.empty:
            filtered_sa_df = filter_data_by_period(sa_df, period)
            fig.add_trace(
                go.Scatter(
                    x=filtered_sa_df.index,
                    y=filtered_sa_df["u_rate"],
                    name="Seasonally Adjusted",
                    line=dict(color=colors["info"], width=2, dash="dash"),
                    hovertemplate="<b>%{x}</b><br>SA Rate: %{y:.1f}%<extra></extra>",
                ),
                row=1,
                col=1,
            )
    except:
        pass

    fig.update_layout(
        height=550,  # Increased height
        showlegend=True,
        hovermode="x unified",
        template="plotly_white",
        plot_bgcolor="rgba(245,242,237,0.8)",
        paper_bgcolor="rgba(245,242,237,0.8)",
        # Fixed legend positioning with proper spacing
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.08,  # Moved higher to create separation
            xanchor="right",
            x=1,
            bgcolor="rgba(255,255,255,0.8)",
            bordercolor="rgba(0,0,0,0.1)",
            borderwidth=1,
        ),
        # Add top margin to accommodate legend
        margin=dict(t=80, b=50, l=50, r=50),
    )

    fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor="rgba(74,102,112,0.2)")
    fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor="rgba(74,102,112,0.2)")

    return fig


def filter_data_by_period(df, period):
    """Filter dataframe based on time period"""
    if df is None or df.empty:
        return df

    current_date = datetime.now()

    if period == "1Y":
        cutoff_date = current_date - timedelta(days=365)
    elif period == "3Y":
        cutoff_date = current_date - timedelta(days=365 * 3)
    elif period == "5Y":
        cutoff_date = current_date - timedelta(days=365 * 5)
    else:  # "all"
        return df

    # Filter based on index (assuming it's datetime)
    try:
        if hasattr(df.index, "to_pydatetime"):
            # If index is datetime-like
            return df[df.index >= cutoff_date]
        else:
            # If index needs conversion or is string-based dates
            return df.tail(int(365 * int(period[0]) / 30)) if period != "all" else df
    except:
        # Fallback: return last N months of data
        months_map = {"1Y": 12, "3Y": 36, "5Y": 60}
        if period in months_map:
            return df.tail(months_map[period])
        return df


def create_error_display(error_message, colors):
    """Create error display for overview page"""
    return dbc.Alert(
        [
            html.I(
                className="fas fa-exclamation-triangle", style={"marginRight": "8px"}
            ),
            error_message,
        ],
        color="danger",
        className="status-card",
    )

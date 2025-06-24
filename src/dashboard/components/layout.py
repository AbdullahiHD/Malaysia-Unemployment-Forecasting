"""
Main layout components for Malaysia Unemployment Dashboard.
Handles sidebar, navigation, and main content area.
"""

from dash import html, dcc
import dash_bootstrap_components as dbc


def create_sidebar(colors):
    """Create professional sidebar with navigation"""
    return html.Div(
        [
            # Header
            html.Div(
                [
                    html.H4(
                        "Labor Force",
                        style={
                            "color": "white",
                            "margin": 0,
                            "fontWeight": "bold",
                            "fontSize": "24px",
                        },
                    ),
                    html.P(
                        "Malaysian Analytics Platform",
                        style={
                            "color": "rgba(255,255,255,0.9)",
                            "margin": 0,
                            "fontSize": "14px",
                        },
                    ),
                ],
                className="sidebar-header",
            ),
            # Live Data Status
            html.Div(
                [
                    html.Div(
                        [
                            html.Span(
                                "🟢", style={"marginRight": "8px", "fontSize": "14px"}
                            ),
                            html.Span(
                                "Live Data",
                                style={
                                    "color": "white",
                                    "fontWeight": "bold",
                                    "fontSize": "16px",
                                },
                            ),
                        ]
                    )
                ],
                style={
                    "background": "rgba(107,142,90,0.3)",
                    "margin": "15px",
                    "padding": "12px",
                    "borderRadius": "10px",
                    "border": "1px solid rgba(107,142,90,0.4)",
                },
            ),
            # Navigation items
            html.Div(
                [
                    _create_nav_item(
                        "nav-overview",
                        "fas fa-chart-line",
                        "Market Overview",
                        "Live",
                        colors["success"],
                    ),
                    _create_nav_item(
                        "nav-explorer",
                        "fas fa-search",
                        "Data Explorer",
                        None,
                        None,
                        "Interactive analysis",
                    ),
                    _create_nav_item(
                        "nav-statistics",
                        "fas fa-chart-bar",
                        "Statistical Analysis",
                        None,
                        None,
                        "ACF/PACF & Tests",
                    ),
                    _create_nav_item(
                        "nav-transform",
                        "fas fa-cogs",
                        "Data Transformations",
                        "New",
                        colors["warning"],
                    ),
                    _create_nav_item(
                        "nav-forecasting",
                        "fas fa-crystal-ball",
                        "Forecasting Hub",
                        None,
                        None,
                        "SARIMA & LSTM",
                    ),
                ]
            ),
            # Quick Stats
            html.Div(
                [
                    html.H6(
                        "Quick Stats",
                        style={
                            "color": "white",
                            "marginBottom": "15px",
                            "fontWeight": "bold",
                        },
                    ),
                    html.Div(
                        "3.0%",
                        style={
                            "fontSize": "22px",
                            "fontWeight": "bold",
                            "color": "white",
                        },
                    ),
                    html.Div(
                        "Unemployment Rate",
                        style={
                            "fontSize": "12px",
                            "opacity": 0.9,
                            "color": "white",
                        },
                    ),
                    html.Hr(
                        style={
                            "borderColor": "rgba(255,255,255,0.3)",
                            "margin": "12px 0",
                        }
                    ),
                    html.Div(
                        "17.3M",
                        style={
                            "fontSize": "18px",
                            "fontWeight": "bold",
                            "color": "white",
                        },
                    ),
                    html.Div(
                        "Labor Force",
                        style={
                            "fontSize": "12px",
                            "opacity": 0.9,
                            "color": "white",
                        },
                    ),
                    html.Hr(
                        style={
                            "borderColor": "rgba(255,255,255,0.3)",
                            "margin": "12px 0",
                        }
                    ),
                    html.Div(
                        "Live",
                        style={
                            "color": colors["success"],
                            "fontWeight": "bold",
                            "fontSize": "16px",
                        },
                    ),
                    html.Div(
                        "Data Status",
                        style={
                            "fontSize": "12px",
                            "opacity": 0.9,
                            "color": "white",
                        },
                    ),
                ],
                className="quick-stats",
            ),
            # Auto-refresh indicator
            html.Div(
                [
                    dbc.Badge(
                        [
                            html.I(
                                className="fas fa-sync-alt",
                                style={"marginRight": "8px"},
                            ),
                            html.Span(id="last-updated", children="Starting..."),
                        ],
                        color="light",
                        className="w-100",
                        style={"color": colors["primary"], "fontSize": "12px"},
                    )
                ],
                style={"padding": "15px"},
            ),
        ],
        className="sidebar",
    )


def _create_nav_item(
    nav_id, icon_class, title, badge_text=None, badge_color=None, subtitle=None
):
    """Create individual navigation item"""
    nav_content = [
        html.A(
            [
                html.I(
                    className=icon_class,
                    style={"marginRight": "12px", "fontSize": "16px"},
                ),
                title,
            ],
            id=nav_id,
            className="nav-link",
        )
    ]

    if badge_text:
        nav_content.append(
            html.Span(
                badge_text,
                style={
                    "background": badge_color,
                    "color": "white",
                    "padding": "2px 8px",
                    "borderRadius": "12px",
                    "fontSize": "10px",
                    "marginLeft": "10px",
                },
            )
        )

    if subtitle:
        nav_content.append(
            html.P(
                subtitle,
                style={
                    "color": "rgba(255,255,255,0.8)",
                    "fontSize": "12px",
                    "margin": 0,
                    "paddingLeft": "32px",
                },
            )
        )

    return html.Div(nav_content, className="nav-item", id=f"{nav_id}-item")


def create_main_content():
    """Create main content area"""
    return html.Div(
        [
            html.Div(id="page-content"),
            dcc.Interval(id="interval-component", interval=30 * 1000, n_intervals=0),
            dcc.Store(id="current-page-store", data="overview"),
        ],
        className="main-content",
    )


def create_main_layout(colors):
    """Create complete dashboard layout"""
    return html.Div([create_sidebar(colors), create_main_content()])


def create_page_header(title, subtitle, icon_class, colors):
    """Create consistent page headers"""
    return html.Div(
        [
            html.H2(
                [html.I(className=icon_class, style={"marginRight": "15px"}), title],
                style={
                    "color": colors["primary"],
                    "marginBottom": "10px",
                    "fontWeight": "bold",
                },
            ),
            html.P(
                subtitle,
                style={"color": colors["text"], "fontSize": "16px", "opacity": 0.8},
            ),
        ],
        className="mb-4",
    )


def create_control_card(title, icon_class, children, colors):
    """Create consistent control card layouts"""
    return dbc.Card(
        [
            dbc.CardHeader(
                [html.I(className=icon_class, style={"marginRight": "8px"}), title],
                style={
                    "background": f'linear-gradient(90deg, {colors["secondary"]} 0%, {colors["primary"]} 100%)',
                    "color": "white",
                    "fontWeight": "bold",
                },
            ),
            dbc.CardBody(children, style={"backgroundColor": "rgba(255,255,255,0.5)"}),
        ],
        className="status-card",
    )


def create_chart_container(title, chart_id, icon_class, colors, height="400px"):
    """Create consistent chart container"""
    return html.Div(
        [
            html.H5(
                [html.I(className=icon_class, style={"marginRight": "10px"}), title],
                style={"color": colors["primary"], "fontWeight": "bold"},
            ),
            dcc.Graph(id=chart_id, style={"height": height}),
        ],
        className="chart-container",
    )

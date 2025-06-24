"""
Professional Forecasting Hub page with auto-prediction and improved UX.
Automatically generates forecasts when dropdowns change - no button needed.
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
    """Create professional forecasting hub page with auto-prediction"""
    header = create_page_header(
        "Forecasting Hub",
        "Advanced AI unemployment rate predictions with real-time updates",
        "fas fa-crystal-ball",
        colors,
    )

    controls = create_forecasting_controls(colors)
    forecast_section = create_forecast_section(colors)

    return html.Div([header, controls, forecast_section])


def create_forecasting_controls(colors):
    """Create professional forecasting controls with auto-prediction"""
    controls_content = [
        # Dataset Selection
        dbc.Row(
            [
                dbc.Col(
                    [
                        dbc.Label(
                            "Select Dataset:",
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
                                    "label": "General Labor Force (Raw Data)",
                                    "value": "general",
                                },
                                {
                                    "label": "Seasonally Adjusted Labor Force",
                                    "value": "sa",
                                },
                                {
                                    "label": "Youth Unemployment Data",
                                    "value": "youth",
                                },
                            ],
                            value="general",  # Default to general
                            style={"marginBottom": "15px"},
                        ),
                        html.P(
                            "Choose dataset type for forecasting",
                            style={
                                "color": colors["text"],
                                "fontSize": "12px",
                                "opacity": 0.8,
                            },
                        ),
                    ],
                    width=4,
                ),
                # Model Selection - LSTM as default
                dbc.Col(
                    [
                        dbc.Label(
                            "Select AI Model:",
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
                                    "label": "LSTM - Deep Learning Neural Network (Best Performance)",
                                    "value": "lstm",
                                },
                                {
                                    "label": "SARIMA - Seasonal Time Series",
                                    "value": "sarima",
                                },
                                {
                                    "label": "ARIMA - Statistical Time Series",
                                    "value": "arima",
                                },
                            ],
                            value="lstm",  
                            style={"marginBottom": "15px"},
                        ),
                        html.P(
                            "LSTM achieves the highest accuracy with 1.45% MAPE on SA data",
                            style={
                                "color": colors["text"],
                                "fontSize": "12px",
                                "opacity": 0.8,
                            },
                        ),
                    ],
                    width=4,
                ),
                # Forecast Period - 3 months as default
                dbc.Col(
                    [
                        dbc.Label(
                            "Forecast Period:",
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
                            value=3,  # Default to 3 months
                            style={"marginBottom": "15px"},
                        ),
                        html.P(
                            "Forecasts update automatically when selections change",
                            style={
                                "color": colors["text"],
                                "fontSize": "12px",
                                "opacity": 0.8,
                            },
                        ),
                    ],
                    width=4,
                ),
            ],
            className="mb-4",
        ),
        # Target Variable Info
        dbc.Row(
            [
                dbc.Col(
                    [
                        dbc.Label(
                            "Target Variable:",
                            style={
                                "fontWeight": "bold",
                                "color": colors["text"],
                                "fontSize": "16px",
                            },
                        ),
                        dcc.Dropdown(
                            id="forecast-target-dropdown",
                            options=[
                                {"label": "Overall Unemployment Rate (%)", "value": "u_rate"},
                                {"label": "Youth Unemployment Rate 15-24 (%)", "value": "u_rate_15_24"},
                                {"label": "Youth Unemployment Rate 15-30 (%)", "value": "u_rate_15_30"},
                            ],
                            value="u_rate",  # Default to overall unemployment rate
                            style={"marginBottom": "15px"},
                        ),
                        html.P(
                            "Select the target variable to forecast",
                            style={
                                "color": colors["text"],
                                "fontSize": "12px",
                                "opacity": 0.8,
                            },
                        ),
                    ],
                    width=6,
                ),
                # Auto-Update Status
                dbc.Col(
                    [
                        dbc.Label(
                            "Forecast Status:",
                            style={
                                "fontWeight": "bold",
                                "color": colors["text"],
                                "fontSize": "16px",
                            },
                        ),
                        html.Div(
                            [
                                dbc.Badge(
                                    "Auto-Updating",
                                    color="success",
                                    style={
                                        "fontSize": "14px",
                                        "padding": "10px 15px",
                                        "backgroundColor": colors["success"],
                                        "borderColor": colors["success"],
                                    },
                                )
                            ],
                            style={"marginBottom": "15px"},
                        ),
                        html.P(
                            "Predictions generate automatically when parameters change",
                            style={
                                "color": colors["text"],
                                "fontSize": "12px",
                                "opacity": 0.8,
                            },
                        ),
                    ],
                    width=6,
                ),
            ]
        ),
    ]

    return create_control_card(
        "Forecast Configuration", "fas fa-cogs", controls_content, colors
    )


def create_forecast_section(colors):
    """Create forecast results and visualization section"""
    return html.Div(
        [
            html.Div(id="forecast-summary-cards", className="mb-4"),
            dbc.Row(
                [
                    dbc.Col(
                        [
                            html.Div(
                                [
                                    html.H5(
                                        ["Unemployment Rate Forecast"],
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
            # Forecast Table and Model Info
            dbc.Row(
                [
                    dbc.Col(
                        [
                            html.Div(
                                [
                                    html.H5(
                                        ["Detailed Forecast Values"],
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
                                        ["Model Information"],
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
    """Create summary cards showing key forecast insights with proper colors"""
    try:
        current_rate = forecast_data["current_rate"]
        forecast_values = forecast_data["forecast_values"]
        trend_direction = forecast_data["trend_direction"]
        trend_magnitude = forecast_data["trend_magnitude"]
        confidence_level = forecast_data["confidence_level"]

        if trend_direction == "Rising":
            trend_color = colors["danger"]
        elif trend_direction == "Falling":
            trend_color = colors["success"]
        else:
            trend_color = colors["info"]

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
    """Create comprehensive forecast visualization with fixed color handling"""
    try:
        # Extract data with proper validation
        hist_dates = historical_data.get('dates', [])
        hist_values = historical_data.get('values', [])
        forecast_dates = forecast_data.get('dates', [])
        forecast_values = forecast_data.get('forecast_values', [])
        confidence_upper = forecast_data.get('confidence_upper', [])
        confidence_lower = forecast_data.get('confidence_lower', [])
        target_variable = forecast_data.get('target_variable', 'u_rate')
        
        if target_variable == "u_rate":
            target_name = "Malaysia Unemployment Rate"
        elif target_variable == "u_rate_15_24":
            target_name = "Malaysia Youth Unemployment Rate (15-24)"
        elif target_variable == "u_rate_15_30":
            target_name = "Malaysia Youth Unemployment Rate (15-30)"
        else:
            target_name = "Unemployment Rate"
        
        if hasattr(hist_dates, 'tolist'):
            hist_dates = hist_dates.tolist()
        if hasattr(forecast_dates, 'tolist'):
            forecast_dates = forecast_dates.tolist()
            
        if hasattr(hist_values, 'tolist'):
            hist_values = hist_values.tolist()
        if hasattr(forecast_values, 'tolist'):
            forecast_values = forecast_values.tolist()
        if hasattr(confidence_upper, 'tolist'):
            confidence_upper = confidence_upper.tolist()
        if hasattr(confidence_lower, 'tolist'):
            confidence_lower = confidence_lower.tolist()
        
        if not hist_dates or not hist_values or len(hist_dates) == 0 or len(hist_values) == 0:
            return create_empty_forecast_chart(colors, "No historical data available", target_variable)
        
        if not forecast_dates or not forecast_values or len(forecast_dates) == 0 or len(forecast_values) == 0:
            return create_empty_forecast_chart(colors, "No forecast data available", target_variable)
        
        # Create subplot
        fig = go.Figure()
        
        # Historical data with simple colors
        fig.add_trace(go.Scatter(
            x=hist_dates,
            y=hist_values,
            mode='lines',
            name='Historical Data',
            line=dict(color=colors['primary'], width=3),
            hovertemplate='<b>%{x}</b><br>Rate: %{y:.2f}%<extra></extra>'
        ))
        
        # Confidence interval 
        if (confidence_upper and confidence_lower and 
            len(confidence_upper) > 0 and len(confidence_lower) > 0 and
            len(confidence_upper) == len(forecast_dates) and 
            len(confidence_lower) == len(forecast_dates)):
            
            # Lower bound  
            fig.add_trace(go.Scatter(
                x=forecast_dates,
                y=confidence_lower,
                mode='lines',
                name='Lower Confidence',
                line=dict(color='rgba(0,0,0,0)', width=0),
                showlegend=False,
                hovertemplate='Lower CI: %{y:.2f}%<extra></extra>'
            ))
            
            # Upper bound 
            fig.add_trace(go.Scatter(
                x=forecast_dates,
                y=confidence_upper,
                mode='lines',
                name='95% Confidence Interval',
                line=dict(color='rgba(0,0,0,0)', width=0),
                fill='tonexty',
                fillcolor='rgba(184,92,87,0.2)', 
                hovertemplate='Upper CI: %{y:.2f}%<extra></extra>'
            ))
        
        # Forecast line with simple colors
        fig.add_trace(go.Scatter(
            x=forecast_dates,
            y=forecast_values,
            mode='lines+markers',
            name='AI Forecast',
            line=dict(color=colors['danger'], width=4, dash='dash'),
            marker=dict(size=8, color=colors['danger'], symbol='diamond'),
            hovertemplate='<b>%{x}</b><br>Forecast: %{y:.2f}%<extra></extra>'
        ))
        
        # Add vertical line at forecast start with simple colors
        if len(hist_dates) > 0 and len(forecast_dates) > 0:
            try:
                first_forecast_date = forecast_dates[0]
                fig.add_vline(
                    x=first_forecast_date,
                    line_dash="solid",
                    line_color=colors['warning'],
                    line_width=2,
                    annotation_text="AI Forecast Begins",
                    annotation_position="top",
                    annotation=dict(
                        font=dict(size=12, color=colors['warning']),
                        bgcolor="rgba(255,255,255,0.8)",
                        bordercolor=colors['warning'],
                        borderwidth=1
                    )
                )
            except Exception as vline_error:
                print(f"Warning: Could not add vertical line: {vline_error}")
        
        fig.update_layout(
            title={
                'text': f'{target_name}: AI-Powered Forecast',
                'x': 0.5,
                'font': {'size': 22, 'color': colors['primary'], 'family': 'Arial Black'}
            },
            xaxis_title="Date",
            yaxis_title="Unemployment Rate (%)",
            hovermode='x unified',
            height=500,
            template='plotly_white',
            plot_bgcolor='rgba(245,242,237,0.8)',
            paper_bgcolor='rgba(245,242,237,0.8)',
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1,
                bgcolor="rgba(255,255,255,0.8)",
                bordercolor="rgba(0,0,0,0.2)",
                borderwidth=1
            )
        )
        
        fig.update_xaxes(
            # showgrid=True, 
            gridwidth=1, 
            gridcolor='rgba(74,102,112,0.2)',
            tickformat="%b %Y"
        )
        fig.update_yaxes(
            # showgrid=True, 
            gridwidth=1, 
            gridcolor='rgba(74,102,112,0.2)',
            tickformat=".1f"
        )
        
        return fig
        
    except Exception as e:
        print(f"Chart creation error: {str(e)}")
        return create_error_forecast_chart(colors, str(e), target_variable)


def create_empty_forecast_chart(colors, message="Select parameters to see forecast", target_variable="u_rate"):
    """Create empty forecast chart with simple colors"""
    if target_variable == "u_rate":
        target_name = "Unemployment Rate"
    elif target_variable == "u_rate_15_24":
        target_name = "Youth Unemployment Rate (15-24)"
    elif target_variable == "u_rate_15_30":
        target_name = "Youth Unemployment Rate (15-30)"
    else:
        target_name = "Unemployment Rate"
        
    fig = go.Figure()
    fig.add_annotation(
        text=message,
        xref="paper", 
        yref="paper",
        x=0.5, 
        y=0.5,
        font=dict(size=16, color=colors['text']),
        showarrow=False
    )
    fig.update_layout(
        title=f"{target_name} Forecast",
        template='plotly_white',
        height=500,
        plot_bgcolor='rgba(245,242,237,0.8)',
        paper_bgcolor='rgba(245,242,237,0.8)',
        xaxis=dict(showgrid=False, showticklabels=False),
        yaxis=dict(showgrid=False, showticklabels=False)
    )
    return fig


def create_error_forecast_chart(colors, error_message, target_variable="u_rate"):
    """Create error forecast chart with simple colors"""
    if target_variable == "u_rate":
        target_name = "Unemployment Rate"
    elif target_variable == "u_rate_15_24":
        target_name = "Youth Unemployment Rate (15-24)"
    elif target_variable == "u_rate_15_30":
        target_name = "Youth Unemployment Rate (15-30)"
    else:
        target_name = "Unemployment Rate"
        
    fig = go.Figure()
    fig.add_annotation(
        text=f"Chart Error: {error_message}",
        xref="paper", 
        yref="paper",
        x=0.5, 
        y=0.5,
        font=dict(size=14, color=colors['danger']),
        showarrow=False
    )
    fig.update_layout(
        title=f"{target_name} Forecast - Error",
        template='plotly_white',
        height=500,
        plot_bgcolor='rgba(245,242,237,0.8)',
        paper_bgcolor='rgba(245,242,237,0.8)',
        xaxis=dict(showgrid=False, showticklabels=False),
        yaxis=dict(showgrid=False, showticklabels=False)
    )
    return fig

def create_forecast_table(forecast_data, colors):
    """Create detailed forecast table with page colors"""
    try:
        from dash import dash_table

        # Ptable data
        dates = forecast_data["dates"]
        values = forecast_data["forecast_values"]
        confidence_upper = forecast_data.get("confidence_upper", [None] * len(values))
        confidence_lower = forecast_data.get("confidence_lower", [None] * len(values))

        table_data = []
        for i, (date, value) in enumerate(zip(dates, values)):
            # Calculating change from previous period
            if i == 0:
                change = 0
                change_text = "Baseline"
            else:
                change = value - values[i - 1]
                if abs(change) < 0.05:
                    change_text = "Stable"
                elif change > 0:
                    change_text = f"+{change:.2f}%"
                else:
                    change_text = f"{change:.2f}%"

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

        # Creating table 
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
                "backgroundColor": colors["light"],
                "color": colors["text"],
                "fontSize": "14px",
                "border": f'1px solid {colors["primary"]}33',
            },
            style_header={
                "backgroundColor": colors["primary"],
                "fontWeight": "bold",
                "color": "white",
                "fontSize": "16px",
                "border": f'1px solid {colors["primary"]}',
            },
            style_data_conditional=[
                {
                    "if": {"filter_query": "{Status} = Improving"},
                    "backgroundColor": f'{colors["success"]}33',
                },
                {
                    "if": {"filter_query": "{Status} = Worsening"},
                    "backgroundColor": f'{colors["danger"]}33',
                },
                {
                    "if": {"filter_query": "{Status} = Stable"},
                    "backgroundColor": f'{colors["info"]}33',
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
    """Create professional model information display with page colors"""
    try:
        return dbc.Card(
            [
                dbc.CardBody(
                    [
                        html.H6(
                            ["Model Details"],
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
                                        model_info.get("model_type", "LSTM"),
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
                                # Model accuracy with page colors
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
                "backgroundColor": colors["light"],
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

    if "WRAPPER" in model_type:
        iteration = model_info.get("iteration", 4)
        target_variable = model_info.get("target_variable", "")
        mae = model_info.get("mae", 0.0)
        mape = model_info.get("mape", 0.0)
        rmse = model_info.get("rmse", 0.0)
        accuracy = model_info.get("accuracy", 0.0)
        rank = model_info.get("evaluation_rank", "N/A")
        
        return html.Div(
            [
                html.P(
                    [html.Strong("Model Type: ")],
                    style={"marginBottom": "8px", "fontSize": "12px"},
                ),
                html.P(
                    [html.Strong("Iteration: "), f"i{iteration}"],
                    style={"marginBottom": "8px", "fontSize": "12px"},
                ),
                html.P(
                    [html.Strong("Target: "), target_variable.replace("u_rate_", "Ages ")],
                    style={"marginBottom": "8px", "fontSize": "12px"},
                ),
                html.Hr(),
                html.P(
                    [html.Strong("Model Metrics")],
                    style={
                        "marginBottom": "8px",
                        "fontSize": "14px",
                        "color": colors["primary"],
                    },
                ),
                html.P(
                    [html.Strong("MAE: "), f"{mae:.4f}"],
                    style={"marginBottom": "5px", "fontSize": "12px"},
                ),
                html.P(
                    [html.Strong("MAPE: "), f"{mape:.2f}%"],
                    style={"marginBottom": "5px", "fontSize": "12px"},
                ),
                html.P(
                    [html.Strong("RMSE: "), f"{rmse:.4f}"],
                    style={"marginBottom": "5px", "fontSize": "12px"},
                ),
                html.P(
                    [html.Strong("Accuracy: "), f"{accuracy:.2f}%"],
                    style={"marginBottom": "8px", "fontSize": "12px", "color": colors["success"]},
                ),
                html.P(
                    [html.Strong("Rank: "), rank],
                    style={
                        "marginBottom": "8px",
                        "fontSize": "12px",
                        "color": colors["success"] if "Best" in rank else colors["info"],
                    },
                ),
            ]
        )
    elif "SARIMA" in model_type:
        order = model_info.get("order", (0, 1, 1))
        seasonal_order = model_info.get("seasonal_order", (0, 1, 1, 12))
        youth_age_group = model_info.get("youth_age_group", "")
        
        # Special display for youth SARIMA models
        if youth_age_group:
            return html.Div(
                [
                    html.P(
                        [html.Strong("Model Type: "), f"Youth SARIMA Model"],
                        style={"marginBottom": "8px", "fontSize": "12px"},
                    ),
                    html.P(
                        [html.Strong("Age Group: "), f"{youth_age_group}"],
                        style={"marginBottom": "8px", "fontSize": "12px"},
                    ),
                    html.P(
                        [html.Strong("Model Order: "), f"SARIMA{order}×{seasonal_order}"],
                        style={"marginBottom": "8px", "fontSize": "12px"},
                    ),
                    html.P(
                        [html.Strong("Seasonality: "), "Monthly (12 periods)"],
                        style={"marginBottom": "8px", "fontSize": "12px"},
                    ),
                    html.P(
                        [
                            html.Strong("Note: "), 
                            "This model is specifically trained on youth unemployment data"
                        ],
                        style={
                            "marginBottom": "8px", 
                            "fontSize": "12px",
                            "color": colors["info"]
                        },
                    ),
                ]
            )
        # Standard SARIMA display
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
        youth_age_group = model_info.get("youth_age_group", "")
        
        # display for youth ARIMA models
        if youth_age_group:
            return html.Div(
                [
                    html.P(
                        [html.Strong("Model Type: "), f"Youth ARIMA Model"],
                        style={"marginBottom": "8px", "fontSize": "12px"},
                    ),
                    html.P(
                        [html.Strong("Age Group: "), f"{youth_age_group}"],
                        style={"marginBottom": "8px", "fontSize": "12px"},
                    ),
                    html.P(
                        [html.Strong("Model Order: "), f"ARIMA{order}"],
                        style={"marginBottom": "8px", "fontSize": "12px"},
                    ),
                    html.P(
                        [html.Strong("Type: "), "Non-seasonal"],
                        style={"marginBottom": "8px", "fontSize": "12px"},
                    ),
                    html.P(
                        [
                            html.Strong("Note: "), 
                            "This model is specifically trained on youth unemployment data"
                        ],
                        style={
                            "marginBottom": "8px", 
                            "fontSize": "12px",
                            "color": colors["info"]
                        },
                    ),
                ]
            )
        # Standard ARIMA display
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
                html.P(
                    [html.Strong("Performance: "), "Best Overall Model"],
                    style={
                        "marginBottom": "8px",
                        "fontSize": "12px",
                        "color": colors["success"],
                    },
                ),
            ]
        )
    else:
        return html.Div()


def create_loading_state(colors):
    """Create loading state for forecast generation with page colors"""
    return dbc.Card(
        [
            dbc.CardBody(
                [
                    html.Div(
                        [
                            dbc.Spinner(
                                size="lg",
                                color="primary",
                                spinner_style={"color": colors["primary"]},
                            ),
                            html.H4(
                                "Generating AI Forecast...",
                                className="mt-3 text-center",
                                style={"color": colors["primary"]},
                            ),
                            html.P(
                                "Processing unemployment data with advanced neural networks",
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
    """Create error state display with page colors"""
    return dbc.Alert(
        [
            f"Forecast Error: {error_message}",
            html.Hr(),
            html.P("Please check model availability and try again.", className="mb-0"),
        ],
        color="danger",
        className="status-card",
        style={"borderColor": colors["danger"], "color": colors["danger"]},
    )

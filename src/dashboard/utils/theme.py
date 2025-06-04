"""
Theme and styling management for Malaysia Unemployment Dashboard.
Centralized color scheme and CSS management.
"""


class ThemeManager:
    """Manages dashboard theme, colors, and styling"""

    def __init__(self):
        self.colors = {
            "primary": "#4A6670",
            "secondary": "#C3B59F",
            "accent": "#8FA6B1",
            "success": "#6B8E5A",
            "danger": "#B85C57",
            "warning": "#D4A574",
            "info": "#7B9AA8",
            "light": "#E8E1D5",
            "dark": "#3A4A52",
            "background": "#F5F2ED",
            "text": "#3A4A52",
            "gradient_start": "#C3B59F",
            "gradient_end": "#4A6670",
        }

    def get_custom_css(self):
        """Return custom CSS with full styling"""
        return f"""
        <!DOCTYPE html>
        <html>
            <head>
                {{%metas%}}
                <title>{{%title%}}</title>
                {{%favicon%}}
                {{%css%}}
                <style>
                    body {{
                        font-family: 'Arial', sans-serif;
                        background: linear-gradient(135deg, {self.colors['gradient_start']} 0%, {self.colors['gradient_end']} 100%);
                        margin: 0;
                        padding: 0;
                        min-height: 100vh;
                    }}
                    
                    .sidebar {{
                        background: linear-gradient(135deg, {self.colors['gradient_start']} 0%, {self.colors['gradient_end']} 100%);
                        min-height: 100vh;
                        box-shadow: 2px 0 15px rgba(0,0,0,0.2);
                        position: fixed;
                        z-index: 1000;
                        width: 280px;
                    }}
                    
                    .sidebar-header {{
                        background: rgba(255,255,255,0.15);
                        backdrop-filter: blur(15px);
                        border-radius: 12px;
                        margin: 15px;
                        padding: 25px;
                        text-align: center;
                        border: 1px solid rgba(255,255,255,0.2);
                    }}
                    
                    .nav-item {{
                        margin: 10px 15px;
                        border-radius: 10px;
                        transition: all 0.3s ease;
                        cursor: pointer;
                    }}
                    
                    .nav-item:hover {{
                        background: rgba(255,255,255,0.2);
                        transform: translateX(8px);
                        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
                    }}
                    
                    .nav-item.active {{
                        background: rgba(255,255,255,0.25);
                        border-left: 4px solid #fff;
                    }}
                    
                    .nav-link {{
                        color: white !important;
                        padding: 15px 20px;
                        border-radius: 10px;
                        text-decoration: none;
                        display: block;
                        font-weight: 500;
                    }}
                    
                    .quick-stats {{
                        background: rgba(255,255,255,0.15);
                        backdrop-filter: blur(15px);
                        border-radius: 12px;
                        margin: 15px;
                        padding: 20px;
                        border: 1px solid rgba(255,255,255,0.2);
                    }}
                    
                    .main-content {{
                        background: rgba(245,242,237,0.95);
                        min-height: 100vh;
                        margin-left: 280px;
                        padding: 30px;
                        backdrop-filter: blur(10px);
                    }}
                    
                    .metric-card {{
                        background: linear-gradient(135deg, rgba(255,255,255,0.9) 0%, rgba(232,225,213,0.9) 100%);
                        border-radius: 15px;
                        box-shadow: 0 8px 25px rgba(0,0,0,0.1);
                        transition: all 0.3s ease;
                        border: 1px solid rgba(196,181,159,0.3);
                    }}
                    
                    .metric-card:hover {{
                        transform: translateY(-5px);
                        box-shadow: 0 15px 35px rgba(0,0,0,0.15);
                    }}
                    
                    .chart-container {{
                        background: linear-gradient(135deg, rgba(255,255,255,0.9) 0%, rgba(232,225,213,0.9) 100%);
                        border-radius: 15px;
                        box-shadow: 0 8px 25px rgba(0,0,0,0.1);
                        padding: 25px;
                        margin: 25px 0;
                        border: 1px solid rgba(196,181,159,0.3);
                    }}
                    
                    .status-card {{
                        background: linear-gradient(135deg, rgba(255,255,255,0.9) 0%, rgba(232,225,213,0.9) 100%);
                        border-radius: 12px;
                        border: 1px solid rgba(196,181,159,0.3);
                    }}
                    
                    .transform-header {{
                        background: linear-gradient(135deg, {self.colors['gradient_start']} 0%, {self.colors['gradient_end']} 100%);
                        border-radius: 15px;
                        margin-bottom: 30px;
                        box-shadow: 0 8px 25px rgba(0,0,0,0.15);
                    }}
                    
                    .btn-primary {{
                        background: linear-gradient(135deg, {self.colors['primary']} 0%, {self.colors['secondary']} 100%);
                        border: none;
                        border-radius: 8px;
                    }}
                    
                    .btn-secondary {{
                        background: linear-gradient(135deg, {self.colors['accent']} 0%, {self.colors['warning']} 100%);
                        border: none;
                        border-radius: 8px;
                    }}
                </style>
            </head>
            <body>
                {{%app_entry%}}
                <footer>
                    {{%config%}}
                    {{%scripts%}}
                    {{%renderer%}}
                </footer>
            </body>
        </html>
        """

    def get_chart_style(self):
        """Return consistent chart styling configuration"""
        return {
            "template": "plotly_white",
            "plot_bgcolor": "rgba(245,242,237,0.8)",
            "paper_bgcolor": "rgba(245,242,237,0.8)",
            "gridcolor": "rgba(74,102,112,0.2)",
            "height": 400,
        }

    def get_metric_colors(self):
        """Return color mapping for different metric types"""
        return {
            "unemployment": self.colors["danger"],
            "labor_force": self.colors["info"],
            "employment": self.colors["success"],
            "youth": self.colors["warning"],
            "participation": self.colors["primary"],
        }

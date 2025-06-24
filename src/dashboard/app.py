import dash
from dash import html, dcc
import dash_bootstrap_components as dbc
from datetime import datetime
import sys
import os
from pathlib import Path

# Setup project paths
current_dir = Path(__file__).parent
project_root = current_dir.parent.parent
sys.path.insert(0, str(project_root / "src"))

# Import internal modules
from dashboard.utils.data_manager import DashboardDataManager
from dashboard.utils.theme import ThemeManager
from dashboard.components.layout import create_main_layout
from dashboard.components.callbacks import (
    register_all_callbacks,
    register_overview_callbacks,
)


class MalaysiaUnemploymentDashboard:
    """Malaysia Labor Force Analytics Dashboard"""

    def __init__(self):
        self.app = dash.Dash(
            __name__,
            external_stylesheets=[
                dbc.themes.BOOTSTRAP,
                "https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0/css/all.min.css",
            ],
            suppress_callback_exceptions=True,
            meta_tags=[
                {"name": "viewport", "content": "width=device-width, initial-scale=1"},
                {
                    "name": "description",
                    "content": "Malaysia Labor Force Analytics Dashboard",
                },
                {"name": "author", "content": "Malaysia Unemployment Analytics Team"},
            ],
        )

        self.app.title = "Malaysia Labor Force Analytics"

        # Initialize core components
        self.theme = ThemeManager()
        self.data_manager = DashboardDataManager()

        # Configure the app
        self._setup_layout()
        self._register_callbacks()
        self._initialize_data()

    def _setup_layout(self):
        """Configure the main layout"""
        try:
            self.app.index_string = self.theme.get_custom_css()
            self.app.layout = create_main_layout(self.theme.colors)
            print("Layout setup completed successfully.")
        except Exception as e:
            self.app.layout = html.Div(
                [
                    html.H1("Malaysia Unemployment Dashboard"),
                    html.P(f"Error loading main layout: {str(e)}"),
                ]
            )

    def _register_callbacks(self):
        """Register callbacks"""
        try:
            print("Registering callbacks...")

            register_all_callbacks(self.app, self.data_manager, self.theme.colors)
            print("Main callbacks registered.")

            register_overview_callbacks(self.app, self.data_manager, self.theme.colors)
            print("Overview callbacks registered.")

            print("All callbacks registered successfully.")

        except ImportError as e:
            print(f"Import error registering callbacks: {e}")
            print("Ensure all required modules are available.")
        except Exception as e:
            print(f"Error registering callbacks: {e}")
            print("The dashboard will run with limited functionality.")

    def _initialize_data(self):
        """Initialize dataset on startup"""
        try:
            print("Initializing data manager...")
            success = self.data_manager.initialize()

            if success:
                print("Data initialized successfully.")
                print("Available datasets:")
                for dataset_name, df in self.data_manager.datasets.items():
                    if df is not None and not df.empty:
                        print(f"  - {dataset_name}: {len(df)} records")
                    else:
                        print(f"  - {dataset_name}: No data available")
            else:
                print("Using fallback data. Some features may be limited.")

        except Exception as e:
            print(f"Error initializing data: {e}")
            print("Running with mock or fallback data.")

    def run(self, host="127.0.0.1", port=8050, debug=True):
        """Run in local development mode"""
        print("=" * 70)
        print("MALAYSIA UNEMPLOYMENT ANALYTICS DASHBOARD")
        print("=" * 70)
        print(f"Starting server on http://{host}:{port}")
        print("Architecture: Modular and maintainable")
        print("Styling: Bootstrap-based layout")
        print("Features: Interactive data visualization")
        print("=" * 70)

        if self.data_manager.initialized:
            print("Data Manager: READY")
        else:
            print("Data Manager: LIMITED (using fallback data)")

        print("Theme Manager: READY")
        print("Layout: READY")
        print("=" * 70)

        try:
            if hasattr(self.app, "run"):
                self.app.run(host=host, port=port, debug=debug)
            else:
                self.app.run_server(host=host, port=port, debug=debug)

        except KeyboardInterrupt:
            print("Dashboard stopped by user.")
        except Exception as e:
            print(f"Error running dashboard: {e}")
            print("Consider changing host/port settings.")

    def run_server_production(self):
        """Run server in production mode"""
        port = int(os.environ.get("PORT", 8050))
        host = os.environ.get("HOST", "0.0.0.0")

        print("=" * 70)
        print("PRODUCTION DEPLOYMENT - MALAYSIA UNEMPLOYMENT DASHBOARD")
        print("=" * 70)
        print(f"Starting production server on {host}:{port}")
        print("Mode: Production (debugging disabled)")
        print("Optimizations: Enabled")
        print("=" * 70)

        if self.data_manager.initialized:
            print("Data Manager: READY")
        else:
            print("Data Manager: LIMITED (using fallback data)")

        print("Theme Manager: READY")
        print("Layout: READY")
        print("=" * 70)

        try:
            self.app.run_server(
                host=host,
                port=port,
                debug=False,
                dev_tools_ui=False,
                dev_tools_props_check=False,
            )
        except Exception as e:
            print(f"Production server error: {e}")
            raise

    def get_app(self):
        """Expose Dash app for WSGI servers"""
        return self.app

    def get_server(self):
        """Expose Flask server for deployment"""
        return self.app.server


# Create dashboard instance
dashboard_app = MalaysiaUnemploymentDashboard()

# Export for deployment platforms
app = dashboard_app.get_app()
server = dashboard_app.get_server()


def main():
    """Entry point for running dashboard"""
    dashboard_app.run()


def run_production(host="0.0.0.0", port=None):
    """Run in production mode"""
    if port is None:
        port = int(os.environ.get("PORT", 8050))

    print("Starting production dashboard...")
    dashboard_app.run(host=host, port=port, debug=False)


def run_development():
    """Run in development mode"""
    print("Starting in development mode...")
    dashboard_app.run(host="127.0.0.1", port=8050, debug=True)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Malaysia Unemployment Analytics Dashboard"
    )
    parser.add_argument(
        "--mode",
        choices=["dev", "prod", "render"],
        default="dev",
        help="Run mode: dev (development), prod (production), or render (deployment)",
    )
    parser.add_argument(
        "--host", default="127.0.0.1", help="Host address (default: 127.0.0.1)"
    )
    parser.add_argument(
        "--port",
        type=int,
        default=None,
        help="Port number (default: from environment or 8050)",
    )

    args = parser.parse_args()

    if args.mode == "render":
        dashboard_app.run_server_production()
    elif args.mode == "prod":
        run_production(host=args.host, port=args.port)
    else:
        port = args.port if args.port else 8050
        if args.host != "127.0.0.1" or port != 8050:
            dashboard_app.run(host=args.host, port=port, debug=True)
        else:
            run_development()

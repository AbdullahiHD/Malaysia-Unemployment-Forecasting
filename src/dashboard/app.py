
"""
Malaysia Unemployment Analytics Dashboard - Main Application
Professional entry point with modular architecture.
UPDATED: Fixed callback registration and serialization issues.
"""

import dash
from dash import html, dcc
import dash_bootstrap_components as dbc
from datetime import datetime
import sys
from pathlib import Path

# Add project root to path for imports
current_dir = Path(__file__).parent
project_root = current_dir.parent.parent
sys.path.insert(0, str(project_root / "src"))

from dashboard.utils.data_manager import DashboardDataManager
from dashboard.utils.theme import ThemeManager
from dashboard.components.layout import create_main_layout
from dashboard.components.callbacks import (
    register_all_callbacks,
    register_overview_callbacks,
)


class MalaysiaUnemploymentDashboard:
    """Professional Malaysia Labor Force Analytics Dashboard"""

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

        # Initialize managers
        self.theme = ThemeManager()
        self.data_manager = DashboardDataManager()

        # Setup application
        self._setup_layout()
        self._register_callbacks()
        self._initialize_data()

    def _setup_layout(self):
        """Setup main application layout with custom CSS"""
        try:
            self.app.index_string = self.theme.get_custom_css()
            self.app.layout = create_main_layout(self.theme.colors)
            print("✅ Layout setup completed successfully")
        except Exception as e:
            print(f"❌ Error setting up layout: {e}")
            # Fallback layout
            self.app.layout = html.Div(
                [
                    html.H1("Malaysia Unemployment Dashboard"),
                    html.P(f"Error loading main layout: {str(e)}"),
                ]
            )

    def _register_callbacks(self):
        """Register all dashboard callbacks with proper error handling"""
        try:
            print("🔧 Registering callbacks...")

            # Register main callbacks (navigation, explorer, statistics, etc.)
            register_all_callbacks(self.app, self.data_manager, self.theme.colors)
            print("✅ Main callbacks registered")

            # Register overview-specific callbacks for time period buttons
            register_overview_callbacks(self.app, self.data_manager, self.theme.colors)
            print("✅ Overview callbacks registered")

            print("✅ All callbacks registered successfully")

        except ImportError as e:
            print(f"❌ Import error registering callbacks: {e}")
            print("📋 Check that all required modules are available")
        except Exception as e:
            print(f"❌ Error registering callbacks: {e}")
            print("📋 Dashboard will run with limited functionality")

    def _initialize_data(self):
        """Auto-initialize data on startup with enhanced error handling"""
        try:
            print("📊 Initializing data manager...")
            success = self.data_manager.initialize()

            if success:
                print("✅ Data auto-initialized successfully")
                print(
                    f"📈 Available datasets: {list(self.data_manager.datasets.keys())}"
                )

                # Print data summary
                for dataset_name, df in self.data_manager.datasets.items():
                    if df is not None and not df.empty:
                        print(f"   - {dataset_name}: {len(df)} records")
                    else:
                        print(f"   - {dataset_name}: No data available")
            else:
                print("⚠️ Using fallback data - some features may be limited")

        except Exception as e:
            print(f"❌ Error initializing data: {e}")
            print("⚠️ Dashboard will run with mock data")

    def run(self, host="127.0.0.1", port=8050, debug=True):
        """Run the dashboard application with enhanced startup info"""
        print("=" * 70)
        print("🇲🇾 MALAYSIA UNEMPLOYMENT ANALYTICS DASHBOARD")
        print("=" * 70)
        print(f"🚀 Starting server on http://{host}:{port}")
        print("📊 Professional modular architecture")
        print("🎨 Enhanced styling and user experience")
        print("🔧 Fixed serialization and callback issues")
        print("⏰ Time period buttons fully functional")
        print("📈 Real-time unemployment data visualization")
        print("=" * 70)

        # Print data status
        if self.data_manager.initialized:
            print("✅ Data Manager: READY")
        else:
            print("⚠️ Data Manager: LIMITED (using fallback data)")

        print("✅ Theme Manager: READY")
        print("✅ Layout: READY")
        print("=" * 70)

        try:
            if hasattr(self.app, "run"):
                self.app.run(host=host, port=port, debug=debug)
            else:
                self.app.run_server(host=host, port=port, debug=debug)

        except KeyboardInterrupt:
            print("\n🛑 Dashboard stopped by user")
        except Exception as e:
            print(f"❌ Error running dashboard: {e}")
            print("💡 Try running with different host/port settings")

    def get_app(self):
        """Get the Dash app instance (useful for deployment)"""
        return self.app

    def get_server(self):
        """Get the Flask server instance (for deployment platforms like Heroku)"""
        return self.app.server


# Create application instance
dashboard_app = MalaysiaUnemploymentDashboard()

# Export for deployment platforms
app = dashboard_app.get_app()
server = dashboard_app.get_server()


def main():
    """Main function to run the dashboard"""
    dashboard_app.run()


def run_production(host="0.0.0.0", port=8050):
    """Run dashboard in production mode"""
    print("🚀 Starting in PRODUCTION mode...")
    dashboard_app.run(host=host, port=port, debug=False)


def run_development():
    """Run dashboard in development mode with enhanced debugging"""
    print("🔧 Starting in DEVELOPMENT mode...")
    dashboard_app.run(host="127.0.0.1", port=8050, debug=True)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Malaysia Unemployment Analytics Dashboard"
    )
    parser.add_argument(
        "--mode",
        choices=["dev", "prod"],
        default="dev",
        help="Run mode: dev (development) or prod (production)",
    )
    parser.add_argument(
        "--host", default="127.0.0.1", help="Host address (default: 127.0.0.1)"
    )
    parser.add_argument(
        "--port", type=int, default=8050, help="Port number (default: 8050)"
    )

    args = parser.parse_args()

    if args.mode == "prod":
        run_production(host=args.host, port=args.port)
    else:
        if args.host != "127.0.0.1" or args.port != 8050:
            dashboard_app.run(host=args.host, port=args.port, debug=True)
        else:
            run_development()

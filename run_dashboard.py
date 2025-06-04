#!/usr/bin/env python3
"""
Malaysia Unemployment Forecasting Dashboard Runner
Professional entry point for the analytics platform.
Fixed compatibility with modern Dash versions.
"""
import sys
import os
from pathlib import Path
import argparse

# Add src to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root / "src"))


def main():
    """Main function to run the dashboard."""
    parser = argparse.ArgumentParser(
        description="Malaysia Labor Force Analytics Platform"
    )
    parser.add_argument("--host", default="127.0.0.1", help="Host to run the server on")
    parser.add_argument(
        "--port", type=int, default=8050, help="Port to run the server on"
    )
    parser.add_argument("--debug", action="store_true", help="Run in debug mode")
    parser.add_argument(
        "--no-debug", dest="debug", action="store_false", help="Run in production mode"
    )
    parser.set_defaults(debug=True)

    args = parser.parse_args()

    try:
        # Import dashboard after setting up path
        from dashboard.app import MalaysiaUnemploymentDashboard

        print("=" * 70)
        print("MALAYSIA LABOR FORCE ANALYTICS PLATFORM")
        print("=" * 70)
        print(f"🚀 Starting dashboard server...")
        print(f"🌐 Host: {args.host}")
        print(f"🔌 Port: {args.port}")
        print(f"🐛 Debug mode: {args.debug}")
        print(f"📊 Dashboard URL: http://{args.host}:{args.port}")
        print("=" * 70)
        print("💡 Dashboard Features:")
        print("   📈 Market Overview - Key unemployment metrics")
        print("   🔍 Data Explorer - Interactive visualization")
        print("   📊 Statistical Analysis - Comprehensive testing")
        print("   🔮 Forecasting Hub - Advanced modeling")
        print("=" * 70)

        # Create dashboard instance
        dashboard = MalaysiaUnemploymentDashboard()

        # Try modern Dash method first, fallback to older if needed
        try:
            print("🔧 Using app.run() method...")
            dashboard.app.run(host=args.host, port=args.port, debug=args.debug)
        except AttributeError:
            print("🔧 Falling back to app.run_server() method...")
            dashboard.app.run_server(host=args.host, port=args.port, debug=args.debug)

    except ImportError as e:
        print(f"❌ Error: Failed to import dashboard modules: {e}")
        print("\n💡 Please ensure all dependencies are installed:")
        print("pip install -r requirements.txt")
        print("\n📦 Missing dependencies? Try:")
        print(
            "pip install dash dash-bootstrap-components plotly pandas numpy scipy statsmodels"
        )
        print("\n📁 Directory structure should be:")
        print("   project_root/")
        print("   ├── src/")
        print("   │   ├── dashboard/")
        print("   │   │   └── app.py")
        print("   │   ├── data/")
        print("   │   │   └── data_loader.py")
        print("   │   └── analysis/")
        print("   │       └── statistical_tests.py")
        print("   └── run_dashboard.py")
        sys.exit(1)

    except OSError as e:
        if "Address already in use" in str(e):
            print(f"❌ Error: Port {args.port} is already in use")
            print(
                f"💡 Try a different port: python run_dashboard.py --port {args.port + 1}"
            )
            print("🔍 Or kill the process using that port:")
            print(f"   netstat -ano | findstr :{args.port}")
        else:
            print(f"❌ Network error: {e}")
        sys.exit(1)

    except KeyboardInterrupt:
        print("\n👋 Dashboard shutdown gracefully")
        sys.exit(0)

    except Exception as e:
        print(f"❌ Error: Failed to start dashboard: {e}")
        print("\n🔧 Troubleshooting tips:")
        print("1. Ensure all source files are in the correct directories")
        print("2. Check that the port is not already in use")
        print("3. Verify internet connection for DOSM data access")
        print("4. Check Python version (3.7+ required)")
        print("5. Ensure all dependencies are installed")
        print("\n🐛 For debug information, run with --debug flag")
        sys.exit(1)


if __name__ == "__main__":
    main()

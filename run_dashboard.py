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

# Adding src to Python path
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
        from dashboard.app import MalaysiaUnemploymentDashboard

        print("=" * 70)
        print("MALAYSIA LABOR FORCE ANALYTICS PLATFORM")
        print("=" * 70)
        print(f" Starting dashboard server...")
        print(f" Host: {args.host}")
        print(f" Port: {args.port}")
        print(f" Debug mode: {args.debug}")
        print(f" Dashboard URL: http://{args.host}:{args.port}")
        print("=" * 70)
        print(" Dashboard Features:")
        print("    Market Overview - Key unemployment metrics")
        print("    Data Explorer - Interactive visualization")
        print("    Statistical Analysis - Comprehensive testing")
        print("   Forecasting Hub - Advanced modeling")
        print("=" * 70)

        # Create dashboard instance
        dashboard = MalaysiaUnemploymentDashboard()

        try:
            print("🔧 Using app.run() method...")
            dashboard.app.run(host=args.host, port=args.port, debug=args.debug)
        except AttributeError:
            print("🔧 Falling back to app.run_server() method...")
            dashboard.app.run_server(host=args.host, port=args.port, debug=args.debug)

    except ImportError as e:
        print(f" Error: Failed to import dashboard modules: {e}")
    
        sys.exit(1)

    except OSError as e:
        if "Address already in use" in str(e):
            print(f" Error: Port {args.port} is already in use")
            print(
                f" Try a different port: python run_dashboard.py --port {args.port + 1}"
            )
            print(" Or kill the process using that port:")
            print(f"   netstat -ano | findstr :{args.port}")
        else:
            print(f" Network error: {e}")
        sys.exit(1)

    except KeyboardInterrupt:
        print("\n Dashboard shutdown gracefully")
        sys.exit(0)

    except Exception as e:
        print(f" Error: Failed to start dashboard: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()

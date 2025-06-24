#!/usr/bin/env python3
"""
Render deployment entry point for Malaysia Unemployment Dashboard
This file should be in your PROJECT ROOT (same level as requirements.txt)
"""
import sys
import os
from pathlib import Path

# Add src to Python path so we can import from src/dashboard
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root / "src"))

try:
    # Import the dashboard from src/dashboard/app.py
    from dashboard.app import MalaysiaUnemploymentDashboard

    print("✅ Successfully imported MalaysiaUnemploymentDashboard")

    # Create the dashboard instance
    dashboard_app = MalaysiaUnemploymentDashboard()

    # These are what Render will use
    app = dashboard_app.get_app()
    server = dashboard_app.get_server()

    print("✅ Dashboard app and server created successfully")

except ImportError as e:
    print(f"❌ Import error: {e}")
    print(f"📁 Current working directory: {os.getcwd()}")
    print(f"📁 Project root: {project_root}")
    print(f"📁 Src path: {project_root / 'src'}")
    print("📁 Contents of project root:")
    for item in project_root.iterdir():
        print(f"   - {item.name}")
    raise

if __name__ == "__main__":
    # For local testing of the deployment version
    port = int(os.environ.get("PORT", 8050))
    host = os.environ.get("HOST", "0.0.0.0")

    print("=" * 70)
    print("🚀 RENDER DEPLOYMENT TEST - MALAYSIA UNEMPLOYMENT DASHBOARD")
    print("=" * 70)
    print(f"🌐 Testing deployment server on {host}:{port}")
    print("🔧 This simulates how Render will run your app")
    print("=" * 70)

    try:
        app.run(
            host=host,
            port=port,
            debug=False,
            dev_tools_ui=False,
            dev_tools_props_check=False,
        )
    except Exception as e:
        print(f"❌ Error running deployment test: {e}")
        print("💡 Make sure all dependencies are installed")
        raise

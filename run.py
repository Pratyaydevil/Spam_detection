import sys
import threading
import time
import webbrowser

def open_browser():
    """Wait for the server to spin up, then open the web UI in default browser."""
    time.sleep(1.5)
    webbrowser.open("http://127.0.0.1:8000")

def main():
    print("=" * 65)
    print("  🚀 Starting Hybrid AI Email Defender (Backend + Frontend)")
    print("=" * 65)
    print("  • Web UI:           http://127.0.0.1:8000")
    print("  • API Documentation: http://127.0.0.1:8000/docs")
    print("  • Health Endpoint:   http://127.0.0.1:8000/health")
    print("=" * 65)
    print("  Press Ctrl+C to stop the server.\n")

    # Open browser automatically in a background thread
    threading.Thread(target=open_browser, daemon=True).start()

    try:
        import uvicorn
        uvicorn.run("app.api.main:app", host="127.0.0.1", port=8000, reload=True)
    except ImportError:
        print("[ERROR] uvicorn is not installed in the current environment.")
        print("Please activate your virtual environment: .\\.venv\\Scripts\\Activate.ps1")
        print("Or run: pip install -r requirements.txt")
        sys.exit(1)

if __name__ == "__main__":
    main()

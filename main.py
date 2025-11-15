import os

import uvicorn


def main() -> None:
    """Start the FastAPI server for the Muse metrics backend."""
    host = os.environ.get("HOST", "0.0.0.0")
    port = int(os.environ.get("PORT", "8000"))
    reload_enabled = os.environ.get("RELOAD", "false").lower() in {"1", "true", "yes"}
    uvicorn.run("backend:app", host=host, port=port, reload=reload_enabled)


if __name__ == "__main__":
    main()

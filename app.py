"""
Consensus Engine API Serverless Entrypoint.
Exports the FastAPI app for Vercel and ASGI runners.
"""
from src.api.main import app

__all__ = ["app"]

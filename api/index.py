"""
Vercel Serverless Function entrypoint for Consensus Engine FastAPI backend.
Exports the ASGI app object for @vercel/python.
"""
from src.api.main import app

__all__ = ["app"]

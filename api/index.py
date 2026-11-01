"""
Vercel Serverless Function entrypoint for Consensus Engine FastAPI backend.
Exports the ASGI app object from app.py in the repository root for @vercel/python.
"""
from app import app

__all__ = ["app"]

from __future__ import annotations
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
from pydantic import ValidationError
from groq import RateLimitError, APIError, APIConnectionError

from src.api.routes import router
from src.persistence.database import init_db

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(levelname)s %(message)s",
)
logger = logging.getLogger("consensus_engine_api")


@asynccontextmanager
async def lifespan(app: FastAPI):
    await init_db()
    yield


app = FastAPI(
    title="Consensus Engine",
    description="Multi-Agent Negotiation Under Private Information",
    version="0.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Exception handlers ensuring JSON {detail, error_type} and CORS headers ────

@app.exception_handler(RateLimitError)
async def rate_limit_handler(request: Request, exc: RateLimitError):
    return JSONResponse(
        status_code=429,
        content={"detail": str(exc), "error_type": "rate_limit"},
        headers={"Access-Control-Allow-Origin": "*"},
    )


@app.exception_handler(APIConnectionError)
@app.exception_handler(APIError)
async def upstream_error_handler(request: Request, exc: Exception):
    return JSONResponse(
        status_code=502,
        content={"detail": str(exc), "error_type": "upstream_error"},
        headers={"Access-Control-Allow-Origin": "*"},
    )


@app.exception_handler(RequestValidationError)
@app.exception_handler(ValidationError)
async def validation_error_handler(request: Request, exc: Exception):
    return JSONResponse(
        status_code=422,
        content={"detail": str(exc), "error_type": "validation_error"},
        headers={"Access-Control-Allow-Origin": "*"},
    )


@app.exception_handler(ValueError)
async def value_error_handler(request: Request, exc: ValueError):
    return JSONResponse(
        status_code=400,
        content={"detail": str(exc), "error_type": "value_error"},
        headers={"Access-Control-Allow-Origin": "*"},
    )


@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    logger.exception("Unhandled server error: %s", exc)
    return JSONResponse(
        status_code=500,
        content={"detail": str(exc), "error_type": "internal_error"},
        headers={"Access-Control-Allow-Origin": "*"},
    )


app.include_router(router, prefix="/api")


@app.get("/api/health")
def health():
    return {"status": "ok", "service": "consensus-engine"}


@app.get("/")
def root():
    return {
        "status": "ok",
        "service": "consensus-engine",
        "docs": "/docs",
        "health": "/api/health",
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("src.api.main:app", host="0.0.0.0", port=8000, reload=True)

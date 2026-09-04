import os
import gradio as gr
from fastapi.responses import JSONResponse
from src.api.main import app as fastapi_app

# ZeroGPU hook to satisfy Hugging Face startup check
try:
    import spaces
    @spaces.GPU
    def _zero_gpu_worker():
        return True
except Exception:
    pass

# Root endpoint for API discovery
@fastapi_app.get("/")
def api_root():
    return {
        "status": "ok",
        "service": "consensus-engine",
        "docs": "/docs",
        "health": "/api/health",
        "ui": "/ui"
    }

# Create Gradio UI mounted at /ui
with gr.Blocks(title="Consensus Engine API") as demo:
    gr.Markdown("# 🤖 Consensus Engine API Service")
    gr.Markdown(
        """
        Backend API server for **Consensus Engine: Multi-Agent Negotiation Under Private Information**.
        
        ### 🔗 Live API Endpoints
        - **Interactive API Docs (Swagger UI)**: [/docs](/docs)
        - **Health Probe**: [/api/health](/api/health)
        - **Scenarios List**: [/api/scenarios](/api/scenarios)
        - **Next.js Frontend**: [consensus-engine-opal.vercel.app](https://consensus-engine-opal.vercel.app)
        """
    )

# Mount Gradio onto the existing FastAPI application at /ui
# All FastAPI endpoints (/api/*, /docs, /openapi.json) remain fully active at root
app = gr.mount_gradio_app(fastapi_app, demo, path="/ui")

import os
import gradio as gr
from fastapi.responses import JSONResponse
from src.api.main import app as fastapi_app

# ZeroGPU hook explicitly attached to a Gradio event listener
try:
    import spaces
    @spaces.GPU
    def gpu_health_check(text: str) -> str:
        return f"ZeroGPU active: {text}"
except Exception:
    def gpu_health_check(text: str) -> str:
        return f"Engine active: {text}"

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

# Create Gradio UI with explicit component hooked to @spaces.GPU
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
    with gr.Row():
        inp = gr.Textbox(value="health_check", label="Engine Probe")
        out = gr.Textbox(label="Status")
    btn = gr.Button("Verify ZeroGPU Engine")
    btn.click(fn=gpu_health_check, inputs=inp, outputs=out)

# Mount Gradio onto the existing FastAPI application at /ui
app = gr.mount_gradio_app(fastapi_app, demo, path="/ui")

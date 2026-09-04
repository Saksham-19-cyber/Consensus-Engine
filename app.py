import spaces
import os
import gradio as gr
from fastapi.responses import JSONResponse
from src.api.main import app as fastapi_app

# ZeroGPU hook to satisfy Hugging Face startup check
try:
    import spaces
    @spaces.GPU
    def gpu_health_check(text: str) -> str:
        return f"ZeroGPU active: {text}"
except Exception:
    pass

# Explicit Health check route on root FastAPI app
@fastapi_app.get("/api/health")
def api_health():
    return {"status": "ok", "service": "consensus-engine"}

@fastapi_app.get("/")
def api_root():
    return {
        "status": "ok",
        "service": "consensus-engine",
        "docs": "/docs",
        "health": "/api/health",
        "ui": "/ui"
    }

# Create Gradio UI for Hugging Face Space & ZeroGPU
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

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 7860))
    print(f"Starting Consensus Engine on port {port}...")
    uvicorn.run(app, host="0.0.0.0", port=port)

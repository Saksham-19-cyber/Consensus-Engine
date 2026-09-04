import spaces
import os
import gradio as gr
from fastapi.middleware.cors import CORSMiddleware
from src.api.main import app as fastapi_app
from src.persistence.database import init_db

# ZeroGPU hook to satisfy Hugging Face startup check
@spaces.GPU
def gpu_health_check(text: str) -> str:
    return f"ZeroGPU active: {text}"

# Health check route on FastAPI
@fastapi_app.get("/api/health")
def api_health():
    return {"status": "ok", "service": "consensus-engine"}

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

# ASGI Routing Middleware: Intercepts /api/* and /docs to route directly to FastAPI
class FastAPIRoutingMiddleware:
    def __init__(self, app, fastapi_target):
        self.app = app
        self.fastapi_target = fastapi_target

    async def __call__(self, scope, receive, send):
        if scope["type"] in ("http", "websocket"):
            path = scope.get("path", "")
            if path.startswith("/api") or path in ("/docs", "/openapi.json"):
                return await self.fastapi_target(scope, receive, send)
        return await self.app(scope, receive, send)

demo.app.add_middleware(FastAPIRoutingMiddleware, fastapi_target=fastapi_app)

@demo.app.on_event("startup")
async def on_startup():
    await init_db()

if __name__ == "__main__":
    demo.queue().launch()

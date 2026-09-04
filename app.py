import spaces
import os
import gradio as gr
from fastapi.openapi.docs import get_swagger_ui_html
from fastapi.middleware.cors import CORSMiddleware
from src.api.routes import router
from src.persistence.database import init_db

@spaces.GPU
def gpu_health_check(text: str) -> str:
    return f"ZeroGPU active: {text}"

# Create Gradio UI
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

# 1. Add CORS middleware
demo.app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 2. Add Swagger UI at /docs
@demo.app.get("/docs", include_in_schema=False)
async def custom_swagger():
    return get_swagger_ui_html(
        openapi_url="/openapi.json",
        title="Consensus Engine - Swagger UI"
    )

# 3. Add Health check at /api/health
@demo.app.get("/api/health")
def api_health():
    return {"status": "ok", "service": "consensus-engine"}

# 4. Include all Consensus Engine API routes
demo.app.include_router(router, prefix="/api")

@demo.app.on_event("startup")
async def on_startup():
    await init_db()

if __name__ == "__main__":
    demo.queue().launch()

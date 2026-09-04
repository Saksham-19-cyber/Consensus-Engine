import spaces
import gradio as gr
from fastapi.middleware.cors import CORSMiddleware
from src.api.routes import router
from src.persistence.database import init_db

@spaces.GPU
def gpu_health_check(text: str) -> str:
    return f"ZeroGPU active: {text}"

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

# Mount CORS and FastAPI routes directly onto Gradio's internal FastAPI app
demo.app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
demo.app.include_router(router, prefix="/api")

# Put /api routes at the front of the routing table so they are never intercepted by Gradio catch-all
api_routes = [r for r in demo.app.routes if getattr(r, 'path', '').startswith('/api')]
other_routes = [r for r in demo.app.routes if not getattr(r, 'path', '').startswith('/api')]
demo.app.routes = api_routes + other_routes

@demo.app.on_event("startup")
async def on_startup():
    await init_db()

if __name__ == "__main__":
    demo.queue().launch()

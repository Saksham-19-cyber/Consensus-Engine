import os
import asyncio
import gradio as gr
from fastapi.middleware.cors import CORSMiddleware
from src.api.routes import router
from src.persistence.database import init_db

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

# Mount CORS and FastAPI routes directly onto Gradio's FastAPI application
demo.app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
demo.app.include_router(router, prefix="/api")

@demo.app.on_event("startup")
async def on_startup():
    await init_db()

if __name__ == "__main__":
    # Launch natively via Gradio - Hugging Face automatically manages port binding and ZeroGPU
    demo.queue().launch()

import os
import gradio as gr
from src.api.main import app as fastapi_app

# Create an informative Gradio Blocks UI for the Hugging Face Space frontend
with gr.Blocks(title="Consensus Engine API") as demo:
    gr.Markdown("# 🤖 Consensus Engine API Service")
    gr.Markdown(
        """
        Backend API server for **Consensus Engine: Multi-Agent Negotiation Under Private Information**.
        
        ### 🔗 Endpoints & Documentation
        - **Interactive API Docs (Swagger UI)**: [/docs](/docs)
        - **Health Probe**: [/api/health](/api/health)
        - **Scenarios List**: [/api/scenarios](/api/scenarios)
        - **Frontend App**: [consensus-engine-opal.vercel.app](https://consensus-engine-opal.vercel.app)
        """
    )

# Mount Gradio onto the existing FastAPI application
# All FastAPI endpoints (/api/..., /docs, /openapi.json) remain fully active and accessible
app = gr.mount_gradio_app(fastapi_app, demo, path="/")

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 7860))
    uvicorn.run(app, host="0.0.0.0", port=port)

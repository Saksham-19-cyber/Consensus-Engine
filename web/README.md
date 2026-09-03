# Consensus Engine — Web Frontend

A high-performance, data-dense Next.js dashboard for multi-agent autonomous negotiation under private information.

---

## 🚀 Deployment Architecture & Split

Consensus Engine is architected as a **decoupled client-server deployment**:

```
┌──────────────────────────────────────┐       ┌──────────────────────────────────────┐
│        Vercel (Edge / CDN)           │       │    Render / Railway / Fly.io Host    │
│                                      │ HTTP  │                                      │
│  Next.js 15 App Router Frontend      │──────▶│  FastAPI + LangGraph Orchestrator    │
│  - Interactive Negotiation Runner    │  & WS │  - ChromaDB Vector Store             │
│  - 95% Bootstrap CI Chart Dashboard  │       │  - Groq LLM API Streaming Agent Loop │
│  - Privacy Leakage Probe Visualizer  │       │  - Auditable JSONL Log Streamer      │
│  - Trial Log Explorer                │       │                                      │
└──────────────────────────────────────┘       └──────────────────────────────────────┘
```

### Why This Split is Necessary
* **Frontend (`/web`)**: Deploys as a standard Next.js application on **Vercel** with global CDN caching and client-side rendering for rich Recharts visualisations.
* **Backend (`src/api`)**: Runs **LangGraph negotiation graphs**, SQLite session persistence, persistent ChromaDB precedent memory, and multi-round iterative Groq LLM queries. Because negotiation runs can exceed serverless execution limits (taking 15–45s across 10 rounds), **the backend cannot run as Vercel serverless functions** and must be deployed on a service supporting long-lived stateful processes (e.g., Render, Railway, or Fly.io).

---

## 🛠 Setup & Local Development

### 1. Configure Environment
Copy `.env.example` to `.env.local`:
```bash
cp .env.example .env.local
```

Configure `NEXT_PUBLIC_API_URL` to point to your FastAPI backend:
```env
NEXT_PUBLIC_API_URL=http://localhost:8000
```

### 2. Install Dependencies
```bash
npm install
```

### 3. Run Development Server
```bash
npm run dev
```
Open [http://localhost:3000](http://localhost:3000) to view the UI.

---

## 📦 Deploying to Vercel

1. Push your repository to GitHub.
2. In the [Vercel Dashboard](https://vercel.com/new), import the repository.
3. Set **Root Directory** to `web`.
4. In **Environment Variables**, add:
   - `NEXT_PUBLIC_API_URL`: Your deployed FastAPI backend URL (e.g. `https://consensus-engine-api.onrender.com`).
5. Click **Deploy**.

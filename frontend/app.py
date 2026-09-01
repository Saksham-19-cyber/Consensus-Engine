import streamlit as st
import requests
import json
import time

API_BASE = "http://localhost:8000/api"

st.set_page_config(
    page_title="Consensus Engine",
    page_icon="🤝",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
    .stApp {
        background: linear-gradient(135deg, #0f0c29 0%, #302b63 50%, #24243e 100%);
    }
    .main-header {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 2.5rem;
        font-weight: 800;
        text-align: center;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        color: #a0aec0;
        text-align: center;
        font-size: 1.1rem;
        margin-bottom: 2rem;
    }
    .message-card {
        background: rgba(255, 255, 255, 0.05);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 12px;
        padding: 1rem;
        margin-bottom: 0.75rem;
        backdrop-filter: blur(10px);
    }
    .mediator-card {
        border-left: 4px solid #667eea;
    }
    .stakeholder-card {
        border-left: 4px solid #48bb78;
    }
    .metric-card {
        background: rgba(255, 255, 255, 0.08);
        border-radius: 12px;
        padding: 1.5rem;
        text-align: center;
        border: 1px solid rgba(255, 255, 255, 0.1);
    }
    .status-agreed {
        color: #48bb78;
        font-weight: 700;
    }
    .status-impasse {
        color: #f56565;
        font-weight: 700;
    }
    .status-max_rounds {
        color: #ed8936;
        font-weight: 700;
    }
</style>
""", unsafe_allow_html=True)

st.markdown('<p class="main-header">🤝 Consensus Engine</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Multi-Agent Negotiation Under Private Information</p>', unsafe_allow_html=True)

tab1, tab2, tab3 = st.tabs(["🎯 Negotiate", "📊 Evaluate", "📋 History"])

with tab1:
    with st.sidebar:
        st.markdown("### ⚙️ Configuration")
        scenario = st.selectbox("Scenario", ["roommate", "business_deal", "trip_planning"])
        n_agents = st.slider("Agents", 2, 5, 3 if scenario == "trip_planning" else 2)
        max_rounds = st.slider("Max Rounds", 3, 20, 10)
        seed = st.number_input("Random Seed", value=42, min_value=0)
        start_btn = st.button("🚀 Start Negotiation", type="primary", use_container_width=True)

    if start_btn:
        with st.spinner("Running negotiation..."):
            try:
                resp = requests.post(f"{API_BASE}/negotiate", json={
                    "scenario": scenario,
                    "n_agents": n_agents,
                    "max_rounds": max_rounds,
                    "seed": seed,
                }, timeout=300)
                data = resp.json()
                st.session_state["last_result"] = data
            except Exception as e:
                st.error(f"Error: {e}")
                data = None

        if data:
            outcome = data.get("outcome", {})
            status = outcome.get("status", "unknown")
            status_class = f"status-{status}"

            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.markdown(f'<div class="metric-card"><h3>Status</h3><p class="{status_class}">{status.upper()}</p></div>', unsafe_allow_html=True)
            with col2:
                st.markdown(f'<div class="metric-card"><h3>Rounds</h3><p>{outcome.get("rounds_taken", "?")}</p></div>', unsafe_allow_html=True)
            with col3:
                utils = outcome.get("per_agent_utilities", {})
                avg_util = sum(utils.values()) / len(utils) if utils else 0
                st.markdown(f'<div class="metric-card"><h3>Avg Utility</h3><p>{avg_util:.3f}</p></div>', unsafe_allow_html=True)
            with col4:
                st.markdown(f'<div class="metric-card"><h3>Agreement</h3><p>{"✅" if outcome.get("agreement_reached") else "❌"}</p></div>', unsafe_allow_html=True)

            st.markdown("---")

            if outcome.get("per_agent_utilities"):
                st.markdown("### 📈 Agent Utilities")
                for agent, util in outcome["per_agent_utilities"].items():
                    col_a, col_b = st.columns([1, 4])
                    with col_a:
                        st.markdown(f"**{agent}**")
                    with col_b:
                        st.progress(min(1.0, max(0.0, util)), text=f"{util:.3f}")

            if outcome.get("final_proposal"):
                st.markdown("### 📋 Final Proposal")
                st.json(outcome["final_proposal"])

            st.markdown("### 💬 Negotiation Transcript")
            messages = data.get("messages", [])
            for msg in messages:
                role = msg.get("role", "")
                card_class = "mediator-card" if role == "mediator" else "stakeholder-card"
                icon = "🧑‍⚖️" if role == "mediator" else "🗣️"
                agent = msg.get("agent_name", "")
                content = msg.get("content", "")
                rnd = msg.get("round_number", "")
                meta = msg.get("metadata", {})
                reasoning = meta.get("reasoning", "")

                st.markdown(f"""
                <div class="message-card {card_class}">
                    <strong>{icon} {agent}</strong> <span style="color:#718096">Round {rnd}</span><br/>
                    {content}
                    {"<br/><em style='color:#a0aec0'>" + reasoning[:200] + "...</em>" if reasoning and len(reasoning) > 10 else ""}
                </div>
                """, unsafe_allow_html=True)

with tab2:
    st.markdown("### 📊 Batch Evaluation")
    eval_col1, eval_col2 = st.columns(2)
    with eval_col1:
        eval_scenario = st.selectbox("Eval Scenario", ["roommate", "business_deal", "trip_planning"], key="eval_sc")
        eval_trials = st.slider("Number of Trials", 5, 100, 20, key="eval_trials")
    with eval_col2:
        eval_methods = st.multiselect(
            "Baseline Methods",
            ["naive_average", "nash_bargaining", "single_llm_oracle"],
            default=["naive_average", "nash_bargaining"],
            key="eval_methods",
        )
        eval_seed = st.number_input("Eval Seed", value=42, min_value=0, key="eval_seed")

    if st.button("🔬 Run Evaluation", type="primary"):
        with st.spinner(f"Running {eval_trials} trials..."):
            try:
                resp = requests.post(f"{API_BASE}/eval/run", json={
                    "scenario": eval_scenario,
                    "n_trials": eval_trials,
                    "methods": eval_methods,
                    "seed": eval_seed,
                }, timeout=600)
                eval_data = resp.json()
                st.session_state["last_eval"] = eval_data
            except Exception as e:
                st.error(f"Error: {e}")
                eval_data = None

        if eval_data:
            st.markdown(eval_data.get("report_markdown", ""))
            if eval_data.get("summary"):
                st.markdown("### 📈 Raw Summary")
                st.json(eval_data["summary"])

with tab3:
    st.markdown("### 📋 Previous Sessions")
    session_id = st.text_input("Session ID", placeholder="Enter session ID to load")
    if session_id and st.button("Load Session"):
        try:
            resp = requests.get(f"{API_BASE}/sessions/{session_id}", timeout=30)
            if resp.status_code == 200:
                session_data = resp.json()
                st.json(session_data)
            else:
                st.error("Session not found")
        except Exception as e:
            st.error(f"Error: {e}")

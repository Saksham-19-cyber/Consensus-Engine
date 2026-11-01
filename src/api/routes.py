from __future__ import annotations
import asyncio
import json
import logging
import uuid
from fastapi import APIRouter, WebSocket, WebSocketDisconnect, HTTPException

from src.api.schemas import (
    NegotiateRequest,
    EvalRequest,
    SessionResponse,
    EvalResponse,
    ParseScenarioRequest,
    ParseScenarioResponse,
    ParsedIssueOut,
    ParsedStakeholderOut,
)
from src.protocol.graph import run_negotiation
from src.models.negotiation import NegotiationMessage
from src.eval.runner import run_batch_baselines, evaluate_outcome, aggregate_results
from src.eval.report import generate_markdown_report
from src.persistence.database import (
    create_session,
    update_session_outcome,
    store_message,
    get_session,
    get_session_messages,
)
from src.memory.store import store_negotiation_outcome
from scenarios.generator import get_scenario, make_scenario_generator

logger = logging.getLogger(__name__)
router = APIRouter()


@router.get("/scenarios")
async def list_scenarios():
    from scenarios.generator import SCENARIO_REGISTRY
    return {
        "scenarios": [
            {"name": name, "description": cls.description if hasattr(cls, "description") else ""}
            for name, cls in SCENARIO_REGISTRY.items()
        ]
    }


@router.post("/negotiate", response_model=SessionResponse)
async def start_negotiation(req: NegotiateRequest):
    session_id = str(uuid.uuid4())[:8]

    # ── Free-form mode: decode pre-parsed token ──────────────────────────────
    if req.parsed_scenario_token:
        from scenarios.from_text import ParseReport
        try:
            # Security Note: Token is unsigned base64-encoded JSON; schema is verified
            # via Pydantic model validation on decode. Semantic tampering (e.g. altered
            # weights) is an accepted limitation for this auth-less research/demo tool.
            profiles, issues = ParseReport.decode_token(req.parsed_scenario_token)
        except Exception as e:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid parsed_scenario_token: {e}",
            )
        scenario_name = "free_form"
    else:
        # ── Standard named-scenario mode ─────────────────────────────────────
        kwargs = {}
        if req.scenario == "trip_planning":
            kwargs["n_agents"] = req.n_agents
        scenario = get_scenario(req.scenario, **kwargs)
        profiles, issues = scenario.generate(seed=req.seed)
        scenario_name = req.scenario

    await create_session(session_id, scenario_name, {
        "n_agents": len(profiles),
        "max_rounds": req.max_rounds,
        "seed": req.seed,
    })

    collected_messages = []

    def on_message(msg: NegotiationMessage):
        collected_messages.append(msg)

    loop = asyncio.get_event_loop()
    outcome = await loop.run_in_executor(
        None,
        lambda: run_negotiation(
            profiles=profiles,
            issues=issues,
            max_rounds=req.max_rounds,
            on_message=on_message,
            scenario_name=scenario_name,
            protocol=req.protocol,
        ),
    )

    await update_session_outcome(session_id, outcome.status.value, outcome.model_dump())

    for msg in collected_messages:
        await store_message(
            session_id=session_id,
            round_number=msg.round_number,
            agent_name=msg.agent_name,
            role=msg.role,
            content=msg.content,
            message_type=msg.message_type,
            metadata=msg.metadata,
        )

    try:
        store_negotiation_outcome(
            session_id=session_id,
            scenario_name=scenario_name,
            outcome=outcome.model_dump(),
            profiles=[p.model_dump() for p in profiles],
            issues=issues,
        )
    except Exception as e:
        logger.warning("failed to store in chromadb: %s", e)

    return SessionResponse(
        session_id=session_id,
        scenario=scenario_name,
        status=outcome.status.value,
        outcome=outcome.model_dump(),
        messages=[m.model_dump() for m in collected_messages],
    )


@router.post("/scenario/parse", response_model=ParseScenarioResponse)
async def parse_scenario(req: ParseScenarioRequest):
    """
    Parse a free-text negotiation description into structured scenario objects.

    IMPORTANT: This endpoint does NOT run a negotiation. It returns a
    validation report (ParseScenarioResponse) that MUST be shown to the
    user for confirmation before calling POST /api/negotiate.

    Free-form scenarios are unvalidated / exploratory and are NOT covered
    by the N=30 statistical benchmarks in the README.
    """
    from scenarios.from_text import generate_from_description

    loop = asyncio.get_event_loop()
    try:
        result = await loop.run_in_executor(
            None,
            lambda: generate_from_description(req.description, seed=req.seed),
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    profiles, issues_meta, report = result

    # Build per-stakeholder output objects (with source provenance)
    stakeholders_out = []
    for profile in profiles:
        source = report.stakeholder_sources.get(profile.name, "llm_inferred")
        stakeholders_out.append(ParsedStakeholderOut(
            name=profile.name,
            role=profile.role,
            persona=profile.persona,
            source=source,
            weights=profile.utility_function.weights,
            ideal_values=profile.utility_function.ideal_values,
            reservation_value=profile.reservation_value,
        ))

    issues_out = [
        ParsedIssueOut(
            name=i["name"],
            min_value=i["range"][0],
            max_value=i["range"][1],
            description=i.get("description", ""),
        )
        for i in issues_meta
    ]

    return ParseScenarioResponse(
        issues=issues_out,
        stakeholders=stakeholders_out,
        field_notes=report.field_notes,
        warnings=report.warnings,
        pareto_mode=report.pareto_mode,
        issue_count=report.issue_count,
        parsed_scenario_token=report.to_token(),
    )


@router.get("/sessions/{session_id}", response_model=SessionResponse)
async def get_session_detail(session_id: str):
    session = await get_session(session_id)
    if not session:
        from fastapi import HTTPException
        raise HTTPException(status_code=404, detail="Session not found")

    messages = await get_session_messages(session_id)
    outcome = json.loads(session["outcome_json"]) if session.get("outcome_json") else None

    return SessionResponse(
        session_id=session_id,
        scenario=session["scenario"],
        status=session["status"],
        outcome=outcome,
        messages=messages,
    )


@router.post("/eval/run", response_model=EvalResponse)
async def run_evaluation(req: EvalRequest):
    batch_id = str(uuid.uuid4())[:8]

    kwargs = {}
    if req.scenario == "trip_planning":
        kwargs["n_agents"] = req.n_agents

    generator = make_scenario_generator(req.scenario, **kwargs)

    def _execute():
        from src.eval.runner import run_batch_baselines, run_batch_engine
        from src.eval.log_writer import TrialLogWriter
        results = []
        writer = TrialLogWriter(scenario_name=req.scenario, tag=req.protocol)
        writer.open()
        try:
            baseline_methods = [m for m in req.methods if m != "consensus_engine"]
            if baseline_methods:
                baselines = run_batch_baselines(
                    scenario_generator=generator,
                    methods=baseline_methods,
                    n_trials=req.n_trials,
                    scenario_name=req.scenario,
                    seed=req.seed,
                    model_config_name=req.model_config_name,
                    log_writer=writer,
                )
                results.extend(baselines)

            if "consensus_engine" in req.methods:
                engines = run_batch_engine(
                    scenario_generator=generator,
                    n_trials=req.n_trials,
                    scenario_name=req.scenario,
                    seed=req.seed,
                    protocol=req.protocol,
                    model_config_name=req.model_config_name,
                    run_privacy_probe=req.run_privacy_probe,
                    log_writer=writer,
                )
                results.extend(engines)
        finally:
            writer.close()
        return results

    loop = asyncio.get_event_loop()
    results = await loop.run_in_executor(None, _execute)

    summary = aggregate_results(results)
    report_md = generate_markdown_report(req.scenario, summary, results)

    return EvalResponse(
        batch_id=batch_id,
        scenario=req.scenario,
        n_trials=req.n_trials,
        summary=summary,
        report_markdown=report_md,
    )


@router.websocket("/ws/negotiate/{session_id}")
async def websocket_negotiate(websocket: WebSocket, session_id: str):
    await websocket.accept()
    try:
        data = await websocket.receive_json()
        scenario_name = data.get("scenario", "roommate")
        seed = data.get("seed", 42)
        max_rounds = data.get("max_rounds", 10)
        n_agents = data.get("n_agents", 2)

        kwargs = {}
        if scenario_name == "trip_planning":
            kwargs["n_agents"] = n_agents

        scenario = get_scenario(scenario_name, **kwargs)
        profiles, issues = scenario.generate(seed=seed)

        await create_session(session_id, scenario_name, {
            "n_agents": len(profiles),
            "max_rounds": max_rounds,
            "seed": seed,
        })

        async def on_message_ws(msg: NegotiationMessage):
            try:
                await websocket.send_json(msg.model_dump(mode="json"))
            except Exception:
                pass

        loop = asyncio.get_event_loop()

        collected = []

        def on_message_sync(msg: NegotiationMessage):
            collected.append(msg)

        outcome = await loop.run_in_executor(
            None,
            lambda: run_negotiation(
                profiles=profiles,
                issues=issues,
                max_rounds=max_rounds,
                on_message=on_message_sync,
            ),
        )

        for msg in collected:
            await websocket.send_json(msg.model_dump(mode="json"))

        await websocket.send_json({
            "type": "outcome",
            "data": outcome.model_dump(mode="json"),
        })

        await update_session_outcome(session_id, outcome.status.value, outcome.model_dump())

    except WebSocketDisconnect:
        logger.info("websocket disconnected: %s", session_id)
    except Exception as e:
        logger.error("websocket error: %s", e)
        try:
            await websocket.send_json({"type": "error", "message": str(e)})
        except Exception:
            pass


@router.get("/eval/report/{scenario}")
async def get_eval_report(scenario: str):
    """Return the aggregated benchmark report and metrics for a scenario."""
    from src.config import settings
    from src.eval.log_writer import load_trial_log
    from src.eval.runner import aggregate_results
    log_dir = settings.data_dir / "logs"
    if log_dir.exists():
        matching = sorted(log_dir.glob(f"{scenario}*.jsonl"), reverse=True)
        if matching:
            results = load_trial_log(matching[0])
            if results:
                summary = aggregate_results(results)
                report_md = generate_markdown_report(scenario, summary, results)
                return {"scenario": scenario, "summary": summary, "report_markdown": report_md, "source_log": matching[0].name}

    return {"scenario": scenario, "summary": {}, "report_markdown": f"No benchmark run found for {scenario}"}


@router.get("/logs")
async def list_logs():
    """List available JSONL trial logs in data/logs/."""
    from src.config import settings
    log_dir = settings.data_dir / "logs"
    files = []
    if log_dir.exists():
        for p in sorted(log_dir.glob("*.jsonl"), reverse=True):
            files.append({
                "filename": p.name,
                "size_bytes": p.stat().st_size,
                "modified": p.stat().st_mtime,
            })
    return {"logs": files}


@router.get("/logs/{filename}")
async def read_log(filename: str):
    """Read records from a specific JSONL log file."""
    from src.config import settings
    log_file = settings.data_dir / "logs" / filename
    if not log_file.exists() or not filename.endswith(".jsonl"):
        from fastapi import HTTPException
        raise HTTPException(status_code=404, detail="Log file not found")
    
    records = []
    with open(log_file, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                try:
                    records.append(json.loads(line))
                except Exception:
                    pass
    return {"filename": filename, "record_count": len(records), "records": records[:100]}


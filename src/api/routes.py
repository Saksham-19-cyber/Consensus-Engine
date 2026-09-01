from __future__ import annotations
import asyncio
import json
import logging
import uuid
from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from src.api.schemas import NegotiateRequest, EvalRequest, SessionResponse, EvalResponse
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
    kwargs = {}
    if req.scenario == "trip_planning":
        kwargs["n_agents"] = req.n_agents

    scenario = get_scenario(req.scenario, **kwargs)
    profiles, issues = scenario.generate(seed=req.seed)

    await create_session(session_id, req.scenario, {
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
            scenario_name=req.scenario,
            outcome=outcome.model_dump(),
            profiles=[p.model_dump() for p in profiles],
            issues=issues,
        )
    except Exception as e:
        logger.warning("failed to store in chromadb: %s", e)

    return SessionResponse(
        session_id=session_id,
        scenario=req.scenario,
        status=outcome.status.value,
        outcome=outcome.model_dump(),
        messages=[m.model_dump() for m in collected_messages],
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

    loop = asyncio.get_event_loop()
    results = await loop.run_in_executor(
        None,
        lambda: run_batch_baselines(
            scenario_generator=generator,
            methods=req.methods,
            n_trials=req.n_trials,
            scenario_name=req.scenario,
            seed=req.seed,
        ),
    )

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

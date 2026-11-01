from __future__ import annotations
import aiosqlite
import json
import logging
from pathlib import Path
from datetime import datetime, timezone
from src.config import settings

logger = logging.getLogger(__name__)


def _get_db_path() -> str:
    path = settings.sqlite_path
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
    except Exception as e:
        logger.warning("Could not create database parent directory %s: %s", path.parent, e)
    return str(path)


async def init_db():
    db_path = _get_db_path()
    async with aiosqlite.connect(db_path) as db:
        await db.execute("""
            CREATE TABLE IF NOT EXISTS sessions (
                id TEXT PRIMARY KEY,
                scenario TEXT NOT NULL,
                created_at TEXT NOT NULL,
                status TEXT DEFAULT 'in_progress',
                config_json TEXT,
                outcome_json TEXT
            )
        """)
        await db.execute("""
            CREATE TABLE IF NOT EXISTS messages (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT NOT NULL,
                round_number INTEGER,
                agent_name TEXT,
                role TEXT,
                content TEXT,
                message_type TEXT,
                metadata_json TEXT,
                created_at TEXT NOT NULL,
                FOREIGN KEY (session_id) REFERENCES sessions(id)
            )
        """)
        await db.execute("""
            CREATE TABLE IF NOT EXISTS trial_results (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                batch_id TEXT,
                trial_id INTEGER,
                scenario TEXT,
                method TEXT,
                agreement INTEGER,
                rounds INTEGER,
                utilities_json TEXT,
                proposal_json TEXT,
                pareto_json TEXT,
                fairness_json TEXT,
                created_at TEXT NOT NULL
            )
        """)
        await db.commit()
    logger.info("database initialized at %s", db_path)


async def create_session(session_id: str, scenario: str, config: dict):
    try:
        async with aiosqlite.connect(_get_db_path()) as db:
            await db.execute(
                "INSERT INTO sessions (id, scenario, created_at, config_json) VALUES (?, ?, ?, ?)",
                (session_id, scenario, datetime.now(timezone.utc).isoformat(), json.dumps(config)),
            )
            await db.commit()
    except Exception as e:
        logger.warning("create_session failed: %s", e)


async def update_session_outcome(session_id: str, status: str, outcome: dict):
    try:
        async with aiosqlite.connect(_get_db_path()) as db:
            await db.execute(
                "UPDATE sessions SET status = ?, outcome_json = ? WHERE id = ?",
                (status, json.dumps(outcome), session_id),
            )
            await db.commit()
    except Exception as e:
        logger.warning("update_session_outcome failed: %s", e)


async def store_message(
    session_id: str,
    round_number: int,
    agent_name: str,
    role: str,
    content: str,
    message_type: str,
    metadata: dict,
):
    try:
        async with aiosqlite.connect(_get_db_path()) as db:
            await db.execute(
                "INSERT INTO messages (session_id, round_number, agent_name, role, content, message_type, metadata_json, created_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (session_id, round_number, agent_name, role, content, message_type, json.dumps(metadata), datetime.now(timezone.utc).isoformat()),
            )
            await db.commit()
    except Exception as e:
        logger.warning("store_message failed: %s", e)


async def get_session(session_id: str) -> dict | None:
    try:
        async with aiosqlite.connect(_get_db_path()) as db:
            db.row_factory = aiosqlite.Row
            cursor = await db.execute("SELECT * FROM sessions WHERE id = ?", (session_id,))
            row = await cursor.fetchone()
            if row:
                return dict(row)
    except Exception as e:
        logger.warning("get_session failed: %s", e)
    return None


async def get_session_messages(session_id: str) -> list[dict]:
    try:
        async with aiosqlite.connect(_get_db_path()) as db:
            db.row_factory = aiosqlite.Row
            cursor = await db.execute(
                "SELECT * FROM messages WHERE session_id = ? ORDER BY id ASC",
                (session_id,),
            )
            rows = await cursor.fetchall()
            return [dict(r) for r in rows]
    except Exception as e:
        logger.warning("get_session_messages failed: %s", e)
    return []


async def store_trial_result(batch_id: str, result: dict):
    try:
        async with aiosqlite.connect(_get_db_path()) as db:
            await db.execute(
                "INSERT INTO trial_results (batch_id, trial_id, scenario, method, agreement, rounds, utilities_json, proposal_json, pareto_json, fairness_json, created_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    batch_id,
                    result.get("trial_id", 0),
                    result.get("scenario_name", ""),
                    result.get("method", ""),
                    1 if result.get("agreement_reached") else 0,
                    result.get("rounds_taken", 0),
                    json.dumps(result.get("per_agent_utilities", {})),
                    json.dumps(result.get("final_proposal", {})),
                    json.dumps(result.get("pareto", {})),
                    json.dumps(result.get("fairness", {})),
                    datetime.now(timezone.utc).isoformat(),
                ),
            )
            await db.commit()
    except Exception as e:
        logger.warning("store_trial_result failed: %s", e)

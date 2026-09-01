from __future__ import annotations
import json
import logging
from pathlib import Path
import chromadb
from src.config import settings

logger = logging.getLogger(__name__)

_client: chromadb.PersistentClient | None = None
_collection = None


def get_collection():
    global _client, _collection
    if _collection is None:
        path = str(settings.chroma_dir)
        Path(path).mkdir(parents=True, exist_ok=True)
        _client = chromadb.PersistentClient(path=path)
        _collection = _client.get_or_create_collection(
            name="negotiation_history",
            metadata={"hnsw:space": "cosine"},
        )
    return _collection


def store_negotiation_outcome(
    session_id: str,
    scenario_name: str,
    outcome: dict,
    profiles: list[dict],
    issues: list[dict],
):
    collection = get_collection()
    doc = json.dumps({
        "scenario": scenario_name,
        "outcome": outcome,
        "issues": issues,
        "agent_count": len(profiles),
    })
    metadata = {
        "scenario": scenario_name,
        "agreement": str(outcome.get("agreement_reached", False)),
        "rounds": outcome.get("rounds_taken", 0),
        "agent_count": len(profiles),
    }
    collection.add(
        documents=[doc],
        metadatas=[metadata],
        ids=[session_id],
    )
    logger.info("stored outcome for session %s", session_id)


def retrieve_similar_outcomes(
    scenario_name: str,
    n_results: int = 5,
) -> list[dict]:
    collection = get_collection()
    try:
        results = collection.query(
            query_texts=[f"negotiation outcome for {scenario_name}"],
            n_results=n_results,
            where={"scenario": scenario_name},
        )
        outcomes = []
        for doc in results.get("documents", [[]])[0]:
            try:
                outcomes.append(json.loads(doc))
            except json.JSONDecodeError:
                continue
        return outcomes
    except Exception as e:
        logger.warning("retrieval failed: %s", e)
        return []

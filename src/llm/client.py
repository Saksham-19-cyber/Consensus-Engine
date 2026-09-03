from __future__ import annotations
import json
import logging
import time
from typing import TypeVar, Type

from groq import Groq, APIError, RateLimitError, APIConnectionError
from pydantic import BaseModel, ValidationError
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
)

from src.config import settings

logger = logging.getLogger(__name__)
T = TypeVar("T", bound=BaseModel)

_client: Groq | None = None


def get_client() -> Groq:
    global _client
    if _client is None:
        _client = Groq(api_key=settings.groq_api_key)
    return _client


def _pydantic_to_json_schema(model_class: Type[BaseModel]) -> dict:
    schema = model_class.model_json_schema()

    def strip_extras(s: dict) -> dict:
        s.pop("title", None)
        s.pop("description", None)
        s.pop("$defs", None)
        if "properties" in s:
            for prop in s["properties"].values():
                strip_extras(prop)
                if "anyOf" in prop:
                    for variant in prop["anyOf"]:
                        strip_extras(variant)
                if "items" in prop and isinstance(prop["items"], dict):
                    strip_extras(prop["items"])
                if "additionalProperties" in prop and isinstance(prop["additionalProperties"], dict):
                    strip_extras(prop["additionalProperties"])
        if "items" in s and isinstance(s["items"], dict):
            strip_extras(s["items"])
        return s

    defs = schema.pop("$defs", {})

    def resolve_refs(obj):
        if isinstance(obj, dict):
            if "$ref" in obj:
                ref_name = obj["$ref"].split("/")[-1]
                if ref_name in defs:
                    resolved = defs[ref_name].copy()
                    resolved.pop("title", None)
                    resolved.pop("description", None)
                    return resolve_refs(resolved)
            return {k: resolve_refs(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [resolve_refs(item) for item in obj]
        return obj

    schema = resolve_refs(schema)
    schema = strip_extras(schema)
    schema["additionalProperties"] = False
    return schema


@retry(
    retry=retry_if_exception_type((RateLimitError, APIConnectionError)),
    wait=wait_exponential(multiplier=2, min=2, max=30),
    stop=stop_after_attempt(5),
)
def _call_groq(
    messages: list[dict],
    model: str,
    temperature: float,
    response_format: dict | None = None,
) -> str:
    client = get_client()
    kwargs = {
        "messages": messages,
        "model": model,
        "temperature": temperature,
        "max_tokens": 4096,
    }
    if response_format:
        kwargs["response_format"] = response_format

    start = time.time()
    completion = client.chat.completions.create(**kwargs)
    elapsed = time.time() - start

    content = completion.choices[0].message.content or ""
    logger.debug(
        "groq call model=%s tokens_in=%s tokens_out=%s elapsed=%.2fs",
        model,
        getattr(completion.usage, "prompt_tokens", "?"),
        getattr(completion.usage, "completion_tokens", "?"),
        elapsed,
    )
    return content


def structured_completion(
    messages: list[dict],
    response_model: Type[T],
    model: str | None = None,
    temperature: float | None = None,
    max_retries: int | None = None,
) -> T:
    model = model or settings.negotiator_model
    temperature = temperature if temperature is not None else settings.temperature
    max_retries = max_retries if max_retries is not None else settings.max_retries

    schema = _pydantic_to_json_schema(response_model)
    response_format = {
        "type": "json_object",
    }

    schema_instruction = (
        f"\n\nYou MUST respond with valid JSON matching this schema:\n"
        f"```json\n{json.dumps(schema, indent=2)}\n```"
    )

    patched_messages = list(messages)
    if patched_messages and patched_messages[0]["role"] == "system":
        patched_messages[0] = {
            **patched_messages[0],
            "content": patched_messages[0]["content"] + schema_instruction,
        }
    else:
        patched_messages.insert(0, {"role": "system", "content": schema_instruction})

    last_error = None
    for attempt in range(max_retries):
        try:
            raw = _call_groq(patched_messages, model, temperature, response_format)
            cleaned_raw = raw.strip()
            if cleaned_raw.startswith("```"):
                lines = cleaned_raw.splitlines()
                if lines and lines[0].startswith("```"):
                    lines = lines[1:]
                if lines and lines[-1].strip() == "```":
                    lines = lines[:-1]
                cleaned_raw = "\n".join(lines).strip()
            parsed = json.loads(cleaned_raw)
            result = response_model.model_validate(parsed)
            return result
        except (json.JSONDecodeError, ValidationError) as e:
            last_error = e
            logger.warning("attempt %d/%d failed: %s", attempt + 1, max_retries, e)
            error_msg = f"Your previous response was invalid: {e}. Respond ONLY with raw JSON object matching the schema. No markdown, no backticks."
            patched_messages.append({"role": "assistant", "content": raw if "raw" in dir() else ""})
            patched_messages.append({"role": "user", "content": error_msg})
        except APIError as e:
            last_error = e
            logger.error("groq api error: %s", e)
            if "json_validate_failed" in str(e) or "response_format" in str(e):
                response_format = None
            if attempt == max_retries - 1:
                raise

    raise ValueError(f"Failed after {max_retries} retries: {last_error}")


def plain_completion(
    messages: list[dict],
    model: str | None = None,
    temperature: float | None = None,
) -> str:
    model = model or settings.negotiator_model
    temperature = temperature if temperature is not None else settings.temperature
    return _call_groq(messages, model, temperature)

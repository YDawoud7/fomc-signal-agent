"""LLM providers with structured output, caching and cost tracking.

Three backends share one interface, ``complete_json(system, user, schema)``:

* ``AnthropicProvider``: Messages API with a forced tool call, so the reply
  is guaranteed to match the JSON schema.
* ``OpenAIProvider``: Chat Completions with ``response_format=json_schema``.
* ``ClaudeCodeProvider``: shells out to the ``claude`` CLI in print mode,
  which lets the pipeline run on a Claude subscription without an API key.
  Output is parsed and validated, since the CLI can't enforce a schema.

Every call is cached on disk by a hash of (provider, model, prompt version,
prompt, run index), and token usage and estimated cost are logged, so
re-running the pipeline is free and every score can be traced back to the
exact prompt and model that produced it.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess
import threading
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

# USD per million tokens (input, output). Rough list prices; edit as needed.
PRICES = {
    "claude-haiku-5-5": (1.0, 5.0),
    "claude-sonnet-5-5": (3.0, 15.0),
    "gpt-4o-mini": (0.15, 0.60),
    "gpt-4o": (2.5, 10.0),
}


class LLMError(RuntimeError):
    pass


_LOCK = threading.Lock()


@dataclass
class Usage:
    calls: int = 0
    cached: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    cost_usd: float = 0.0

    def add(self, model: str, inp: int, out: int) -> None:
        p_in, p_out = PRICES.get(model, (0.0, 0.0))
        with _LOCK:
            self.calls += 1
            self.input_tokens += inp
            self.output_tokens += out
            self.cost_usd += (inp * p_in + out * p_out) / 1e6

    def hit(self) -> None:
        with _LOCK:
            self.cached += 1


def validate(obj: Any, schema: dict) -> list[str]:
    """Minimal JSON-schema check (types, required keys, enums, numeric bounds, arrays)."""
    errors: list[str] = []

    def check(value, sch, path):
        t = sch.get("type")
        types = {"object": dict, "array": list, "string": str, "integer": int, "number": (int, float), "boolean": bool}
        if t and not isinstance(value, types[t]) or (t in ("integer", "number") and isinstance(value, bool)):
            errors.append(f"{path}: expected {t}")
            return
        if "enum" in sch and value not in sch["enum"]:
            errors.append(f"{path}: {value!r} not in {sch['enum']}")
        if t in ("integer", "number"):
            if "minimum" in sch and value < sch["minimum"]:
                errors.append(f"{path}: below minimum {sch['minimum']}")
            if "maximum" in sch and value > sch["maximum"]:
                errors.append(f"{path}: above maximum {sch['maximum']}")
        if t == "object":
            for k in sch.get("required", []):
                if k not in value:
                    errors.append(f"{path}.{k}: missing")
            for k, sub in sch.get("properties", {}).items():
                if k in value:
                    check(value[k], sub, f"{path}.{k}")
        if t == "array":
            if "maxItems" in sch and len(value) > sch["maxItems"]:
                errors.append(f"{path}: more than {sch['maxItems']} items")
            if "minItems" in sch and len(value) < sch["minItems"]:
                errors.append(f"{path}: fewer than {sch['minItems']} items")
            for i, item in enumerate(value):
                check(item, sch.get("items", {}), f"{path}[{i}]")

    check(obj, schema, "$")
    return errors


def extract_json(text: str) -> Any:
    """Parse the first JSON object in a model reply (tolerates code fences)."""
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if not m:
        raise LLMError("no JSON object in reply")
    return json.loads(m.group(0))


@dataclass
class Provider:
    model: str
    cache_dir: Path = Path("data/llm_cache")
    prompt_version: str = "v1"
    usage: Usage = field(default_factory=Usage)
    name: str = "base"
    max_retries: int = 3

    def _key(self, system: str, user: str, run: int) -> str:
        h = hashlib.sha256(json.dumps([self.name, self.model, self.prompt_version, system, user, run]).encode())
        return h.hexdigest()[:24]

    def complete_json(self, system: str, user: str, schema: dict, run: int = 0, temperature: float = 0.0) -> dict:
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        path = self.cache_dir / f"{self._key(system, user, run)}.json"
        if path.exists():
            self.usage.hit()
            return json.loads(path.read_text())["output"]

        last_err = None
        feedback = ""
        for attempt in range(self.max_retries):
            try:
                out, inp_tok, out_tok = self._call(system, user + feedback, schema, temperature if run == 0 else 1.0)
                self.usage.add(self.model, inp_tok, out_tok)
                errs = validate(out, schema)
                if errs:
                    raise LLMError("schema: " + "; ".join(errs[:5]))
                record = {"provider": self.name, "model": self.model, "prompt_version": self.prompt_version,
                          "run": run, "input_tokens": inp_tok, "output_tokens": out_tok, "output": out,
                          "time": time.strftime("%Y-%m-%dT%H:%M:%S")}
                path.write_text(json.dumps(record))
                return out
            except (LLMError, json.JSONDecodeError) as e:
                last_err = e
                feedback = f"\n\nYour previous reply was invalid ({e}). Reply again following the schema exactly."
            except urllib.error.HTTPError as e:
                last_err = e
                if e.code in (429, 500, 502, 503, 529):
                    time.sleep(2 ** attempt * 5)
                else:
                    raise
        raise LLMError(f"failed after {self.max_retries} attempts: {last_err}")

    def _call(self, system: str, user: str, schema: dict, temperature: float) -> tuple[dict, int, int]:
        raise NotImplementedError


def _post(url: str, headers: dict, body: dict, timeout: float = 120) -> dict:
    req = urllib.request.Request(url, data=json.dumps(body).encode(), headers={**headers, "content-type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode())


@dataclass
class AnthropicProvider(Provider):
    model: str = "claude-haiku-5-5"
    name: str = "anthropic"

    def _call(self, system, user, schema, temperature):
        key = os.environ.get("ANTHROPIC_API_KEY")
        if not key:
            raise LLMError("ANTHROPIC_API_KEY is not set")
        body = {
            "model": self.model, "max_tokens": 2048, "temperature": temperature, "system": system,
            "messages": [{"role": "user", "content": user}],
            "tools": [{"name": "record", "description": "Record the structured answer.", "input_schema": schema}],
            "tool_choice": {"type": "tool", "name": "record"},
        }
        resp = _post("https://api.anthropic.com/v1/messages", {"x-api-key": key, "anthropic-version": "2023-06-01"}, body)
        block = next((b for b in resp["content"] if b.get("type") == "tool_use"), None)
        if block is None:
            raise LLMError("no tool_use block in response")
        u = resp.get("usage", {})
        return block["input"], u.get("input_tokens", 0), u.get("output_tokens", 0)


@dataclass
class OpenAIProvider(Provider):
    model: str = "gpt-4o-mini"
    name: str = "openai"

    def _call(self, system, user, schema, temperature):
        key = os.environ.get("OPENAI_API_KEY")
        if not key:
            raise LLMError("OPENAI_API_KEY is not set")
        body = {
            "model": self.model, "temperature": temperature,
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
            "response_format": {"type": "json_schema", "json_schema": {"name": "record", "schema": schema}},
        }
        resp = _post("https://api.openai.com/v1/chat/completions", {"authorization": f"Bearer {key}"}, body)
        u = resp.get("usage", {})
        return json.loads(resp["choices"][0]["message"]["content"]), u.get("prompt_tokens", 0), u.get("completion_tokens", 0)


@dataclass
class ClaudeCodeProvider(Provider):
    """Uses the local ``claude`` CLI (Claude Code) in non-interactive print mode.

    Runs with tools disabled, no MCP servers and no saved session, and passes
    the schema through ``--json-schema`` so the CLI returns validated
    structured output. Lets the pipeline run on a Claude subscription without
    an API key.
    """

    model: str = "haiku"
    name: str = "claude-code"

    def _call(self, system, user, schema, temperature):
        cmd = ["claude", "-p", user, "--system-prompt", system, "--json-schema", json.dumps(schema),
               "--output-format", "json", "--model", self.model, "--tools", "", "--strict-mcp-config",
               "--no-session-persistence"]
        env = {k: v for k, v in os.environ.items() if k != "CLAUDE_CODE_SESSION_ID"}
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=600, env=env, cwd="/tmp" if os.path.isdir("/tmp") else None)
        if proc.returncode != 0:
            raise LLMError(f"claude CLI failed: {(proc.stderr or proc.stdout)[:300]}")
        envelope = json.loads(proc.stdout)
        if envelope.get("is_error"):
            raise LLMError(f"claude CLI error: {str(envelope.get('result'))[:300]}")
        u = envelope.get("usage", {})
        inp = u.get("input_tokens", 0) + u.get("cache_creation_input_tokens", 0) + u.get("cache_read_input_tokens", 0)
        with _LOCK:
            self.usage.cost_usd += float(envelope.get("total_cost_usd") or 0.0)
        out = envelope.get("structured_output")
        if out is None:
            out = extract_json(envelope.get("result", ""))
        return out, inp, u.get("output_tokens", 0)


def make_provider(name: str, model: str | None = None, cache_dir: str | Path = "data/llm_cache") -> Provider:
    classes = {"anthropic": AnthropicProvider, "openai": OpenAIProvider, "claude-code": ClaudeCodeProvider}
    if name not in classes:
        raise ValueError(f"unknown provider {name!r}; choose from {sorted(classes)}")
    kwargs = {"cache_dir": Path(cache_dir)}
    if model:
        kwargs["model"] = model
    return classes[name](**kwargs)

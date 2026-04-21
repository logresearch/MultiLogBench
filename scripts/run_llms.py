#!/usr/bin/env python3
from __future__ import annotations

import argparse
import concurrent.futures
import json
import logging
import math
import os
import re
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

try:
    from tqdm import tqdm
except ImportError:
    class tqdm:
        def __init__(self, total: int, desc: str) -> None:
            self.total = total
            self.desc = desc

        def set_postfix_str(self, value: str, refresh: bool = False) -> None:
            return None

        def update(self, count: int = 1) -> None:
            return None

        def close(self) -> None:
            return None


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATASET = REPO_ROOT / "data" / "revision_history" / "accepted_pool.jsonl"
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "runs"
DEFAULT_GENERATIONS_NAME = "generations.jsonl"
DEFAULT_GOLD_SUBSET_NAME = "selected_gold.jsonl"
DEFAULT_OPENAI_PROVIDER = "openai-compatible"
DEFAULT_GOOGLE_GENAI_PROVIDER = "google-genai"
DEFAULT_GOOGLE_GENAI_API_VERSION = "v1alpha"
DEFAULT_GOOGLE_GENAI_API_KEY_ENVS = ("GEMINI_API_KEY", "GOOGLE_API_KEY")
LOG = logging.getLogger("run_llms")


DEFAULT_RETRIEVAL_DATASET = REPO_ROOT / "data" / "repository_snapshot" / "splits" / "valid.jsonl"


SYSTEM_PROMPT = """
You are a coding assistant that helps developers add appropriate logging statements to their code. The following function input misses a logging statement, please help me add a logging statement to the function to the appropriate place.
"""


GOOGLE_GENAI_MODEL_ALIASES = {
    "gemini-3.1": "gemini-3-pro-preview",
    "gemini-3.1-pro": "gemini-3-pro-preview",
    "gemini-3.1-pro-preview": "gemini-3-pro-preview",
    "gemini-3.1-flash": "gemini-3-flash-preview",
    "gemini-3.1-flash-preview": "gemini-3-flash-preview",
}


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass(frozen=True)
class RunnerConfig:
    dataset_path: Path
    retrieval_dataset_path: Path
    output_root: Path
    run_name: str
    model: str
    provider: str
    strategy: str
    temperature: float
    max_completion_tokens: int
    timeout: float
    max_workers: int
    abort_after_consecutive_non_200: int
    api_key_env: str
    base_url: str | None
    resume: bool
    overwrite: bool
    dry_run: bool


@dataclass(frozen=True)
class RunArtifacts:
    run_dir: Path
    generations_path: Path
    gold_subset_path: Path
    config_path: Path


@dataclass(frozen=True)
class PromptContext:
    user_prompt: str
    example_index: int | None = None
    example_repo_name: str | None = None


@dataclass(frozen=True)
class OpenAICompatibleInvocationContext:
    client: Any
    resolved_model_name: str
    temperature: float
    max_completion_tokens: int


@dataclass(frozen=True)
class GoogleGenAIInvocationContext:
    client: Any
    generation_config: Any
    resolved_model_name: str


class ModelRequestError(RuntimeError):
    def __init__(self, details: str, *, status_code: int | None) -> None:
        super().__init__(details)
        self.status_code = status_code


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run LLM-based logging generation on the local multilingual benchmark and "
            "store raw full-code generations for later post-processing."
        ),
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        default=DEFAULT_DATASET,
        help=(
            "JSONL benchmark split to run. Defaults to data/revision_history/accepted_pool.jsonl."
        ),
    )
    parser.add_argument(
        "--model",
        type=str,
        default=os.getenv("LLM_MODEL"),
        help="Model identifier passed to the selected provider. For google-genai, local Gemini aliases like gemini-3.1-pro are also supported.",
    )
    parser.add_argument(
        "--provider",
        choices=(DEFAULT_OPENAI_PROVIDER, DEFAULT_GOOGLE_GENAI_PROVIDER),
        default=os.getenv("LLM_PROVIDER") or DEFAULT_OPENAI_PROVIDER,
        help=(
            "Inference provider. openai-compatible uses the OpenAI Python client; "
            "google-genai uses the Google Gen AI SDK."
        ),
    )
    parser.add_argument(
        "--base-url",
        type=str,
        default=os.getenv("OPENAI_BASE_URL") or os.getenv("LLM_BASE_URL"),
        help=(
            "Optional base URL for the openai-compatible provider. Ignored by google-genai."
        ),
    )
    parser.add_argument(
        "--api-key-env",
        type=str,
        default="OPENAI_API_KEY",
        help="Environment variable that stores the API key. Defaults to OPENAI_API_KEY.",
    )
    parser.add_argument(
        "--strategy",
        choices=("base", "rag"),
        default="rag",
        help="Prompting strategy. Defaults to rag to match the original template's main experiment.",
    )
    parser.add_argument(
        "--retrieval-dataset",
        type=Path,
        default=DEFAULT_RETRIEVAL_DATASET,
        help="JSONL split used to retrieve RAG exemplars. Defaults to data/repository_snapshot/splits/valid.jsonl.",
    )
    parser.add_argument(
        "--languages",
        type=str,
        default="",
        help="Optional comma-separated language filter, for example java,python,go.",
    )
    parser.add_argument(
        "--sample-indexes",
        type=str,
        default="",
        help="Optional comma-separated sample indexes for quick debugging.",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Optional cap for smoke tests.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Sampling temperature. Defaults to 0.0 to match the shared baseline protocol.",
    )
    parser.add_argument(
        "--max-completion-tokens",
        type=int,
        default=2048,
        help="Maximum completion tokens for each request.",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=180.0,
        help="Per-request timeout in seconds. Defaults to 180s to better tolerate slower providers like Gemini.",
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=50,
        help="Maximum number of concurrent sample requests. Defaults to 50 to match the previous runner behavior.",
    )
    parser.add_argument(
        "--abort-after-consecutive-non-200",
        type=int,
        default=2,
        help=(
            "Abort the entire script after this many consecutive request-level failures that did not return "
            "HTTP 200. Set to 0 to disable."
        ),
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT,
        help="Root directory for run artifacts. Each run is stored under runs/<model>/<run_name>/.",
    )
    parser.add_argument(
        "--run-name",
        type=str,
        default=None,
        help="Optional explicit run name. Defaults to a dataset-aware name like historical_test.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume an existing run by keeping completed samples and rerunning missing or failed samples.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Delete existing artifacts in the target run directory before writing a new run.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Skip model requests and emit placeholder records so the local I/O pipeline can be smoke-tested.",
    )
    parser.add_argument(
        "--verbosity",
        type=str,
        default="INFO",
        help="Python logging verbosity level.",
    )
    return parser.parse_args()


def sanitize_component(text: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", text.strip())
    return cleaned or "unnamed"


def infer_run_name(dataset_path: Path) -> str:
    path_text = dataset_path.as_posix().casefold()
    if dataset_path.name == "accepted_pool.jsonl" and "transformed" in path_text:
        base = "transformed_historical_test"
    elif dataset_path.name == "accepted_pool.jsonl" and "historical" in path_text:
        base = "historical_test"
    elif dataset_path.name == "test.jsonl":
        base = "core_test"
    elif dataset_path.name == "valid.jsonl":
        base = "core_valid"
    elif dataset_path.name == "train.jsonl":
        base = "core_train"
    else:
        base = dataset_path.stem

    return base


def parse_csv_items(raw: str) -> list[str]:
    return [item.strip() for item in raw.split(",") if item.strip()]


def parse_index_filter(raw: str) -> set[int]:
    indexes: set[int] = set()
    for item in parse_csv_items(raw):
        indexes.add(int(item))
    return indexes


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            line = raw_line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON at {path}:{line_number}: {exc}") from exc
    return rows


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def load_selected_samples(
    dataset_path: Path,
    *,
    languages: set[str] | None,
    sample_indexes: set[int] | None,
    max_samples: int | None,
) -> list[dict[str, Any]]:
    rows = read_jsonl(dataset_path)
    selected: list[dict[str, Any]] = []
    for row in rows:
        language = str(row.get("language", "")).strip().casefold()
        index = int(row["index"])
        if languages and language not in languages:
            continue
        if sample_indexes and index not in sample_indexes:
            continue
        selected.append(row)
        if max_samples is not None and len(selected) >= max_samples:
            break
    return selected


def normalize_code_block(text: str) -> str:
    return str(text).replace("\r\n", "\n").strip()


def build_base_user_prompt(sample: dict[str, Any]) -> str:
    code = normalize_code_block(sample.get("input_initial", ""))
    return f"code: {code}"


def build_example_output_json(example: dict[str, Any]) -> str:
    revised_code = normalize_code_block(example.get("function_content", ""))
    return json.dumps(
        {"revised_code": revised_code},
        ensure_ascii=False,
        indent=2,
    )


def build_rag_user_prompt(sample: dict[str, Any], example: dict[str, Any]) -> str:
    code = normalize_code_block(sample.get("input_initial", ""))
    example_input = normalize_code_block(example.get("input_initial", ""))
    example_output = build_example_output_json(example)
    return (
        f"example input :\n{example_input}\n"
        f"example output :\n{example_output}\n"
        "Return the answer for the next code in the same JSON format and output only that JSON object.\n"
        f"code: {code}"
    )


def tokenize_for_retrieval(text: str) -> list[str]:
    tokens = re.findall(r"\b\w+\b", str(text).lower())
    return tokens if tokens else [""]


class BM25Retriever:
    def __init__(self, rows: list[dict[str, Any]]) -> None:
        self.rows = rows
        self.documents = [tokenize_for_retrieval(row.get("input_initial", "")) for row in rows]
        self.avgdl = (
            sum(len(document) for document in self.documents) / len(self.documents)
            if self.documents
            else 0.0
        )
        self.k1 = 1.5
        self.b = 0.75
        self.doc_freqs: list[dict[str, int]] = []
        self.term_doc_counts: dict[str, int] = {}
        for document in self.documents:
            doc_freq: dict[str, int] = {}
            for token in document:
                doc_freq[token] = doc_freq.get(token, 0) + 1
            self.doc_freqs.append(doc_freq)
            for token in doc_freq:
                self.term_doc_counts[token] = self.term_doc_counts.get(token, 0) + 1

    def score(self, query: str, *, exclude_index: int | None = None) -> dict[str, Any] | None:
        if not self.rows:
            return None
        query_tokens = tokenize_for_retrieval(query)
        total_docs = len(self.rows)
        best_row: dict[str, Any] | None = None
        best_score = float("-inf")
        for row, doc_freq, document in zip(self.rows, self.doc_freqs, self.documents):
            if exclude_index is not None and int(row.get("index", -1)) == exclude_index:
                continue
            document_length = len(document)
            score = 0.0
            for token in query_tokens:
                term_frequency = doc_freq.get(token, 0)
                if term_frequency <= 0:
                    continue
                doc_count = self.term_doc_counts.get(token, 0)
                idf = math.log(1 + (total_docs - doc_count + 0.5) / (doc_count + 0.5))
                numerator = term_frequency * (self.k1 + 1)
                denominator = term_frequency + self.k1 * (
                    1 - self.b + self.b * document_length / max(self.avgdl, 1.0)
                )
                score += idf * numerator / denominator
            if score > best_score:
                best_score = score
                best_row = row
        return best_row


def extract_text_content(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, dict):
                text = item.get("text")
                if text is not None:
                    parts.append(str(text))
            else:
                text = getattr(item, "text", None)
                if text is not None:
                    parts.append(str(text))
        return "".join(parts)
    return str(content)


def strip_outer_code_fence(text: str) -> str:
    stripped = text.strip()
    fence_match = re.fullmatch(r"```(?:[A-Za-z0-9_+-]+)?\s*(.*?)\s*```", stripped, flags=re.DOTALL)
    if fence_match:
        return fence_match.group(1).strip()
    return stripped


def extract_first_json_object(text: str) -> dict[str, Any]:
    stripped = text.strip()
    if not stripped:
        raise ValueError("Model response is empty.")

    try:
        parsed = json.loads(stripped)
    except json.JSONDecodeError:
        parsed = None
    if isinstance(parsed, dict):
        return parsed
    if isinstance(parsed, list) and parsed and isinstance(parsed[0], dict):
        return parsed[0]

    decoder = json.JSONDecoder()
    for match in re.finditer(r"\{", stripped):
        candidate = stripped[match.start() :]
        try:
            parsed, _ = decoder.raw_decode(candidate)
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, dict):
            return parsed
    raise ValueError("Could not find a JSON object in the model response.")


def extract_revised_code(text: str) -> str:
    try:
        payload = extract_first_json_object(text)
    except ValueError:
        return strip_outer_code_fence(text)

    for key in ("revised_code", "revisedCode", "code", "updated_code"):
        value = payload.get(key)
        if isinstance(value, str) and value.strip():
            return strip_outer_code_fence(value)

    string_values = [value for value in payload.values() if isinstance(value, str) and value.strip()]
    if len(string_values) == 1:
        return strip_outer_code_fence(string_values[0])
    raise ValueError("Model JSON response does not contain revised_code.")


def build_success_record(index: int, revised_code: str) -> dict[str, Any]:
    return {
        "index": index,
        "status": "ok",
        "revised_code": revised_code,
    }


def build_error_record(index: int, status: str, error: str) -> dict[str, Any]:
    return {
        "index": index,
        "status": status,
        "error": error,
        "revised_code": "",
    }


def redact_sensitive_text(text: str) -> str:
    key_prefix = "sk" + "-"
    bearer_prefix = "Bear" + "er"
    redacted = re.sub(rf"\b{re.escape(key_prefix)}[A-Za-z0-9._-]+\b", f"{key_prefix}***REDACTED***", str(text))
    redacted = re.sub(rf"({bearer_prefix}\s+)[A-Za-z0-9._-]+", r"\1***REDACTED***", redacted, flags=re.IGNORECASE)
    return redacted


def truncate_text(text: str, limit: int = 2000) -> str:
    if len(text) <= limit:
        return text
    return f"{text[: limit - 15]}...<truncated>"


def format_error_payload(payload: Any) -> str | None:
    if payload is None:
        return None
    if isinstance(payload, (dict, list)):
        rendered = json.dumps(payload, ensure_ascii=False, sort_keys=True)
    else:
        rendered = str(payload).strip()
    if not rendered:
        return None
    return truncate_text(redact_sensitive_text(rendered))


def extract_response_body(response: Any) -> str | None:
    if response is None:
        return None
    try:
        payload = response.json()
    except Exception:
        payload = None
    rendered = format_error_payload(payload)
    if rendered:
        return rendered
    try:
        text = response.text
    except Exception:
        text = None
    return format_error_payload(text)


def extract_exception_status_code(exc: BaseException) -> int | None:
    seen_exceptions: set[int] = set()
    current: BaseException | None = exc
    level = 0

    while current is not None and id(current) not in seen_exceptions and level < 5:
        seen_exceptions.add(id(current))
        for candidate in (
            getattr(current, "status_code", None),
            getattr(getattr(current, "response", None), "status_code", None),
        ):
            if candidate is None:
                continue
            try:
                return int(candidate)
            except (TypeError, ValueError):
                continue
        current = current.__cause__ or current.__context__
        level += 1
    return None


def format_exception_details(exc: BaseException) -> str:
    details: list[str] = []
    seen_sections: set[str] = set()
    seen_exceptions: set[int] = set()
    current: BaseException | None = exc
    level = 0

    while current is not None and id(current) not in seen_exceptions and level < 5:
        seen_exceptions.add(id(current))
        prefix = "exception" if level == 0 else f"cause_{level}"
        summary = truncate_text(redact_sensitive_text(f"{type(current).__name__}: {current}"))
        section = f"{prefix}={summary}"
        if section not in seen_sections:
            details.append(section)
            seen_sections.add(section)

        status_code = getattr(current, "status_code", None)
        if status_code is not None:
            section = f"{prefix}_status_code={status_code}"
            if section not in seen_sections:
                details.append(section)
                seen_sections.add(section)

        body = format_error_payload(getattr(current, "body", None))
        if body:
            section = f"{prefix}_body={body}"
            if section not in seen_sections:
                details.append(section)
                seen_sections.add(section)

        response = getattr(current, "response", None)
        if response is not None:
            response_status = getattr(response, "status_code", None)
            if response_status is not None:
                section = f"{prefix}_response_status={response_status}"
                if section not in seen_sections:
                    details.append(section)
                    seen_sections.add(section)

            request = getattr(response, "request", None)
            request_method = getattr(request, "method", None)
            request_url = getattr(request, "url", None) or getattr(response, "url", None)
            if request_method or request_url:
                section = f"{prefix}_request={request_method or 'UNKNOWN'} {request_url}"
                section = truncate_text(redact_sensitive_text(section))
                if section not in seen_sections:
                    details.append(section)
                    seen_sections.add(section)

            headers = getattr(response, "headers", None)
            if headers:
                request_id = headers.get("x-request-id") or headers.get("request-id") or headers.get("cf-ray")
                if request_id:
                    section = f"{prefix}_request_id={request_id}"
                    if section not in seen_sections:
                        details.append(section)
                        seen_sections.add(section)

            response_body = extract_response_body(response)
            if response_body:
                section = f"{prefix}_response_body={response_body}"
                if section not in seen_sections:
                    details.append(section)
                    seen_sections.add(section)

        current = current.__cause__ or current.__context__
        level += 1

    return " | ".join(details)


def sample_progress_label(sample: dict[str, Any], prompt_context: PromptContext) -> str:
    parts = [
        f"idx={int(sample['index'])}",
        f"lang={str(sample.get('language', '')).strip() or 'unknown'}",
    ]
    repo_name = str(sample.get("repo_name", "")).strip()
    if repo_name:
        parts.append(f"repo={repo_name}")
    if prompt_context.example_index is not None:
        parts.append(f"rag={prompt_context.example_index}")
    return " ".join(parts)


def basic_sample_label(sample: dict[str, Any]) -> str:
    parts = [
        f"idx={int(sample['index'])}",
        f"lang={str(sample.get('language', '')).strip() or 'unknown'}",
    ]
    repo_name = str(sample.get("repo_name", "")).strip()
    if repo_name:
        parts.append(f"repo={repo_name}")
    return " ".join(parts)


def ensure_run_dir(config: RunnerConfig) -> RunArtifacts:
    run_dir = config.output_root / sanitize_component(config.model) / sanitize_component(config.run_name)
    generations_path = run_dir / DEFAULT_GENERATIONS_NAME
    gold_subset_path = run_dir / DEFAULT_GOLD_SUBSET_NAME
    config_path = run_dir / "run_config.json"

    if config.resume and config.overwrite:
        raise SystemExit("Choose either --resume or --overwrite, not both.")

    if config.overwrite and run_dir.exists():
        for path in (generations_path, gold_subset_path, config_path):
            if path.exists():
                path.unlink()

    run_dir.mkdir(parents=True, exist_ok=True)
    return RunArtifacts(
        run_dir=run_dir,
        generations_path=generations_path,
        gold_subset_path=gold_subset_path,
        config_path=config_path,
    )


def verify_subset_for_resume(artifacts: RunArtifacts, samples: list[dict[str, Any]]) -> None:
    if not artifacts.gold_subset_path.exists():
        return
    existing = read_jsonl(artifacts.gold_subset_path)
    existing_indexes = [int(row["index"]) for row in existing]
    current_indexes = [int(row["index"]) for row in samples]
    current_index_set = set(current_indexes)
    if not set(existing_indexes).issubset(current_index_set):
        raise SystemExit(
            "Existing selected_gold.jsonl contains indexes that are not included in the current filtered sample set. "
            "Use a different --run-name, or rerun with --overwrite.",
        )


def should_auto_resume(artifacts: RunArtifacts, config: RunnerConfig) -> bool:
    if config.overwrite:
        return False
    return artifacts.generations_path.exists()


def existing_generation_indexes(path: Path) -> set[int]:
    if not path.exists():
        return set()
    indexes: set[int] = set()
    for row in read_jsonl(path):
        indexes.add(int(row["index"]))
    return indexes


def deduplicate_generation_rows(rows: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    latest_by_index: dict[int, dict[str, Any]] = {}
    ordered_indexes: list[int] = []
    for row in rows:
        index = int(row["index"])
        if index not in latest_by_index:
            ordered_indexes.append(index)
        latest_by_index[index] = row
    return [latest_by_index[index] for index in ordered_indexes]


def load_existing_generation_rows(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return deduplicate_generation_rows(read_jsonl(path))


def generation_row_status(row: dict[str, Any]) -> str:
    return str(row.get("status", "")).strip()


def select_non_ok_retry_indexes(
    rows: Iterable[dict[str, Any]],
    *,
    allowed_indexes: set[int] | None = None,
) -> set[int]:
    retry_indexes: set[int] = set()
    for row in rows:
        index = int(row["index"])
        if allowed_indexes is not None and index not in allowed_indexes:
            continue
        if generation_row_status(row) != "ok":
            retry_indexes.add(index)
    return retry_indexes


def api_key_env_candidates(config: RunnerConfig) -> list[str]:
    candidates: list[str] = []
    seen: set[str] = set()
    if config.provider == DEFAULT_GOOGLE_GENAI_PROVIDER and str(config.api_key_env).strip() == "OPENAI_API_KEY":
        raw_candidates = (*DEFAULT_GOOGLE_GENAI_API_KEY_ENVS, config.api_key_env)
    elif config.provider == DEFAULT_GOOGLE_GENAI_PROVIDER:
        raw_candidates = (config.api_key_env, *DEFAULT_GOOGLE_GENAI_API_KEY_ENVS)
    else:
        raw_candidates = (config.api_key_env,)

    for env_name in raw_candidates:
        normalized = str(env_name).strip()
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        candidates.append(normalized)
    return candidates


def resolve_api_key(config: RunnerConfig) -> str | None:
    for env_name in api_key_env_candidates(config):
        value = os.getenv(env_name)
        if value:
            return value
    return None


def resolve_google_genai_model_name(model_name: str) -> str:
    normalized = str(model_name).strip()
    return GOOGLE_GENAI_MODEL_ALIASES.get(normalized, normalized)


def google_genai_thinking_mode(resolved_model_name: str) -> tuple[str, str, int | None]:
    model_name = str(resolved_model_name).strip().casefold()
    if model_name.startswith("gemini-3-pro"):
        return (
            "best_effort_low",
            "Gemini 3.1 Pro does not support full thinking-off; using thinking_level='low' instead.",
            None,
        )
    if model_name.startswith("gemini-3-flash-lite") or model_name.startswith("gemini-3-flash"):
        return (
            "minimal",
            "Gemini 3 Flash models do not support guaranteed thinking-off; using thinking_level='minimal'.",
            None,
        )
    if model_name.startswith("gemini-2.5-flash") or model_name.startswith("gemini-2.5-flash-lite"):
        return (
            "disabled",
            "Disabled Gemini 2.5 Flash-family thinking with thinking_budget=0.",
            0,
        )
    if model_name.startswith("gemini-2.5-pro"):
        return (
            "best_effort_default",
            "Gemini 2.5 Pro does not support thinking-off; leaving thinking config unset.",
            None,
        )
    return (
        "best_effort_default",
        "Unknown Gemini family; leaving thinking config unset.",
        None,
    )


def normalize_openai_base_url(base_url: str | None) -> str | None:
    if base_url is None:
        return None
    normalized = base_url.strip().rstrip("/")
    return normalized or None


def build_openai_compatible_invocation_context(config: RunnerConfig) -> OpenAICompatibleInvocationContext:
    try:
        from openai import OpenAI
    except ImportError as exc:
        raise SystemExit(
            "openai is required for --provider openai-compatible. "
            "Install dependencies with `pip install -r requirements.txt`."
        ) from exc
    api_key = resolve_api_key(config)
    if not api_key:
        raise SystemExit(
            f"Environment variable {config.api_key_env} is not set. "
            "Set it before running the real LLM inference, or use --dry-run for a local smoke test.",
        )
    client_kwargs: dict[str, Any] = {
        "api_key": api_key,
        "timeout": config.timeout,
    }
    base_url = normalize_openai_base_url(config.base_url)
    if base_url is not None:
        client_kwargs["base_url"] = base_url
    return OpenAICompatibleInvocationContext(
        client=OpenAI(**client_kwargs),
        resolved_model_name=config.model,
        temperature=config.temperature,
        max_completion_tokens=config.max_completion_tokens,
    )


def build_google_genai_invocation_context(config: RunnerConfig) -> GoogleGenAIInvocationContext:
    try:
        from google import genai
        from google.genai import types
    except ImportError as exc:
        raise SystemExit(
            "google-genai is required for --provider google-genai. "
            "Install it with `pip install google-genai` before running this provider.",
        ) from exc

    api_key = resolve_api_key(config)
    if not api_key:
        raise SystemExit(
            "No Gemini API key was found. Set one of "
            f"{', '.join(api_key_env_candidates(config))} before running --provider google-genai.",
        )

    resolved_model_name = resolve_google_genai_model_name(config.model)
    thinking_mode, thinking_note, thinking_budget = google_genai_thinking_mode(resolved_model_name)
    client = genai.Client(
        api_key=api_key,
        http_options=types.HttpOptions(
            api_version=DEFAULT_GOOGLE_GENAI_API_VERSION,
            timeout=int(config.timeout * 1000),
        ),
    )
    thinking_config = None
    if thinking_mode == "best_effort_low":
        thinking_config = types.ThinkingConfig(thinking_level="low")
    elif thinking_mode == "minimal":
        thinking_config = types.ThinkingConfig(thinking_level="minimal")
    elif thinking_mode == "disabled" and thinking_budget is not None:
        thinking_config = types.ThinkingConfig(thinking_budget=thinking_budget)

    generation_config = types.GenerateContentConfig(
        system_instruction=SYSTEM_PROMPT.strip(),
        temperature=config.temperature,
        max_output_tokens=config.max_completion_tokens,
        response_mime_type="application/json",
        response_schema={
            "type": "OBJECT",
            "required": ["revised_code"],
            "properties": {
                "revised_code": {
                    "type": "STRING",
                },
            },
        },
        thinking_config=thinking_config,
    )
    LOG.info("%s model=%s", thinking_note, resolved_model_name)
    return GoogleGenAIInvocationContext(
        client=client,
        generation_config=generation_config,
        resolved_model_name=resolved_model_name,
    )


def build_invocation_context(
    config: RunnerConfig,
) -> OpenAICompatibleInvocationContext | GoogleGenAIInvocationContext:
    if config.provider == DEFAULT_GOOGLE_GENAI_PROVIDER:
        return build_google_genai_invocation_context(config)
    return build_openai_compatible_invocation_context(config)


def build_user_prompt(
    config: RunnerConfig,
    sample: dict[str, Any],
    retriever: BM25Retriever | None,
) -> str:
    return prepare_prompt_context(config, sample, retriever).user_prompt


def prepare_prompt_context(
    config: RunnerConfig,
    sample: dict[str, Any],
    retriever: BM25Retriever | None,
) -> PromptContext:
    if config.strategy == "base":
        return PromptContext(user_prompt=build_base_user_prompt(sample))
    if config.strategy == "rag":
        if retriever is None:
            raise ValueError("RAG strategy requires a retriever.")
        exclude_index = None
        if config.dataset_path == config.retrieval_dataset_path:
            exclude_index = int(sample.get("index", -1))
        example = retriever.score(str(sample.get("input_initial", "")), exclude_index=exclude_index)
        if example is None:
            raise ValueError("RAG strategy could not retrieve an exemplar.")
        return PromptContext(
            user_prompt=build_rag_user_prompt(sample, example),
            example_index=int(example.get("index", -1)) if example.get("index") is not None else None,
            example_repo_name=str(example.get("repo_name", "")).strip() or None,
        )
    raise ValueError(f"Unsupported strategy: {config.strategy}")


def invoke_model(
    invocation_context: OpenAICompatibleInvocationContext | GoogleGenAIInvocationContext,
    prompt_context: PromptContext,
) -> str:
    if isinstance(invocation_context, GoogleGenAIInvocationContext):
        try:
            response = invocation_context.client.models.generate_content(
                model=invocation_context.resolved_model_name,
                contents=prompt_context.user_prompt,
                config=invocation_context.generation_config,
            )
        except Exception as exc:
            raise ModelRequestError(
                format_exception_details(exc),
                status_code=extract_exception_status_code(exc),
            ) from exc
        text = getattr(response, "text", None)
        if isinstance(text, str) and text.strip():
            return text
        raise ValueError("google-genai response does not contain a usable text field.")

    try:
        response = invocation_context.client.chat.completions.create(
            model=invocation_context.resolved_model_name,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT.strip()},
                {"role": "user", "content": prompt_context.user_prompt},
            ],
            temperature=invocation_context.temperature,
            max_tokens=invocation_context.max_completion_tokens,
        )
    except Exception as exc:
        raise ModelRequestError(
            format_exception_details(exc),
            status_code=extract_exception_status_code(exc),
        ) from exc
    text = response.choices[0].message.content if response.choices else None
    if isinstance(text, str) and text.strip():
        return text
    raise ValueError("Model response does not contain a usable text field.")


def predict_sample(
    invocation_context: OpenAICompatibleInvocationContext | GoogleGenAIInvocationContext | None,
    config: RunnerConfig,
    sample: dict[str, Any],
    prompt_context: PromptContext,
) -> dict[str, Any]:
    index = int(sample["index"])
    if config.dry_run:
        return build_error_record(index, "dry_run", "model invocation skipped by --dry-run")

    assert invocation_context is not None
    raw_text = invoke_model(invocation_context, prompt_context)
    try:
        revised_code = extract_revised_code(raw_text)
    except Exception as exc:
        return build_error_record(index, "parse_error", format_exception_details(exc))
    return build_success_record(index, revised_code)


def process_sample_task(
    invocation_context: OpenAICompatibleInvocationContext | GoogleGenAIInvocationContext | None,
    config: RunnerConfig,
    sample: dict[str, Any],
    retriever: BM25Retriever | None,
) -> tuple[dict[str, Any], PromptContext, dict[str, Any]]:
    prompt_context = prepare_prompt_context(config, sample, retriever)
    record = predict_sample(invocation_context, config, sample, prompt_context)
    return sample, prompt_context, record


def write_run_config(
    artifacts: RunArtifacts,
    config: RunnerConfig,
    *,
    sample_count: int,
    started_at: str,
) -> None:
    payload = {
        **asdict(config),
        "dataset_path": str(config.dataset_path),
        "retrieval_dataset_path": str(config.retrieval_dataset_path),
        "output_root": str(config.output_root),
        "sample_count": sample_count,
        "output_schema": {
            "file": DEFAULT_GENERATIONS_NAME,
            "record": {
                "index": "int",
                "status": "ok | request_error | dry_run | parse_error",
                "revised_code": "string",
                "error": "optional string",
            },
        },
        "started_at": started_at,
    }
    artifacts.config_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def update_run_config_finished(artifacts: RunArtifacts, finished_at: str) -> None:
    payload = json.loads(artifacts.config_path.read_text(encoding="utf-8"))
    payload["finished_at"] = finished_at
    artifacts.config_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def update_run_config_aborted(artifacts: RunArtifacts, aborted_at: str, reason: str) -> None:
    payload = json.loads(artifacts.config_path.read_text(encoding="utf-8"))
    payload["aborted_at"] = aborted_at
    payload["abort_reason"] = reason
    artifacts.config_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def flush_all_log_handlers() -> None:
    for logger in (logging.getLogger(), LOG):
        for handler in logger.handlers:
            try:
                handler.flush()
            except Exception:
                continue


def abort_run_immediately(
    *,
    artifacts: RunArtifacts,
    handle: Any,
    progress: Any,
    reason: str,
) -> None:
    LOG.error(reason)
    update_run_config_aborted(artifacts, utc_now_iso(), reason)
    try:
        handle.flush()
        os.fsync(handle.fileno())
    except Exception:
        pass
    try:
        handle.close()
    except Exception:
        pass
    try:
        progress.close()
    except Exception:
        pass
    flush_all_log_handlers()
    os._exit(2)


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=getattr(logging, args.verbosity.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    if not args.model and not args.dry_run:
        raise SystemExit("--model is required unless --dry-run is enabled.")

    languages = {item.casefold() for item in parse_csv_items(args.languages)} or None
    sample_indexes = parse_index_filter(args.sample_indexes) or None

    config = RunnerConfig(
        dataset_path=args.dataset.resolve(),
        retrieval_dataset_path=args.retrieval_dataset.resolve(),
        output_root=args.output_root.resolve(),
        run_name=args.run_name or infer_run_name(args.dataset),
        model=args.model or "dry-run",
        provider=args.provider,
        strategy=args.strategy,
        temperature=args.temperature,
        max_completion_tokens=args.max_completion_tokens,
        timeout=args.timeout,
        max_workers=args.max_workers,
        abort_after_consecutive_non_200=max(0, args.abort_after_consecutive_non_200),
        api_key_env=args.api_key_env,
        base_url=args.base_url,
        resume=args.resume,
        overwrite=args.overwrite,
        dry_run=args.dry_run,
    )

    samples = load_selected_samples(
        config.dataset_path,
        languages=languages,
        sample_indexes=sample_indexes,
        max_samples=args.max_samples,
    )
    if not samples:
        raise SystemExit("No samples matched the current filters.")
    LOG.info("Loaded %s samples from %s", len(samples), config.dataset_path)

    artifacts = ensure_run_dir(config)
    auto_resume = should_auto_resume(artifacts, config)
    if auto_resume or config.resume:
        verify_subset_for_resume(artifacts, samples)

    write_jsonl(artifacts.gold_subset_path, samples)
    started_at = utc_now_iso()
    write_run_config(artifacts, config, sample_count=len(samples), started_at=started_at)

    existing_rows = load_existing_generation_rows(artifacts.generations_path) if auto_resume else []
    selected_index_set = {int(sample["index"]) for sample in samples}
    existing_selected_indexes = {
        int(row["index"])
        for row in existing_rows
        if int(row["index"]) in selected_index_set
    }
    retry_non_ok_indexes = (
        select_non_ok_retry_indexes(
            existing_rows,
            allowed_indexes=selected_index_set,
        )
        if auto_resume
        else set()
    )
    missing_indexes = selected_index_set - existing_selected_indexes
    completed_indexes = {
        int(row["index"])
        for row in existing_rows
        if int(row["index"]) in selected_index_set and generation_row_status(row) == "ok"
    }
    retained_rows = [row for row in existing_rows if int(row["index"]) not in retry_non_ok_indexes]
    pending_samples = [sample for sample in samples if int(sample["index"]) not in completed_indexes]
    if existing_rows:
        LOG.info(
            "Detected %s existing generations in %s; preserving completed ok cases automatically.",
            len(existing_rows),
            artifacts.generations_path,
        )
    if retry_non_ok_indexes:
        LOG.info(
            "Retrying %s selected samples whose latest status is not ok.",
            len(retry_non_ok_indexes),
        )
    if missing_indexes:
        LOG.info(
            "Retrying %s selected samples that do not have any recorded generation yet.",
            len(missing_indexes),
        )
    if not pending_samples:
        update_run_config_finished(artifacts, utc_now_iso())
        LOG.info("All %s selected samples already have generations. Nothing to run.", len(samples))
        return

    retriever = None
    if config.strategy == "rag":
        retrieval_rows = read_jsonl(config.retrieval_dataset_path)
        if not retrieval_rows:
            raise SystemExit(f"Retrieval dataset is empty: {config.retrieval_dataset_path}")
        retriever = BM25Retriever(retrieval_rows)
        LOG.info("Loaded %s retrieval exemplars from %s", len(retrieval_rows), config.retrieval_dataset_path)

    LOG.info(
        "Running provider=%s model=%s strategy=%s samples=%s pending=%s max_workers=%s abort_after_consecutive_non_200=%s output=%s",
        config.provider,
        config.model,
        config.strategy,
        len(samples),
        len(pending_samples),
        config.max_workers,
        config.abort_after_consecutive_non_200,
        artifacts.run_dir,
    )

    invocation_context = None if config.dry_run else build_invocation_context(config)
    if invocation_context is not None:
        LOG.info("Resolved %s model selection to %s", config.provider, invocation_context.resolved_model_name)
    if auto_resume and (retry_non_ok_indexes or len(retained_rows) != len(existing_rows)):
        write_jsonl(artifacts.generations_path, retained_rows)
    file_mode = "a" if auto_resume else "w"
    with artifacts.generations_path.open(file_mode, encoding="utf-8") as handle:
        progress = tqdm(total=len(pending_samples), desc=f"{config.model}:{config.run_name}")
        future_to_sample: dict[concurrent.futures.Future[tuple[dict[str, Any], PromptContext, dict[str, Any]]], dict[str, Any]] = {}
        with concurrent.futures.ThreadPoolExecutor(max_workers=config.max_workers) as executor:
            consecutive_non_200 = 0
            for ordinal, sample in enumerate(pending_samples, start=1):
                LOG.info("Queued sample %s/%s %s", ordinal, len(pending_samples), basic_sample_label(sample))
                future = executor.submit(
                    process_sample_task,
                    invocation_context,
                    config,
                    sample,
                    retriever,
                )
                future_to_sample[future] = sample

            completed = 0
            for future in concurrent.futures.as_completed(future_to_sample):
                sample = future_to_sample[future]
                index = int(sample["index"])
                try:
                    completed_sample, prompt_context, record = future.result()
                    progress_label = sample_progress_label(completed_sample, prompt_context)
                    progress.set_postfix_str(progress_label, refresh=False)
                    completed += 1
                    consecutive_non_200 = 0
                    LOG.info(
                        "Completed %s/%s %s status=%s revised_code_chars=%s",
                        completed,
                        len(pending_samples),
                        progress_label,
                        record["status"],
                        len(record.get("revised_code", "")),
                    )
                except Exception as exc:
                    completed += 1
                    progress_label = basic_sample_label(sample)
                    progress.set_postfix_str(progress_label, refresh=False)
                    is_request_failure = isinstance(exc, ModelRequestError)
                    error_details = str(exc) if is_request_failure else format_exception_details(exc)
                    LOG.exception(
                        "Failed sample %s/%s %s | %s",
                        completed,
                        len(pending_samples),
                        progress_label,
                        error_details,
                    )
                    record = build_error_record(index, "request_error", error_details)
                    if is_request_failure:
                        consecutive_non_200 += 1
                        if config.abort_after_consecutive_non_200 > 0:
                            LOG.warning(
                                "Consecutive non-200 request failures: %s/%s last_status=%s sample=%s",
                                consecutive_non_200,
                                config.abort_after_consecutive_non_200,
                                exc.status_code if exc.status_code is not None else "unknown",
                                progress_label,
                            )
                    if (
                        is_request_failure
                        and config.abort_after_consecutive_non_200 > 0
                        and consecutive_non_200 >= config.abort_after_consecutive_non_200
                    ):
                        handle.write(json.dumps(record, ensure_ascii=False) + "\n")
                        handle.flush()
                        progress.update(1)
                        abort_run_immediately(
                            artifacts=artifacts,
                            handle=handle,
                            progress=progress,
                            reason=(
                                "Aborting run after "
                                f"{consecutive_non_200} consecutive request-level failures without HTTP 200. "
                                f"Last sample={progress_label} last_status="
                                f"{exc.status_code if exc.status_code is not None else 'unknown'}."
                            ),
                        )
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
                handle.flush()
                progress.update(1)

    update_run_config_finished(artifacts, utc_now_iso())
    LOG.info("Raw generations written to %s", artifacts.generations_path)
    LOG.info("Selected gold subset written to %s", artifacts.gold_subset_path)


if __name__ == "__main__":
    main()

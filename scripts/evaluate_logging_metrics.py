#!/usr/bin/env python3
from __future__ import annotations

import argparse
import difflib
import json
import math
import re
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from multilogbench.logging_dataset.config import LANGUAGE_SPECS
from multilogbench.logging_dataset.core import (
    CallableIdentity,
    callable_identity,
    extract_detections,
    find_named_callable,
    iter_named_callables,
    node_text,
)
from multilogbench.logging_dataset.runtime import get_tree_sitter_parser


PREDICTION_CONTAINER_KEYS = (
    "pred_log",
    "prediction_log",
    "prediction",
    "predicted_log",
    "model_output",
    "output",
    "target_log",
)

FIELD_ALIASES = {
    "line": ("line", "position", "pred_line", "logging_point"),
    "level": ("level", "pred_level", "logging_level"),
    "message": ("message", "payload_text", "pred_message", "logging_text"),
    "vars": ("vars", "variables", "pred_vars", "logging_variables"),
    "statement": ("statement", "full_statement", "pred_statement"),
}

DEFAULT_LEVEL_ORDER = (
    "trace",
    "debug",
    "info",
    "notice",
    "warn",
    "error",
    "fatal",
)

LEVEL_ALIASES = {
    "trace": "trace",
    "verbose": "trace",
    "debug": "debug",
    "dbg": "debug",
    "info": "info",
    "information": "info",
    "notice": "notice",
    "warn": "warn",
    "warning": "warn",
    "error": "error",
    "err": "error",
    "exception": "error",
    "fatal": "fatal",
    "critical": "fatal",
    "panic": "fatal",
}

TOKEN_RE = re.compile(
    r"[A-Za-z_][A-Za-z0-9_]*|\d+|==|!=|<=|>=|&&|\|\||::|->|[^\s]",
)
CPP_LOG_MACRO_RE = re.compile(
    r"^\s*((?:ABSL_LOG|[VD]?LOG)"
    r"(?:_IF_EVERY_N|_EVERY_N|_FIRST_N|_IF)?)\s*\("
)
DEFAULT_RUN_GOLD_NAME = "selected_gold.jsonl"
DEFAULT_RUN_GENERATIONS_NAME = "generations.jsonl"
DEFAULT_RUN_METRICS_NAME = "metrics.json"
DEFAULT_RUN_DETAILS_NAME = "metrics_details.jsonl"
DEFAULT_RUN_PREDICTIONS_NAME = "predictions.jsonl"
METRIC_NAMES = (
    "PA",
    "FA",
    "LA",
    "AOD",
    "BLEU-4",
    "ROUGE-L",
    "PMR",
    "Precision",
    "Recall",
    "F1",
    "CCS",
)
@dataclass(frozen=True)
class LoggingRecord:
    index: int
    language: str
    repo_name: str
    line: int | None
    level: str | None
    message: str
    vars: tuple[str, ...]
    statement: str


@dataclass(frozen=True)
class ParsedDetection:
    owner_identity: CallableIdentity | None
    line: int | None
    abs_line: int
    abs_end_line: int
    level: str | None
    message: str
    vars: tuple[str, ...]
    statement: str


@dataclass(frozen=True)
class SampleScore:
    pa: float
    fa: float
    la: float
    aod: float
    rouge_l: float
    pmr: float
    precision: float
    recall: float
    f1: float
    missing_prediction: bool


@dataclass(frozen=True)
class PAResult:
    hit: bool
    prediction_log: LoggingRecord | None
    debug: dict[str, Any] | None


def exact_line_pa_hit(pred_line: int | None, gold_line: int | None, *, tolerance: int = 1) -> bool:
    if pred_line is None or gold_line is None:
        return False
    return abs(pred_line - gold_line) <= tolerance


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate logging-generation predictions against the local JSONL benchmark "
            "using PA, FA, LA, AOD, BLEU-4, ROUGE-L, PMR, Precision, Recall, and F1."
        ),
    )
    parser.add_argument(
        "--run-dir",
        type=Path,
        default=None,
        help=(
            "Optional run directory that contains selected_gold.jsonl and generations.jsonl. "
            "When provided, the evaluator will also default outputs to metrics.json, "
            "metrics_details.jsonl, and predictions.jsonl inside that run directory."
        ),
    )
    parser.add_argument(
        "--gold",
        type=Path,
        default=None,
        help="Gold JSONL dataset, for example dataset/core/v1/samples.jsonl.",
    )
    parser.add_argument(
        "--pred",
        type=Path,
        default=None,
        help=(
            "Prediction JSONL file keyed by index. Supported layouts include top-level "
            "fields or nested dicts such as pred_log / prediction / target_log, as well as "
            "raw generations.jsonl files that only store revised_code."
        ),
    )
    parser.add_argument(
        "--pred-format",
        choices=("auto", "structured", "generations"),
        default="auto",
        help=(
            "Prediction file format. 'structured' expects pred_log-like fields, "
            "'generations' expects revised_code, and 'auto' infers the layout."
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional JSON file that stores the aggregated metrics.",
    )
    parser.add_argument(
        "--details-out",
        type=Path,
        default=None,
        help="Optional JSONL file with per-sample metric details.",
    )
    parser.add_argument(
        "--predictions-out",
        type=Path,
        default=None,
        help=(
            "Optional JSONL file that stores normalized pred_log records. This is especially "
            "useful when the input is a raw generations.jsonl file."
        ),
    )
    parser.add_argument(
        "--include-by-repo",
        action="store_true",
        help="Also emit repo-level metric breakdowns.",
    )
    parser.add_argument(
        "--level-order",
        type=str,
        default=",".join(DEFAULT_LEVEL_ORDER),
        help=(
            "Comma-separated ordinal logging-level order for AOD, for example "
            "'trace,debug,info,warn,error'. Defaults to a dataset-friendly superset "
            "that also includes notice and fatal."
        ),
    )
    parser.add_argument(
        "--pa-mode",
        choices=("auto", "target-line-change", "exact-line"),
        default="auto",
        help=(
            "How to compute PA. auto uses target-line-change for raw generations and exact-line "
            "for structured predictions. target-line-change checks whether the gold target line "
            "in revised_code contains a newly added or changed detectable logging statement. "
            "exact-line accepts callable-relative line predictions within +/-1 of gold.line."
        ),
    )
    parser.add_argument(
        "--condition-metrics-on-pa",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Keep PA over all samples, but compute FA, LA, AOD, BLEU-4, ROUGE-L, "
            "PMR, Precision, Recall, and F1 using only samples where PA=1. "
            "Enabled by default."
        ),
    )
    return parser.parse_args()


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
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


def load_gold_rows(path: Path) -> dict[int, dict[str, Any]]:
    rows_by_index: dict[int, dict[str, Any]] = {}
    for raw_record in _read_jsonl(path):
        index = int(raw_record["index"])
        if index in rows_by_index:
            raise ValueError(f"Duplicate gold index {index} found in {path}")
        rows_by_index[index] = raw_record
    return rows_by_index


def normalize_level(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip().casefold()
    if not text:
        return None
    if text in LEVEL_ALIASES:
        return LEVEL_ALIASES[text]
    for alias, canonical in LEVEL_ALIASES.items():
        if re.search(rf"\b{re.escape(alias)}\b", text):
            return canonical
    return text


def normalize_line(value: Any) -> int | None:
    if value is None or value == "":
        return None
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, int):
        return value
    text = str(value).strip()
    if not text:
        return None
    return int(text)


def normalize_message(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip()


def canonicalize_var(value: str) -> str:
    return re.sub(r"\s+", "", value.strip())


def normalize_vars(value: Any) -> tuple[str, ...]:
    if value is None:
        return ()
    raw_items: Iterable[Any]
    if isinstance(value, list):
        raw_items = value
    elif isinstance(value, tuple):
        raw_items = value
    elif isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            return ()
        if stripped.startswith("[") and stripped.endswith("]"):
            try:
                parsed = json.loads(stripped)
            except json.JSONDecodeError:
                raw_items = (stripped,)
            else:
                raw_items = parsed if isinstance(parsed, list) else (stripped,)
        else:
            raw_items = (stripped,)
    else:
        raw_items = (value,)

    seen: set[str] = set()
    ordered: list[str] = []
    for item in raw_items:
        normalized = canonicalize_var(str(item))
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        ordered.append(normalized)
    return tuple(ordered)


def tokenize_message(text: str) -> list[str]:
    if not text:
        return []
    return TOKEN_RE.findall(text)


def extract_framework_anchor(statement: str, language: str | None = None) -> str | None:
    """Extract the strict lexical receiver or macro anchor.

    Under the strict-anchor taxonomy, member-based logging APIs are represented
    by the receiver expression before the earliest member-access separator.
    C++ stream-style logging APIs are represented by their exact leading macro
    symbol, including conditional and rate-limited variants.
    """
    text = statement.strip()
    if not text:
        return None

    # C++ stream payloads commonly contain '.', '->', or '::'. The leading
    # macro must therefore be recognized before inspecting payload separators.
    if language == "cpp":
        macro_match = re.match(r"^\s*([A-Za-z_][A-Za-z0-9_]*)\s*\(", text)
        if macro_match:
            return macro_match.group(1)
    elif language is None:
        # Preserve useful standalone behavior for known C++ logging macros.
        macro_match = CPP_LOG_MACRO_RE.match(text)
        if macro_match:
            return macro_match.group(1)

    separators = [
        (text.find(separator), separator)
        for separator in (".", "->", "::")
        if separator in text
    ]
    if separators:
        position, _ = min(separators)
        prefix = text[:position].strip()
        return prefix or None

    macro_match = re.match(r"^\s*([A-Za-z_][A-Za-z0-9_]*)\s*\(", text)
    if macro_match:
        return macro_match.group(1)

    return None


def _container_from_prediction(raw_record: dict[str, Any]) -> dict[str, Any]:
    for key in PREDICTION_CONTAINER_KEYS:
        value = raw_record.get(key)
        if isinstance(value, dict):
            return value
    return raw_record


def _has_nonempty_prediction_payload(container: dict[str, Any]) -> bool:
    if not container:
        return False
    for field_name in FIELD_ALIASES:
        value = _extract_field(container, field_name)
        if field_name == "line":
            if normalize_line(value) is not None:
                return True
            continue
        if field_name == "level":
            if normalize_level(value) is not None:
                return True
            continue
        if field_name == "vars":
            if normalize_vars(value):
                return True
            continue
        if normalize_message(value):
            return True
    return bool(container)


def _extract_field(container: dict[str, Any], field_name: str) -> Any:
    for alias in FIELD_ALIASES[field_name]:
        if alias in container:
            return container[alias]
    return None


def load_gold_records_from_rows(rows_by_index: dict[int, dict[str, Any]]) -> dict[int, LoggingRecord]:
    records: dict[int, LoggingRecord] = {}
    for index in sorted(rows_by_index):
        raw_record = rows_by_index[index]
        target_log = raw_record.get("target_log")
        if not isinstance(target_log, dict):
            raise ValueError(f"Gold record {index} is missing target_log")
        records[index] = LoggingRecord(
            index=index,
            language=str(raw_record.get("language", "")).strip(),
            repo_name=str(raw_record.get("repo_name", "")).strip(),
            line=normalize_line(target_log.get("line")),
            level=normalize_level(target_log.get("level")),
            message=normalize_message(target_log.get("message")),
            vars=normalize_vars(target_log.get("vars")),
            statement=normalize_message(target_log.get("statement")),
        )
    return records


def load_gold_records(path: Path) -> dict[int, LoggingRecord]:
    return load_gold_records_from_rows(load_gold_rows(path))


def detect_prediction_format(
    rows: list[dict[str, Any]],
    explicit_format: str,
) -> str:
    if explicit_format != "auto":
        return explicit_format
    for raw_record in rows:
        if "revised_code" in raw_record:
            return "generations"
        container = _container_from_prediction(raw_record)
        if any(_extract_field(container, field_name) is not None for field_name in FIELD_ALIASES):
            return "structured"
    return "structured"


def collapse_statement(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def _language_spec_for_row(row: dict[str, Any]) -> Any:
    language = str(row.get("language", "")).strip()
    if language not in LANGUAGE_SPECS:
        raise ValueError(f"Unsupported language for prediction extraction: {language!r}")
    return LANGUAGE_SPECS[language]


def resolve_parser_name(spec: Any, row: dict[str, Any]) -> str:
    file_path = str(row.get("file_path", "")).strip()
    if file_path:
        suffix = Path(file_path).suffix.lower()
        if suffix in spec.parser_by_suffix:
            return spec.parser_name_for_path(Path(file_path))
    return next(iter(spec.parser_by_suffix.values()))


def parse_logged_snippet(row: dict[str, Any], code_text: str) -> tuple[list[ParsedDetection], list[CallableIdentity]]:
    if not str(code_text).strip():
        return [], []
    spec = _language_spec_for_row(row)
    parser = get_tree_sitter_parser(resolve_parser_name(spec, row))
    source = str(code_text).encode("utf-8")
    tree = parser.parse(source)

    callables = [
        callable_identity(owner)
        for owner in iter_named_callables(spec, tree.root_node, source)
    ]
    detections: list[ParsedDetection] = []
    for detection in extract_detections(spec, tree.root_node, source):
        owner = find_named_callable(spec, detection.statement_node, source)
        detections.append(
            ParsedDetection(
                owner_identity=None if owner is None else callable_identity(owner),
                line=(
                    None
                    if owner is None
                    else detection.statement_node.start_point[0] - owner.node.start_point[0] + 1
                ),
                abs_line=detection.statement_node.start_point[0] + 1,
                abs_end_line=detection.statement_node.end_point[0] + 1,
                level=normalize_level(detection.level),
                message=normalize_message(detection.payload_text),
                vars=normalize_vars(detection.vars),
                statement=normalize_message(node_text(source, detection.statement_node)),
            )
        )
    detections.sort(key=lambda item: (item.abs_line, item.abs_end_line, item.statement))
    return detections, callables


def changed_line_ranges(before_text: str, after_text: str) -> list[tuple[int, int]]:
    matcher = difflib.SequenceMatcher(
        a=str(before_text).splitlines(),
        b=str(after_text).splitlines(),
        autojunk=False,
    )
    ranges: list[tuple[int, int]] = []
    for tag, _, _, j1, j2 in matcher.get_opcodes():
        if tag == "equal" or j1 == j2:
            continue
        ranges.append((j1 + 1, j2))
    return ranges


def overlaps_changed_lines(detection: ParsedDetection, ranges: list[tuple[int, int]]) -> bool:
    return any(
        not (detection.abs_end_line < start_line or end_line < detection.abs_line)
        for start_line, end_line in ranges
    )


def resolve_target_owner_identity(gold_row: dict[str, Any]) -> CallableIdentity | None:
    gold_code = str(gold_row.get("function_content", ""))
    detections, callables = parse_logged_snippet(gold_row, gold_code)
    target_log = gold_row.get("target_log")
    if not isinstance(target_log, dict):
        return callables[0] if len(callables) == 1 else None

    target_statement = collapse_statement(str(target_log.get("statement", "")))
    target_line = normalize_line(target_log.get("line"))
    candidates = [item for item in detections if collapse_statement(item.statement) == target_statement]
    if target_line is not None:
        line_matches = [item for item in candidates if item.line == target_line]
        if line_matches:
            candidates = line_matches
    for candidate in candidates:
        if candidate.owner_identity is not None:
            return candidate.owner_identity

    function_name = str(gold_row.get("function_name", "")).strip()
    if function_name:
        matches = [identity for identity in callables if identity.name == function_name]
        if len(matches) == 1:
            return matches[0]
    return callables[0] if len(callables) == 1 else None


def select_changed_detection(
    before_detections: list[ParsedDetection],
    after_detections: list[ParsedDetection],
    *,
    target_owner_identity: CallableIdentity | None,
    change_ranges: list[tuple[int, int]],
) -> ParsedDetection | None:
    candidate_before = before_detections
    candidate_after = after_detections
    if target_owner_identity is not None:
        owner_after = [item for item in after_detections if item.owner_identity == target_owner_identity]
        if owner_after:
            candidate_after = owner_after
            candidate_before = [
                item for item in before_detections if item.owner_identity == target_owner_identity
            ]

    if not candidate_after:
        return None

    base_signatures = [collapse_statement(item.statement) for item in candidate_before]
    pred_signatures = [collapse_statement(item.statement) for item in candidate_after]
    matcher = difflib.SequenceMatcher(a=base_signatures, b=pred_signatures, autojunk=False)
    candidate_indexes: list[int] = []
    for tag, _, _, j1, j2 in matcher.get_opcodes():
        if tag == "equal":
            continue
        candidate_indexes.extend(range(j1, j2))

    seen_indexes: set[int] = set()
    ordered_candidates = [index for index in candidate_indexes if not (index in seen_indexes or seen_indexes.add(index))]
    for index in ordered_candidates:
        detection = candidate_after[index]
        if overlaps_changed_lines(detection, change_ranges):
            return detection
    for index in ordered_candidates:
        return candidate_after[index]

    overlap_candidates = [
        detection for detection in candidate_after if overlaps_changed_lines(detection, change_ranges)
    ]
    if overlap_candidates:
        return overlap_candidates[0]
    return None


def build_empty_prediction_record(raw_record: dict[str, Any], gold_row: dict[str, Any] | None) -> LoggingRecord:
    return LoggingRecord(
        index=int(raw_record["index"]),
        language=(
            str(gold_row.get("language", "")).strip()
            if gold_row is not None
            else str(raw_record.get("language", "")).strip()
        ),
        repo_name=(
            str(gold_row.get("repo_name", "")).strip()
            if gold_row is not None
            else str(raw_record.get("repo_name", "")).strip()
        ),
        line=None,
        level=None,
        message="",
        vars=(),
        statement="",
    )


def build_log_payload(record: LoggingRecord) -> dict[str, Any]:
    return {
        "line": record.line,
        "level": record.level,
        "message": record.message,
        "vars": list(record.vars),
        "statement": record.statement,
    }


def build_prediction_row(
    record: LoggingRecord,
    *,
    status: str | None = None,
    error: str | None = None,
    extraction_status: str | None = None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "index": record.index,
        "pred_log": build_log_payload(record),
    }
    if status:
        payload["status"] = status
    if error:
        payload["error"] = error
    if extraction_status:
        payload["extraction_status"] = extraction_status
    return payload


def extract_prediction_from_generation(
    raw_record: dict[str, Any],
    gold_row: dict[str, Any] | None,
) -> tuple[LoggingRecord, str]:
    empty_record = build_empty_prediction_record(raw_record, gold_row)
    if gold_row is None:
        return empty_record, "missing_gold_context"

    revised_code = str(raw_record.get("revised_code", ""))
    status = str(raw_record.get("status", "")).strip()
    if status and status != "ok":
        return empty_record, f"model_status:{status}"
    if not revised_code.strip():
        return empty_record, "empty_revised_code"

    before_code = str(gold_row.get("input_initial", ""))
    if not before_code.strip():
        return empty_record, "missing_input_initial"

    try:
        before_detections, _ = parse_logged_snippet(gold_row, before_code)
        after_detections, _ = parse_logged_snippet(gold_row, revised_code)
        target_owner_identity = resolve_target_owner_identity(gold_row)
    except Exception:
        return empty_record, "parse_failure"

    if not after_detections:
        return empty_record, "no_log_detected"

    change_ranges = changed_line_ranges(before_code, revised_code)
    if not change_ranges:
        return empty_record, "no_code_change"

    selected = select_changed_detection(
        before_detections,
        after_detections,
        target_owner_identity=target_owner_identity,
        change_ranges=change_ranges,
    )
    if selected is None:
        return empty_record, "no_changed_log_detected"

    return (
        LoggingRecord(
            index=empty_record.index,
            language=empty_record.language,
            repo_name=empty_record.repo_name,
            line=selected.line,
            level=selected.level,
            message=selected.message,
            vars=selected.vars,
            statement=selected.statement,
        ),
        "extracted",
    )


def load_structured_prediction_records(
    rows: list[dict[str, Any]],
) -> tuple[dict[int, LoggingRecord], list[dict[str, Any]], dict[str, Any]]:
    records: dict[int, LoggingRecord] = {}
    normalized_rows: list[dict[str, Any]] = []
    for raw_record in rows:
        if "index" not in raw_record:
            raise ValueError(f"Prediction record is missing index: {raw_record}")
        index = int(raw_record["index"])
        if index in records:
            raise ValueError(f"Duplicate prediction index {index} found in prediction file")
        container = _container_from_prediction(raw_record)
        record = LoggingRecord(
            index=index,
            language=str(raw_record.get("language", "")).strip(),
            repo_name=str(raw_record.get("repo_name", "")).strip(),
            line=normalize_line(_extract_field(container, "line")),
            level=normalize_level(_extract_field(container, "level")),
            message=normalize_message(_extract_field(container, "message")),
            vars=normalize_vars(_extract_field(container, "vars")),
            statement=normalize_message(_extract_field(container, "statement")),
        )
        records[index] = record
        normalized_rows.append(
            build_prediction_row(
                record,
                status=str(raw_record.get("status", "")).strip() or None,
                error=str(raw_record.get("error", "")).strip() or None,
                extraction_status="structured",
            )
        )
    return records, normalized_rows, {"format": "structured"}


def load_generation_prediction_records(
    rows: list[dict[str, Any]],
    gold_rows: dict[int, dict[str, Any]],
) -> tuple[dict[int, LoggingRecord], list[dict[str, Any]], dict[str, Any]]:
    records: dict[int, LoggingRecord] = {}
    normalized_rows: list[dict[str, Any]] = []
    extraction_counter: Counter[str] = Counter()
    model_status_counter: Counter[str] = Counter()

    for raw_record in rows:
        if "index" not in raw_record:
            raise ValueError(f"Prediction record is missing index: {raw_record}")
        index = int(raw_record["index"])
        if index in records:
            raise ValueError(f"Duplicate prediction index {index} found in prediction file")
        gold_row = gold_rows.get(index)
        record, extraction_status = extract_prediction_from_generation(raw_record, gold_row)
        records[index] = record
        extraction_counter[extraction_status] += 1
        model_status = str(raw_record.get("status", "")).strip() or "unknown"
        model_status_counter[model_status] += 1
        normalized_rows.append(
            build_prediction_row(
                record,
                status=model_status,
                error=str(raw_record.get("error", "")).strip() or None,
                extraction_status=extraction_status,
            )
        )
    return records, normalized_rows, {
        "format": "generations",
        "extraction_status_counts": dict(sorted(extraction_counter.items())),
        "model_status_counts": dict(sorted(model_status_counter.items())),
    }


def load_prediction_records_with_metadata(
    path: Path,
    *,
    gold_rows: dict[int, dict[str, Any]] | None = None,
    pred_format: str = "auto",
) -> tuple[dict[int, LoggingRecord], list[dict[str, Any]], dict[str, Any]]:
    rows = _read_jsonl(path)
    resolved_format = detect_prediction_format(rows, pred_format)
    if resolved_format == "structured":
        return load_structured_prediction_records(rows)
    if gold_rows is None:
        raise ValueError("Raw generations.jsonl evaluation requires gold rows with input_initial context.")
    return load_generation_prediction_records(rows, gold_rows)


def load_prediction_records(
    path: Path,
    *,
    gold_rows: dict[int, dict[str, Any]] | None = None,
    pred_format: str = "auto",
) -> dict[int, LoggingRecord]:
    records, _, _ = load_prediction_records_with_metadata(
        path,
        gold_rows=gold_rows,
        pred_format=pred_format,
    )
    return records


def _code_line_text(code_text: str, target_line: int) -> str:
    lines = str(code_text).splitlines()
    if 1 <= target_line <= len(lines):
        return lines[target_line - 1]
    return ""


def _statement_has_target_line_text(statement: str, target_line_text: str) -> bool:
    target_norm = collapse_statement(target_line_text)
    if not target_norm:
        return False
    return any(
        collapse_statement(line) == target_norm
        for line in str(statement).splitlines()
    )


def _select_target_line_detection(
    code_text: str,
    detections: list[ParsedDetection],
    *,
    target_line: int,
    target_owner_identity: CallableIdentity | None,
) -> tuple[ParsedDetection | None, bool]:
    candidate_detections = detections
    if target_owner_identity is not None:
        candidate_detections = [item for item in detections if item.owner_identity == target_owner_identity]

    line_matches = [item for item in candidate_detections if item.line == target_line]
    if line_matches:
        return line_matches[0], False

    line_less_candidates = [item for item in candidate_detections if item.line is None]
    if not line_less_candidates:
        return None, False

    target_line_text = _code_line_text(code_text, target_line)
    for detection in line_less_candidates:
        if _statement_has_target_line_text(detection.statement, target_line_text):
            return detection, True
    return None, True


def _build_target_line_prediction_log(
    raw_record: dict[str, Any],
    gold_row: dict[str, Any],
    *,
    target_line: int,
    detection: ParsedDetection,
) -> LoggingRecord:
    return LoggingRecord(
        index=int(raw_record["index"]),
        language=str(gold_row.get("language", "")).strip(),
        repo_name=str(gold_row.get("repo_name", "")).strip(),
        line=target_line,
        level=detection.level,
        message=detection.message,
        vars=detection.vars,
        statement=detection.statement,
    )


def extract_exact_line_target_prediction(
    raw_record: dict[str, Any],
    gold_row: dict[str, Any] | None,
) -> LoggingRecord | None:
    if gold_row is None:
        return None

    target_log = gold_row.get("target_log")
    if not isinstance(target_log, dict):
        return None

    target_line = normalize_line(target_log.get("line"))
    if target_line is None or target_line <= 0:
        return None

    revised_code = str(raw_record.get("revised_code", ""))
    if not revised_code.strip():
        return None

    try:
        detections, _ = parse_logged_snippet(gold_row, revised_code)
    except Exception:
        return None

    target_owner_identity = resolve_target_owner_identity(gold_row)
    if target_owner_identity is not None:
        detections = [item for item in detections if item.owner_identity == target_owner_identity]

    line_matches = [item for item in detections if item.line == target_line]
    if not line_matches:
        return None
    selected = line_matches[0]

    return LoggingRecord(
        index=int(raw_record["index"]),
        language=str(gold_row.get("language", "")).strip(),
        repo_name=str(gold_row.get("repo_name", "")).strip(),
        line=target_line,
        level=selected.level,
        message=selected.message,
        vars=selected.vars,
        statement=selected.statement,
    )


def extract_target_line_change_result(
    raw_record: dict[str, Any],
    gold_row: dict[str, Any] | None,
) -> PAResult:
    debug: dict[str, Any] = {
        "target_line": None,
        "after_target_line_has_log": False,
        "before_target_line_has_log": False,
        "used_text_line_fallback": False,
        "target_line_change_hit": False,
    }
    if gold_row is None:
        debug["reason"] = "missing_gold_context"
        return PAResult(hit=False, prediction_log=None, debug=debug)

    target_log = gold_row.get("target_log")
    if not isinstance(target_log, dict):
        debug["reason"] = "missing_target_log"
        return PAResult(hit=False, prediction_log=None, debug=debug)

    target_line = normalize_line(target_log.get("line"))
    debug["target_line"] = target_line
    if target_line is None or target_line <= 0:
        debug["reason"] = "invalid_target_line"
        return PAResult(hit=False, prediction_log=None, debug=debug)

    revised_code = str(raw_record.get("revised_code", ""))
    if not revised_code.strip():
        debug["reason"] = "empty_revised_code"
        return PAResult(hit=False, prediction_log=None, debug=debug)

    before_code = str(gold_row.get("input_initial", ""))
    if not before_code.strip():
        debug["reason"] = "missing_input_initial"
        return PAResult(hit=False, prediction_log=None, debug=debug)

    try:
        before_detections, _ = parse_logged_snippet(gold_row, before_code)
        after_detections, _ = parse_logged_snippet(gold_row, revised_code)
    except Exception:
        debug["reason"] = "parse_failure"
        return PAResult(hit=False, prediction_log=None, debug=debug)

    try:
        target_owner_identity = resolve_target_owner_identity(gold_row)
    except Exception:
        target_owner_identity = None

    before_detection, before_used_fallback = _select_target_line_detection(
        before_code,
        before_detections,
        target_line=target_line,
        target_owner_identity=target_owner_identity,
    )
    after_detection, after_used_fallback = _select_target_line_detection(
        revised_code,
        after_detections,
        target_line=target_line,
        target_owner_identity=target_owner_identity,
    )

    debug.update(
        {
            "before_target_line_has_log": before_detection is not None,
            "after_target_line_has_log": after_detection is not None,
            "used_text_line_fallback": before_used_fallback or after_used_fallback,
            "before_used_text_line_fallback": before_used_fallback,
            "after_used_text_line_fallback": after_used_fallback,
            "before_target_line_statement": "" if before_detection is None else before_detection.statement,
            "after_target_line_statement": "" if after_detection is None else after_detection.statement,
        }
    )

    if after_detection is None:
        debug["reason"] = "after_target_line_no_log"
        return PAResult(hit=False, prediction_log=None, debug=debug)

    if before_detection is not None and collapse_statement(before_detection.statement) == collapse_statement(after_detection.statement):
        debug["reason"] = "unchanged_target_line_log"
        return PAResult(hit=False, prediction_log=None, debug=debug)

    debug["target_line_change_hit"] = True
    debug["reason"] = "changed_target_line_log" if before_detection is not None else "new_target_line_log"
    return PAResult(
        hit=True,
        prediction_log=_build_target_line_prediction_log(
            raw_record,
            gold_row,
            target_line=target_line,
            detection=after_detection,
        ),
        debug=debug,
    )


def compute_pa_results(
    pred_path: Path,
    *,
    gold_rows: dict[int, dict[str, Any]] | None = None,
    pred_format: str = "auto",
    pa_mode: str = "target-line-change",
) -> dict[int, PAResult]:
    rows = _read_jsonl(pred_path)
    resolved_format = detect_prediction_format(rows, pred_format)
    if resolved_format != "generations":
        raise ValueError("target-line-change PA requires a raw generations.jsonl prediction file.")
    pa_results: dict[int, PAResult] = {}

    for raw_record in rows:
        if "index" not in raw_record:
            raise ValueError(f"Prediction record is missing index: {raw_record}")
        index = int(raw_record["index"])
        if index in pa_results:
            raise ValueError(f"Duplicate prediction index {index} found in prediction file")

        gold_row = gold_rows.get(index) if gold_rows else None
        if pa_mode == "target-line-change":
            pa_results[index] = extract_target_line_change_result(raw_record, gold_row)
        elif pa_mode == "exact-line":
            prediction_log = extract_exact_line_target_prediction(raw_record, gold_row)
            pa_results[index] = PAResult(
                hit=prediction_log is not None,
                prediction_log=prediction_log,
                debug=None,
            )
        else:
            raise ValueError(f"Unsupported PA mode for raw generations: {pa_mode}")

    return pa_results


def attach_target_line_prediction_logs(
    rows: list[dict[str, Any]],
    pa_results: dict[int, PAResult],
) -> None:
    rows_by_index = {int(row["index"]): row for row in rows if "index" in row}
    for index, result in pa_results.items():
        if result.prediction_log is None:
            continue
        row = rows_by_index.get(index)
        if row is None:
            continue
        row["prediction_log"] = build_log_payload(result.prediction_log)


def count_ngrams(tokens: list[str], n: int) -> Counter[tuple[str, ...]]:
    counter: Counter[tuple[str, ...]] = Counter()
    if len(tokens) < n:
        return counter
    for index in range(len(tokens) - n + 1):
        counter[tuple(tokens[index : index + n])] += 1
    return counter


def compute_bleu_4(reference_tokens: list[list[str]], hypothesis_tokens: list[list[str]]) -> float:
    if len(reference_tokens) != len(hypothesis_tokens):
        raise ValueError("Reference and hypothesis token lists must have the same length")
    if not reference_tokens:
        return 0.0

    precisions: list[float] = []
    for n in range(1, 5):
        matched = 0
        total = 0
        for ref_tokens, hyp_tokens in zip(reference_tokens, hypothesis_tokens):
            hyp_counts = count_ngrams(hyp_tokens, n)
            ref_counts = count_ngrams(ref_tokens, n)
            total += sum(hyp_counts.values())
            matched += sum(min(count, ref_counts[ngram]) for ngram, count in hyp_counts.items())
        if total == 0 or matched == 0:
            return 0.0
        precisions.append(matched / total)

    ref_length = sum(len(tokens) for tokens in reference_tokens)
    hyp_length = sum(len(tokens) for tokens in hypothesis_tokens)
    if hyp_length == 0:
        return 0.0
    brevity_penalty = 1.0 if hyp_length > ref_length else math.exp(1.0 - (ref_length / hyp_length))
    geometric_mean = math.exp(sum(math.log(value) for value in precisions) / 4.0)
    return brevity_penalty * geometric_mean


def lcs_length(left: list[str], right: list[str]) -> int:
    if not left or not right:
        return 0
    prev = [0] * (len(right) + 1)
    for left_token in left:
        curr = [0]
        for idx, right_token in enumerate(right, start=1):
            if left_token == right_token:
                curr.append(prev[idx - 1] + 1)
            else:
                curr.append(max(curr[-1], prev[idx]))
        prev = curr
    return prev[-1]


def compute_rouge_l(reference_tokens: list[str], hypothesis_tokens: list[str]) -> float:
    if not reference_tokens and not hypothesis_tokens:
        return 1.0
    if not reference_tokens or not hypothesis_tokens:
        return 0.0
    lcs = lcs_length(reference_tokens, hypothesis_tokens)
    precision = lcs / len(hypothesis_tokens)
    recall = lcs / len(reference_tokens)
    if precision + recall == 0:
        return 0.0
    return (2.0 * precision * recall) / (precision + recall)


def compute_aod(
    gold_level: str | None,
    pred_level: str | None,
    level_positions: dict[str, int],
) -> float:
    if gold_level is None or pred_level is None:
        return 0.0
    if gold_level not in level_positions or pred_level not in level_positions:
        return 0.0
    gold_index = level_positions[gold_level]
    pred_index = level_positions[pred_level]
    max_index = len(level_positions) - 1
    max_distance = max(gold_index, max_index - gold_index)
    if max_distance == 0:
        return 1.0
    distance = abs(gold_index - pred_index)
    return 1.0 - (distance / max_distance)


def compute_variable_metrics(gold_vars: tuple[str, ...], pred_vars: tuple[str, ...]) -> tuple[float, float, float, float]:
    gold_set = set(gold_vars)
    pred_set = set(pred_vars)
    overlap = len(gold_set & pred_set)
    pmr = 1.0 if gold_set == pred_set else 0.0

    if not pred_set and not gold_set:
        return pmr, 1.0, 1.0, 1.0
    if not pred_set:
        return pmr, 0.0, 0.0, 0.0
    if not gold_set:
        return pmr, 0.0, 0.0, 0.0

    precision = overlap / len(pred_set)
    recall = overlap / len(gold_set)
    if precision + recall == 0:
        f1 = 0.0
    else:
        f1 = (2.0 * precision * recall) / (precision + recall)
    return pmr, precision, recall, f1


def score_sample(
    gold: LoggingRecord,
    pred: LoggingRecord | None,
    level_positions: dict[str, int],
    *,
    pa_mode: str = "exact-line",
    target_line_change_hit: bool | None = None,
) -> tuple[SampleScore, list[str], list[str]]:
    if pred is None:
        pred = LoggingRecord(
            index=gold.index,
            language=gold.language,
            repo_name=gold.repo_name,
            line=None,
            level=None,
            message="",
            vars=(),
            statement="",
        )
        missing_prediction = True
    else:
        missing_prediction = False

    if pa_mode == "target-line-change":
        pa = 1.0 if target_line_change_hit else 0.0
    else:
        pa = 1.0 if exact_line_pa_hit(pred.line, gold.line) else 0.0
    gold_framework = extract_framework_anchor(gold.statement, gold.language)
    # Prediction-only JSONL records do not always repeat the language. The
    # benchmark language is defined by the matched gold sample.
    pred_framework = extract_framework_anchor(pred.statement, gold.language)
    fa = 1.0 if gold_framework is not None and pred_framework == gold_framework else 0.0
    la = 1.0 if pred.level is not None and pred.level == gold.level else 0.0
    aod = compute_aod(gold.level, pred.level, level_positions)
    rouge_l = compute_rouge_l(tokenize_message(gold.message), tokenize_message(pred.message))
    pmr, precision, recall, f1 = compute_variable_metrics(gold.vars, pred.vars)

    return (
        SampleScore(
            pa=pa,
            fa=fa,
            la=la,
            aod=aod,
            rouge_l=rouge_l,
            pmr=pmr,
            precision=precision,
            recall=recall,
            f1=f1,
            missing_prediction=missing_prediction,
        ),
        tokenize_message(gold.message),
        tokenize_message(pred.message),
    )


def round_metric(value: float) -> float:
    return round(value, 6)


def compute_ccs(pa: float, fa: float, la: float, rouge_l: float, f1: float) -> float:
    residual_quality = (la + rouge_l + f1) / 3.0
    return pa * (0.5 + 0.25 * fa + 0.25 * residual_quality)


def build_level_positions(level_order: tuple[str, ...]) -> dict[str, int]:
    deduped: list[str] = []
    seen: set[str] = set()
    for item in level_order:
        normalized = normalize_level(item)
        if normalized is None or normalized in seen:
            continue
        seen.add(normalized)
        deduped.append(normalized)
    return {level: index for index, level in enumerate(deduped)}


def aggregate_group(
    sample_payloads: list[tuple[LoggingRecord, LoggingRecord | None, SampleScore, list[str], list[str]]],
    *,
    condition_metrics_on_pa: bool = False,
) -> dict[str, Any]:
    count = len(sample_payloads)
    pa_correct_payloads = [payload for payload in sample_payloads if payload[2].pa == 1.0]
    metric_payloads = pa_correct_payloads if condition_metrics_on_pa else sample_payloads
    metric_count = len(metric_payloads)
    if count == 0:
        return {
            "sample_count": 0,
            "pa_correct_count": 0,
            "metric_sample_count": 0,
            "missing_predictions": 0,
            "metrics": {
                "PA": 0.0,
                "FA": 0.0,
                "LA": 0.0,
                "AOD": 0.0,
                "BLEU-4": 0.0,
                "ROUGE-L": 0.0,
                "PMR": 0.0,
                "Precision": 0.0,
                "Recall": 0.0,
                "F1": 0.0,
                "CCS": 0.0,
            },
        }

    missing_predictions = sum(1 for payload in sample_payloads if payload[2].missing_prediction)
    mean_pa = sum(payload[2].pa for payload in sample_payloads) / count
    if metric_count == 0:
        bleu = 0.0
        mean_fa = 0.0
        mean_la = 0.0
        mean_aod = 0.0
        mean_rouge = 0.0
        mean_pmr = 0.0
        mean_precision = 0.0
        mean_recall = 0.0
        mean_f1 = 0.0
    else:
        bleu = compute_bleu_4(
            [payload[3] for payload in metric_payloads],
            [payload[4] for payload in metric_payloads],
        )
        mean_fa = sum(payload[2].fa for payload in metric_payloads) / metric_count
        mean_la = sum(payload[2].la for payload in metric_payloads) / metric_count
        mean_aod = sum(payload[2].aod for payload in metric_payloads) / metric_count
        mean_rouge = sum(payload[2].rouge_l for payload in metric_payloads) / metric_count
        mean_pmr = sum(payload[2].pmr for payload in metric_payloads) / metric_count
        mean_precision = sum(payload[2].precision for payload in metric_payloads) / metric_count
        mean_recall = sum(payload[2].recall for payload in metric_payloads) / metric_count
        mean_f1 = sum(payload[2].f1 for payload in metric_payloads) / metric_count

    ccs = compute_ccs(mean_pa, mean_fa, mean_la, mean_rouge, mean_f1)
    return {
        "sample_count": count,
        "pa_correct_count": len(pa_correct_payloads),
        "metric_sample_count": metric_count,
        "missing_predictions": missing_predictions,
        "metrics": {
            "PA": round_metric(mean_pa),
            "FA": round_metric(mean_fa),
            "LA": round_metric(mean_la),
            "AOD": round_metric(mean_aod),
            "BLEU-4": round_metric(bleu),
            "ROUGE-L": round_metric(mean_rouge),
            "PMR": round_metric(mean_pmr),
            "Precision": round_metric(mean_precision),
            "Recall": round_metric(mean_recall),
            "F1": round_metric(mean_f1),
            "CCS": round_metric(ccs),
        },
    }


def macro_average_aggregates(
    groups: Iterable[dict[str, Any]],
    *,
    condition_metrics_on_pa: bool = False,
) -> dict[str, Any]:
    group_list = list(groups)
    if not group_list:
        return {
            "group_count": 0,
            "metric_group_count": 0,
            "metrics": {metric_name: 0.0 for metric_name in METRIC_NAMES},
        }
    metric_groups = (
        [group for group in group_list if group.get("metric_sample_count", group.get("sample_count", 0)) > 0]
        if condition_metrics_on_pa
        else group_list
    )
    metrics: dict[str, float] = {}
    for metric_name in METRIC_NAMES:
        source_groups = group_list if metric_name == "PA" else metric_groups
        if not source_groups:
            metrics[metric_name] = 0.0
            continue
        metrics[metric_name] = round_metric(
            sum(group["metrics"][metric_name] for group in source_groups) / len(source_groups)
        )
    return {
        "group_count": len(group_list),
        "metric_group_count": len(metric_groups),
        "metrics": metrics,
    }


def aggregate_repo_macro_language(
    sample_payloads: list[tuple[LoggingRecord, LoggingRecord | None, SampleScore, list[str], list[str]]],
    repo_groups: list[dict[str, Any]],
    *,
    condition_metrics_on_pa: bool = False,
) -> dict[str, Any]:
    macro_payload = macro_average_aggregates(
        repo_groups,
        condition_metrics_on_pa=condition_metrics_on_pa,
    )
    return {
        "sample_count": len(sample_payloads),
        "pa_correct_count": sum(1 for payload in sample_payloads if payload[2].pa == 1.0),
        "metric_sample_count": (
            sum(1 for payload in sample_payloads if payload[2].pa == 1.0)
            if condition_metrics_on_pa
            else len(sample_payloads)
        ),
        "repo_count": len(repo_groups),
        "metric_repo_count": macro_payload["metric_group_count"],
        "missing_predictions": sum(1 for payload in sample_payloads if payload[2].missing_prediction),
        "metrics": macro_payload["metrics"],
    }


def pa_mode_note(pa_mode: str) -> str:
    if pa_mode == "target-line-change":
        return (
            "PA is computed with target-line-change mode: the gold target line "
            "in revised_code must contain a detectable log, and that target-line log must be "
            "new or changed relative to input_initial."
        )
    return "PA is computed by relative line match with a +/-1 tolerance around gold.line."


def evaluate_predictions(
    gold_records: dict[int, LoggingRecord],
    pred_records: dict[int, LoggingRecord],
    *,
    level_order: tuple[str, ...] = DEFAULT_LEVEL_ORDER,
    pa_mode: str = "exact-line",
    target_line_change_hits: dict[int, bool] | None = None,
    target_line_change_debug: dict[int, dict[str, Any]] | None = None,
    condition_metrics_on_pa: bool = False,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    level_positions = build_level_positions(level_order)
    extra_prediction_indexes = sorted(set(pred_records) - set(gold_records))

    sample_payloads: list[tuple[LoggingRecord, LoggingRecord | None, SampleScore, list[str], list[str]]] = []
    details: list[dict[str, Any]] = []
    by_language: dict[str, list[tuple[LoggingRecord, LoggingRecord | None, SampleScore, list[str], list[str]]]] = {}
    by_repo: dict[str, list[tuple[LoggingRecord, LoggingRecord | None, SampleScore, list[str], list[str]]]] = {}

    for index in sorted(gold_records):
        gold = gold_records[index]
        pred = pred_records.get(index)
        score, ref_tokens, hyp_tokens = score_sample(
            gold,
            pred,
            level_positions,
            pa_mode=pa_mode,
            target_line_change_hit=(
                None if target_line_change_hits is None else target_line_change_hits.get(index, False)
            ),
        )
        payload = (gold, pred, score, ref_tokens, hyp_tokens)
        sample_payloads.append(payload)
        by_language.setdefault(gold.language or "unknown", []).append(payload)
        by_repo.setdefault(gold.repo_name or "unknown", []).append(payload)

        details.append(
            {
                "index": gold.index,
                "language": gold.language,
                "repo_name": gold.repo_name,
                "missing_prediction": score.missing_prediction,
                "target_line_change_debug": (
                    None if target_line_change_debug is None else target_line_change_debug.get(index)
                ),
                "gold": {
                    "line": gold.line,
                    "framework": extract_framework_anchor(gold.statement, gold.language),
                    "level": gold.level,
                    "message": gold.message,
                    "vars": list(gold.vars),
                    "statement": gold.statement,
                },
                "pred": {
                    "line": None if pred is None else pred.line,
                    "framework": (
                        None
                        if pred is None
                        else extract_framework_anchor(pred.statement, gold.language)
                    ),
                    "level": None if pred is None else pred.level,
                    "message": "" if pred is None else pred.message,
                    "vars": [] if pred is None else list(pred.vars),
                    "statement": "" if pred is None else pred.statement,
                },
                "metrics": {
                    "PA": round_metric(score.pa),
                    "FA": round_metric(score.fa),
                    "LA": round_metric(score.la),
                    "AOD": round_metric(score.aod),
                    "ROUGE-L": round_metric(score.rouge_l),
                    "PMR": round_metric(score.pmr),
                    "Precision": round_metric(score.precision),
                    "Recall": round_metric(score.recall),
                    "F1": round_metric(score.f1),
                },
            }
        )

    by_repo_results = {
        repo_name: aggregate_group(payloads, condition_metrics_on_pa=condition_metrics_on_pa)
        for repo_name, payloads in sorted(by_repo.items())
    }
    repo_breakdown_by_language: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for repo_name, payloads in sorted(by_repo.items()):
        language = payloads[0][0].language or "unknown"
        repo_breakdown_by_language[language].append(by_repo_results[repo_name])

    by_language_sample_micro = {
        language: aggregate_group(payloads, condition_metrics_on_pa=condition_metrics_on_pa)
        for language, payloads in sorted(by_language.items())
    }
    by_language_repo_macro = {
        language: aggregate_repo_macro_language(
            payloads,
            repo_breakdown_by_language.get(language, []),
            condition_metrics_on_pa=condition_metrics_on_pa,
        )
        for language, payloads in sorted(by_language.items())
    }

    results: dict[str, Any] = {
        "counts": {
            "gold_samples": len(gold_records),
            "prediction_samples": len(pred_records),
            "missing_predictions": sum(1 for payload in sample_payloads if payload[2].missing_prediction),
            "extra_predictions": len(extra_prediction_indexes),
            "extra_prediction_indexes": extra_prediction_indexes,
        },
        "config": {
            "level_order": list(level_positions.keys()),
            "pa_mode": pa_mode,
            "pa_note": pa_mode_note(pa_mode),
            "condition_metrics_on_pa": condition_metrics_on_pa,
            "metric_conditioning_note": (
                "PA is always averaged over all samples. FA, LA, AOD, BLEU-4, ROUGE-L, "
                "PMR, Precision, Recall, and F1 are computed only on samples with PA=1. "
                "CCS is then derived from the aggregate PA, FA, LA, ROUGE-L, and F1 values."
                if condition_metrics_on_pa
                else "All metrics are computed on all samples."
            ),
            "framework_accuracy_note": (
                "FA is computed from target_log.statement by comparing the extracted "
                "strict lexical anchor. C++ stream-style logging uses the exact leading "
                "macro symbol; member-based APIs use the receiver expression before the "
                "earliest '.', '->', or '::' separator."
            ),
            "message_field_note": (
                "BLEU-4 and ROUGE-L are computed on target_log.message. In the current "
                "dataset schema this field corresponds to the extracted logging payload text."
            ),
            "variable_average": "macro-average over per-statement Precision/Recall/F1",
            "ccs_formula": "CCS = PA * (0.5 + 0.25 * FA + 0.25 * ((LA + ROUGE-L + F1) / 3))",
        },
        "overall": aggregate_group(
            sample_payloads,
            condition_metrics_on_pa=condition_metrics_on_pa,
        ),
        "by_language": by_language_repo_macro,
        "language_macro_average": {},
        "by_language_sample_micro": by_language_sample_micro,
        "language_sample_micro_average": {},
        "by_repo": by_repo_results,
    }
    results["language_macro_average"] = macro_average_aggregates(
        results["by_language"].values(),
        condition_metrics_on_pa=condition_metrics_on_pa,
    )
    results["language_sample_micro_average"] = macro_average_aggregates(
        results["by_language_sample_micro"].values(),
        condition_metrics_on_pa=condition_metrics_on_pa,
    )
    return results, details


def write_details(path: Path, details: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for item in details:
            handle.write(json.dumps(item, ensure_ascii=False) + "\n")


def write_predictions(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for item in rows:
            handle.write(json.dumps(item, ensure_ascii=False) + "\n")


def resolve_paths(args: argparse.Namespace) -> tuple[Path, Path, Path | None, Path | None, Path | None]:
    run_dir = args.run_dir
    gold_path = args.gold or (run_dir / DEFAULT_RUN_GOLD_NAME if run_dir is not None else None)
    pred_path = args.pred or (run_dir / DEFAULT_RUN_GENERATIONS_NAME if run_dir is not None else None)
    output_path = args.output or (run_dir / DEFAULT_RUN_METRICS_NAME if run_dir is not None else None)
    details_path = args.details_out or (run_dir / DEFAULT_RUN_DETAILS_NAME if run_dir is not None else None)
    predictions_path = (
        args.predictions_out
        or (run_dir / DEFAULT_RUN_PREDICTIONS_NAME if run_dir is not None else None)
    )
    if gold_path is None or pred_path is None:
        raise SystemExit("Please provide either --run-dir or both --gold and --pred.")
    return gold_path, pred_path, output_path, details_path, predictions_path


def print_metric_table(title: str, groups: dict[str, dict[str, Any]], average_key: str, average_label: str) -> None:
    ordered_names = [group_name for group_name in sorted(groups) if group_name != average_key]
    metric_header = " ".join(f"{metric_name:>9}" for metric_name in METRIC_NAMES)
    print(title, flush=True)
    print(
        f"{'group':<14} {'samples':>7} {'pa_ok':>7} {'metric_n':>9} {'missing':>7} {metric_header}",
        flush=True,
    )
    for group_name in ordered_names:
        payload = groups[group_name]
        sample_count = payload.get("sample_count", payload.get("group_count", 0))
        pa_correct_count = payload.get("pa_correct_count", sample_count)
        metric_sample_count = payload.get(
            "metric_sample_count",
            payload.get("metric_group_count", sample_count),
        )
        missing_predictions = payload.get("missing_predictions", 0)
        metrics_text = " ".join(f"{payload['metrics'][metric_name]:>9.6f}" for metric_name in METRIC_NAMES)
        print(
            (
                f"{group_name:<14} {sample_count:>7} {pa_correct_count:>7} "
                f"{metric_sample_count:>9} {missing_predictions:>7} {metrics_text}"
            ),
            flush=True,
        )
    average_payload = groups[average_key] if average_key in groups else None
    if average_payload is None:
        return
    average_count = average_payload.get("sample_count", average_payload.get("group_count", 0))
    average_pa_ok = average_payload.get("pa_correct_count", average_count)
    average_metric_count = average_payload.get(
        "metric_sample_count",
        average_payload.get("metric_group_count", average_count),
    )
    average_missing = average_payload.get("missing_predictions", 0)
    metrics_text = " ".join(
        f"{average_payload['metrics'][metric_name]:>9.6f}" for metric_name in METRIC_NAMES
    )
    print(
        (
            f"{average_label:<14} {average_count:>7} {average_pa_ok:>7} "
            f"{average_metric_count:>9} {average_missing:>7} {metrics_text}"
        ),
        flush=True,
    )


def main() -> None:
    args = parse_args()
    level_order = tuple(item.strip() for item in args.level_order.split(",") if item.strip())
    gold_path, pred_path, output_path, details_path, predictions_path = resolve_paths(args)

    gold_rows = load_gold_rows(gold_path)
    gold_records = load_gold_records_from_rows(gold_rows)
    pred_records, normalized_prediction_rows, prediction_metadata = load_prediction_records_with_metadata(
        pred_path,
        gold_rows=gold_rows,
        pred_format=args.pred_format,
    )
    effective_pa_mode = args.pa_mode
    if effective_pa_mode == "auto":
        effective_pa_mode = (
            "target-line-change"
            if prediction_metadata.get("format") == "generations"
            else "exact-line"
        )

    target_line_change_hits: dict[int, bool] | None = None
    target_line_change_debug: dict[int, dict[str, Any]] | None = None
    if effective_pa_mode == "target-line-change":
        pa_results = compute_pa_results(
            pred_path,
            gold_rows=gold_rows,
            pred_format=args.pred_format,
            pa_mode=effective_pa_mode,
        )
        target_line_change_hits = {index: result.hit for index, result in pa_results.items()}
        target_line_change_debug = {
            index: result.debug
            for index, result in pa_results.items()
            if result.debug is not None
        }
        attach_target_line_prediction_logs(normalized_prediction_rows, pa_results)
    results, details = evaluate_predictions(
        gold_records,
        pred_records,
        level_order=level_order or DEFAULT_LEVEL_ORDER,
        pa_mode=effective_pa_mode,
        target_line_change_hits=target_line_change_hits,
        target_line_change_debug=target_line_change_debug,
        condition_metrics_on_pa=args.condition_metrics_on_pa,
    )
    results["prediction_input"] = {
        "gold_path": str(gold_path),
        "pred_path": str(pred_path),
        "requested_pa_mode": args.pa_mode,
        **prediction_metadata,
    }

    if not args.include_by_repo:
        results.pop("by_repo", None)

    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(results, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    if details_path is not None:
        write_details(details_path, details)

    if predictions_path is not None:
        write_predictions(predictions_path, normalized_prediction_rows)

    overall = results["overall"]
    metric_summary = ", ".join(
        f"{name}={value:.6f}" for name, value in overall["metrics"].items()
    )
    print(
        (
            f"Evaluated {overall['sample_count']} samples "
            f"(pa_correct={overall['pa_correct_count']}, metric_sample_count={overall['metric_sample_count']}, "
            f"missing_predictions={overall['missing_predictions']}, "
            f"extra_predictions={results['counts']['extra_predictions']}). "
            f"{metric_summary}"
        ),
        flush=True,
    )
    print_metric_table(
        title="By-language repo-macro results:",
        groups={
            **results["by_language"],
            "__avg__": results["language_macro_average"],
        },
        average_key="__avg__",
        average_label="lang-avg",
    )


if __name__ == "__main__":
    main()

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Iterable, Iterator

from .config import LanguageSpec


DIRECT_CALL_NODE_TYPES = {
    "java": {"method_invocation"},
    "python": {"call"},
    "go": {"call_expression"},
    "cpp": {"call_expression"},
    "javascript": {"call_expression"},
    "csharp": {"invocation_expression"},
}

IDENTIFIER_NODE_TYPES = {
    "java": {"identifier"},
    "python": {"identifier"},
    "go": {"identifier", "field_identifier"},
    "cpp": {"identifier", "field_identifier", "namespace_identifier", "type_identifier"},
    "javascript": {"identifier", "property_identifier", "shorthand_property_identifier_pattern"},
    "csharp": {"identifier"},
}

MEMBER_ACCESS_NODE_TYPES = {
    "java": {"field_access"},
    "python": {"attribute"},
    "go": {"selector_expression", "index_expression"},
    "cpp": {
        "field_expression",
        "qualified_identifier",
        "scoped_identifier",
        "subscript_expression",
    },
    "javascript": {"member_expression", "subscript_expression"},
    "csharp": {
        "member_access_expression",
        "conditional_access_expression",
        "element_binding_expression",
        "element_access_expression",
    },
}

CALL_LIKE_NODE_TYPES = {
    "java": {"method_invocation", "object_creation_expression"},
    "python": {"call"},
    "go": {"call_expression"},
    "cpp": {"call_expression"},
    "javascript": {"call_expression", "new_expression"},
    "csharp": {"invocation_expression", "object_creation_expression"},
}

STRINGISH_NODE_FRAGMENTS = (
    "string",
    "template",
    "char",
)

LITERAL_NODE_TYPES = {
    "number",
    "integer",
    "float",
    "true",
    "false",
    "null",
    "none",
    "nil",
}

CPP_STREAM_MACROS = {"LOG", "DLOG", "VLOG", "ABSL_LOG"}

CPP_STD_STREAM_SENTINELS = {"std::endl", "endl", "std::flush", "flush"}


@dataclass(frozen=True)
class CallableOwner:
    node: Any
    name: str
    class_name: str | None


@dataclass(frozen=True)
class CallableIdentity:
    name: str
    class_name: str | None


@dataclass(frozen=True)
class LogDetection:
    statement_node: Any
    log_node: Any
    level: str
    framework: str
    payload_text: str
    vars: list[str]
    is_exception_related: bool


def iter_nodes(root: Any) -> Iterator[Any]:
    stack = [root]
    while stack:
        node = stack.pop()
        yield node
        children = getattr(node, "children", None) or []
        for child in reversed(children):
            stack.append(child)


def node_text(source: bytes, node: Any) -> str:
    return source[node.start_byte : node.end_byte].decode("utf-8", errors="ignore")


def unique_in_order(items: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    ordered: list[str] = []
    for item in items:
        cleaned = item.strip()
        if not cleaned or cleaned in seen:
            continue
        seen.add(cleaned)
        ordered.append(cleaned)
    return ordered


def trim_argument_list(raw_text: str) -> str:
    stripped = raw_text.strip()
    if stripped.startswith("(") and stripped.endswith(")"):
        return stripped[1:-1].strip()
    return stripped


def looks_like_string_literal(raw_text: str) -> bool:
    stripped = raw_text.strip()
    return bool(
        stripped.startswith(('"', "'", '`', 'L"', 'u8"', 'R"', '@"'))
        or stripped.endswith(('"', "'", '`'))
    )


def is_stringish_node(node: Any) -> bool:
    lowered = node.type.lower()
    return any(fragment in lowered for fragment in STRINGISH_NODE_FRAGMENTS)


def is_literal_node(node: Any) -> bool:
    lowered = node.type.lower()
    return lowered in LITERAL_NODE_TYPES or is_stringish_node(node)


def normalize_level(raw_level: str) -> str:
    lowered = raw_level.casefold()
    if "information" in lowered:
        return "info"
    if lowered in {"warn", "warning"} or "warning" in lowered:
        return "warn"
    if lowered in {"critical", "fatal", "panic"} or "fatal" in lowered or "critical" in lowered:
        return "fatal"
    if lowered == "exception":
        return "error"
    if lowered.startswith("infos"):
        return "info"
    if lowered.startswith("errors"):
        return "error"
    if lowered.startswith("vlog"):
        return "debug"
    for suffix in ("f", "ln", "s"):
        if lowered.endswith(suffix) and len(lowered) > len(suffix):
            trimmed = lowered[: -len(suffix)]
            if trimmed != lowered:
                normalized_trimmed = normalize_level(trimmed)
                if normalized_trimmed != trimmed or trimmed in {
                    "trace",
                    "debug",
                    "info",
                    "warn",
                    "warning",
                    "error",
                    "fatal",
                    "critical",
                    "panic",
                }:
                    return normalized_trimmed
    if lowered.endswith("s") and lowered[:-1] in {"info", "error", "warn"}:
        return lowered[:-1]
    return lowered


def extract_callee_and_arguments(node: Any, source: bytes, language: str) -> tuple[str, Any | None]:
    if language == "java":
        receiver = node.child_by_field_name("object")
        name_node = node.child_by_field_name("name")
        args_node = node.child_by_field_name("arguments")
        name = node_text(source, name_node) if name_node else ""
        if receiver:
            return f"{node_text(source, receiver)}.{name}", args_node
        return name, args_node

    function_node = node.child_by_field_name("function")
    args_node = node.child_by_field_name("arguments")
    callee = node_text(source, function_node) if function_node else ""
    return callee, args_node


def last_symbol_token(callee_text: str) -> str:
    matches = re.findall(r"[A-Za-z_][A-Za-z0-9_]*", callee_text)
    return matches[-1] if matches else ""


def direct_call_framework_and_level(
    language: str,
    callee_text: str,
    full_call_text: str,
) -> tuple[str, str] | None:
    lowered = callee_text.casefold()
    token = normalize_level(last_symbol_token(callee_text))
    full_lower = full_call_text.casefold()

    if language == "java":
        if token in {"trace", "debug", "info", "warn", "error", "fatal"} and any(
            marker in lowered for marker in ("log", "logger")
        ):
            return "slf4j-like", token
        return None

    if language == "python":
        if token in {"debug", "info", "warn", "error", "fatal"} and any(
            marker in lowered for marker in ("logging", "logger", ".log")
        ):
            return "python-logging", token
        return None

    if language == "go":
        if token in {"debug", "info", "warn", "error", "fatal"} and (
            any(marker in lowered for marker in ("log", "logger", "slog", "zap", "klog"))
            or ".l()" in lowered
            or ".logger" in lowered
        ):
            if "zap" in lowered or "zap." in full_lower:
                return "zap", token
            if "slog" in lowered or "slog." in full_lower:
                return "slog", token
            if "klog" in lowered:
                return "klog", token
            if lowered.startswith("log.") or lowered == "log":
                return "stdlib-log", token
            return "generic-logger", token
        return None

    if language == "cpp":
        if token in {"trace", "debug", "info", "warn", "error", "fatal"} and (
            "spdlog" in lowered or "log" in lowered or "logger" in lowered
        ):
            framework = "spdlog" if "spdlog" in lowered else "generic-cpp-logger"
            return framework, token
        return None

    if language == "javascript":
        if token in {"log", "debug", "info", "warn", "error", "fatal"} and "console" in lowered:
            return "console", "info" if token == "log" else token
        if token in {"trace", "debug", "info", "warn", "error", "fatal"} and any(
            marker in lowered for marker in ("log", "logger")
        ):
            return "generic-js-logger", token
        return None

    if language == "csharp":
        if "loginformation" in lowered:
            return "microsoft-extensions-logging", "info"
        if "logwarning" in lowered:
            return "microsoft-extensions-logging", "warn"
        if "logerror" in lowered:
            return "microsoft-extensions-logging", "error"
        if "logdebug" in lowered:
            return "microsoft-extensions-logging", "debug"
        if "logtrace" in lowered:
            return "microsoft-extensions-logging", "trace"
        if "logcritical" in lowered:
            return "microsoft-extensions-logging", "fatal"
        if lowered.startswith("log.") or lowered == "log":
            if token in {"debug", "info", "warn", "error", "fatal"}:
                return "serilog-like", token
        if token in {"debug", "info", "warn", "error", "fatal"} and any(
            marker in lowered for marker in ("logger", "_logger", "log")
        ):
            return "generic-csharp-logger", token
        return None

    return None


def unwrap_argument_node(node: Any, language: str) -> Any:
    if language == "csharp" and node.type == "argument" and node.named_children:
        return node.named_children[-1]
    if language == "python" and node.type == "keyword_argument":
        value = node.child_by_field_name("value")
        return value or node
    if language == "javascript" and node.type == "pair":
        value = node.child_by_field_name("value")
        return value or node
    return node


def collect_var_texts(node: Any, source: bytes, language: str) -> list[str]:
    current = unwrap_argument_node(node, language)
    if current is None:
        return []
    if is_stringish_node(current):
        nested = []
        for child in current.named_children:
            nested.extend(collect_var_texts(child, source, language))
        return unique_in_order(nested)
    if is_literal_node(current):
        return []

    current_text = node_text(source, current).strip()
    if language == "go" and current.type == "call_expression":
        callee_text, args_node = extract_callee_and_arguments(current, source, language)
        callee_lower = callee_text.casefold()
        if args_node and (
            callee_lower.startswith("zap.")
            or ".zap." in callee_lower
            or callee_lower.startswith("slog.")
            or ".slog." in callee_lower
        ):
            vars_from_fields: list[str] = []
            for child in list(args_node.named_children)[1:]:
                vars_from_fields.extend(collect_var_texts(child, source, language))
            return unique_in_order(vars_from_fields)

    if current.type in CALL_LIKE_NODE_TYPES[language]:
        detected = direct_call_framework_and_level(language, extract_callee_and_arguments(current, source, language)[0], current_text)
        if detected is None:
            return [current_text]

    if current.type in MEMBER_ACCESS_NODE_TYPES[language]:
        return [current_text]
    if current.type in IDENTIFIER_NODE_TYPES[language]:
        return [current_text]

    nested: list[str] = []
    for child in current.named_children:
        nested.extend(collect_var_texts(child, source, language))
    return unique_in_order(nested)


def nearest_statement_node(node: Any) -> Any | None:
    current = node
    while current is not None:
        if current.type.endswith("statement"):
            return current
        current = current.parent
    return None


def is_exception_related(level: str, vars_list: list[str], payload_text: str) -> bool:
    if level == "error" and "exception" in payload_text.casefold():
        return True
    if re.search(r"\b(exception|error|err|traceback|stacktrace|throwable|ex)\b", payload_text.casefold()):
        return True
    return any(
        re.search(r"\b(exception|error|err|throwable|traceback|ex)\b", item.casefold())
        for item in vars_list
    )


def cpp_stream_detection(statement_node: Any, source: bytes) -> LogDetection | None:
    statement_text = node_text(source, statement_node).strip()
    if not statement_text:
        return None
    expr_text = statement_text[:-1] if statement_text.endswith(";") else statement_text
    macro_match = re.match(r"^(?P<macro>LOG|DLOG|VLOG|ABSL_LOG)\s*\((?P<level>[^)]*)\)\s*(?P<body>.*)$", expr_text, re.S)
    if not macro_match:
        return None
    body = macro_match.group("body").strip()
    if "<<" not in body:
        return None

    raw_level = macro_match.group("level").split(",", 1)[0].strip()
    normalized_level = normalize_level(raw_level)
    if macro_match.group("macro") == "VLOG":
        normalized_level = "debug"
    payload_text = body
    if payload_text.startswith("<<"):
        payload_text = payload_text[2:].strip()
    chunks = [chunk.strip() for chunk in payload_text.split("<<")]
    vars_list = [
        chunk
        for chunk in chunks
        if chunk
        and chunk not in CPP_STD_STREAM_SENTINELS
        and not looks_like_string_literal(chunk)
    ]
    vars_list = unique_in_order(vars_list)
    return LogDetection(
        statement_node=statement_node,
        log_node=statement_node,
        level=normalized_level,
        framework="glog-like",
        payload_text=payload_text,
        vars=vars_list,
        is_exception_related=is_exception_related(normalized_level, vars_list, payload_text),
    )


def declarator_name(node: Any, source: bytes) -> str | None:
    current = node
    visited: set[int] = set()
    while current is not None and current.id not in visited:
        visited.add(current.id)
        name_node = current.child_by_field_name("name")
        if name_node is not None:
            return node_text(source, name_node)
        if current.type in {"identifier", "field_identifier", "property_identifier", "type_identifier", "namespace_identifier"}:
            return node_text(source, current)
        next_node = (
            current.child_by_field_name("declarator")
            or current.child_by_field_name("function")
            or current.child_by_field_name("field")
        )
        if next_node is None:
            for child in current.named_children:
                if child.type in {"identifier", "field_identifier", "property_identifier", "type_identifier", "qualified_identifier", "scoped_identifier"}:
                    return node_text(source, child)
            return None
        current = next_node
    return None


def callable_identity(owner: CallableOwner) -> CallableIdentity:
    return CallableIdentity(name=owner.name, class_name=owner.class_name)


def extract_detections(spec: LanguageSpec, root: Any, source: bytes) -> list[LogDetection]:
    detections: list[LogDetection] = []
    seen_statements: set[tuple[int, int]] = set()
    for node in iter_nodes(root):
        if spec.key == "cpp" and node.type == "expression_statement":
            cpp_detection = cpp_stream_detection(node, source)
            if cpp_detection is not None:
                key = (node.start_byte, node.end_byte)
                if key not in seen_statements:
                    detections.append(cpp_detection)
                    seen_statements.add(key)
                continue

        if node.type not in DIRECT_CALL_NODE_TYPES[spec.key]:
            continue
        callee_text, args_node = extract_callee_and_arguments(node, source, spec.key)
        direct_info = direct_call_framework_and_level(spec.key, callee_text, node_text(source, node))
        if direct_info is None or args_node is None:
            continue
        statement_node = nearest_statement_node(node)
        if statement_node is None:
            continue
        key = (statement_node.start_byte, statement_node.end_byte)
        if key in seen_statements:
            continue

        arg_nodes = list(args_node.named_children)
        payload_text = trim_argument_list(node_text(source, args_node))
        vars_list: list[str] = []
        for arg_node in arg_nodes:
            vars_list.extend(collect_var_texts(arg_node, source, spec.key))
        vars_list = unique_in_order(vars_list)

        level = direct_info[1]
        detections.append(
            LogDetection(
                statement_node=statement_node,
                log_node=node,
                level=level,
                framework=direct_info[0],
                payload_text=payload_text,
                vars=vars_list,
                is_exception_related=is_exception_related(level, vars_list, payload_text),
            )
        )
        seen_statements.add(key)
    return detections


def callable_name_from_node(node: Any, spec: LanguageSpec, source: bytes) -> str | None:
    name_node = node.child_by_field_name("name")
    if name_node is not None:
        return node_text(source, name_node)

    declarator = node.child_by_field_name("declarator")
    if declarator is not None:
        return declarator_name(declarator, source)

    if spec.key == "javascript":
        parent = node.parent
        if parent is not None:
            if parent.type == "variable_declarator":
                name_node = parent.child_by_field_name("name") or (
                    parent.named_children[0] if parent.named_children else None
                )
                if name_node is not None:
                    return node_text(source, name_node)
            if parent.type == "pair":
                key_node = parent.child_by_field_name("key")
                if key_node is not None:
                    return node_text(source, key_node)
            if parent.type == "assignment_expression":
                left = parent.child_by_field_name("left")
                if left is not None:
                    return node_text(source, left)
    return None


def nearest_class_name(node: Any | None, spec: LanguageSpec, source: bytes) -> str | None:
    if spec.key not in {"java", "python", "javascript", "csharp", "cpp"}:
        return None
    current = node
    while current is not None:
        if current.type in spec.class_types:
            name_node = current.child_by_field_name("name")
            if name_node is None and current.named_children:
                for child in current.named_children:
                    if child.type in {
                        "identifier",
                        "type_identifier",
                        "property_identifier",
                    }:
                        name_node = child
                        break
            if name_node is not None:
                return node_text(source, name_node)
            return "A"
        current = current.parent
    return None


def find_named_callable(spec: LanguageSpec, node: Any, source: bytes) -> CallableOwner | None:
    current = node
    while current is not None:
        if current.type in spec.callable_types:
            name = callable_name_from_node(current, spec, source)
            if name:
                class_name = nearest_class_name(current.parent, spec, source)
                return CallableOwner(node=current, name=name, class_name=class_name)
        current = current.parent
    return None


def iter_named_callables(spec: LanguageSpec, root: Any, source: bytes) -> Iterator[CallableOwner]:
    for node in iter_nodes(root):
        if node.type not in spec.callable_types:
            continue
        name = callable_name_from_node(node, spec, source)
        if not name:
            continue
        yield CallableOwner(
            node=node,
            name=name,
            class_name=nearest_class_name(node.parent, spec, source),
        )


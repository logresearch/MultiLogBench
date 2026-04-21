from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

@dataclass(frozen=True)
class LanguageSpec:
    key: str
    parser_by_suffix: dict[str, str]
    callable_types: set[str]
    class_types: set[str]
    candidate_patterns: tuple[re.Pattern[str], ...]

    @property
    def extensions(self) -> set[str]:
        return set(self.parser_by_suffix)

    def parser_name_for_path(self, path: Path) -> str:
        return self.parser_by_suffix[path.suffix.lower()]


LANGUAGE_SPECS: dict[str, LanguageSpec] = {
    "java": LanguageSpec(
        key="java",
        parser_by_suffix={".java": "java"},
        callable_types={"method_declaration", "constructor_declaration"},
        class_types={
            "class_declaration",
            "interface_declaration",
            "enum_declaration",
            "record_declaration",
        },
        candidate_patterns=(
            re.compile(
                r"\b(?:logger|log|LOGGER|LOG)\s*\.\s*(?:trace|debug|info|warn|error|fatal)\s*\(",
                re.I,
            ),
        ),
    ),
    "python": LanguageSpec(
        key="python",
        parser_by_suffix={".py": "python"},
        callable_types={"function_definition", "async_function_definition"},
        class_types={"class_definition"},
        candidate_patterns=(
            re.compile(
                r"\b(?:logging|logger|log)\s*\.\s*(?:debug|info|warning|warn|error|exception|critical)\s*\(",
                re.I,
            ),
        ),
    ),
    "go": LanguageSpec(
        key="go",
        parser_by_suffix={".go": "go"},
        callable_types={"function_declaration", "method_declaration"},
        class_types=set(),
        candidate_patterns=(
            re.compile(r"\bslog\s*\.\s*(?:Debug|Info|Warn|Error)\s*\("),
            re.compile(r"\bklog\s*\.\s*(?:Info|InfoS|Warning|Error|ErrorS|Fatal)\s*\("),
            re.compile(r"\bzap\s*\.\s*L\s*\(\s*\)\s*\.\s*(?:Debug|Info|Warn|Error|Fatal|Panic)\s*\("),
            re.compile(
                r"\b(?:logger|log|Logger|Log)\s*\.\s*(?:Trace|Debug|Info|Warn|Error|Fatal|Panic|InfoS|ErrorS)\s*\(",
            ),
        ),
    ),
    "cpp": LanguageSpec(
        key="cpp",
        parser_by_suffix={
            ".c": "cpp",
            ".cc": "cpp",
            ".cpp": "cpp",
            ".cxx": "cpp",
            ".h": "cpp",
            ".hh": "cpp",
            ".hpp": "cpp",
            ".hxx": "cpp",
        },
        callable_types={"function_definition"},
        class_types={"class_specifier", "struct_specifier"},
        candidate_patterns=(
            re.compile(r"\b(?:LOG|DLOG|VLOG|ABSL_LOG)\s*\("),
            re.compile(r"\bspdlog\s*::\s*(?:trace|debug|info|warn|error|critical)\s*\("),
            re.compile(
                r"\b(?:logger|log|LOGGER|LOG)[A-Za-z_0-9]*\s*(?:->|\.)\s*(?:trace|debug|info|warn|error|critical)\s*\(",
                re.I,
            ),
        ),
    ),
    "javascript": LanguageSpec(
        key="javascript",
        parser_by_suffix={
            ".js": "javascript",
            ".jsx": "javascript",
            ".mjs": "javascript",
            ".cjs": "javascript",
            ".ts": "typescript",
            ".tsx": "tsx",
        },
        callable_types={
            "function_declaration",
            "method_definition",
            "generator_function_declaration",
            "function",
            "arrow_function",
        },
        class_types={"class_declaration", "class"},
        candidate_patterns=(
            re.compile(r"\bconsole\s*\.\s*(?:log|debug|info|warn|error)\s*\("),
            re.compile(
                r"\b(?:logger|log|LOGGER|LOG)\s*\.\s*(?:trace|debug|info|warn|error|fatal)\s*\(",
                re.I,
            ),
        ),
    ),
    "csharp": LanguageSpec(
        key="csharp",
        parser_by_suffix={".cs": "c_sharp"},
        callable_types={
            "method_declaration",
            "constructor_declaration",
            "local_function_statement",
        },
        class_types={
            "class_declaration",
            "struct_declaration",
            "record_declaration",
        },
        candidate_patterns=(
            re.compile(
                r"\b(?:_logger|logger|Logger|LOGGER)\s*\.\s*Log(?:Trace|Debug|Information|Warning|Error|Critical)\s*\(",
            ),
            re.compile(
                r"\b(?:_logger|logger|Logger|LOGGER|Log)\s*\.\s*(?:Verbose|Debug|Information|Warning|Error|Fatal)\s*\(",
            ),
        ),
    ),
}

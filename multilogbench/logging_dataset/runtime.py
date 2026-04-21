from __future__ import annotations

import warnings
from functools import lru_cache


def ensure_vendor_path() -> None:
    return None


@lru_cache(maxsize=None)
def get_tree_sitter_parser(language_name: str):
    ensure_vendor_path()
    warnings.filterwarnings("ignore", category=FutureWarning, module="tree_sitter")
    from tree_sitter_languages import get_parser

    return get_parser(language_name)

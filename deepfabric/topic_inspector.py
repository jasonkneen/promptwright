"""Topic file inspection utilities for deepfabric CLI."""

import json

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from .graph import Graph
from .utils import read_topic_tree_from_jsonl


@dataclass
class TopicInspectionResult:
    """Result of inspecting a topic file."""

    format: Literal["tree", "graph"]
    total_paths: int
    max_depth: int
    paths_at_level: list[list[str]] | None
    expanded_paths: list[list[str]] | None  # Paths from level onwards (with --expand)
    all_paths: list[list[str]] | None
    metadata: dict[str, Any]
    source_file: str


def detect_format(file_path: str) -> Literal["tree", "graph"]:
    """Auto-detect topic file format based on content.

    Args:
        file_path: Path to the topic file

    Returns:
        "tree" for JSONL format, "graph" for JSON format

    Raises:
        ValueError: If format cannot be detected
        FileNotFoundError: If file doesn't exist
    """
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    with open(file_path, encoding="utf-8") as f:
        content = f.read().strip()
        if not content:
            raise ValueError("Empty file")

        # Try to parse as a complete JSON object (Graph format)
        try:
            data = json.loads(content)
            if isinstance(data, dict):
                # Graph format has "nodes" and "root_id" keys
                if "nodes" in data and "root_id" in data:
                    return "graph"
        except json.JSONDecodeError:
            pass

        # Try to parse first line as JSONL (Tree format)
        first_line = content.split("\n")[0].strip()
        try:
            first_obj = json.loads(first_line)
            if isinstance(first_obj, dict) and "path" in first_obj:
                return "tree"
        except json.JSONDecodeError:
            pass

    raise ValueError(f"Unable to detect format for: {file_path}")


def _load_tree_paths(file_path: str) -> list[list[str]]:
    """Load tree paths directly from JSONL without initializing LLM.

    Args:
        file_path: Path to the JSONL file

    Returns:
        List of paths (each path is a list of topic strings)
    """
    dict_list = read_topic_tree_from_jsonl(file_path)
    return [d["path"] for d in dict_list if "path" in d]


def _load_graph_data(file_path: str) -> tuple[list[list[str]], dict[str, Any]]:
    """Load graph data and extract paths and metadata.

    Args:
        file_path: Path to the JSON file

    Returns:
        Tuple of (paths, metadata)
    """
    # Load Graph - need minimal params for from_json
    params = {
        "topic_prompt": "loaded",
        "model_name": "placeholder/model",
        "degree": 3,
        "depth": 2,
        "temperature": 0.7,
    }
    graph = Graph.from_json(file_path, params)

    all_paths = graph.get_all_paths()
    metadata: dict[str, Any] = {
        "total_nodes": len(graph.nodes),
        "has_cycles": graph.has_cycle(),
        "root_topic": graph.root.topic if graph.root else None,
    }

    # Read graph-level metadata directly from the JSON file
    # since Graph.from_json doesn't restore provider/model
    with open(file_path, encoding="utf-8") as f:
        raw_data = json.load(f)

    if "metadata" in raw_data and raw_data["metadata"]:
        file_metadata = raw_data["metadata"]
        if file_metadata.get("created_at"):
            metadata["created_at"] = file_metadata["created_at"]
        if file_metadata.get("provider"):
            metadata["provider"] = file_metadata["provider"]
        if file_metadata.get("model"):
            metadata["model"] = file_metadata["model"]

    return all_paths, metadata


def inspect_topic_file(
    file_path: str,
    level: int | None = None,
    expand_depth: int | None = None,
    show_all: bool = False,
) -> TopicInspectionResult:
    """Inspect a topic file and return structured results.

    Args:
        file_path: Path to the topic file
        level: Specific level to show (0=root), or None
        expand_depth: Number of sublevels to show (-1 for all), or None for no expansion
        show_all: Whether to include all paths in result

    Returns:
        TopicInspectionResult with inspection data
    """
    format_type = detect_format(file_path)

    # Load paths and metadata based on format
    if format_type == "graph":
        all_paths, metadata = _load_graph_data(file_path)
    else:
        all_paths = _load_tree_paths(file_path)
        # Extract root topic from paths
        metadata = {}
        if all_paths:
            metadata["root_topic"] = all_paths[0][0]

    max_depth = max(len(p) for p in all_paths) if all_paths else 0

    # Get unique topics at specific level if requested
    # Level 0 = root, Level 1 = children of root, etc.
    paths_at_level = None
    expanded_paths = None

    if level is not None:
        # Extract unique topic names at the given depth position
        seen_topics: set[str] = set()
        unique_topics: list[str] = []
        for path in all_paths:
            if len(path) > level:
                topic_at_level = path[level]
                if topic_at_level not in seen_topics:
                    seen_topics.add(topic_at_level)
                    unique_topics.append(topic_at_level)
        # Store as single-element paths for consistency
        paths_at_level = [[t] for t in unique_topics]

        # If expand_depth is set, get paths from level onwards
        if expand_depth is not None:
            seen_paths: set[tuple[str, ...]] = set()
            expanded_paths = []
            for path in all_paths:
                if len(path) > level:
                    # Trim path to start from the specified level
                    trimmed_path = path[level:]
                    # Limit depth if expand_depth is not -1
                    if expand_depth != -1 and len(trimmed_path) > expand_depth + 1:
                        trimmed_path = trimmed_path[: expand_depth + 1]
                    # Deduplicate paths (after trimming, many may be identical)
                    path_key = tuple(trimmed_path)
                    if path_key not in seen_paths:
                        seen_paths.add(path_key)
                        expanded_paths.append(trimmed_path)

    return TopicInspectionResult(
        format=format_type,
        total_paths=len(all_paths),
        max_depth=max_depth,
        paths_at_level=paths_at_level,
        expanded_paths=expanded_paths,
        all_paths=all_paths if show_all else None,
        metadata=metadata,
        source_file=file_path,
    )

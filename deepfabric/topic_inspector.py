"""Topic file inspection utilities for deepfabric CLI."""

import hashlib
import json

from dataclasses import dataclass, field
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
    # Maps path tuple to UUID/topic_id (for --uuid flag)
    path_to_uuid: dict[tuple[str, ...], str] = field(default_factory=dict)
    # Maps topic name to UUID (for graph format, all nodes)
    topic_to_uuid: dict[str, str] = field(default_factory=dict)


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
            if isinstance(data, dict) and "nodes" in data and "root_id" in data:
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


def _load_tree_paths(file_path: str) -> tuple[list[list[str]], dict[tuple[str, ...], str]]:
    """Load tree paths directly from JSONL without initializing LLM.

    Args:
        file_path: Path to the JSONL file

    Returns:
        Tuple of (paths, path_to_uuid mapping)
    """
    dict_list = read_topic_tree_from_jsonl(file_path)
    paths = []
    path_to_uuid: dict[tuple[str, ...], str] = {}

    for d in dict_list:
        if "path" not in d:
            continue
        path = d["path"]
        paths.append(path)
        # Generate hash-based ID from path (same as tree.py)
        path_str = " > ".join(path)
        topic_id = hashlib.sha256(path_str.encode()).hexdigest()[:16]
        path_to_uuid[tuple(path)] = topic_id

    return paths, path_to_uuid


def _load_graph_data(
    file_path: str,
) -> tuple[list[list[str]], dict[str, Any], dict[tuple[str, ...], str], dict[str, str]]:
    """Load graph data and extract paths and metadata.

    Args:
        file_path: Path to the JSON file

    Returns:
        Tuple of (paths, metadata, path_to_uuid mapping, topic_to_uuid mapping)
    """
    graph = Graph.load(file_path)

    # Get paths with UUIDs (for leaf nodes)
    paths_with_ids = graph.get_all_paths_with_ids()
    all_paths = [tp.path for tp in paths_with_ids]
    path_to_uuid: dict[tuple[str, ...], str] = {
        tuple(tp.path): tp.topic_id for tp in paths_with_ids
    }

    # Build topic name to UUID mapping for ALL nodes (not just leaves)
    topic_to_uuid: dict[str, str] = {}
    for node in graph.nodes.values():
        node_uuid = node.metadata.get("uuid", "")
        if node_uuid:
            topic_to_uuid[node.topic] = node_uuid

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

    return all_paths, metadata, path_to_uuid, topic_to_uuid


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
    topic_to_uuid: dict[str, str] = {}
    if format_type == "graph":
        all_paths, metadata, path_to_uuid, topic_to_uuid = _load_graph_data(file_path)
    else:
        all_paths, path_to_uuid = _load_tree_paths(file_path)
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
                    # If this topic is a leaf (path ends at level+1), map single-topic to UUID
                    if len(path) == level + 1:
                        original_uuid = path_to_uuid.get(tuple(path), "")
                        if original_uuid:
                            path_to_uuid[(topic_at_level,)] = original_uuid
        # Store as single-element paths for consistency
        paths_at_level = [[t] for t in unique_topics]

        # If expand_depth is set, get paths from level onwards
        if expand_depth is not None:
            seen_paths: set[tuple[str, ...]] = set()
            expanded_paths = []
            for path in all_paths:
                if len(path) > level:
                    original_uuid = path_to_uuid.get(tuple(path), "")
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
                        # Map trimmed path to original UUID (for --uuid display)
                        if original_uuid and path_key not in path_to_uuid:
                            path_to_uuid[path_key] = original_uuid

    return TopicInspectionResult(
        format=format_type,
        total_paths=len(all_paths),
        max_depth=max_depth,
        paths_at_level=paths_at_level,
        expanded_paths=expanded_paths,
        all_paths=all_paths if show_all else None,
        metadata=metadata,
        source_file=file_path,
        path_to_uuid=path_to_uuid,
        topic_to_uuid=topic_to_uuid,
    )

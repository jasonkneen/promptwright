"""Graph pruning operations for deepfabric CLI."""

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from .graph import Graph


@dataclass
class PruneResult:
    """Result of a pruning operation."""

    operation: Literal["level", "uuid"]
    removed_count: int
    removed_node_ids: list[int]
    remaining_nodes: int
    remaining_paths: int
    output_path: str


def load_graph_for_pruning(file_path: str) -> Graph:
    """Load a graph from JSON for pruning operations.

    Args:
        file_path: Path to the graph JSON file.

    Returns:
        Loaded Graph instance.

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If the file is not a JSON graph file.
    """
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"Graph file not found: {file_path}")
    if path.suffix != ".json":
        raise ValueError(
            f"Expected a JSON graph file, got: {path.suffix}. "
            "Pruning is only supported for graph format files."
        )
    return Graph.load(file_path)


def prune_graph_at_level(
    file_path: str,
    max_depth: int,
    output_path: str | None = None,
) -> PruneResult:
    """Prune a graph file by removing all nodes below a depth level.

    Args:
        file_path: Path to the input graph JSON file.
        max_depth: Maximum depth to keep (0=root only, 1=root+children, etc.).
        output_path: Output file path. If None, derives from input filename.

    Returns:
        PruneResult with operation details.
    """
    graph = load_graph_for_pruning(file_path)
    removed_ids = graph.prune_at_level(max_depth)

    final_output = output_path or _derive_output_path(file_path, f"pruned_level{max_depth}")
    graph.save(final_output)

    return PruneResult(
        operation="level",
        removed_count=len(removed_ids),
        removed_node_ids=removed_ids,
        remaining_nodes=len(graph.nodes),
        remaining_paths=len(graph.get_all_paths()),
        output_path=final_output,
    )


def prune_graph_by_uuid(
    file_path: str,
    uuid: str,
    output_path: str | None = None,
) -> PruneResult:
    """Remove a node (by UUID) and its entire subtree from a graph file.

    Args:
        file_path: Path to the input graph JSON file.
        uuid: UUID of the node to remove.
        output_path: Output file path. If None, derives from input filename.

    Returns:
        PruneResult with operation details.

    Raises:
        ValueError: If UUID not found or targets the root node.
    """
    graph = load_graph_for_pruning(file_path)
    node = graph.find_node_by_uuid(uuid)

    if node is None:
        raise ValueError(f"No node found with UUID: {uuid}")

    removed_ids = graph.remove_subtree(node.id)

    final_output = output_path or _derive_output_path(file_path, "pruned")
    graph.save(final_output)

    return PruneResult(
        operation="uuid",
        removed_count=len(removed_ids),
        removed_node_ids=removed_ids,
        remaining_nodes=len(graph.nodes),
        remaining_paths=len(graph.get_all_paths()),
        output_path=final_output,
    )


def _derive_output_path(input_path: str, suffix: str) -> str:
    """Derive a non-destructive output path from the input path.

    Example: topic_graph.json -> topic_graph_pruned_level2.json
    """
    p = Path(input_path)
    return str(p.with_stem(f"{p.stem}_{suffix}"))

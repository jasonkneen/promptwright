"""Tests for graph pruning operations."""

import json
import tempfile
from pathlib import Path

import pytest
from click.testing import CliRunner

from deepfabric.cli import cli
from deepfabric.graph import Graph
from deepfabric.graph_pruner import (
    PruneResult,
    _derive_output_path,
    load_graph_for_pruning,
    prune_graph_at_level,
    prune_graph_by_uuid,
)


@pytest.fixture
def deep_graph_json_file():
    """Create a 9-node graph with depth 3 for pruning tests.

    Structure:
        Root (0)
        ├── A (1)
        │   ├── A1 (3)
        │   │   ├── A1a (7)
        │   │   └── A1b (8)
        │   └── A2 (4)
        └── B (2)
            ├── B1 (5)
            └── B2 (6)
    """
    content = {
        "nodes": {
            "0": {
                "id": 0,
                "topic": "Root",
                "children": [1, 2],
                "parents": [],
                "metadata": {"uuid": "uuid-root"},
            },
            "1": {
                "id": 1,
                "topic": "A",
                "children": [3, 4],
                "parents": [0],
                "metadata": {"uuid": "uuid-a"},
            },
            "2": {
                "id": 2,
                "topic": "B",
                "children": [5, 6],
                "parents": [0],
                "metadata": {"uuid": "uuid-b"},
            },
            "3": {
                "id": 3,
                "topic": "A1",
                "children": [7, 8],
                "parents": [1],
                "metadata": {"uuid": "uuid-a1"},
            },
            "4": {
                "id": 4,
                "topic": "A2",
                "children": [],
                "parents": [1],
                "metadata": {"uuid": "uuid-a2"},
            },
            "5": {
                "id": 5,
                "topic": "B1",
                "children": [],
                "parents": [2],
                "metadata": {"uuid": "uuid-b1"},
            },
            "6": {
                "id": 6,
                "topic": "B2",
                "children": [],
                "parents": [2],
                "metadata": {"uuid": "uuid-b2"},
            },
            "7": {
                "id": 7,
                "topic": "A1a",
                "children": [],
                "parents": [3],
                "metadata": {"uuid": "uuid-a1a"},
            },
            "8": {
                "id": 8,
                "topic": "A1b",
                "children": [],
                "parents": [3],
                "metadata": {"uuid": "uuid-a1b"},
            },
        },
        "root_id": 0,
        "metadata": {
            "provider": "openai",
            "model": "gpt-4",
            "temperature": 0.7,
            "created_at": "2024-01-01T00:00:00+00:00",
        },
    }
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        json.dump(content, f)
        path = f.name
    yield path
    Path(path).unlink(missing_ok=True)


@pytest.fixture
def cli_runner():
    return CliRunner()


# --- Graph method tests ---


class TestFindNodeByUuid:
    def test_find_existing(self, deep_graph_json_file):
        graph = Graph.load(deep_graph_json_file)
        node = graph.find_node_by_uuid("uuid-a1")
        assert node is not None
        assert node.topic == "A1"
        assert node.id == 3

    def test_find_root(self, deep_graph_json_file):
        graph = Graph.load(deep_graph_json_file)
        node = graph.find_node_by_uuid("uuid-root")
        assert node is not None
        assert node.topic == "Root"

    def test_find_nonexistent(self, deep_graph_json_file):
        graph = Graph.load(deep_graph_json_file)
        assert graph.find_node_by_uuid("no-such-uuid") is None


class TestRemoveNode:
    def test_remove_leaf(self, deep_graph_json_file):
        graph = Graph.load(deep_graph_json_file)
        graph.remove_node(7)  # A1a
        assert 7 not in graph.nodes
        # Parent A1 should no longer list A1a as child
        a1 = graph.nodes[3]
        assert all(c.id != 7 for c in a1.children)

    def test_remove_internal_node(self, deep_graph_json_file):
        graph = Graph.load(deep_graph_json_file)
        graph.remove_node(3)  # A1 (has children A1a, A1b)
        assert 3 not in graph.nodes
        # Parent A should no longer list A1
        a = graph.nodes[1]
        assert all(c.id != 3 for c in a.children)
        # Children A1a, A1b should no longer list A1 as parent
        assert all(p.id != 3 for p in graph.nodes[7].parents)
        assert all(p.id != 3 for p in graph.nodes[8].parents)

    def test_remove_root_raises(self, deep_graph_json_file):
        graph = Graph.load(deep_graph_json_file)
        with pytest.raises(ValueError, match="Cannot remove the root node"):
            graph.remove_node(0)

    def test_remove_nonexistent_raises(self, deep_graph_json_file):
        graph = Graph.load(deep_graph_json_file)
        with pytest.raises(ValueError, match="not found"):
            graph.remove_node(999)


class TestRemoveSubtree:
    def test_remove_leaf_subtree(self, deep_graph_json_file):
        graph = Graph.load(deep_graph_json_file)
        removed = graph.remove_subtree(7)  # A1a (leaf)
        assert removed == [7]
        assert 7 not in graph.nodes
        assert len(graph.nodes) == 8

    def test_remove_internal_subtree(self, deep_graph_json_file):
        graph = Graph.load(deep_graph_json_file)
        removed = graph.remove_subtree(1)  # A -> A1 -> A1a, A1b + A2
        assert set(removed) == {1, 3, 4, 7, 8}
        assert len(graph.nodes) == 4  # Root, B, B1, B2
        # Root should no longer have A as child
        assert all(c.id != 1 for c in graph.root.children)

    def test_remove_root_raises(self, deep_graph_json_file):
        graph = Graph.load(deep_graph_json_file)
        with pytest.raises(ValueError, match="Cannot remove the root node"):
            graph.remove_subtree(0)

    def test_remove_subtree_parent_becomes_leaf(self, deep_graph_json_file):
        graph = Graph.load(deep_graph_json_file)
        graph.remove_subtree(3)  # A1 and its children
        # A should now have only A2 as child
        a = graph.nodes[1]
        assert len(a.children) == 1
        assert a.children[0].id == 4


class TestPruneAtLevel:
    def test_prune_level_0(self, deep_graph_json_file):
        graph = Graph.load(deep_graph_json_file)
        removed = graph.prune_at_level(0)
        assert len(removed) == 8  # Everything except root
        assert len(graph.nodes) == 1
        assert graph.root.children == []

    def test_prune_level_1(self, deep_graph_json_file):
        graph = Graph.load(deep_graph_json_file)
        removed = graph.prune_at_level(1)
        # Should keep Root, A, B (3 nodes), remove the rest (6 nodes)
        assert len(removed) == 6
        assert len(graph.nodes) == 3
        assert set(graph.nodes.keys()) == {0, 1, 2}
        # A and B should now be leaves
        assert graph.nodes[1].children == []
        assert graph.nodes[2].children == []

    def test_prune_level_2(self, deep_graph_json_file):
        graph = Graph.load(deep_graph_json_file)
        removed = graph.prune_at_level(2)
        # Should keep Root, A, B, A1, A2, B1, B2 (7 nodes), remove A1a, A1b (2 nodes)
        assert len(removed) == 2
        assert len(graph.nodes) == 7
        # A1 should now be a leaf
        assert graph.nodes[3].children == []

    def test_prune_level_beyond_max(self, deep_graph_json_file):
        graph = Graph.load(deep_graph_json_file)
        removed = graph.prune_at_level(10)
        assert len(removed) == 0
        assert len(graph.nodes) == 9

    def test_prune_negative_raises(self, deep_graph_json_file):
        graph = Graph.load(deep_graph_json_file)
        with pytest.raises(ValueError, match="non-negative"):
            graph.prune_at_level(-1)


# --- Pruner module tests ---


class TestLoadGraphForPruning:
    def test_load_valid(self, deep_graph_json_file):
        graph = load_graph_for_pruning(deep_graph_json_file)
        assert len(graph.nodes) == 9

    def test_file_not_found(self):
        with pytest.raises(FileNotFoundError):
            load_graph_for_pruning("/nonexistent/file.json")

    def test_non_json_file(self, tmp_path):
        jsonl_file = tmp_path / "test.jsonl"
        jsonl_file.write_text('{"path": ["a", "b"]}')
        with pytest.raises(ValueError, match="JSON graph file"):
            load_graph_for_pruning(str(jsonl_file))


class TestPruneGraphAtLevel:
    def test_prune_creates_output(self, deep_graph_json_file, tmp_path):
        output = str(tmp_path / "pruned.json")
        result = prune_graph_at_level(deep_graph_json_file, 1, output)
        assert Path(output).exists()
        assert result.operation == "level"
        assert result.removed_count == 6
        assert result.remaining_nodes == 3
        assert result.output_path == output

    def test_prune_output_is_valid_graph(self, deep_graph_json_file, tmp_path):
        output = str(tmp_path / "pruned.json")
        prune_graph_at_level(deep_graph_json_file, 1, output)
        # Should be loadable
        reloaded = Graph.load(output)
        assert len(reloaded.nodes) == 3

    def test_auto_derived_filename(self, deep_graph_json_file):
        result = prune_graph_at_level(deep_graph_json_file, 2)
        assert "pruned_level2" in result.output_path
        # Clean up auto-generated file
        Path(result.output_path).unlink(missing_ok=True)

    def test_preserves_metadata(self, deep_graph_json_file, tmp_path):
        output = str(tmp_path / "pruned.json")
        prune_graph_at_level(deep_graph_json_file, 1, output)
        with open(output) as f:
            data = json.load(f)
        assert data["metadata"]["provider"] == "openai"
        assert data["metadata"]["model"] == "gpt-4"


class TestPruneGraphByUuid:
    def test_prune_leaf(self, deep_graph_json_file, tmp_path):
        output = str(tmp_path / "pruned.json")
        result = prune_graph_by_uuid(deep_graph_json_file, "uuid-b1", output)
        assert result.removed_count == 1
        assert result.remaining_nodes == 8

    def test_prune_subtree(self, deep_graph_json_file, tmp_path):
        output = str(tmp_path / "pruned.json")
        result = prune_graph_by_uuid(deep_graph_json_file, "uuid-a", output)
        # A + A1 + A2 + A1a + A1b = 5 removed
        assert result.removed_count == 5
        assert result.remaining_nodes == 4

    def test_uuid_not_found(self, deep_graph_json_file, tmp_path):
        output = str(tmp_path / "pruned.json")
        with pytest.raises(ValueError, match="No node found"):
            prune_graph_by_uuid(deep_graph_json_file, "no-such-uuid", output)

    def test_root_uuid_raises(self, deep_graph_json_file, tmp_path):
        output = str(tmp_path / "pruned.json")
        with pytest.raises(ValueError, match="Cannot remove the root"):
            prune_graph_by_uuid(deep_graph_json_file, "uuid-root", output)


class TestDeriveOutputPath:
    def test_level_suffix(self):
        assert _derive_output_path("/tmp/graph.json", "pruned_level2") == "/tmp/graph_pruned_level2.json"

    def test_uuid_suffix(self):
        assert _derive_output_path("/tmp/graph.json", "pruned") == "/tmp/graph_pruned.json"


# --- CLI tests ---


class TestTopicPruneCLI:
    def test_prune_level(self, cli_runner, deep_graph_json_file, tmp_path):
        output = str(tmp_path / "out.json")
        result = cli_runner.invoke(
            cli, ["topic", "prune", deep_graph_json_file, "--level", "1", "-o", output]
        )
        assert result.exit_code == 0
        assert "pruned successfully" in result.output
        assert Path(output).exists()

    def test_prune_uuid(self, cli_runner, deep_graph_json_file, tmp_path):
        output = str(tmp_path / "out.json")
        result = cli_runner.invoke(
            cli, ["topic", "prune", deep_graph_json_file, "--uuid", "uuid-b", "-o", output]
        )
        assert result.exit_code == 0
        assert "pruned successfully" in result.output

    def test_dry_run_level(self, cli_runner, deep_graph_json_file):
        result = cli_runner.invoke(
            cli, ["topic", "prune", deep_graph_json_file, "--level", "1", "--dry-run"]
        )
        assert result.exit_code == 0
        assert "DRY RUN" in result.output
        assert "9 unique nodes" in result.output
        assert "Would remove" in result.output
        assert "Would keep" in result.output

    def test_dry_run_uuid(self, cli_runner, deep_graph_json_file):
        result = cli_runner.invoke(
            cli, ["topic", "prune", deep_graph_json_file, "--uuid", "uuid-a", "--dry-run"]
        )
        assert result.exit_code == 0
        assert "DRY RUN" in result.output
        assert "9 unique nodes" in result.output
        assert "5 nodes" in result.output  # A subtree = 5 nodes

    def test_no_mode_error(self, cli_runner, deep_graph_json_file):
        result = cli_runner.invoke(cli, ["topic", "prune", deep_graph_json_file])
        assert result.exit_code == 1

    def test_both_modes_error(self, cli_runner, deep_graph_json_file):
        result = cli_runner.invoke(
            cli, ["topic", "prune", deep_graph_json_file, "--level", "1", "--uuid", "uuid-a"]
        )
        assert result.exit_code == 1

    def test_uuid_not_found_error(self, cli_runner, deep_graph_json_file, tmp_path):
        output = str(tmp_path / "out.json")
        result = cli_runner.invoke(
            cli, ["topic", "prune", deep_graph_json_file, "--uuid", "nonexistent", "-o", output]
        )
        assert result.exit_code == 1

    def test_force_overwrites(self, cli_runner, deep_graph_json_file):
        result = cli_runner.invoke(
            cli, ["topic", "prune", deep_graph_json_file, "--level", "1", "--force"]
        )
        assert result.exit_code == 0
        # Original file should now have only 3 nodes
        reloaded = Graph.load(deep_graph_json_file)
        assert len(reloaded.nodes) == 3

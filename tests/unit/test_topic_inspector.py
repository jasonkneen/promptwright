"""Tests for topic inspector module."""

import json
import tempfile

from pathlib import Path

import pytest

from click.testing import CliRunner

from deepfabric.cli import cli
from deepfabric.topic_inspector import (
    TopicInspectionResult,
    detect_format,
    inspect_topic_file,
)


@pytest.fixture
def tree_jsonl_file():
    """Create a temporary tree JSONL file."""
    content = [
        {"path": ["Root", "Child1", "Grandchild1"]},
        {"path": ["Root", "Child1", "Grandchild2"]},
        {"path": ["Root", "Child2"]},
    ]
    with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
        for item in content:
            f.write(json.dumps(item) + "\n")
        path = f.name
    yield path
    Path(path).unlink(missing_ok=True)


@pytest.fixture
def graph_json_file():
    """Create a temporary graph JSON file."""
    content = {
        "nodes": {
            "0": {
                "id": 0,
                "topic": "Root",
                "children": [1, 2],
                "parents": [],
                "metadata": {"uuid": "test-uuid-0"},
            },
            "1": {
                "id": 1,
                "topic": "Child1",
                "children": [],
                "parents": [0],
                "metadata": {"uuid": "test-uuid-1"},
            },
            "2": {
                "id": 2,
                "topic": "Child2",
                "children": [],
                "parents": [0],
                "metadata": {"uuid": "test-uuid-2"},
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
def empty_file():
    """Create an empty file."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
        path = f.name
    yield path
    Path(path).unlink(missing_ok=True)


class TestDetectFormat:
    """Tests for format detection."""

    def test_detect_tree_format(self, tree_jsonl_file):
        assert detect_format(tree_jsonl_file) == "tree"

    def test_detect_graph_format(self, graph_json_file):
        assert detect_format(graph_json_file) == "graph"

    def test_detect_format_file_not_found(self):
        with pytest.raises(FileNotFoundError):
            detect_format("/nonexistent/file.jsonl")

    def test_detect_format_empty_file(self, empty_file):
        with pytest.raises(ValueError, match="Empty file"):
            detect_format(empty_file)


class TestInspectTopicFile:
    """Tests for topic file inspection."""

    def test_inspect_tree_file(self, tree_jsonl_file):
        result = inspect_topic_file(tree_jsonl_file)

        assert result.format == "tree"
        assert result.total_paths == 3
        assert result.max_depth == 3
        assert result.paths_at_level is None
        assert result.all_paths is None
        assert result.metadata.get("root_topic") == "Root"

    def test_inspect_tree_file_with_level(self, tree_jsonl_file):
        result = inspect_topic_file(tree_jsonl_file, level=2)

        assert result.paths_at_level is not None
        assert len(result.paths_at_level) == 2  # Grandchild1, Grandchild2

    def test_inspect_tree_file_with_level_0(self, tree_jsonl_file):
        result = inspect_topic_file(tree_jsonl_file, level=0)

        # Level 0 shows the root topic
        assert result.paths_at_level is not None
        assert len(result.paths_at_level) == 1
        assert result.paths_at_level[0] == ["Root"]

    def test_inspect_tree_file_with_level_1(self, tree_jsonl_file):
        result = inspect_topic_file(tree_jsonl_file, level=1)

        assert result.paths_at_level is not None
        # Level 1 shows unique topics at depth 1: Child1, Child2
        assert len(result.paths_at_level) == 2

    def test_inspect_tree_file_show_all(self, tree_jsonl_file):
        result = inspect_topic_file(tree_jsonl_file, show_all=True)

        assert result.all_paths is not None
        assert len(result.all_paths) == 3

    def test_inspect_graph_file(self, graph_json_file):
        result = inspect_topic_file(graph_json_file)

        assert result.format == "graph"
        assert result.total_paths == 2  # Two leaf paths
        assert "total_nodes" in result.metadata
        assert result.metadata["total_nodes"] == 3
        assert result.metadata["root_topic"] == "Root"
        assert result.metadata["has_cycles"] is False


class TestTopicInspectCLI:
    """Tests for the CLI command."""

    @pytest.fixture
    def cli_runner(self):
        return CliRunner()

    def test_inspect_basic(self, cli_runner, tree_jsonl_file):
        result = cli_runner.invoke(cli, ["topic", "inspect", tree_jsonl_file])
        assert result.exit_code == 0
        assert "Total Paths" in result.output

    def test_inspect_with_level(self, cli_runner, tree_jsonl_file):
        result = cli_runner.invoke(
            cli, ["topic", "inspect", tree_jsonl_file, "--level", "2"]
        )
        assert result.exit_code == 0
        assert "Level 2" in result.output

    def test_inspect_show_all(self, cli_runner, tree_jsonl_file):
        result = cli_runner.invoke(cli, ["topic", "inspect", tree_jsonl_file, "--all"])
        assert result.exit_code == 0
        assert "Full Tree Structure" in result.output

    def test_inspect_json_format(self, cli_runner, tree_jsonl_file):
        result = cli_runner.invoke(
            cli, ["topic", "inspect", tree_jsonl_file, "--format", "json"]
        )
        assert result.exit_code == 0
        # Extract JSON from output (may have update notice before it)
        json_start = result.output.find("{")
        json_output = result.output[json_start:]
        output = json.loads(json_output)
        assert output["format"] == "tree"
        assert output["total_paths"] == 3

    def test_inspect_table_format(self, cli_runner, tree_jsonl_file):
        result = cli_runner.invoke(
            cli, ["topic", "inspect", tree_jsonl_file, "--all", "--format", "table"]
        )
        assert result.exit_code == 0
        # Table output should have depth column
        assert "Depth" in result.output

    def test_inspect_nonexistent_file(self, cli_runner):
        result = cli_runner.invoke(cli, ["topic", "inspect", "/nonexistent/file.jsonl"])
        # Click will fail with exit code 2 for invalid path
        assert result.exit_code != 0

    def test_inspect_graph_file(self, cli_runner, graph_json_file):
        result = cli_runner.invoke(cli, ["topic", "inspect", graph_json_file])
        assert result.exit_code == 0
        assert "Graph (JSON)" in result.output
        assert "Total Nodes" in result.output

    def test_inspect_graph_json_format(self, cli_runner, graph_json_file):
        result = cli_runner.invoke(
            cli, ["topic", "inspect", graph_json_file, "--format", "json"]
        )
        assert result.exit_code == 0
        # Extract JSON from output (may have update notice before it)
        json_start = result.output.find("{")
        json_output = result.output[json_start:]
        output = json.loads(json_output)
        assert output["format"] == "graph"
        assert output["metadata"]["total_nodes"] == 3

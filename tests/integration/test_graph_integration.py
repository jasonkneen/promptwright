"""Integration tests for Graph with real API calls."""

import json

import pytest  # pyright: ignore[reportMissingImports]

from deepfabric import Graph

from .conftest import assert_topic_build_result, requires_gemini, requires_openai


@requires_openai
class TestGraphOpenAI:
    """Integration tests for Graph with OpenAI provider."""

    @pytest.mark.openai
    @pytest.mark.flaky(reruns=1, reruns_delay=3)
    async def test_graph_builds_basic(self, openai_config):
        """Test basic graph building with OpenAI."""
        degree = 2
        depth = 1
        topic = "Machine Learning"

        graph = Graph(
            topic_prompt=topic,
            provider=openai_config["provider"],
            model_name=openai_config["model_name"],
            temperature=openai_config["temperature"],
            degree=degree,
            depth=depth,
        )

        events = [event async for event in graph.build_async()]

        # Verify build completion
        completes = [e for e in events if e.get("event") == "build_complete"]
        assert len(completes) == 1

        # Verify paths were built
        paths = graph.get_all_paths()
        assert_topic_build_result(paths, graph, min_paths=1, context="graph basic build (OpenAI)")
        assert all(p[0] == topic for p in paths)

    @pytest.mark.openai
    @pytest.mark.flaky(reruns=1, reruns_delay=3)
    async def test_graph_save_and_load_roundtrip(self, tmp_path, openai_config):
        """Test saving and loading a graph with OpenAI."""
        degree = 2
        depth = 1
        topic = "Data Science"

        graph = Graph(
            topic_prompt=topic,
            provider=openai_config["provider"],
            model_name=openai_config["model_name"],
            temperature=openai_config["temperature"],
            degree=degree,
            depth=depth,
        )

        async for _ in graph.build_async():
            pass

        # Save to file
        out_path = tmp_path / "graph.json"
        graph.save(str(out_path))

        # Verify file structure
        data = json.loads(out_path.read_text())
        assert "nodes" in data
        assert "root_id" in data
        assert len(data["nodes"]) >= 1

        # Load into new graph and verify
        graph_params = {
            "topic_prompt": topic,
            "provider": openai_config["provider"],
            "model_name": openai_config["model_name"],
            "degree": degree,
            "depth": depth,
        }
        new_graph = Graph.from_json(str(out_path), graph_params)

        assert new_graph.get_all_paths() == graph.get_all_paths()


@requires_gemini
class TestGraphGemini:
    """Integration tests for Graph with Gemini provider."""

    @pytest.mark.gemini
    @pytest.mark.flaky(reruns=1, reruns_delay=3)
    async def test_graph_builds_basic(self, gemini_config):
        """Test basic graph building with Gemini."""
        degree = 2
        depth = 1
        topic = "Cloud Computing"

        graph = Graph(
            topic_prompt=topic,
            provider=gemini_config["provider"],
            model_name=gemini_config["model_name"],
            temperature=gemini_config["temperature"],
            degree=degree,
            depth=depth,
        )

        events = [event async for event in graph.build_async()]

        # Verify build completion
        completes = [e for e in events if e.get("event") == "build_complete"]
        assert len(completes) == 1

        # Verify paths were built
        paths = graph.get_all_paths()
        assert_topic_build_result(paths, graph, min_paths=1, context="graph basic build (Gemini)")
        assert all(p[0] == topic for p in paths)

    @pytest.mark.gemini
    @pytest.mark.flaky(reruns=1, reruns_delay=3)
    async def test_graph_no_cycles(self, gemini_config):
        """Test that generated graphs have no cycles."""
        graph = Graph(
            topic_prompt="Software Engineering",
            provider=gemini_config["provider"],
            model_name=gemini_config["model_name"],
            temperature=gemini_config["temperature"],
            degree=2,
            depth=2,
        )

        async for _ in graph.build_async():
            pass

        # Verify no cycles
        assert not graph.has_cycle()
        # Verify we got paths
        paths = graph.get_all_paths()
        assert_topic_build_result(paths, graph, min_paths=1, context="graph no cycles (Gemini)")

"""Integration tests for DataSetGenerator with real API calls."""

import asyncio

import pytest

from deepfabric import DataSetGenerator, Graph

from .conftest import (
    assert_generation_result,
    assert_topic_build_result,
    requires_gemini,
    requires_openai,
)


@pytest.fixture
def openai_generator(openai_config):
    """Create a DataSetGenerator configured for OpenAI."""
    return DataSetGenerator(
        provider=openai_config["provider"],
        model_name=openai_config["model_name"],
        generation_system_prompt="You are a helpful assistant that generates training data.",
        temperature=openai_config["temperature"],
    )


@pytest.fixture
def gemini_generator(gemini_config):
    """Create a DataSetGenerator configured for Gemini."""
    return DataSetGenerator(
        provider=gemini_config["provider"],
        model_name=gemini_config["model_name"],
        generation_system_prompt="You are a helpful assistant that generates training data.",
        temperature=gemini_config["temperature"],
    )


@pytest.fixture
def small_topic_graph(openai_config):
    """Create a small topic graph for generator tests."""
    graph = Graph(
        topic_prompt="Python Programming",
        provider=openai_config["provider"],
        model_name=openai_config["model_name"],
        temperature=openai_config["temperature"],
        degree=2,
        depth=1,
    )

    async def build_graph():
        async for _ in graph.build_async():
            pass

    asyncio.run(build_graph())
    assert_topic_build_result(
        graph.get_all_paths(), graph, min_paths=1, context="small_topic_graph fixture"
    )
    return graph


@requires_openai
class TestDataSetGeneratorOpenAI:
    """Integration tests for DataSetGenerator with OpenAI provider."""

    @pytest.mark.openai
    @pytest.mark.flaky(reruns=2, reruns_delay=5)
    def test_basic_generation(self, openai_generator):
        """Test basic dataset generation without topic model."""
        result = openai_generator.create_data(
            num_steps=1,
            batch_size=2,
        )

        assert result is not None
        assert_generation_result(
            result, openai_generator, context="basic generation (no topic model)"
        )

    @pytest.mark.openai
    @pytest.mark.flaky(reruns=2, reruns_delay=5)
    def test_generation_with_topic_model(self, openai_generator, small_topic_graph):
        """Test dataset generation with a topic graph."""
        result = openai_generator.create_data(
            num_steps=1,
            batch_size=2,
            topic_model=small_topic_graph,
        )

        assert result is not None
        assert_generation_result(result, openai_generator, context="generation with topic model")

    @pytest.mark.openai
    @pytest.mark.flaky(reruns=2, reruns_delay=5)
    async def test_async_generation(self, openai_generator):
        """Test async dataset generation."""
        result = await openai_generator.create_data_async(
            num_steps=1,
            batch_size=2,
        )

        assert result is not None
        assert_generation_result(result, openai_generator, context="async generation")

    @pytest.mark.openai
    @pytest.mark.flaky(reruns=2, reruns_delay=5)
    def test_generation_with_cot(self, openai_config):
        """Test generation with cot conversation type."""
        generator = DataSetGenerator(
            provider=openai_config["provider"],
            model_name=openai_config["model_name"],
            generation_system_prompt="You are a reasoning assistant.",
            temperature=openai_config["temperature"],
            conversation_type="cot",
            reasoning_style="freetext",
        )

        result = generator.create_data(
            num_steps=1,
            batch_size=1,
        )

        assert result is not None
        assert_generation_result(result, generator, context="cot generation")


@requires_gemini
class TestDataSetGeneratorGemini:
    """Integration tests for DataSetGenerator with Gemini provider."""

    @pytest.mark.gemini
    @pytest.mark.flaky(reruns=2, reruns_delay=5)
    async def test_basic_generation(self, gemini_generator):
        """Test basic dataset generation with Gemini."""
        result = await gemini_generator.create_data_async(
            num_steps=1,
            batch_size=2,
        )

        assert result is not None
        assert_generation_result(result, gemini_generator, context="Gemini basic generation")

    @pytest.mark.gemini
    @pytest.mark.flaky(reruns=2, reruns_delay=5)
    async def test_generation_saves_to_file(self, tmp_path, gemini_generator):
        """Test that generated data can be saved."""
        await gemini_generator.create_data_async(
            num_steps=1,
            batch_size=2,
        )

        # Save dataset
        out_path = tmp_path / "dataset.jsonl"
        gemini_generator.save_dataset(str(out_path))

        # Verify file was created
        assert out_path.exists()
        lines = out_path.read_text().strip().split("\n")
        assert len(lines) >= 1

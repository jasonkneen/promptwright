"""Integration tests for LLMClient with real API calls."""

import pytest  # pyright: ignore[reportMissingImports]

from deepfabric.llm.client import LLMClient
from deepfabric.schemas import ChatMessage, ChatTranscript, TopicList

from .conftest import requires_gemini, requires_openai


@pytest.fixture
def openai_client(openai_config):
    """Create an LLMClient configured for OpenAI."""
    return LLMClient(
        provider=openai_config["provider"],
        model_name=openai_config["model_name"],
    )


@pytest.fixture
def gemini_client(gemini_config):
    """Create an LLMClient configured for Gemini."""
    return LLMClient(
        provider=gemini_config["provider"],
        model_name=gemini_config["model_name"],
    )


@requires_openai
class TestLLMClientOpenAI:
    """Integration tests for LLMClient with OpenAI provider."""

    @pytest.mark.openai
    @pytest.mark.flaky(reruns=1, reruns_delay=3)
    def test_basic_structured_output(self, openai_client):
        """Test basic structured output generation with OpenAI."""
        prompt = "Generate a short greeting conversation between two people."
        result = openai_client.generate(prompt, ChatTranscript)

        assert isinstance(result, ChatTranscript), (
            f"Expected ChatTranscript, got {type(result).__name__}: {result}"
        )
        assert len(result.messages) >= 1, (
            f"Expected >= 1 messages, got {len(result.messages)}: {result}"
        )
        assert all(isinstance(m, ChatMessage) for m in result.messages)

    @pytest.mark.openai
    @pytest.mark.flaky(reruns=1, reruns_delay=3)
    async def test_async_structured_output(self, openai_client):
        """Test async structured output generation with OpenAI."""
        prompt = "List 3 subtopics about machine learning."
        result = await openai_client.generate_async(prompt, TopicList)

        assert isinstance(result, TopicList), (
            f"Expected TopicList, got {type(result).__name__}: {result}"
        )
        assert len(result.subtopics) >= 1, (
            f"Expected >= 1 subtopics, got {len(result.subtopics)}: {result}"
        )
        assert all(isinstance(s, str) for s in result.subtopics)

    @pytest.mark.openai
    @pytest.mark.flaky(reruns=1, reruns_delay=3)
    async def test_async_streaming(self, openai_client):
        """Test async streaming generation with OpenAI."""
        prompt = "Generate a brief Q&A about Python programming."
        chunks = []
        final_result = None

        # generate_async_stream yields tuples: (chunk, None) or (None, result)
        async for chunk, result in openai_client.generate_async_stream(prompt, ChatTranscript):
            if chunk is not None:
                chunks.append(chunk)
            if result is not None:
                final_result = result

        # Should have received streaming chunks
        assert len(chunks) >= 1, f"Expected streaming chunks, got {len(chunks)}"
        # Final result should be valid
        assert isinstance(final_result, ChatTranscript), (
            f"Expected ChatTranscript, got {type(final_result).__name__}: {final_result}"
        )
        assert len(final_result.messages) >= 1, (
            f"Expected >= 1 messages, got {len(final_result.messages)}: {final_result}"
        )


@requires_gemini
class TestLLMClientGemini:
    """Integration tests for LLMClient with Gemini provider.

    Note: Gemini only supports async generation, so all tests use generate_async().
    """

    @pytest.mark.gemini
    @pytest.mark.flaky(reruns=1, reruns_delay=3)
    async def test_basic_structured_output(self, gemini_client):
        """Test basic structured output generation with Gemini."""
        prompt = "Generate a short greeting conversation between two people."
        result = await gemini_client.generate_async(prompt, ChatTranscript)

        assert isinstance(result, ChatTranscript), (
            f"Expected ChatTranscript, got {type(result).__name__}: {result}"
        )
        assert len(result.messages) >= 1, (
            f"Expected >= 1 messages, got {len(result.messages)}: {result}"
        )
        assert all(isinstance(m, ChatMessage) for m in result.messages)

    @pytest.mark.gemini
    @pytest.mark.flaky(reruns=1, reruns_delay=3)
    async def test_async_topic_list(self, gemini_client):
        """Test async structured output generation with Gemini."""
        prompt = "List 3 subtopics about data science."
        result = await gemini_client.generate_async(prompt, TopicList)

        assert isinstance(result, TopicList), (
            f"Expected TopicList, got {type(result).__name__}: {result}"
        )
        assert len(result.subtopics) >= 1, (
            f"Expected >= 1 subtopics, got {len(result.subtopics)}: {result}"
        )
        assert all(isinstance(s, str) for s in result.subtopics)

    @pytest.mark.gemini
    @pytest.mark.flaky(reruns=1, reruns_delay=3)
    async def test_gemini_schema_handling(self, gemini_client):
        """Test that Gemini correctly handles schema conversion.

        Gemini has specific requirements around JSON schemas (no additionalProperties,
        specific array constraints). This test verifies the schema conversion works.
        """
        # TopicList has min_length constraint which tests array handling
        prompt = "List exactly 2 subtopics about cloud computing."
        result = await gemini_client.generate_async(prompt, TopicList)

        assert isinstance(result, TopicList), (
            f"Expected TopicList, got {type(result).__name__}: {result}"
        )
        assert len(result.subtopics) >= 2, (  # noqa: PLR2004
            f"Expected >= 2 subtopics, got {len(result.subtopics)}: {result}"
        )

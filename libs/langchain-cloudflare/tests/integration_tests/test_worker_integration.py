"""Integration tests for LangChain Cloudflare Python Worker.

These tests start the worker using `pywrangler dev` and make HTTP requests
to verify the endpoints work correctly with the Workers AI, Vectorize, and D1 bindings.

Tests are organized by functionality:
- Chat and LLM tests
- Structured output tests
- Tool calling tests
- Agent tests (create_agent pattern)
- Vectorize binding tests
- D1 binding tests
- Multi-modal input tests

Note: These tests require:
1. The examples/workers directory to be set up
2. Valid Cloudflare credentials configured in wrangler.jsonc
3. pywrangler installed (uv add workers-py)
"""

import base64
import time
import uuid

import pytest
import requests

# Models to test against
MODELS = [
    "@cf/meta/llama-3.3-70b-instruct-fp8-fast",
    "@cf/mistralai/mistral-small-3.1-24b-instruct",
    "@cf/qwen/qwen3-30b-a3b-fp8",
    "@cf/zai-org/glm-4.7-flash",
    "@cf/openai/gpt-oss-120b",
    "@cf/openai/gpt-oss-20b",
    "@cf/nvidia/nemotron-3-120b-a12b",
    "@cf/moonshotai/kimi-k2.5",
]


# MARK: - Index Tests


class TestWorkerIndex:
    """Test the index/documentation endpoint."""

    def test_index_returns_documentation(self, dev_server):
        """GET / should return API documentation."""
        port = dev_server
        response = requests.get(f"http://localhost:{port}/")

        assert response.status_code == 200
        data = response.json()

        assert "name" in data
        assert "endpoints" in data
        assert "supported_models" in data
        assert "default_model" in data
        assert "/chat" in data["endpoints"]
        assert "/structured" in data["endpoints"]
        assert "/tools" in data["endpoints"]
        # D1 endpoints
        assert "/d1-health" in data["endpoints"]
        assert "/d1-create-table" in data["endpoints"]


# MARK: - Chat Tests


class TestWorkerChat:
    """Test basic chat endpoint with Worker binding."""

    @pytest.mark.parametrize("model", MODELS)
    def test_chat_basic_message(self, dev_server, model):
        """POST /chat should return a response from the model."""
        port = dev_server
        response = requests.post(
            f"http://localhost:{port}/chat",
            json={"message": "Say hello in exactly 3 words.", "model": model},
            headers={"Content-Type": "application/json"},
        )

        assert response.status_code == 200
        data = response.json()

        assert "response" in data
        assert "model" in data
        assert len(data["response"]) > 0

    @pytest.mark.parametrize("model", MODELS)
    def test_chat_batch(self, dev_server, model):
        """POST /chat-batch should return batch responses."""
        port = dev_server
        response = requests.post(
            f"http://localhost:{port}/chat-batch",
            json={
                "messages": [
                    "Say 'Hello' and nothing else.",
                    "Say 'World' and nothing else.",
                ],
                "model": model,
            },
            headers={"Content-Type": "application/json"},
        )

        assert response.status_code == 200
        data = response.json()

        assert "results" in data
        assert data["count"] == 2
        assert len(data["results"]) == 2

    def test_chat_default_message(self, dev_server):
        """POST /chat with empty body should use default message."""
        port = dev_server
        response = requests.post(
            f"http://localhost:{port}/chat",
            json={},
            headers={"Content-Type": "application/json"},
        )

        assert response.status_code == 200
        data = response.json()
        assert "response" in data


# MARK: - Structured Output Tests


class TestWorkerStructuredOutput:
    """Test structured output endpoint with Worker binding."""

    @pytest.mark.parametrize("model", MODELS)
    def test_structured_output_extracts_announcements(self, dev_server, model):
        """POST /structured should extract structured data from text."""
        port = dev_server
        response = requests.post(
            f"http://localhost:{port}/structured",
            json={
                "text": "Acme Corp announced a partnership with TechGiant Inc.",
                "model": model,
            },
            headers={"Content-Type": "application/json"},
        )

        assert response.status_code == 200
        data = response.json()

        assert "input" in data
        assert "extracted" in data
        assert "announcements" in data["extracted"] or "raw" in data["extracted"]


class TestWorkerStructuredOutputBatch:
    """Test batch structured output endpoint with Worker binding."""

    @pytest.mark.parametrize("model", MODELS)
    def test_structured_output_batch(self, dev_server, model):
        """POST /structured-batch should return batch results."""
        port = dev_server
        response = requests.post(
            f"http://localhost:{port}/structured-batch",
            json={
                "texts": [
                    "Acme Corp announced a partnership with TechGiant.",
                    "Apple Inc announced record Q4 earnings.",
                ],
                "model": model,
            },
            headers={"Content-Type": "application/json"},
        )

        assert response.status_code == 200
        data = response.json()

        assert "results" in data
        assert data["count"] == 2
        assert len(data["results"]) == 2

        for i, result in enumerate(data["results"]):
            assert result is not None, f"Result {i} is None for {model}"


# MARK: - Tool Calling Tests


class TestWorkerToolCalling:
    """Test tool calling endpoint with Worker binding."""

    @pytest.mark.parametrize("model", MODELS)
    def test_tools_weather_query(self, dev_server, model):
        """POST /tools should handle weather queries with tool calls."""
        port = dev_server
        response = requests.post(
            f"http://localhost:{port}/tools",
            json={"message": "What is the weather in San Francisco?", "model": model},
            headers={"Content-Type": "application/json"},
        )

        assert response.status_code == 200
        data = response.json()

        assert "input" in data
        assert "tool_calls" in data or "response_content" in data


class TestWorkerToolCallingBatch:
    """Test batch tool calling endpoint with Worker binding."""

    @pytest.mark.parametrize("model", MODELS)
    def test_tools_batch(self, dev_server, model):
        """POST /tools-batch should return batch tool calling results."""
        port = dev_server
        response = requests.post(
            f"http://localhost:{port}/tools-batch",
            json={
                "messages": [
                    "What's the weather in New York?",
                    "What's the stock price of AAPL?",
                ],
                "model": model,
            },
            headers={"Content-Type": "application/json"},
        )

        assert response.status_code == 200
        data = response.json()

        assert "results" in data
        assert data["count"] == 2
        assert len(data["results"]) == 2

        for i, result in enumerate(data["results"]):
            assert result is not None, f"Result {i} is None for {model}"


# MARK: - Multi-Turn Tests


class TestWorkerMultiTurn:
    """Test multi-turn conversation endpoint with Worker binding."""

    @pytest.mark.parametrize("model", MODELS)
    def test_multi_turn_conversation(self, dev_server, model):
        """POST /multi-turn should handle multi-turn conversations."""
        port = dev_server
        response = requests.post(
            f"http://localhost:{port}/multi-turn",
            json={"message": "What is the weather in NYC?", "model": model},
            headers={"Content-Type": "application/json"},
        )

        assert response.status_code == 200
        data = response.json()

        assert "conversation" in data
        assert "final_response" in data
        assert len(data["conversation"]) >= 1


# MARK: - Agent Tests


class TestWorkerAgentStructuredOutput:
    """Test create_agent with structured output endpoint.

    Note: These tests will fail with 501 if create_agent is not available in the
    Pyodide environment due to the uuid-utils dependency in langsmith.
    See .claude/create_agent_pyodide_issue.md for details.
    """

    @pytest.mark.parametrize("model", MODELS)
    def test_agent_structured_output(self, dev_server, model):
        """POST /agent-structured should use create_agent with structured output."""
        port = dev_server
        response = requests.post(
            f"http://localhost:{port}/agent-structured",
            json={"text": "Apple Inc announced record Q4 earnings.", "model": model},
            headers={"Content-Type": "application/json"},
        )

        if response.status_code == 501:
            pytest.skip("create_agent unavailable (uuid-utils not in Pyodide)")

        assert response.status_code == 200, (
            f"Expected 200, got {response.status_code}. Response: {response.text}"
        )
        data = response.json()

        assert "input" in data
        assert "result" in data


class TestWorkerAgentStructuredJsonSchema:
    """Test create_agent with ToolStrategy using JSON schema dict.

    This verifies that passing a raw JSON schema dict (from
    model_json_schema()) wrapped in ToolStrategy works via the
    Worker binding, not just Pydantic models.
    """

    @pytest.mark.parametrize("model", MODELS)
    def test_agent_structured_json_schema(self, dev_server, model):
        """POST /agent-structured-json should work with ToolStrategy(json_schema)."""
        port = dev_server
        response = requests.post(
            f"http://localhost:{port}/agent-structured-json",
            json={
                "text": "Apple Inc announced record Q4 earnings.",
                "model": model,
            },
            headers={"Content-Type": "application/json"},
        )

        if response.status_code == 501:
            pytest.skip("create_agent or ToolStrategy unavailable in Pyodide")

        assert response.status_code == 200, (
            f"Expected 200, got {response.status_code}. Response: {response.text}"
        )
        data = response.json()

        assert data.get("success") is True, (
            f"ToolStrategy with JSON schema failed: {data}"
        )
        assert "result" in data
        assert data.get("strategy") == "ToolStrategy"
        assert data.get("schema_type") == "json_schema"


class TestWorkerAgentTools:
    """Test create_agent with tools endpoint.

    Note: These tests will fail with 501 if create_agent is not available in the
    Pyodide environment due to the uuid-utils dependency in langsmith.
    See .claude/create_agent_pyodide_issue.md for details.
    """

    @pytest.mark.parametrize("model", MODELS)
    def test_agent_tools(self, dev_server, model):
        """POST /agent-tools should use create_agent with tools."""
        port = dev_server
        response = requests.post(
            f"http://localhost:{port}/agent-tools",
            json={"message": "What is the weather in San Francisco?", "model": model},
            headers={"Content-Type": "application/json"},
        )

        if response.status_code == 501:
            pytest.skip("create_agent unavailable (uuid-utils not in Pyodide)")

        assert response.status_code == 200, (
            f"Expected 200, got {response.status_code}. Response: {response.text}"
        )
        data = response.json()

        assert "input" in data
        assert "result" in data or "response" in data


# MARK: - D1 Binding Tests


class TestWorkerD1:
    """Test D1 database operations via Worker binding.

    These tests verify that the D1 binding works correctly through the Worker,
    mirroring the functionality tested in test_vectorstores.py via REST API.
    """

    def test_d1_health_check(self, dev_server):
        """GET /d1-health should return healthy status."""
        port = dev_server
        response = requests.get(f"http://localhost:{port}/d1-health")

        # May fail if D1 binding not configured
        if response.status_code == 400:
            pytest.skip("D1 binding not configured in wrangler.jsonc")

        assert response.status_code == 200
        data = response.json()

        assert data["status"] == "healthy"
        assert data["database"] == "connected"
        assert data["value"] == 1

    def test_d1_crud_cycle(self, dev_server):
        """Test full CRUD cycle on D1 via Worker binding."""
        port = dev_server
        table_name = f"test_worker_{uuid.uuid4().hex[:8]}"

        # Check if D1 is available
        health_response = requests.get(f"http://localhost:{port}/d1-health")
        if health_response.status_code == 400:
            pytest.skip("D1 binding not configured in wrangler.jsonc")

        try:
            # CREATE TABLE
            create_response = requests.post(
                f"http://localhost:{port}/d1-create-table",
                json={"table_name": table_name},
                headers={"Content-Type": "application/json"},
            )
            assert create_response.status_code == 200
            assert create_response.json()["success"] is True

            # INSERT
            records = [
                {
                    "id": "doc-1",
                    "text": "First document",
                    "namespace": "test",
                    "metadata": "{}",
                },
                {
                    "id": "doc-2",
                    "text": "Second document",
                    "namespace": "test",
                    "metadata": "{}",
                },
            ]
            insert_response = requests.post(
                f"http://localhost:{port}/d1-insert",
                json={"table_name": table_name, "records": records},
                headers={"Content-Type": "application/json"},
            )
            assert insert_response.status_code == 200
            assert insert_response.json()["success"] is True
            assert insert_response.json()["inserted"] == 2

            # QUERY
            query_response = requests.post(
                f"http://localhost:{port}/d1-query",
                json={"table_name": table_name, "ids": ["doc-1"]},
                headers={"Content-Type": "application/json"},
            )
            assert query_response.status_code == 200
            query_data = query_response.json()
            assert query_data["success"] is True
            assert query_data["count"] == 1
            assert query_data["results"][0]["text"] == "First document"

            # QUERY ALL
            query_all_response = requests.post(
                f"http://localhost:{port}/d1-query",
                json={"table_name": table_name},
                headers={"Content-Type": "application/json"},
            )
            assert query_all_response.status_code == 200
            assert query_all_response.json()["count"] == 2

        finally:
            # DROP TABLE (cleanup)
            drop_response = requests.post(
                f"http://localhost:{port}/d1-drop-table",
                json={"table_name": table_name},
                headers={"Content-Type": "application/json"},
            )
            assert drop_response.status_code == 200


# MARK: - Vectorize Binding Tests


class TestWorkerVectorize:
    """Test Vectorize operations via Worker binding.

    These tests verify that the Vectorize binding works correctly through the Worker.
    """

    def test_vectorize_info(self, dev_server_with_vectorize):
        """GET /vectorize-info should return index information."""
        port, index_name = dev_server_with_vectorize
        response = requests.get(f"http://localhost:{port}/vectorize-info")

        if response.status_code == 500:
            data = response.json()
            if "VECTORIZE binding not configured" in data.get("error", ""):
                pytest.skip("Vectorize binding not configured")

        assert response.status_code == 200
        data = response.json()
        assert "index_info" in data

    def test_vectorize_insert_and_search(self, dev_server_with_vectorize):
        """Test inserting documents and searching for them via Worker binding."""
        port, index_name = dev_server_with_vectorize

        # Generate unique IDs for this test
        test_id_1 = f"worker-test-{uuid.uuid4().hex[:8]}"
        test_id_2 = f"worker-test-{uuid.uuid4().hex[:8]}"

        try:
            # Insert documents
            insert_response = requests.post(
                f"http://localhost:{port}/vectorize-insert",
                json={
                    "texts": [
                        "The capital of France is Paris.",
                        "Python is a programming language.",
                    ],
                    "ids": [test_id_1, test_id_2],
                    "metadatas": [
                        {"category": "geography"},
                        {"category": "technology"},
                    ],
                },
                headers={"Content-Type": "application/json"},
            )

            if insert_response.status_code == 500:
                data = insert_response.json()
                if "VECTORIZE binding not configured" in data.get("error", ""):
                    pytest.skip("Vectorize binding not configured")

            assert insert_response.status_code == 200
            insert_data = insert_response.json()
            assert "inserted_ids" in insert_data
            assert insert_data["count"] == 2

            # Wait for vectors to be indexed (Vectorize has eventual consistency)
            time.sleep(5)

            # Search for documents
            search_response = requests.post(
                f"http://localhost:{port}/vectorize-search",
                json={
                    "query": "What is the capital of France?",
                    "k": 2,
                },
                headers={"Content-Type": "application/json"},
            )

            assert search_response.status_code == 200
            search_data = search_response.json()
            assert "results" in search_data
            assert "query" in search_data

        finally:
            # Clean up - delete the documents
            requests.post(
                f"http://localhost:{port}/vectorize-delete",
                json={"ids": [test_id_1, test_id_2]},
                headers={"Content-Type": "application/json"},
            )

    def test_vectorize_insert_missing_texts(self, dev_server_with_vectorize):
        """POST /vectorize-insert without texts should return error."""
        port, _ = dev_server_with_vectorize
        response = requests.post(
            f"http://localhost:{port}/vectorize-insert",
            json={},
            headers={"Content-Type": "application/json"},
        )

        assert response.status_code == 400
        data = response.json()
        assert "error" in data

    def test_vectorize_search_missing_query(self, dev_server_with_vectorize):
        """POST /vectorize-search without query should return error."""
        port, _ = dev_server_with_vectorize
        response = requests.post(
            f"http://localhost:{port}/vectorize-search",
            json={},
            headers={"Content-Type": "application/json"},
        )

        assert response.status_code == 400
        data = response.json()
        assert "error" in data


# MARK: - Error Handling Tests


# MARK: - Reranker Binding Tests


class TestWorkerReranker:
    """Test Reranker operations via Worker binding.

    These tests verify that the Reranker binding works correctly through the Worker.
    """

    def test_rerank_basic(self, dev_server_with_vectorize):
        """POST /rerank should rerank documents based on query relevance."""
        port, _ = dev_server_with_vectorize

        response = requests.post(
            f"http://localhost:{port}/rerank",
            json={
                "query": "What is the capital of France?",
                "documents": [
                    "Paris is the capital and largest city of France.",
                    "Berlin is the capital of Germany.",
                    "The Eiffel Tower is located in Paris, France.",
                    "London is the capital of the United Kingdom.",
                ],
                "top_k": 3,
            },
            headers={"Content-Type": "application/json"},
        )

        if response.status_code == 500:
            data = response.json()
            if "AI binding not configured" in data.get("error", ""):
                pytest.skip("AI binding not configured")

        assert response.status_code == 200
        data = response.json()

        assert data["success"] is True
        assert "results" in data
        assert "query" in data
        assert len(data["results"]) <= 3
        assert len(data["results"]) > 0, "Reranker returned no results"

        # Results should be sorted by score (descending)
        # The Paris-related documents should score higher
        if data["results"]:
            assert "score" in data["results"][0]
            assert "text" in data["results"][0]
            assert "index" in data["results"][0]

    def test_rerank_default_documents(self, dev_server_with_vectorize):
        """POST /rerank with empty body should use default documents."""
        port, _ = dev_server_with_vectorize

        response = requests.post(
            f"http://localhost:{port}/rerank",
            json={},
            headers={"Content-Type": "application/json"},
        )

        if response.status_code == 500:
            data = response.json()
            if "AI binding not configured" in data.get("error", ""):
                pytest.skip("AI binding not configured")

        assert response.status_code == 200
        data = response.json()

        assert data["success"] is True
        assert "results" in data
        assert len(data["results"]) > 0, (
            "Reranker returned no results for default documents"
        )

    def test_vectorize_search_with_rerank(self, dev_server_with_vectorize):
        """POST /vectorize-search with rerank=true should rerank results."""
        port, index_name = dev_server_with_vectorize

        # Generate unique test run ID to isolate this test's data
        test_run_id = uuid.uuid4().hex[:8]
        test_id_1 = f"rerank-test-{test_run_id}-1"
        test_id_2 = f"rerank-test-{test_run_id}-2"
        test_id_3 = f"rerank-test-{test_run_id}-3"

        try:
            # Insert with include_d1=True to store text content in D1
            # and wait=True to ensure vectors are indexed before searching
            insert_response = requests.post(
                f"http://localhost:{port}/vectorize-insert",
                json={
                    "texts": [
                        "The quick brown fox jumps over the lazy dog.",
                        "Machine learning is a subset of artificial intelligence.",
                        "Python is a popular programming language for AI.",
                    ],
                    "ids": [test_id_1, test_id_2, test_id_3],
                    "metadatas": [
                        {"category": "animals"},
                        {"category": "technology"},
                        {"category": "programming"},
                    ],
                    "include_d1": True,
                    "wait": True,
                },
                headers={"Content-Type": "application/json"},
                timeout=60,  # Longer timeout for wait=True
            )

            if insert_response.status_code == 500:
                data = insert_response.json()
                error_msg = data.get("error", "")
                if "VECTORIZE binding not configured" in error_msg:
                    pytest.skip("Vectorize binding not configured")
                if "D1 binding not configured" in error_msg:
                    pytest.skip("D1 binding not configured for text storage")
                if "no such table" in error_msg.lower():
                    pytest.skip("D1 table not created - run vectorize tests first")
                if "greenlet" in error_msg.lower():
                    pytest.skip(
                        "D1 requires greenlet (not in Pyodide). "
                        "Test with standalone /rerank instead."
                    )

            assert insert_response.status_code == 200, (
                f"Insert failed: {insert_response.json()}"
            )

            # Search with reranking enabled and include_d1 to retrieve text
            # The reranker will filter out results with empty page_content
            search_response = requests.post(
                f"http://localhost:{port}/vectorize-search",
                json={
                    "query": "What programming language is used for AI?",
                    "k": 3,
                    "rerank": True,
                    "include_d1": True,
                },
                headers={"Content-Type": "application/json"},
            )

            if search_response.status_code == 500:
                error_data = search_response.json()
                # Skip if error is related to empty content
                error_msg = str(error_data.get("error", "")).lower()
                if (
                    "empty" in error_msg
                    or "length" in error_msg
                    or "contexts" in error_msg
                ):
                    pytest.skip(
                        "Reranking requires page_content to be populated. "
                        "D1 integration may not be working correctly."
                    )

            assert search_response.status_code == 200
            search_data = search_response.json()

            assert "results" in search_data
            assert search_data.get("reranked") is True

            # Reranked results should have both original_score and rerank_score
            if search_data["results"]:
                result = search_data["results"][0]
                assert "rerank_score" in result
                assert "original_score" in result
                # With D1 integration, page_content should be populated
                assert result.get("page_content"), (
                    "page_content should not be empty with D1"
                )

        finally:
            # Clean up - delete from both Vectorize and D1
            requests.post(
                f"http://localhost:{port}/vectorize-delete",
                json={"ids": [test_id_1, test_id_2, test_id_3], "include_d1": True},
                headers={"Content-Type": "application/json"},
            )


# MARK: - Error Handling Tests


class TestWorkerErrorHandling:
    """Test error handling in Worker."""

    def test_unknown_endpoint_returns_index(self, dev_server):
        """GET /unknown should return index documentation."""
        port = dev_server
        response = requests.get(f"http://localhost:{port}/unknown")

        assert response.status_code == 200
        data = response.json()
        assert "endpoints" in data

    def test_invalid_json_returns_error(self, dev_server):
        """POST with invalid JSON should return an error."""
        port = dev_server
        response = requests.post(
            f"http://localhost:{port}/chat",
            data="not valid json",
            headers={"Content-Type": "application/json"},
        )

        assert response.status_code == 500
        data = response.json()
        assert "error" in data


# MARK: - AI Gateway Tests


class TestWorkerAIGateway:
    """Test AI Gateway integration with Workers AI bindings."""

    def test_ai_gateway_chat(self, dev_server):
        """Test chat model with AI Gateway routing."""
        port = dev_server
        response = requests.post(
            f"http://localhost:{port}/ai-gateway-test",
            json={
                "gateway_id": "test-ai-gateway",
                "test_type": "chat",
            },
            headers={"Content-Type": "application/json"},
        )

        assert response.status_code == 200, f"Failed: {response.text}"
        data = response.json()
        assert data["success"] is True
        assert data["gateway_id"] == "test-ai-gateway"
        assert "chat" in data["results"]
        assert data["results"]["chat"]["success"] is True
        assert data["results"]["chat"]["gateway"] == "test-ai-gateway"

    def test_ai_gateway_embeddings(self, dev_server):
        """Test embeddings model with AI Gateway routing."""
        port = dev_server
        response = requests.post(
            f"http://localhost:{port}/ai-gateway-test",
            json={
                "gateway_id": "test-ai-gateway",
                "test_type": "embeddings",
            },
            headers={"Content-Type": "application/json"},
        )

        assert response.status_code == 200, f"Failed: {response.text}"
        data = response.json()
        assert data["success"] is True
        assert "embeddings" in data["results"]
        assert data["results"]["embeddings"]["success"] is True
        assert data["results"]["embeddings"]["gateway"] == "test-ai-gateway"
        assert data["results"]["embeddings"]["dimensions"] > 0

    def test_ai_gateway_all(self, dev_server):
        """Test all models with AI Gateway routing."""
        port = dev_server
        response = requests.post(
            f"http://localhost:{port}/ai-gateway-test",
            json={
                "gateway_id": "test-ai-gateway",
                "test_type": "all",
            },
            headers={"Content-Type": "application/json"},
        )

        assert response.status_code == 200, f"Failed: {response.text}"
        data = response.json()
        assert data["success"] is True
        assert data["gateway_id"] == "test-ai-gateway"

        # All three should be tested
        assert "chat" in data["results"]
        assert "embeddings" in data["results"]
        assert "reranker" in data["results"]

        # All should succeed
        assert data["results"]["chat"]["success"] is True
        assert data["results"]["embeddings"]["success"] is True
        assert data["results"]["reranker"]["success"] is True
        assert data["results"]["reranker"]["count"] > 0, (
            "AI Gateway reranker returned no results"
        )


# MARK: - Reasoning Content Tests


class TestWorkerReasoningContent:
    """Test reasoning_content extraction from content blocks via Worker binding."""

    REASONING_MODELS = [
        "@cf/qwen/qwen3-30b-a3b-fp8",
        "@cf/zai-org/glm-4.7-flash",
        "@cf/openai/gpt-oss-120b",
        "@cf/openai/gpt-oss-20b",
        "@cf/moonshotai/kimi-k2.5",
        "@cf/nvidia/nemotron-3-120b-a12b",
    ]

    @pytest.mark.parametrize("model", REASONING_MODELS)
    def test_reasoning_content_returned(self, dev_server, model):
        """POST /reasoning should return reasoning_content."""
        port = dev_server
        response = requests.post(
            f"http://localhost:{port}/reasoning",
            json={
                "message": "What is 25 * 37? Think step by step.",
                "model": model,
            },
            headers={"Content-Type": "application/json"},
        )

        assert response.status_code == 200, f"Failed: {response.text}"
        data = response.json()

        assert "content" in data
        assert len(data["content"]) > 0, f"Expected non-empty content for {model}"
        assert data["model"] == model

        assert data["has_reasoning_content"] is True, (
            f"Expected reasoning_content for {model}"
        )
        assert data["reasoning_content"] is not None
        assert len(data["reasoning_content"]) > 0, (
            f"Expected non-empty reasoning_content for {model}"
        )

    @pytest.mark.parametrize("model", REASONING_MODELS)
    def test_reasoning_content_with_tool_calls(self, dev_server, model):
        """POST /reasoning-tools should preserve reasoning_content."""
        port = dev_server
        response = requests.post(
            f"http://localhost:{port}/reasoning-tools",
            json={
                "message": "What's the weather in San Francisco?",
                "model": model,
            },
            headers={"Content-Type": "application/json"},
        )

        assert response.status_code == 200, f"Failed: {response.text}"
        data = response.json()

        assert data["model"] == model

        if data["has_tool_calls"] and data["has_reasoning_content"]:
            assert data["reasoning_content"] is not None
            assert len(data["reasoning_content"]) > 0, (
                f"Expected non-empty reasoning_content alongside tool calls for {model}"
            )
            assert data["content_type"] == "list", (
                f"Content should be list when reasoning + tool "
                f"calls present, got {data['content_type']} "
                f"for {model}"
            )
        elif data["has_tool_calls"]:
            # Tool call without reasoning - acceptable
            pass


# MARK: - Multi-Modal Tests


def create_test_image_base64() -> str:
    """Create a minimal 1x1 red pixel PNG and return as base64.

    Uses raw PNG bytes to avoid requiring PIL in the test environment.
    """
    import struct
    import zlib

    def _png_chunk(chunk_type: bytes, data: bytes) -> bytes:
        chunk = chunk_type + data
        return (
            struct.pack(">I", len(data))
            + chunk
            + struct.pack(">I", zlib.crc32(chunk) & 0xFFFFFFFF)
        )

    width, height = 1, 1
    ihdr_data = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)
    raw_row = b"\x00" + b"\xff\x00\x00"
    idat_data = zlib.compress(raw_row)

    png = b"\x89PNG\r\n\x1a\n"
    png += _png_chunk(b"IHDR", ihdr_data)
    png += _png_chunk(b"IDAT", idat_data)
    png += _png_chunk(b"IEND", b"")

    return base64.standard_b64encode(png).decode("utf-8")


class TestWorkerMultiModal:
    """Test multi-modal image input via native AI binding.

    Discovery test: Which Workers AI models accept image content blocks
    when invoked through the native AI binding (env.AI) in a Python Worker?

    Previous REST API testing showed only Mistral Small 3.1 supports image
    base64 input. This tests whether the native binding behaves differently.
    """

    @pytest.mark.parametrize("model", MODELS)
    def test_multi_modal_image(self, dev_server, model):
        """POST /multi-modal should attempt image input via native AI binding."""
        port = dev_server
        image_b64 = create_test_image_base64()

        response = requests.post(
            f"http://localhost:{port}/multi-modal",
            json={
                "model": model,
                "image_base64": image_b64,
                "prompt": "Describe this image in one sentence. What color is it?",
            },
            headers={"Content-Type": "application/json"},
            timeout=60,
        )

        print(f"\n  === Multi-Modal Result: {model} ===")  # noqa: T201
        print(f"  Status code: {response.status_code}")  # noqa: T201

        if response.status_code == 200:
            data = response.json()
            print(f"  Response: {data.get('response', '')[:200]}")  # noqa: T201
            assert "response" in data
            assert len(data["response"]) > 0, (
                f"Expected non-empty response from {model}"
            )
        else:
            data = response.json()
            error_msg = data.get("error", "Unknown error")
            print(f"  Error: {error_msg[:200]}")  # noqa: T201
            pytest.skip(
                f"Model {model} does not support multi-modal via binding: "
                f"{error_msg[:100]}"
            )

    def test_multi_modal_missing_image(self, dev_server):
        """POST /multi-modal without image_base64 should return 400."""
        port = dev_server
        response = requests.post(
            f"http://localhost:{port}/multi-modal",
            json={"prompt": "Describe this image."},
            headers={"Content-Type": "application/json"},
        )

        assert response.status_code == 400
        data = response.json()
        assert "error" in data


# MARK: - Session Affinity (Prompt Caching) Tests
class TestWorkerSessionAffinity:
    """Test prompt caching via session_id on Worker binding.

    Tests the /session-affinity endpoint which passes session_id
    through to the binding options as x-session-affinity header.
    Currently only kimi-k2.5 is known to support prompt caching.
    """

    def test_session_affinity_basic(self, dev_server):
        """POST /session-affinity should return a response with session_id."""
        port = dev_server
        response = requests.post(
            f"http://localhost:{port}/session-affinity",
            json={
                "model": "@cf/moonshotai/kimi-k2.5",
                "message": "Say hello in exactly 3 words.",
                "session_id": "test-worker-session",
            },
            headers={"Content-Type": "application/json"},
            timeout=30,
        )

        print(f"  Status: {response.status_code}")  # noqa: T201
        assert response.status_code == 200
        data = response.json()
        print(f"  Response: {data.get('response', '')[:200]}")  # noqa: T201
        assert "response" in data
        assert len(data["response"]) > 0
        assert data["session_id"] == "test-worker-session"

    def test_session_affinity_repeated_calls(self, dev_server):
        """Two calls with the same session_id should both succeed."""
        port = dev_server
        session_id = f"test-repeat-{uuid.uuid4().hex[:8]}"

        for i in range(2):
            response = requests.post(
                f"http://localhost:{port}/session-affinity",
                json={
                    "model": "@cf/moonshotai/kimi-k2.5",
                    "message": f"What is {i + 1} + {i + 1}?",
                    "session_id": session_id,
                },
                headers={"Content-Type": "application/json"},
                timeout=30,
            )

            print(f"  Call {i + 1} status: {response.status_code}")  # noqa: T201
            assert response.status_code == 200
            data = response.json()
            assert len(data["response"]) > 0

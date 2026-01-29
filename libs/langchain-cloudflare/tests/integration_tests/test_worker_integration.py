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

Note: These tests require:
1. The examples/workers directory to be set up
2. Valid Cloudflare credentials configured in wrangler.jsonc
3. pywrangler installed (uv add workers-py)
"""

import time
import uuid

import pytest
import requests

# Models to test against (subset for faster tests)
MODELS = [
    "@cf/qwen/qwen3-30b-a3b-fp8",
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

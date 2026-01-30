"""LangChain + Cloudflare Python Worker Example.

This example demonstrates how to use langchain-cloudflare with Python Workers,
using the Workers AI and Vectorize bindings directly for optimal performance.

Features demonstrated:
- Basic chat invocation
- Structured output with Pydantic models
- Tool calling
- Multi-turn conversations
- create_agent pattern with structured output and tools
- Vectorize operations (insert, search, delete)
- D1 database operations

All endpoints accept an optional "model" parameter in the request body to specify
which Workers AI model to use. Defaults to Qwen if not specified.
"""

from langchain_core.messages import HumanMessage, ToolMessage
from models import Data
from tools import ALL_TOOLS, get_stock_price, get_weather
from workers import Response, WorkerEntrypoint

from langchain_cloudflare import ChatCloudflareWorkersAI, CloudflareVectorize
from langchain_cloudflare.embeddings import CloudflareWorkersAIEmbeddings
from langchain_cloudflare.rerankers import CloudflareWorkersAIReranker

# MARK: - Models

# Supported Workers AI models for this example
SUPPORTED_MODELS = [
    "@cf/meta/llama-3.3-70b-instruct-fp8-fast",
    "@cf/mistralai/mistral-small-3.1-24b-instruct",
    "@cf/qwen/qwen3-30b-a3b-fp8",
]

DEFAULT_MODEL = "@cf/qwen/qwen3-30b-a3b-fp8"

# Embedding model for Vectorize
EMBEDDING_MODEL = "@cf/baai/bge-base-en-v1.5"

# Reranker model
RERANKER_MODEL = "@cf/baai/bge-reranker-base"


# MARK: - Agent Import

try:
    from langchain.agents import create_agent

    CREATE_AGENT_AVAILABLE = True
except ImportError:
    CREATE_AGENT_AVAILABLE = False

try:
    from langchain.agents.structured_output import ToolStrategy

    TOOL_STRATEGY_AVAILABLE = True
except ImportError:
    TOOL_STRATEGY_AVAILABLE = False


# MARK: - Worker Entrypoint


class Default(WorkerEntrypoint):
    """Main Worker entrypoint for LangChain examples."""

    # MARK: - Request Routing

    async def fetch(self, request, env):
        """Handle incoming HTTP requests."""
        try:
            url = request.url
            path = url.split("/")[-1].split("?")[0] if "/" in url else ""

            if path == "chat":
                return await self.handle_chat(request)
            elif path == "structured":
                return await self.handle_structured_output(request)
            elif path == "tools":
                return await self.handle_tool_calling(request)
            elif path == "multi-turn":
                return await self.handle_multi_turn(request)
            elif path == "agent-structured":
                return await self.handle_agent_structured_output(request)
            elif path == "agent-structured-json":
                return await self.handle_agent_structured_json_schema(request)
            elif path == "agent-tools":
                return await self.handle_agent_tools(request)
            # Vectorize endpoints
            elif path == "vectorize-insert":
                return await self.handle_vectorize_insert(request)
            elif path == "vectorize-search":
                return await self.handle_vectorize_search(request)
            elif path == "vectorize-delete":
                return await self.handle_vectorize_delete(request)
            elif path == "vectorize-info":
                return await self.handle_vectorize_info(request)
            # Reranker endpoint
            elif path == "rerank":
                return await self.handle_rerank(request)
            # AI Gateway test endpoint
            elif path == "ai-gateway-test":
                return await self.handle_ai_gateway_test(request)
            # D1 endpoints
            elif path == "d1-health":
                return await self.handle_d1_health(request)
            elif path == "d1-create-table":
                return await self.handle_d1_create_table(request)
            elif path == "d1-insert":
                return await self.handle_d1_insert(request)
            elif path == "d1-query":
                return await self.handle_d1_query(request)
            elif path == "d1-drop-table":
                return await self.handle_d1_drop_table(request)
            else:
                return await self.handle_index()

        except Exception as e:
            return Response.json(
                {"error": str(e), "type": type(e).__name__},
                status=500,
            )

    # MARK: - Index Handler

    async def handle_index(self):
        """Return API documentation."""
        # Check binding availability
        vectorize_available = hasattr(self.env, "VECTORIZE")
        d1_available = hasattr(self.env, "D1")

        return Response.json(
            {
                "name": "LangChain + Cloudflare Python Workers Example",
                "create_agent_available": CREATE_AGENT_AVAILABLE,
                "vectorize_available": vectorize_available,
                "d1_available": d1_available,
                "supported_models": SUPPORTED_MODELS,
                "default_model": DEFAULT_MODEL,
                "embedding_model": EMBEDDING_MODEL,
                "endpoints": {
                    "/chat": "Basic chat completion",
                    "/structured": "Structured output with Pydantic models",
                    "/tools": "Tool calling example",
                    "/multi-turn": "Multi-turn conversation with tools",
                    "/agent-structured": "create_agent with structured output",
                    "/agent-tools": "create_agent with tools",
                    "/vectorize-insert": "Insert documents into Vectorize",
                    "/vectorize-search": "Similarity search (rerank=true)",
                    "/vectorize-delete": "Delete documents from Vectorize",
                    "/vectorize-info": "Get Vectorize index info",
                    "/rerank": "Rerank documents by query relevance",
                    "/ai-gateway-test": "Test AI Gateway with bindings",
                    "/d1-health": "D1 database health check",
                    "/d1-create-table": "Create a D1 table",
                    "/d1-insert": "Insert records into D1",
                    "/d1-query": "Query D1 table",
                    "/d1-drop-table": "Drop a D1 table",
                },
            }
        )

    # MARK: - Chat Handler

    async def handle_chat(self, request):
        """Handle basic chat completion."""
        data = await request.json()
        message = data.get("message", "Hello!")
        model = data.get("model", DEFAULT_MODEL)

        llm = ChatCloudflareWorkersAI(
            model_name=model,
            binding=self.env.AI,
            temperature=0.7,
        )

        response = await llm.ainvoke(message)

        return Response.json(
            {
                "response": response.content,
                "model": llm.model,
            }
        )

    # MARK: - Structured Output Handler

    async def handle_structured_output(self, request):
        """Handle structured output with Pydantic models."""
        from pydantic import ValidationError

        data = await request.json()
        text = data.get("text", "Acme Corp announced a partnership with TechGiant Inc.")
        model = data.get("model", DEFAULT_MODEL)

        llm = ChatCloudflareWorkersAI(
            model_name=model,
            binding=self.env.AI,
            temperature=0.0,
        )

        structured_llm = llm.with_structured_output(Data)

        prompt = f"""Extract announcements from this text as structured data.

Text: {text}

Return JSON with an "announcements" array. Each announcement should have:
- type: partnership, investment, regulatory, milestone, event, m&a, or none
- context: brief description
- entities: array with name, ticker (optional), and role"""

        try:
            result = await structured_llm.ainvoke(prompt)

            if isinstance(result, Data):
                result_dict = result.model_dump()
            elif isinstance(result, dict):
                try:
                    validated = Data(**result)
                    result_dict = validated.model_dump()
                except ValidationError:
                    result_dict = result
            else:
                result_dict = {"raw": str(result)}

            return Response.json(
                {
                    "input": text,
                    "extracted": result_dict,
                }
            )

        except ValidationError as e:
            return Response.json(
                {
                    "input": text,
                    "extracted": {"announcements": []},
                    "validation_warning": str(e),
                }
            )

    # MARK: - Tool Calling Handler

    async def handle_tool_calling(self, request):
        """Handle tool calling."""
        data = await request.json()
        message = data.get("message", "What's the weather in San Francisco?")
        model = data.get("model", DEFAULT_MODEL)

        llm = ChatCloudflareWorkersAI(
            model_name=model,
            binding=self.env.AI,
            temperature=0.0,
        )

        llm_with_tools = llm.bind_tools(ALL_TOOLS)

        response = await llm_with_tools.ainvoke(message)

        tool_results = []
        if response.tool_calls:
            for tool_call in response.tool_calls:
                tool_name = tool_call["name"]
                tool_args = tool_call["args"]

                if tool_name == "get_weather":
                    result = get_weather.invoke(tool_args)
                elif tool_name == "get_stock_price":
                    result = get_stock_price.invoke(tool_args)
                else:
                    result = f"Unknown tool: {tool_name}"

                tool_results.append(
                    {
                        "tool": tool_name,
                        "args": tool_args,
                        "result": result,
                    }
                )

        return Response.json(
            {
                "input": message,
                "response_content": response.content,
                "tool_calls": response.tool_calls,
                "tool_results": tool_results,
            }
        )

    # MARK: - Multi-Turn Handler

    async def handle_multi_turn(self, request):
        """Handle multi-turn conversation with tools."""
        data = await request.json()
        initial_message = data.get("message", "What's the weather in NYC?")
        model = data.get("model", DEFAULT_MODEL)

        llm = ChatCloudflareWorkersAI(
            model_name=model,
            binding=self.env.AI,
            temperature=0.0,
        )

        llm_with_tools = llm.bind_tools(ALL_TOOLS)

        messages = [HumanMessage(content=initial_message)]
        response1 = await llm_with_tools.ainvoke(messages)

        conversation = [
            {"role": "user", "content": initial_message},
            {
                "role": "assistant",
                "content": response1.content,
                "tool_calls": response1.tool_calls,
            },
        ]

        if response1.tool_calls:
            tool_call = response1.tool_calls[0]
            tool_name = tool_call["name"]
            tool_args = tool_call["args"]

            if tool_name == "get_weather":
                tool_result = get_weather.invoke(tool_args)
            elif tool_name == "get_stock_price":
                tool_result = get_stock_price.invoke(tool_args)
            else:
                tool_result = f"Unknown tool: {tool_name}"

            conversation.append(
                {
                    "role": "tool",
                    "name": tool_name,
                    "content": tool_result,
                }
            )

            messages.append(response1)
            messages.append(
                ToolMessage(
                    content=tool_result,
                    tool_call_id=tool_call.get("id", "unknown"),
                    name=tool_name,
                )
            )

            response2 = await llm_with_tools.ainvoke(messages)

            conversation.append(
                {
                    "role": "assistant",
                    "content": response2.content,
                }
            )

            return Response.json(
                {
                    "conversation": conversation,
                    "final_response": response2.content,
                }
            )

        return Response.json(
            {
                "conversation": conversation,
                "final_response": response1.content,
            }
        )

    # MARK: - Agent Structured Output Handler

    async def handle_agent_structured_output(self, request):
        """Handle create_agent with structured output."""
        if not CREATE_AGENT_AVAILABLE:
            return Response.json(
                {"error": "create_agent is not available. Install langchain>=0.3.0"},
                status=501,
            )

        data = await request.json()
        text = data.get("text", "Acme Corp announced a partnership with TechGiant Inc.")
        model = data.get("model", DEFAULT_MODEL)

        llm = ChatCloudflareWorkersAI(
            model_name=model,
            binding=self.env.AI,
            temperature=0.0,
        )

        system_prompt = (
            "You are a press release analyst. Extract announcements. "
            "Classify as: partnership, investment, regulatory, milestone, "
            "event, m&a, or none. Return in structured format."
        )

        agent = create_agent(
            model=llm,
            response_format=Data,
            system_prompt=system_prompt,
            tools=[],
        )

        result = await agent.ainvoke(
            {"messages": [{"role": "user", "content": f"Text: {text}"}]}
        )

        if isinstance(result, dict):
            if "structured_response" in result:
                structured = result["structured_response"]
                if isinstance(structured, Data):
                    structured = structured.model_dump()
                return Response.json(
                    {
                        "input": text,
                        "result": structured,
                    }
                )
            return Response.json(
                {
                    "input": text,
                    "result": result,
                }
            )

        return Response.json(
            {
                "input": text,
                "result": str(result),
            }
        )

    # MARK: - Agent Structured JSON Schema Handler

    async def handle_agent_structured_json_schema(self, request):
        """Handle create_agent with ToolStrategy using a JSON schema dict.

        This tests the code path where response_format is a raw JSON schema
        dict wrapped in ToolStrategy, rather than a Pydantic model.
        """
        if not CREATE_AGENT_AVAILABLE:
            return Response.json(
                {"error": "create_agent is not available"},
                status=501,
            )
        if not TOOL_STRATEGY_AVAILABLE:
            return Response.json(
                {"error": "ToolStrategy is not available"},
                status=501,
            )

        data = await request.json()
        text = data.get(
            "text",
            "Acme Corp announced a partnership with TechGiant Inc.",
        )
        model = data.get("model", DEFAULT_MODEL)

        llm = ChatCloudflareWorkersAI(
            model_name=model,
            binding=self.env.AI,
            temperature=0.0,
        )

        # Use JSON schema dict instead of Pydantic model
        json_schema = Data.model_json_schema()

        system_prompt = (
            "You are a press release analyst. Extract announcements. "
            "Classify as: partnership, investment, regulatory, milestone, "
            "event, m&a, or none. Return in structured format."
        )

        agent = create_agent(
            model=llm,
            response_format=ToolStrategy(json_schema),
            system_prompt=system_prompt,
            tools=[],
        )

        result = await agent.ainvoke(
            {"messages": [{"role": "user", "content": f"Text: {text}"}]}
        )

        if isinstance(result, dict):
            structured = result.get("structured_response", result)
            if hasattr(structured, "model_dump"):
                structured = structured.model_dump()
            return Response.json(
                {
                    "success": True,
                    "input": text,
                    "result": structured,
                    "strategy": "ToolStrategy",
                    "schema_type": "json_schema",
                }
            )

        return Response.json(
            {
                "success": True,
                "input": text,
                "result": str(result),
                "strategy": "ToolStrategy",
                "schema_type": "json_schema",
            }
        )

    # MARK: - Agent Tools Handler

    async def handle_agent_tools(self, request):
        """Handle create_agent with tools."""
        if not CREATE_AGENT_AVAILABLE:
            return Response.json(
                {"error": "create_agent is not available. Install langchain>=0.3.0"},
                status=501,
            )

        data = await request.json()
        message = data.get("message", "What's the weather in San Francisco?")
        model = data.get("model", DEFAULT_MODEL)

        llm = ChatCloudflareWorkersAI(
            model_name=model,
            binding=self.env.AI,
            temperature=0.0,
        )

        agent = create_agent(
            model=llm,
            tools=ALL_TOOLS,
        )

        result = await agent.ainvoke(
            {"messages": [{"role": "user", "content": message}]}
        )

        if isinstance(result, dict):
            if "messages" in result:
                messages = result["messages"]
                for msg in reversed(messages):
                    if hasattr(msg, "content"):
                        return Response.json(
                            {
                                "input": message,
                                "response": msg.content,
                                "full_result": str(result),
                            }
                        )
            return Response.json(
                {
                    "input": message,
                    "result": result,
                }
            )

        return Response.json(
            {
                "input": message,
                "result": str(result),
            }
        )

    # MARK: - Vectorize Handlers

    def _get_vectorstore(self, include_d1: bool = False):
        """Get a CloudflareVectorize instance with bindings."""
        if not hasattr(self.env, "VECTORIZE"):
            raise ValueError(
                "VECTORIZE binding not configured. "
                "Add a [[vectorize]] section to wrangler.jsonc"
            )

        embeddings = CloudflareWorkersAIEmbeddings(
            model_name=EMBEDDING_MODEL,
            binding=self.env.AI,
        )

        d1_binding = None
        if include_d1:
            if not hasattr(self.env, "D1"):
                raise ValueError(
                    "D1 binding not configured. "
                    "Add a [[d1_databases]] section to wrangler.jsonc"
                )
            d1_binding = self.env.D1

        return CloudflareVectorize(
            embedding=embeddings,
            binding=self.env.VECTORIZE,
            d1_binding=d1_binding,
            index_name="langchain-test-persistent",
        )

    async def handle_vectorize_insert(self, request):
        """Handle inserting documents into Vectorize."""
        data = await request.json()
        texts = data.get("texts", [])
        ids = data.get("ids")
        metadatas = data.get("metadatas")
        upsert = data.get("upsert", False)
        include_d1 = data.get("include_d1", False)
        wait = data.get("wait", False)

        if not texts:
            return Response.json(
                {"error": "texts array is required"},
                status=400,
            )

        vectorstore = self._get_vectorstore(include_d1=include_d1)

        inserted_ids = await vectorstore.aadd_texts(
            texts=texts,
            ids=ids,
            metadatas=metadatas,
            upsert=upsert,
            include_d1=include_d1,
            wait=wait,
        )

        return Response.json(
            {
                "inserted_ids": inserted_ids,
                "count": len(inserted_ids),
                "include_d1": include_d1,
                "wait": wait,
            }
        )

    async def handle_vectorize_search(self, request):
        """Handle similarity search in Vectorize with optional reranking."""
        data = await request.json()
        query = data.get("query", "")
        k = data.get("k", 5)
        return_metadata = data.get("return_metadata", "all")
        include_d1 = data.get("include_d1", False)
        rerank = data.get("rerank", False)
        rerank_top_k = data.get("rerank_top_k", k)
        md_filter = data.get("md_filter")

        if not query:
            return Response.json(
                {"error": "query is required"},
                status=400,
            )

        vectorstore = self._get_vectorstore(include_d1=include_d1)

        results = await vectorstore.asimilarity_search_with_score(
            query=query,
            k=k,
            return_metadata=return_metadata,
            include_d1=include_d1,
            md_filter=md_filter,
        )

        formatted_results = []
        for doc, score in results:
            formatted_results.append(
                {
                    "id": doc.id,
                    "page_content": doc.page_content,
                    "metadata": doc.metadata,
                    "score": score,
                }
            )

        response_data = {
            "query": query,
            "results": formatted_results,
            "count": len(formatted_results),
            "include_d1": include_d1,
        }

        # Rerank results if requested
        if rerank and formatted_results:
            # Filter out results with empty page_content (reranker needs text)
            results_with_content = [
                r for r in formatted_results if r.get("page_content")
            ]

            if results_with_content:
                reranker = CloudflareWorkersAIReranker(
                    model_name=RERANKER_MODEL,
                    binding=self.env.AI,
                )

                # Extract texts for reranking
                texts_to_rerank = [r["page_content"] for r in results_with_content]

                # Rerank the documents
                rerank_results = await reranker.arerank(
                    query=query,
                    documents=texts_to_rerank,
                    top_k=rerank_top_k,
                )

                # Reorder results based on rerank scores
                reranked_results = []
                for rr in rerank_results:
                    original = results_with_content[rr.index]
                    reranked_results.append(
                        {
                            **original,
                            "original_score": original["score"],
                            "rerank_score": rr.score,
                        }
                    )

                response_data["results"] = reranked_results
                response_data["reranked"] = True
                response_data["reranker_model"] = RERANKER_MODEL
                response_data["filtered_empty_content"] = len(formatted_results) - len(
                    results_with_content
                )

        return Response.json(response_data)

    async def handle_vectorize_delete(self, request):
        """Handle deleting documents from Vectorize."""
        data = await request.json()
        ids = data.get("ids", [])
        include_d1 = data.get("include_d1", False)

        if not ids:
            return Response.json(
                {"error": "ids array is required"},
                status=400,
            )

        vectorstore = self._get_vectorstore(include_d1=include_d1)

        success = await vectorstore.adelete(ids=ids, include_d1=include_d1)

        return Response.json(
            {
                "deleted_ids": ids,
                "success": success,
                "include_d1": include_d1,
            }
        )

    async def handle_vectorize_info(self, request):
        """Handle getting Vectorize index info."""
        vectorstore = self._get_vectorstore()

        info = await vectorstore.aget_index_info()

        return Response.json(
            {
                "index_info": info,
            }
        )

    # MARK: - Reranker Handler

    async def handle_rerank(self, request):
        """Handle reranking documents based on query relevance."""
        data = await request.json()

        query = data.get("query", "What is the capital of France?")
        documents = data.get(
            "documents",
            [
                "Paris is the capital and largest city of France.",
                "Berlin is the capital of Germany.",
                "The Eiffel Tower is located in Paris, France.",
                "London is the capital of the United Kingdom.",
            ],
        )
        top_k = data.get("top_k", len(documents))

        reranker = CloudflareWorkersAIReranker(
            model_name=RERANKER_MODEL,
            binding=self.env.AI,
        )

        results = await reranker.arerank(
            query=query,
            documents=documents,
            top_k=top_k,
            return_documents=True,
        )

        formatted_results = [
            {
                "index": r.index,
                "score": r.score,
                "text": r.text,
            }
            for r in results
        ]

        return Response.json(
            {
                "success": True,
                "query": query,
                "model": RERANKER_MODEL,
                "results": formatted_results,
                "count": len(formatted_results),
            }
        )

    # MARK: - AI Gateway Test Handler

    async def handle_ai_gateway_test(self, request):
        """Test AI Gateway with Workers AI bindings.

        Tests that ai_gateway parameter works correctly when using bindings.
        Tests chat, embeddings, and reranker with AI Gateway routing.
        """
        data = await request.json()
        gateway_id = data.get("gateway_id", "test-ai-gateway")
        test_type = data.get("test_type", "all")  # all, chat, embeddings, reranker

        results = {}

        # Test Chat with AI Gateway
        if test_type in ("all", "chat"):
            llm = ChatCloudflareWorkersAI(
                model_name=DEFAULT_MODEL,
                binding=self.env.AI,
                ai_gateway=gateway_id,
                temperature=0.7,
            )
            chat_response = await llm.ainvoke("Say 'AI Gateway test successful'")
            results["chat"] = {
                "success": True,
                "model": DEFAULT_MODEL,
                "gateway": gateway_id,
                "response": chat_response.content[:200],
            }

        # Test Embeddings with AI Gateway
        if test_type in ("all", "embeddings"):
            embeddings = CloudflareWorkersAIEmbeddings(
                model_name=EMBEDDING_MODEL,
                binding=self.env.AI,
                ai_gateway=gateway_id,
            )
            embed_result = await embeddings.aembed_query("AI Gateway test")
            results["embeddings"] = {
                "success": True,
                "model": EMBEDDING_MODEL,
                "gateway": gateway_id,
                "dimensions": len(embed_result),
            }

        # Test Reranker with AI Gateway
        if test_type in ("all", "reranker"):
            reranker = CloudflareWorkersAIReranker(
                model_name=RERANKER_MODEL,
                binding=self.env.AI,
                ai_gateway=gateway_id,
            )
            rerank_result = await reranker.arerank(
                query="test query",
                documents=["First doc", "Second doc"],
                top_k=2,
            )
            results["reranker"] = {
                "success": True,
                "model": RERANKER_MODEL,
                "gateway": gateway_id,
                "count": len(rerank_result),
            }

        return Response.json(
            {
                "success": True,
                "gateway_id": gateway_id,
                "test_type": test_type,
                "results": results,
            }
        )

    # MARK: - D1 Handlers

    async def handle_d1_health(self, request):
        """Health check for D1 binding."""
        if not hasattr(self.env, "D1"):
            return Response.json(
                {"error": "D1 binding not configured"},
                status=400,
            )

        from sqlalchemy_cloudflare_d1 import WorkerConnection

        try:
            conn = WorkerConnection(self.env.D1)
            cursor = conn.cursor()
            await cursor.execute_async("SELECT 1 as value")
            row = cursor.fetchone()
            conn.close()

            return Response.json(
                {
                    "status": "healthy",
                    "database": "connected",
                    "value": row[0] if row else None,
                }
            )
        except Exception as e:
            return Response.json(
                {
                    "status": "unhealthy",
                    "error": str(e),
                },
                status=500,
            )

    async def handle_d1_create_table(self, request):
        """Create a table in D1."""
        if not hasattr(self.env, "D1"):
            return Response.json({"error": "D1 binding not configured"}, status=400)

        data = await request.json()
        table_name = data.get("table_name", "test_table")

        from sqlalchemy_cloudflare_d1 import WorkerConnection

        try:
            conn = WorkerConnection(self.env.D1)
            cursor = conn.cursor()
            await cursor.execute_async(f"""
                CREATE TABLE IF NOT EXISTS {table_name} (
                    id TEXT PRIMARY KEY,
                    text TEXT,
                    namespace TEXT,
                    metadata TEXT
                )
            """)
            conn.close()

            return Response.json(
                {
                    "success": True,
                    "table_name": table_name,
                }
            )
        except Exception as e:
            return Response.json(
                {
                    "success": False,
                    "error": str(e),
                },
                status=500,
            )

    async def handle_d1_insert(self, request):
        """Insert records into D1."""
        if not hasattr(self.env, "D1"):
            return Response.json({"error": "D1 binding not configured"}, status=400)

        data = await request.json()
        table_name = data.get("table_name", "test_table")
        records = data.get("records", [])

        if not records:
            return Response.json({"error": "records array is required"}, status=400)

        from sqlalchemy_cloudflare_d1 import WorkerConnection

        try:
            conn = WorkerConnection(self.env.D1)
            cursor = conn.cursor()

            for record in records:
                sql = (
                    f"INSERT OR REPLACE INTO {table_name} "
                    "(id, text, namespace, metadata) VALUES (?, ?, ?, ?)"
                )
                await cursor.execute_async(
                    sql,
                    (
                        record.get("id"),
                        record.get("text"),
                        record.get("namespace", ""),
                        record.get("metadata", "{}"),
                    ),
                )

            conn.close()

            return Response.json(
                {
                    "success": True,
                    "inserted": len(records),
                }
            )
        except Exception as e:
            return Response.json(
                {
                    "success": False,
                    "error": str(e),
                },
                status=500,
            )

    async def handle_d1_query(self, request):
        """Query D1 table."""
        if not hasattr(self.env, "D1"):
            return Response.json({"error": "D1 binding not configured"}, status=400)

        data = await request.json()
        table_name = data.get("table_name", "test_table")
        ids = data.get("ids", [])

        from sqlalchemy_cloudflare_d1 import WorkerConnection

        try:
            conn = WorkerConnection(self.env.D1)
            cursor = conn.cursor()

            if ids:
                placeholders = ",".join(["?" for _ in ids])
                sql = (
                    f"SELECT id, text, namespace, metadata FROM {table_name} "
                    f"WHERE id IN ({placeholders})"
                )
                await cursor.execute_async(sql, tuple(ids))
            else:
                sql = f"SELECT id, text, namespace, metadata FROM {table_name}"
                await cursor.execute_async(sql)

            rows = cursor.fetchall()
            conn.close()

            results = []
            for row in rows:
                results.append(
                    {
                        "id": row[0],
                        "text": row[1],
                        "namespace": row[2],
                        "metadata": row[3],
                    }
                )

            return Response.json(
                {
                    "success": True,
                    "results": results,
                    "count": len(results),
                }
            )
        except Exception as e:
            return Response.json(
                {
                    "success": False,
                    "error": str(e),
                },
                status=500,
            )

    async def handle_d1_drop_table(self, request):
        """Drop a D1 table."""
        if not hasattr(self.env, "D1"):
            return Response.json({"error": "D1 binding not configured"}, status=400)

        data = await request.json()
        table_name = data.get("table_name", "test_table")

        from sqlalchemy_cloudflare_d1 import WorkerConnection

        try:
            conn = WorkerConnection(self.env.D1)
            cursor = conn.cursor()
            await cursor.execute_async(f"DROP TABLE IF EXISTS {table_name}")
            conn.close()

            return Response.json(
                {
                    "success": True,
                    "table_name": table_name,
                }
            )
        except Exception as e:
            return Response.json(
                {
                    "success": False,
                    "error": str(e),
                },
                status=500,
            )

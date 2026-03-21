# Changelog

All notable changes to the Langchain Cloudflare packages will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).


## langchain-cloudflare

### [0.3.0]

#### Added

- **Reasoning content support**: Models that return `reasoning_content` (Qwen, GLM, GPT-OSS) now surface it as content blocks on `AIMessage`, consistent with langchain-anthropic and langchain-google-genai. When reasoning is present, `content` becomes a list of typed blocks: `[{"type": "thinking", "thinking": "..."}, {"type": "text", "text": "..."}]`.
- **`supports_reasoning_content` flag** on `ModelBehavior` to gate reasoning extraction per model family.
- **GLM model support**: Added `@cf/zai-org/glm-4.7-flash` with `unsupported_params` for `max_tokens`, `top_k`, `repetition_penalty`, and `tool_choice`.
- **GPT-OSS model support**: Added `@cf/openai/gpt-oss-120b` and `@cf/openai/gpt-oss-20b` with `supports_reasoning_content=True`.
- **Nemotron model support**: Added `@cf/nvidia/nemotron-3-120b-a12b` (120B NVIDIA model with 32K context, tool calling, structured output).
- **Kimi K2.5 model support**: Added `@cf/moonshotai/kimi-k2.5` (256K context, tool calling, structured output, reasoning content, vision/multi-modal).
- **MARK sections**: Added `# MARK: -` comments across all modules for IDE minimap navigation.

#### Fixed

- **Reasoning content preserved with tool calls**: Fixed a bug where `reasoning_content` was silently discarded when the model also returned `tool_calls`. The `if tool_calls` branch in `_create_chat_result` took priority over `elif reasoning_content`.
- **OpenAI response_format normalization**: Added `_normalize_response_format_for_cloudflare()` to transform OpenAI-style nested `response_format` to Cloudflare Workers AI format, fixing `create_agent` with `ProviderStrategy` for GPT-OSS models.

#### Breaking Changes

- **`AIMessage.content` type changes for reasoning models**: For models with `supports_reasoning_content=True` (Qwen, GLM, GPT-OSS, Kimi), when the model returns `reasoning_content`, `AIMessage.content` changes from a `str` to a `list[dict]` of typed content blocks: `[{"type": "thinking", "thinking": "..."}, {"type": "text", "text": "..."}]`. Downstream code that assumes `content` is always a string (e.g., `len(result.content)`, string concatenation, `print(result.content)`) will need to handle both types.

#### Changed

- **Version bump**: 0.2.1 → 0.3.0 (new model families and reasoning content feature).

#### Tests

- Added unit tests for reasoning content extraction across Qwen, GLM, and GPT-OSS models (with and without tool calls).
- Added `TestReasoningContent` integration tests with sync/async parametrized tests across all reasoning models.
- Added reasoning content with tool calls tests to both REST API and Worker integration suites.
- Added Nemotron to all model test lists (REST API and Worker integration).

### [0.2.1]

#### Added

- **Reproducible Pyodide dependency setup**: Added `setup_pyodide_deps.sh` script and `package.json` to automate the Python Worker dependency setup for packages without Pyodide wheels (langchain>=1.0.0, langgraph, langgraph-checkpoint).
- **Pure Python stubs for Pyodide**: Added `xxhash` and `ormsgpack` stub packages in `stubs/` with proper `pyproject.toml` files, enabling `create_agent` and langgraph imports in Cloudflare Python Workers.
- **ToolStrategy with JSON schema dict**: Added `/agent-structured-json` Worker endpoint supporting `ToolStrategy` for structured output using raw JSON schema dicts (not just Pydantic models).

#### Fixed

- **Reranker binding response handling**: Fixed `CloudflareWorkersAIReranker.arerank()` returning empty results when using native AI binding (`binding=env.AI`). `convert_reranker_response()` now handles the `response` key returned by `env.AI.run()` for reranker models.
- **Pyodide namespace packages**: Fixed langgraph imports failing in Pyodide due to PEP 420 implicit namespace packages. The setup script now creates missing `__init__.py` files.
- **ToolStrategy import isolation**: Separated `ToolStrategy` import from `create_agent` import so a missing `ToolStrategy` doesn't disable all agent endpoints.

#### Changed

- **Worker dev server flow**: `conftest.py` now uses a 3-step flow (`pywrangler sync` → `setup_pyodide_deps.sh` → `npx wrangler dev`) matching `package.json` scripts.

#### Tests

- Added unit tests for `convert_reranker_response()` covering all known response formats.
- Added `TestReranker` integration tests for the REST API reranker path.
- Added `TestWorkerAgentStructuredJsonSchema` worker integration tests for ToolStrategy with JSON schema.
- Added `TestToolStrategyJsonSchema` REST API integration tests.
- Strengthened reranker assertions in worker integration tests to assert result count > 0.

### [0.2.0]

#### Added

- **Python Workers binding support** for running langchain-cloudflare in Cloudflare Python Workers (Pyodide environment)
  - `binding=` parameter for `ChatCloudflareWorkersAI`, `CloudflareWorkersAIEmbeddings`, and `CloudflareVectorize`
  - `bindings.py` module with Pyodide/JS interop utilities
  - Full example Worker implementation in `examples/workers/`
- `CloudflareWorkersAIReranker` class for document reranking using Cloudflare Workers AI
  - Supports REST API, Worker bindings, and AI Gateway
  - `rerank()` and `arerank()` methods for sync/async reranking
  - `compress_documents()` and `acompress_documents()` methods
  - `RerankResult` dataclass for structured rerank results
- AI Gateway support for binding mode across all Workers AI components
- Integration tests for Workers binding functionality

#### Changed

- D1 async methods now use `create_engine_from_binding()` for Worker compatibility (no greenlet dependency)

#### Security

- Strengthened table name validation to prevent SQL injection (alphanumeric, underscore, hyphen only)
- **CVE-2025-68664**: Updated `langchain-core` dependency to `>=0.3.81` to address critical serialization vulnerability ("LangGrinch") that could allow secret extraction, object instantiation, and RCE via Jinja2

### [0.1.11]

#### Added

- SQLAlchemy integration via `sqlalchemy-cloudflare-d1` for D1 database operations
- New helper methods `_get_d1_engine()` and `_get_d1_table()` for SQLAlchemy engine/table management
- Unit tests for D1 SQLAlchemy integration and input validation

#### Changed

- Refactored all D1 methods to use SQLAlchemy with parameterized queries:
  - `d1_create_table` / `ad1_create_table`
  - `d1_drop_table` / `ad1_drop_table`
  - `d1_upsert_texts` / `ad1_upsert_texts` (now uses batch upsert for better performance)
  - `d1_get_by_ids` / `ad1_get_by_ids`
  - `d1_delete` / `ad1_delete`
  - `d1_metadata_query` / `ad1_metadata_query`

#### Removed

- Removed `_d1_create_upserts()` method (replaced by SQLAlchemy batch operations)

#### Security

- **CVE-TBD**: Fixed SQL injection vulnerability in D1 upsert operations. Previously, nested metadata containing single quotes could escape SQL string literals and inject arbitrary SQL. Now uses SQLAlchemy's parameterized queries which properly separate SQL from data.

### [0.1.10]

#### Added

- ModelBehavior Pydantic class for centralized model-specific configurations
- MODEL_BEHAVIORS registry with entries for Llama, Mistral, Qwen
- _translate_params_for_model() method for model-specific parameter handling
- _format_ai_message_with_tool_calls() helper method
- Integration test suite (33 tests) covering Llama, Mistral, and Qwen models

#### Changed

- Refactored scattered is_llama_model checks to use ModelBehavior registry
- Response parsing now handles both Workers AI format and OpenAI-compatible format

#### Fixed

- Mistral structured output now uses guided_json instead of response_format
- Mistral no longer receives unsupported tool_choice parameter (caused 400 errors)
- Qwen tool calls now correctly parsed from OpenAI-compatible choices[].message.tool_calls format

### [0.1.9]

#### Changed
- Updated Python version requirement from `>=3.9,<4.0` to `>=3.10,<4.0`
- Updated `langchain-core` dependency from `^0.3.15` to `>=0.3.15,<2.0.0` for better compatibility
- Updated `langchain-tests` dependency from `^0.3.17` to `>=0.3.17` for better compatibility

## langgraph-checkpoint-cloudflare-d1

### [0.1.5]

#### Security
- **CVE-2025-68664**: Updated `langchain-core` dependency to `>=0.3.81` to address critical serialization vulnerability ("LangGrinch") that could allow secret extraction, object instantiation, and RCE via Jinja2

### [0.1.4]

#### Added
- Optional logging configuration via `enable_logging` parameter in both `CloudflareD1Saver` and `AsyncCloudflareD1Saver` classes. Defaults to `False` (opt-in).

#### Security
- CVE-2025-64439 security fixes

### [0.1.0] - Initial
- Initial release


## [Unreleased]

### Added
### Changed
### Deprecated
### Removed
### Fixed
### Security

---

## Release Template

When creating a new release, copy this template and fill in the details:

## [Package Name] - [Version] - [Date]

### Added
- New features

### Changed
- Changes in existing functionality

### Deprecated
- Soon-to-be removed features

### Removed
- Now removed features

### Fixed
- Bug fixes

### Security
- Security vulnerability fixes

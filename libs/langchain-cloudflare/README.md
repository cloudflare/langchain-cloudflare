# langchain-cloudflare

This package contains the LangChain integration with CloudflareWorkersAI

## Installation

```bash
pip install -U langchain-cloudflare
```

And you should configure credentials by setting the following environment variables:

- `CF_ACCOUNT_ID`

AND

- `CF_API_TOKEN` (if using a single token scoped for all services)

OR (if using separately scoped tokens)

- `CF_AI_API_TOKEN` (CloudflareWorkersAI, CloudflareWorkersAIEmbeddings, CloudflareBrowserRunLoader, CloudflareBrowserRunTool)
- `CF_AI_SEARCH_API_TOKEN` (CloudflareAISearchRetriever)
- `CF_VECTORIZE_API_TOKEN` (CloudflareVectorize)
- `CF_D1_API_TOKEN` (CloudflareVectorize)
- `CF_D1_DATABASE_ID` (CloudflareVectorize)

> **Browser Run** requires the *Browser Rendering – Edit* permission on your API token. See [Browser Run setup](https://developers.cloudflare.com/browser-run/quick-actions/#before-you-begin).

## Chat Models

`ChatCloudflareWorkersAI` class exposes chat models from [CloudflareWorkersAI](https://developers.cloudflare.com/workers-ai/).

```python
from langchain_cloudflare.chat_models import ChatCloudflareWorkersAI

llm = ChatCloudflareWorkersAI()
llm.invoke("Sing a ballad of LangChain.")
```

### REST endpoint format

By default, `ChatCloudflareWorkersAI` uses the native Workers AI run endpoint:

```python
llm = ChatCloudflareWorkersAI(
    model="@cf/moonshotai/kimi-k2.6",
    endpoint_format="workers_ai",  # default
)
```

For REST calls that need Cloudflare's OpenAI-compatible chat completions API,
set `endpoint_format="openai_compatible"`:

```python
llm = ChatCloudflareWorkersAI(
    model="@cf/moonshotai/kimi-k2.6",
    endpoint_format="openai_compatible",
)
```

When `ai_gateway` is configured, OpenAI-compatible mode routes through the
Workers AI chat completions path on AI Gateway. This option is REST-only; Worker
bindings use `env.AI.run()` and do not expose a chat completions route.

## Embeddings

`CloudflareWorkersAIEmbeddings` class exposes embeddings from [CloudflareWorkersAI](https://developers.cloudflare.com/workers-ai/).

```python
from langchain_cloudflare.embeddings import CloudflareWorkersAIEmbeddings

embeddings = CloudflareWorkersAIEmbeddings(model_name="@cf/baai/bge-base-en-v1.5")
embeddings.embed_query("What is the meaning of life?")
```

## VectorStores
`CloudflareVectorize` class exposes vectorstores from Cloudflare [Vectorize](https://developers.cloudflare.com/vectorize/).

```python
from langchain_cloudflare.vectorstores import CloudflareVectorize

vst = CloudflareVectorize(embedding=embeddings)
vst.create_index(index_name="my-cool-vectorstore")
```

## Retrievers
`CloudflareAISearchRetriever` exposes Cloudflare [AI Search](https://developers.cloudflare.com/ai-search/) (the managed retrieval / RAG service, fka AutoRAG) as a LangChain retriever.

### Prerequisites

- **An AI Search instance with content.** The retriever searches an *existing* instance,
  so create one and add your data first — via the
  [dashboard](https://developers.cloudflare.com/ai-search/),
  [Wrangler](https://developers.cloudflare.com/ai-search/wrangler-commands/), or the
  [Python SDK](https://developers.cloudflare.com/ai-search/get-started/python/).
- **Credentials**, read from the environment:
  - `CF_ACCOUNT_ID`
  - `CF_AI_SEARCH_API_TOKEN` — an `AI Search:Run` token (falls back to `CF_API_TOKEN`)
  - `CF_AI_SEARCH_INSTANCE_NAME` — or pass `instance_name=`

### Usage

```python
from langchain_cloudflare import CloudflareAISearchRetriever

retriever = CloudflareAISearchRetriever(instance_name="my-instance")
docs = retriever.invoke("How do I configure Workers AI?")
```

Inside a Python Worker, pass the dedicated `ai_search` binding instead of REST
credentials (async only):

```python
retriever = CloudflareAISearchRetriever(binding=env.MY_SEARCH)
docs = await retriever.ainvoke("How do I configure Workers AI?")
```

The constructor exposes AI Search's [retrieval options](https://developers.cloudflare.com/ai-search/configuration/retrieval/)
(hybrid search, metadata filters, reranking, query rewriting, …) as parameters, plus an
`ai_search_options` parameter for passing any AI Search option that doesn't have its own
parameter. As a standard `BaseRetriever` it plugs into RAG chains
and becomes an agent tool via `create_retriever_tool`. For multi-tenant
setups, give each tenant its own instance and point a retriever at that instance.

## Browser Run: REST vs. Worker Binding Parity

Browser Run has two distinct APIs: **Quick Actions** (single request/response calls --
markdown extraction, screenshots, structured extraction, etc.) and **full browser
sessions** (stateful, multi-step control via CDP/Puppeteer/Playwright/Stagehand -- click,
type, navigate across pages). This library only implements Quick Actions; full sessions
are JS/npm-only (`@cloudflare/puppeteer`, Playwright) with no Python equivalent, so
they're not reachable from a Python Worker at all, REST or binding.

Every Quick Action is reachable through this library, split across
`CloudflareBrowserRunLoader` (document ingestion) and `CloudflareBrowserRunTool` (agent
actions) — see their sections below for details. `crawl` and `browser="kitesurf"` are
REST-only; every other Quick Action works over both REST and the `binding` parameter,
verified live against the real API and against a real Python Worker:

| Mode | Class | REST | Binding (`quickAction()`) |
|------|-------|:----:|:--------------------------:|
| `markdown` | Loader, Tool | ✅ | ✅ |
| `content` | Loader | ✅ | ✅ |
| `scrape` | Loader | ✅ | ✅ |
| `crawl` | Loader | ✅ | ❌ async job with polling, no `quickAction()` equivalent |
| `json` | Tool | ✅ | ✅ |
| `links` | Tool | ✅ | ✅ |
| `screenshot` | Tool | ✅ | ✅ |
| `pdf` | Tool | ✅ | ✅ |
| `snapshot` | Tool | ✅ | ✅ |
| `accessibility_tree` | Tool | ✅ | ✅ |
| `browser="kitesurf"` | Loader, Tool | ✅ | ❌ URL query param, no binding equivalent |

The binding path is async-only on both classes (`aload()`/`ainvoke()`, not
`load()`/`invoke()`) — calling the sync methods with `binding` set raises
`NotImplementedError`.

## Browser Run (Document Loader)

`CloudflareBrowserRunLoader` loads web pages as LangChain `Document` objects using
[Cloudflare Browser Run](https://developers.cloudflare.com/browser-run/) (formerly
Browser Rendering). It renders JavaScript-heavy pages on Cloudflare's global network and
returns clean content via a REST API or, inside a Python Worker, the `browser` binding.

```python
from langchain_cloudflare import CloudflareBrowserRunLoader

# Single page -> markdown
loader = CloudflareBrowserRunLoader(
    urls=["https://developers.cloudflare.com/workers-ai/"],
    mode="markdown",
)
docs = loader.load()

# Multi-page crawl -> knowledge base (REST-only; async job with polling)
loader = CloudflareBrowserRunLoader(
    urls=["https://developers.cloudflare.com/cloudflare-one/"],
    mode="crawl",
    crawl_limit=50,
    crawl_depth=2,
    crawl_options={"source": "sitemaps"},  # any other /crawl body option
)
docs = loader.load()

# Scrape specific elements with CSS selectors
loader = CloudflareBrowserRunLoader(
    urls=["https://example.com/pricing"],
    mode="scrape",
    elements=[{"selector": "h1"}, {"selector": ".plan-card"}],
)
docs = loader.load()  # one Document per matched selector group

# Async support
docs = await loader.aload()
```

Supported modes:

| Mode | Endpoint | Description |
|------|----------|-------------|
| `markdown` | [`/markdown`](https://developers.cloudflare.com/browser-run/quick-actions/markdown-endpoint/) | Clean markdown from any page |
| `crawl` | [`/crawl`](https://developers.cloudflare.com/browser-run/quick-actions/crawl-endpoint/) | Multi-page crawl with async polling (REST-only) |
| `scrape` | [`/scrape`](https://developers.cloudflare.com/browser-run/quick-actions/scrape-endpoint/) | CSS selector-based element extraction |
| `content` | [`/content`](https://developers.cloudflare.com/browser-run/quick-actions/content-endpoint/) | Raw rendered HTML |

Inside a Python Worker, pass the `browser` binding instead of REST credentials
(async only — use `aload()`/`alazy_load()`, not `load()`/`lazy_load()`):

```python
loader = CloudflareBrowserRunLoader(
    urls=["https://example.com"], mode="markdown", binding=env.BROWSER
)
docs = await loader.aload()
```

Pass `browser="kitesurf"` to use Cloudflare's stateless, agent-optimized browser
runtime instead of full Chromium (REST-only — not reachable via the `quickAction()`
binding, since it's a URL query parameter with no equivalent in the binding's params
object):

```python
loader = CloudflareBrowserRunLoader(
    urls=["https://example.com"], mode="markdown", browser="kitesurf"
)
```

## Browser Run (Agent Tool)

`CloudflareBrowserRunTool` gives [LangGraph](https://langchain-ai.github.io/langgraph/)
agents the ability to interact with the live web.

```python
from langchain_cloudflare import CloudflareBrowserRunTool

# Read any page as markdown
tool = CloudflareBrowserRunTool(mode="markdown")
content = tool.invoke({"url": "https://example.com"})

# AI-powered structured data extraction
tool = CloudflareBrowserRunTool(
    mode="json",
    json_prompt="Extract the company name, pricing plans, and key features.",
)
data = tool.invoke({"url": "https://www.cloudflare.com/plans/"})

# Combined-format snapshot (markdown + screenshot in one call)
tool = CloudflareBrowserRunTool(
    mode="snapshot", snapshot_formats=["markdown", "screenshot"]
)
snapshot = tool.invoke({"url": "https://example.com"})

# Accessibility tree (roles, names, states, hierarchy)
tool = CloudflareBrowserRunTool(mode="accessibility_tree")
tree = tool.invoke({"url": "https://example.com"})

# Use multiple tools in a LangGraph agent
from langgraph.prebuilt import ToolNode

tools = [
    CloudflareBrowserRunTool(mode="markdown"),
    CloudflareBrowserRunTool(mode="json", json_prompt="Extract key facts."),
    CloudflareBrowserRunTool(mode="links"),
]
tool_node = ToolNode(
    tools
)  # each tool auto-named: cloudflare_browser_run_markdown, etc.
```

Supported modes:

| Mode | Endpoint | Description |
|------|----------|-------------|
| `markdown` | [`/markdown`](https://developers.cloudflare.com/browser-run/quick-actions/markdown-endpoint/) | Read any webpage as markdown |
| `json` | [`/json`](https://developers.cloudflare.com/browser-run/quick-actions/json-endpoint/) | AI-powered structured data extraction |
| `links` | [`/links`](https://developers.cloudflare.com/browser-run/quick-actions/links-endpoint/) | Discover all links on a page |
| `screenshot` | [`/screenshot`](https://developers.cloudflare.com/browser-run/quick-actions/screenshot-endpoint/) | Capture screenshot (base64 PNG) |
| `pdf` | [`/pdf`](https://developers.cloudflare.com/browser-run/quick-actions/pdf-endpoint/) | Generate PDF (base64) |
| `snapshot` | [`/snapshot`](https://developers.cloudflare.com/browser-run/quick-actions/snapshot/) | Multiple page formats in one call |
| `accessibility_tree` | [`/accessibilityTree`](https://developers.cloudflare.com/browser-run/quick-actions/accessibility-tree-endpoint/) | Accessibility tree as JSON |

Inside a Python Worker, pass the `browser` binding instead of REST credentials
(async only — use `ainvoke()`, not `invoke()`). This calls Browser Run's
`quickAction()` RPC method instead of the REST API:

```python
tool = CloudflareBrowserRunTool(mode="markdown", binding=env.BROWSER)
result = await tool.ainvoke({"url": "https://example.com"})
```

The `browser` binding requires a `compatibility_date` of `2026-03-24` or later
and, in local development, `"remote": true` (`quickAction()` isn't supported in
local simulation):

```jsonc
// wrangler.jsonc
{
  "compatibility_date": "2026-03-24",
  "browser": { "binding": "BROWSER", "remote": true }
}
```

## Release Notes
v0.1.1 (2025-04-08)

- Added ChatCloudflareWorkersAI integration
- Added CloudflareWorkersAIEmbeddings support
- Added CloudflareVectorize integration

v0.1.3 (2025-04-10)

- Added AI Gateway support for CloudflareWorkersAIEmbeddings
- Added Async support for CloudflareWorkersAIEmbeddings

v0.1.4 (2025-04-14)

- Added support for additional model parameters as explicit class attributes for ChatCloudflareWorkersAI

v0.1.6 (2025-05-01)

- Added Standalone D1 Metadata Filtering Methods
- Update Docs for more clarity around D1 Table/Vectorize Index Names

v0.1.8 (2025-05-11)

- Added support for environmental variables (embeddings, vectorstores)

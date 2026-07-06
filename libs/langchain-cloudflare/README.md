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

- `CF_AI_API_TOKEN` (CloudflareWorkersAI and CloudflareWorkersAIEmbeddings)
- `CF_AI_SEARCH_API_TOKEN` (CloudflareAISearchRetriever)
- `CF_VECTORIZE_API_TOKEN` (CloudflareVectorize)
- `CF_D1_API_TOKEN` (CloudflareVectorize)
- `CF_D1_DATABASE_ID` (CloudflareVectorize)

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

embeddings = CloudflareWorkersAIEmbeddings(
    model_name="@cf/baai/bge-base-en-v1.5"
)
embeddings.embed_query("What is the meaning of life?")
```

## VectorStores
`CloudflareVectorize` class exposes vectorstores from Cloudflare [Vectorize](https://developers.cloudflare.com/vectorize/).

```python
from langchain_cloudflare.vectorstores import CloudflareVectorize

vst = CloudflareVectorize(
    embedding=embeddings
)
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

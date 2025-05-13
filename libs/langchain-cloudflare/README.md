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

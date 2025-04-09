# langchain-cloudflare

This package contains the LangChain integration with CloudflareWorkersAI

## Installation

```bash
pip install -U langchain-cloudflare
```

And you should configure credentials by setting the following environment variables:

- CF_ACCOUNT_ID
- CF_API_TOKEN

## Chat Models

`ChatCloudflareWorkersAI` class exposes chat models from CloudflareWorkersAI.

```python
from langchain_cloudflare.chat_models import ChatCloudflareWorkersAI

llm = ChatCloudflareWorkersAI()
llm.invoke("Sing a ballad of LangChain.")
```

## Embeddings

`CloudflareWorkersAIEmbeddings` class exposes embeddings from CloudflareWorkersAI.

```python
from langchain_cloudflare.embeddings import CloudflareWorkersAIEmbeddings

embeddings = CloudflareWorkersAIEmbeddings()
embeddings.embed_query("What is the meaning of life?")
```

## VectorStores
`CloudflareWorkersAILLM` class exposes LLMs from CloudflareWorkersAI.

```python
from langchain_cloudflare.vectorstores import CloudflareVectorize

vst = CloudflareVectorize()
vst.create_index(index_name="my-cool-vectorstore")
```

## Release Notes
v0.1.1 (2025-04-08)

- Added ChatCloudflareWorkersAI integration
- Added CloudflareWorkersAIEmbeddings support
- Added CloudflareVectorize integration
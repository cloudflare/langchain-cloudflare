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

embeddings = CloudflareWorkersAIEmbeddings(
    account_id="YOUR_ACCOUNT_ID",
    api_token="YOUR_API_TOKEN",
    model_name="@cf/baai/bge-base-en-v1.5"
)
embeddings.embed_query("What is the meaning of life?")
```

## VectorStores
`CloudflareWorkersAILLM` class exposes LLMs from CloudflareWorkersAI.

```python
from langchain_cloudflare.vectorstores import CloudflareVectorize

vst = CloudflareVectorize(
    embedding=embeddings,
    account_id="YOUR_ACCOUNT_ID",
    # api_token="GLOBAL_API_TOKEN",
    d1_api_token="D1_API_TOKEN",  # (Optional if using global token)
    vectorize_api_token="VECTORIZE_API_TOKEN",  # (Optional if using global token)
    d1_database_id="YOUR_D1 DB_ID",  # (Optional if not using D1)
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
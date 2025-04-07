# langchain-cloudflare

This package contains the LangChain integration with CloudflareWorkersAI

## Installation

```bash
pip install -U langchain-cloudflare
```

And you should configure credentials by setting the following environment variables:

* TODO: fill this out

## Chat Models

`ChatCloudflareWorkersAI` class exposes chat models from CloudflareWorkersAI.

```python
from langchain_cloudflare import ChatCloudflareWorkersAI

llm = ChatCloudflareWorkersAI()
llm.invoke("Sing a ballad of LangChain.")
```

## Embeddings

`CloudflareWorkersAIEmbeddings` class exposes embeddings from CloudflareWorkersAI.

```python
from langchain_cloudflare import CloudflareWorkersAIEmbeddings

embeddings = CloudflareWorkersAIEmbeddings()
embeddings.embed_query("What is the meaning of life?")
```

## LLMs
`CloudflareWorkersAILLM` class exposes LLMs from CloudflareWorkersAI.

```python
from langchain_cloudflare import CloudflareWorkersAILLM

llm = CloudflareWorkersAILLM()
llm.invoke("The meaning of life is")
```

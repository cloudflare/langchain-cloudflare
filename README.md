# 🦜️🔗 LangChain Cloudflare

This is a Monorepo containing LangChain packages for Cloudflare.
It includes integrations between WorkersAI, Vectorize, D1, LangChain, and LangGraph.

It contains the following packages.

- `langchain-cloudflare` ([PyPI](https://pypi.org/project/langchain-cloudflare/))
- `langgraph-checkpoint-cloudflare-d1` ([PyPI](https://pypi.org/project/langgraph-checkpoint-cloudflare-d1/))

## Features

### LangChain

#### Components

- [ChatCloudflareWorkersAI](https://python.langchain.com/docs/integrations/chat/cloudflare_workersai/)
- [CloudflareWorkersAIEmbeddings](https://python.langchain.com/docs/integrations/text_embedding/cloudflare_workersai/)
- [CloudflareVectorize](https://python.langchain.com/docs/integrations/vectorstores/cloudflare_vectorize/)

### LangGraph

- Checkpointing (BaseCheckpointSaver)
    - CloudflareD1Saver
    - AsyncCloudflareD1Saver

## Installation

You can install the `langchain-cloudflare` package from PyPI.

```bash
pip install langchain-cloudflare
```

You can install the `langgraph-checkpoint-cloudflare-d1` package from PyPI as well:

```bash
pip install langgraph-checkpoint-cloudflare-d1
```

## Usage

See [langchain-cloudflare usage](libs/langchain-cloudflare/README.md#usage) and [langgraph-checkpoint-cloudflare-d1 usage](libs/langgraph-checkpoint-cloudflare-d1/README.md#usage).

## License

This project is licensed under the [MIT License](LICENSE).

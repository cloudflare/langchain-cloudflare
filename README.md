# 🦜️🔗 LangChain Cloudflare

This is a Monorepo containing LangChain packages for Cloudflare.
It includes integrations between WorkersAI, AI Search, Vectorize, D1, LangChain, and LangGraph.

It contains the following packages.

- `langchain-cloudflare` ([PyPI](https://pypi.org/project/langchain-cloudflare/))
- `langgraph-checkpoint-cloudflare-d1` ([PyPI](https://pypi.org/project/langgraph-checkpoint-cloudflare-d1/))
- `langmem-cloudflare-vectorize` ([PyPI](https://pypi.org/project/langmem-cloudflare-vectorize/))

## Features

### LangChain

#### Components

- [ChatCloudflareWorkersAI](https://python.langchain.com/docs/integrations/chat/cloudflare_workersai/)
- [CloudflareWorkersAIEmbeddings](https://python.langchain.com/docs/integrations/text_embedding/cloudflare_workersai/)
- [CloudflareWorkersAIReranker](https://developers.cloudflare.com/workers-ai/) — document reranking on Workers AI
- [CloudflareVectorize](https://python.langchain.com/docs/integrations/vectorstores/cloudflare_vectorize/)
- [CloudflareAISearchRetriever](libs/langchain-cloudflare/README.md#retrievers) — Cloudflare [AI Search](https://developers.cloudflare.com/ai-search/) (fka AutoRAG)
- [CloudflareBrowserRunLoader](libs/langchain-cloudflare/README.md#browser-run-document-loader) — document loader via [Browser Run](https://developers.cloudflare.com/browser-run/)
- [CloudflareBrowserRunTool](libs/langchain-cloudflare/README.md#browser-run-agent-tool) — agent tool via [Browser Run](https://developers.cloudflare.com/browser-run/)

### LangGraph

- Checkpointing (BaseCheckpointSaver)
    - CloudflareD1Saver
    - AsyncCloudflareD1Saver
- MemoryStore (BaseStore)
    - CloudflareVectorizeBaseStore

## Installation

You can install the `langchain-cloudflare` package from PyPI.

```bash
pip install langchain-cloudflare
```

You can install the `langgraph-checkpoint-cloudflare-d1` package from PyPI as well:

```bash
pip install langgraph-checkpoint-cloudflare-d1
```

And to install the `langmem-cloudflare-vectorize` package from PyPI:

```bash
pip install langmem-cloudflare-vectorize
```

## Usage

See [langchain-cloudflare usage](libs/langchain-cloudflare/README.md#usage) , [langgraph-checkpoint-cloudflare-d1 usage](libs/langgraph-checkpoint-cloudflare-d1/README.md#usage).
and [langmem-cloudflare-vectorize usage](libs/langmem-cloudflare-vectorize/README.md#usage)

## Example notebooks

Example notebooks use a dedicated, locked environment with this checkout of
`langchain-cloudflare` installed. From the repository root:

```bash
make notebook_sync
make notebook_check
make notebook_lab
```

`notebook_lab` loads credentials from the repo-root `.env` before starting
JupyterLab. Open a notebook under [`docs/`](docs/), such as
[`ai_search.ipynb`](docs/ai_search.ipynb) or
[`browser_run.ipynb`](docs/browser_run.ipynb), and run it manually. Stop the
JupyterLab process with `Ctrl-C`.

## License

This project is licensed under the [MIT License](LICENSE).

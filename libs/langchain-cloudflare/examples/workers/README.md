# LangChain Cloudflare Worker Example

This example demonstrates how to use `langchain-cloudflare` with Cloudflare Python Workers, using the Workers AI, Vectorize, and D1 bindings directly for optimal performance.

## Features

- Basic chat completion with Workers AI
- Structured output with Pydantic models
- Tool calling
- Multi-turn conversations
- `create_agent` pattern (requires langchain>=0.3.0)
- Vectorize operations (insert, search, delete)
- D1 database operations

## Prerequisites

1. Cloudflare account with Workers, AI, Vectorize, and D1 access
2. Python 3.12+
3. [uv](https://docs.astral.sh/uv/) package manager
4. [pywrangler](https://pypi.org/project/workers-py/) for local development

## Setup

1. Install dependencies:
   ```bash
   uv sync
   ```

2. Create a Vectorize index (if needed):
   ```bash
   npx wrangler vectorize create langchain-test-persistent --dimensions 768 --metric cosine
   ```

3. Create a D1 database (if needed):
   ```bash
   npx wrangler d1 create test-db
   ```

4. Update `wrangler.jsonc` with your database ID

## Running Locally

```bash
uv run pywrangler dev
```

## Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | API documentation |
| `/chat` | POST | Basic chat completion |
| `/structured` | POST | Structured output extraction |
| `/tools` | POST | Tool calling |
| `/multi-turn` | POST | Multi-turn conversation |
| `/agent-structured` | POST | Agent with structured output |
| `/agent-tools` | POST | Agent with tools |
| `/vectorize-insert` | POST | Insert into Vectorize |
| `/vectorize-search` | POST | Search Vectorize |
| `/vectorize-delete` | POST | Delete from Vectorize |
| `/vectorize-info` | GET | Vectorize index info |
| `/d1-health` | GET | D1 health check |
| `/d1-create-table` | POST | Create D1 table |
| `/d1-insert` | POST | Insert into D1 |
| `/d1-query` | POST | Query D1 |
| `/d1-drop-table` | POST | Drop D1 table |

## Example Usage

```bash
# Chat
curl -X POST http://localhost:8787/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "Hello!"}'

# Structured output
curl -X POST http://localhost:8787/structured \
  -H "Content-Type: application/json" \
  -d '{"text": "Acme Corp announced a partnership with TechGiant."}'

# Tool calling
curl -X POST http://localhost:8787/tools \
  -H "Content-Type: application/json" \
  -d '{"message": "What is the weather in San Francisco?"}'
```

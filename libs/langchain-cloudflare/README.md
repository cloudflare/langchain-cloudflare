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

For chat dynamic routes (`model="dynamic/<route name>"`), message
`response_metadata["model_name"]` identifies the model reported by the provider.
`response_metadata["requested_model"]` retains the requested route. If the
provider does not report a model, `model_name` is `None`. Each message preserves
its model identity when a batch uses different fallback models.

## Extraction Models (Clairic)

`CloudflareWorkersAIExtractionModel` supports the preview
[`@cf/cloudflare/clairic`](https://developers.cloudflare.com/workers-ai/models/clairic/)
model. It returns typed values with source spans, confidence scores and token
usage. Fields accept Cloudflare's native `text`, `number`, `date`, `time`,
`datetime`, `duration`, `list`, `estimate` and `forecast` definitions.

```python
from langchain_cloudflare import CloudflareWorkersAIExtractionModel

model = CloudflareWorkersAIExtractionModel()
fields = {
    "invoice_number": {"type": "text"},
    "total": {"type": "number", "unit": "USD", "decimals": 2},
    "po_number": {"type": "text", "instructions": "Purchase order number"},
}
result = model.extract(state="Invoice INV-2291 totals $1080.", fields=fields)
print(result["answers"]["total"]["value"])

# Async REST accepts the same fields.
result = await model.aextract(state="Invoice INV-2291 totals $1080.", fields=fields)

# Python Workers use the existing AI binding without REST credentials.
model = CloudflareWorkersAIExtractionModel(binding=self.env.AI)
result = await model.aextract(state="Invoice INV-2291 totals $1080.", fields=fields)
```

REST credentials use `CF_ACCOUNT_ID` and `CF_AI_API_TOKEN`; `ai_gateway` selects
an optional AI Gateway. Results preserve the native `model`, `answers`,
`truncated`, optional `state_text`, and `usage` fields. Missing single-value
answers can be `None`; lists can be empty. Typed output and confidence scores
do not guarantee correctness. Text read from the input state includes source
spans. Text read from image pixels can instead have `source: "image"`,
`span: null` and `spell_confidence`; the adapter preserves these native fields.

Pass `reference_date="2026-10-19"` to anchor dates without a year. Native
`options` supports `include_quantiles` and `return_state_text`. Optional `images`
accepts embedded PNG/JPEG/WebP data URLs or `{ "content_type": "image/png",
"base64": "..." }` objects, up to four images; remote URLs are unsupported.
The Worker example exposes this operation at `/extraction`. Image-only input can
use an empty `state`; put the image in `images` and define the desired native
fields. Image fixtures reuse the same labeled PNG and expected ticker (`ACME`),
timeframe (`Q4`) and date (`2026-06-21`) as the OpenAI-compatible structured-vision
test. Tests cover data URLs and base64 objects through sync/async REST and Python
Worker bindings, with no answer text supplied in the state.

In the existing public partnership and earnings fixtures, separate `text`
fields for the announcing company and its partner pass, including source spans,
printed tickers and null answers for absent partners or tickers. For those
fixtures and field definitions, company-list extraction also returns exchange
names, ticker symbols and possessive references, and sentence extraction returns
a company name. These are observed failures, not documented restrictions on
`list` or `text`: the API supports matching lists and source-text spans.

The earnings sentence experiment also asks the model to decide whether an
announcement qualifies before extracting it. That combines classification with
extraction; it is distinct from asking for a missing fact such as a tracking
number. The documented support-email example passes through this adapter,
including its damage list, computed amount and duration, and missing tracking
number. The native field option `null_threshold` controls when a guess becomes
`None`. Setting it to `0.01` makes the earnings-only sentence fixture return
`None`; the default `0.5` did not. This threshold was selected using these two
synthetic inputs, and a lower threshold can also reject valid extractions.
Choose a threshold using representative examples; these fixtures do not establish
general calibration. The change does not fix the full-sentence extraction case.
Confidence scores do not guarantee that an answer is correct.

The integration suites retain the original company-list and sentence expectations
as explicit preview-model quality checks. Only a well-formed but incorrect value
is an expected failure; HTTP errors, invalid spans and malformed responses fail.
These checks fail on an unexpected pass so model improvements trigger revalidation.

### Combining Clairic and Clef

The [combined press-release example](examples/workers/src/system_one.py) produces
the existing nested `Data` structure without a generative chat model:

1. Clairic extracts candidate company mentions from the source.
2. Clef decides eligibility, announcement type, each candidate's role or exclusion,
   and which source line supplies the announcement context.
3. Clairic extracts printed tickers for the selected company names. Python
   assembles and validates `Data → announcements[] → entities[]` with Pydantic.

The function returns `extracted` plus native `stages` containing candidate spans,
probabilities and token usage. Both REST tests and the Worker endpoint
`POST /structured-system-one` invoke this same function. The public partnership
fixture requires exactly Acme Corp and TechGiant Inc, their roles/tickers and the
complete source sentence; the earnings-only fixture requires no announcements.
No expected company names are supplied to the example. Positive results use three
native calls; rejected releases use two. A company omitted from Clairic's
candidates cannot be recovered by Clef's filtering in this example.

This is a focused example for one primary announcement, with each candidate
context sentence on its own input line and at most 61 candidate company mentions.
It does not implement arbitrary Pydantic-schema conversion, multi-event grouping,
or generated summaries. The earlier standalone extraction failures remain
separate checks of Clairic's native answers.

### Pydantic structured output for Clairic and Clef

Both native model classes expose `with_structured_output()` using ordinary
Pydantic v2 `BaseModel` classes. Clairic supplies extracted values; Clef supplies
boolean and categorical decisions. Supply the other model explicitly when a
schema needs both. The wrapper calls the existing native adapters and returns a
LangChain Runnable with `invoke`, `ainvoke`, `batch`, and `abatch`.

```python
from pydantic import BaseModel, Field
from langchain_cloudflare import (
    CloudflareWorkersAIDecisionModel,
    CloudflareWorkersAIExtractionModel,
)


class InvoiceReview(BaseModel):
    invoice_number: str = Field(description="Invoice number")
    total: float = Field(description="Amount due including shipping, in USD")
    paid: bool = Field(
        description="Does the source explicitly say payment is completed?"
    )


clairic = CloudflareWorkersAIExtractionModel()
clef = CloudflareWorkersAIDecisionModel()
structured = clairic.with_structured_output(InvoiceReview, decision_model=clef)
review = structured.invoke("Invoice INV-2291 totals $1080. Payment is pending.")
print(review.model_dump())

# The same schema works from the decision-model side.
structured = clef.with_structured_output(InvoiceReview, extraction_model=clairic)
review = await structured.ainvoke("Invoice INV-2291 totals $1080. Payment is pending.")
```

Extraction-only schemas need only Clairic; boolean/Enum/Literal-only schemas need
only Clef. Field names, paths, and `Field(description=...)` supply instructions.
Supported extraction fields are `str`, `int`, `float`, `date`, `time`, `datetime`,
and `list[str]`, including nullable scalar fields. Enum/Literal choices require
at least two scalar alternatives. Fixed nested models are flattened into native
fields and rebuilt before Pydantic validation; aliases, constraints, and custom
validators apply to the result. Missing native values remain `None`, so an
incompatible field fails validation even if it declares a default.

Non-nullable booleans use Clef's yes/no probability strictly greater than
`boolean_threshold` (default `0.5`). Nullable booleans and nullable choices add an
explicit absence choice. This is a conversion policy, not a guarantee of model
accuracy. No automatic inference retries, output repair, or chat-model fallback
are added.

Pass text directly, or a dictionary containing native inputs such as
`{"state": "", "images": [...]}` for image-only extraction. Extraction fields
also accept `reference_date` and native `options`; audio/video inputs are
supported for decision-only schemas with an appropriate Clef model. Worker
instances use their existing bindings and require `ainvoke` or `abatch`.

`include_raw=True` returns `{"raw": {"extraction": ..., "decision": ...},
"parsed": ..., "parsing_error": ...}` with only the stages used by the schema.
Native spans, probabilities and usage remain in `raw`. Parsing or Pydantic
validation failures return `parsed=None` plus the error; HTTP and transport
failures still raise. Without `include_raw`, validation failures raise normally.

Lists of objects, optional nested objects, dynamic dictionaries, recursive
models, and unions other than a supported type plus `None` are rejected before
inference. Raw JSON Schema dictionaries are not accepted by this wrapper.
The announcement schema's variable lists still use the explicit combined example
above. The `/native-structured` Worker example covers extraction-only,
decision-only, and mixed fixed schemas through the same wrapper.

### Agent-level structured output

Both native classes can also be passed directly to LangChain's `create_agent`.
Configure the companion model in the constructor for mixed schemas, and pass the
ordinary Pydantic class as `response_format`:

```python
from langchain.agents import create_agent

clef = CloudflareWorkersAIDecisionModel()
model = CloudflareWorkersAIExtractionModel(decision_model=clef)
agent = create_agent(model=model, tools=[], response_format=InvoiceReview)
result = agent.invoke(
    {
        "messages": [
            {
                "role": "user",
                "content": "Invoice INV-2291 totals $1080. Payment is pending.",
            }
        ]
    }
)
review = result["structured_response"]  # InvoiceReview instance
```

Decision-only schemas work with a Clef instance directly. A Clef instance with
`extraction_model=clairic` also supports mixed schemas. The agent's `invoke`,
`ainvoke`, `batch`, and `abatch` methods use the same native operations; use async
methods for Worker bindings. Text and embedded image messages are supported.
System messages supply instructions rather than being treated as source facts.

Automatic strategy selection, explicit `ProviderStrategy(MySchema)`, and
`ToolStrategy(MySchema)` with one output schema are supported. The adapter
translates the schema into native requests; LangChain validates the original
Pydantic class and returns it in `structured_response`. Native evidence is kept
in the model's `AIMessage.response_metadata["native"]`, with aggregate token
usage in `usage_metadata`. ToolStrategy's validation feedback stays under
LangChain's retry policy; the native adapters add no inference retry loop.

These models provide structured extraction and decisions. Executable application
tools, selection between multiple output schemas, and unsupported schema shapes
are rejected before inference. Provider-side strict JSON-schema decoding is
unavailable; Pydantic validation runs locally. Install `langchain` to use
`create_agent`; the repository's test and notebook environments include it. The
existing Worker example supports agent mode on `/native-structured`.

## Decision Models (Clef)

`CloudflareWorkersAIDecisionModel` supports
[`@cf/cloudflare/clef`](https://developers.cloudflare.com/workers-ai/models/clef/),
[`@cf/cloudflare/clef-flash`](https://developers.cloudflare.com/workers-ai/models/clef-flash/),
and [`@cf/cloudflare/clef-omni`](https://developers.cloudflare.com/workers-ai/models/clef-omni/).
They evaluate text or JSON state against typed questions and return probabilities.
Question types are `noul` (yes/no probability), `choice` (an option with its
probability distribution), and `score` (a probability-weighted ordered rubric).

```python
from langchain_cloudflare import CloudflareWorkersAIDecisionModel

model = CloudflareWorkersAIDecisionModel(model_name="@cf/cloudflare/clef-flash")
questions = {
    "urgent": {"type": "noul", "instructions": "Is this request urgent?"},
    "team": {
        "type": "choice",
        "instructions": "Which team should handle this request?",
        "criteria": {
            "billing": "Invoices and refunds",
            "technical": "Outages and errors",
        },
    },
    "severity": {
        "type": "score",
        "instructions": "How severe is the impact?",
        "criteria": ["No impact", "Minor", "Major", "Critical"],
    },
}
result = model.evaluate(
    state="Checkout is failing for every customer.", questions=questions
)
print(result["answers"])  # Includes probabilities, confidence, and scores.

# Async REST uses the same input and result shape.
result = await model.aevaluate(
    state={"service": "checkout", "status": "down"}, questions=questions
)

# In a Python Worker, authentication comes from the existing AI binding.
model = CloudflareWorkersAIDecisionModel(binding=self.env.AI)
result = await model.aevaluate(state="Checkout is down.", questions=questions)
```

REST credentials use `CF_ACCOUNT_ID` and `CF_AI_API_TOKEN`. Configure
`ai_gateway` to use an AI Gateway. `reject_if_busy=True` passes live Clef Worker
binding tests; Clef's REST validator currently rejects the documented `options`
field with HTTP 422. Clef and Clef Flash are not currently supported through
dynamic routing. Results preserve `model`, `answers`,
and `usage` fields. Pass optional `images` as embedded PNG/JPEG/WebP data URLs or
`{"content_type": "image/png", "base64": "..."}` objects (up to four images).
The Worker example exposes this operation at `/decision`.

### Clef Omni audio and video

Clef Omni accepts audio and video alongside text, JSON, and images. It returns
the same decision probabilities and usage as the other Clef models.

```python
import base64
from pathlib import Path

from langchain_cloudflare import CloudflareWorkersAIDecisionModel

model = CloudflareWorkersAIDecisionModel(model_name="@cf/cloudflare/clef-omni")
audio = base64.b64encode(Path("recording.wav").read_bytes()).decode("ascii")
result = model.evaluate(
    state="Evaluate the attached recording.",
    questions={
        "speech": {
            "type": "noul",
            "instructions": "Does the recording contain human speech?",
        }
    },
    audio=[f"data:audio/wav;base64,{audio}"],
)
```

Both `evaluate()` and `aevaluate()` accept optional `audio` and `videos` lists.
Each item is a base64 data URL or a `{"content_type": "...", "base64": "..."}`
object. Supported formats are WAV/MP3 audio and MP4/WebM video; remote URLs
are unsupported. Audio and video arguments require Clef Omni. In a Python
Worker, construct the model with `binding=self.env.AI` and use `aevaluate()`.

- Audio: up to four clips, each at most 8 MiB and 300 seconds.
- Video: up to two clips, each at most 16 MiB and 60 seconds, sampled at two
  frames per second. Video soundtracks are used when every supplied video has one.
- Combined decoded audio and video: at most 16 MiB per request.
- Media and questions consume the 64,000-token context window. If they exceed
  it, the request fails; otherwise, text state is truncated to the remaining space.

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

# langgraph-checkpoint-cloudflare-d1

## Installation

```bash
pip install -U langmem-cloudflare-vectorize langgraph langchain-cloudflare
```

## Usage

This package provides both synchronous and asynchronous interfaces for semantic vector search capabilities. Use this
package when you want to use a custom store with langmem.

```python
from langmem_cloudflare_vectorize import CloudflareVectorizeLangmemStore
from langchain_cloudflare.chat_models import ChatCloudflareWorkersAI
from langmem import create_manage_memory_tool, create_search_memory_tool
from langgraph.prebuilt import create_react_agent
from langchain_core.tools import tool
# Cloudflare credentials
account_id = "your-cloudflare-account-id"
vectorize_api_token = "your-vectorize-api-token"
workers_ai_token = "your-workers-ai-api-token"

# Create the langmem vectorize store
agent_store = CloudflareVectorizeLangmemStore.with_cloudflare_embeddings(
    account_id=account_id,
    index_name="cool-vectorize-index-name",
    vectorize_api_token=vectorize_api_token,
    workers_ai_token=workers_ai_token,
    embedding_model="@cf/baai/bge-base-en-v1.5",
    dimensions=768
)
# Create the llm
cloudflare_llm = ChatCloudflareWorkersAI(
    cloudflare_account_id=account_id,
    cloudflare_api_token=workers_ai_token,
    model="@cf/meta/llama-3.3-70b-instruct-fp8-fast"
)
# Create memory tools
manage_memory = create_manage_memory_tool(
    namespace=("memories",)
)
search_memory = create_search_memory_tool(
    namespace=("memories",)
)

@tool
def get_weather(location: str):
    """Get the current weather for a location."""
    if location.lower() in ["sf", "san francisco"]:
        return "It's 60 degrees and foggy in San Francisco."
    elif location.lower() in ["ny", "new york"]:
        return "It's 45 degrees and sunny in New York."
    else:
        return f"It's 75 degrees and partly cloudy in {location}."


#  Create the agent
agent = create_react_agent(
    cloudflare_llm,
    tools=[
        manage_memory,
        search_memory,
    ],
    store=agent_store,  # This is how LangMem gets access to your store
)

config = {"configurable": {"thread_id": "test_session_1"}}

response1 = agent.invoke(
            {"messages": [{"role": "user",
                           "content": "Please remember this important information about me: My name is Sarah, I'm allergic to peanuts, and I love Italian food, especially pasta carbonara. Please use your manage_memory tool to store this."}]},
            config
        )
print(
    "User: Please remember this important information about me: My name is Sarah, I'm allergic to peanuts, and I love Italian food, especially pasta carbonara. Please use your manage_memory tool to store this.")
print(f"Agent: {response1['messages'][-1].content}")

# Test 2: Try to recall the stored information
print("\nCONVERSATION 2: THE MEMORY TEST")
print("-" * 40)
print("Testing if the agent remembers the stored information...")

response2 = agent.invoke(
    {"messages": [{"role": "user",
                   "content": "What do you remember about my dietary restrictions and food preferences? Please search your memory using search_memory tool."}]},
    config
)
print(
    "User: What do you remember about my dietary restrictions and food preferences? Please search your memory using search_memory tool.")
print(f"Agent: {response2['messages'][-1].content}")

# Test 3: Different topic, then back to memory
print("\n📅 CONVERSATION 3: Different topic")
print("-" * 40)

response3 = agent.invoke(
    {"messages": [{"role": "user",
                   "content": "What's the weather like in San Francisco?"}]},
    config
)
print("User: What's the weather like in San Francisco?")
print(f"Agent: {response3['messages'][-1].content}")

response4 = agent.invoke(
    {"messages": [{"role": "user",
                   "content": "What is my name and what am I allergic to?"}]},
    config
)
print(
    "User: What is my name and what am I allergic to?")
print(f"Agent: {response4['messages'][-1].content}")

```
## Release Notes

v0.1.2` (2025-09-25)

- Added support for environmental variables

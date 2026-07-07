"""Integration tests for CloudflareAISearchClient.

These tests create a temporary AI Search instance, upload one fixture document,
query it, and delete the instance during teardown.

Required environment variables:
    - CF_ACCOUNT_ID
    - CF_AI_SEARCH_API_TOKEN (or TEST_CF_API_TOKEN or CF_API_TOKEN)
"""

import os
import uuid

import pytest

from langchain_cloudflare.ai_search import CloudflareAISearchClient

_ACCOUNT_ID = os.environ.get("CF_ACCOUNT_ID") or os.environ.get("CLOUDFLARE_ACCOUNT_ID")
_API_TOKEN = (
    os.environ.get("CF_AI_SEARCH_API_TOKEN")
    or os.environ.get("TEST_CF_API_TOKEN")
    or os.environ.get("CF_API_TOKEN")
    or os.environ.get("CLOUDFLARE_API_TOKEN")
)
_NAMESPACE = os.environ.get("CF_AI_SEARCH_NAMESPACE", "default")
_HAS_CREDS = bool(_ACCOUNT_ID and _API_TOKEN)

pytestmark = pytest.mark.skipif(
    not _HAS_CREDS,
    reason="AI Search credentials not configured",
)


def test_ai_search_client_instance_item_lifecycle() -> None:
    """Create an instance, upload/search an item, and delete the instance."""
    instance_name = f"langchain-cloudflare-client-{uuid.uuid4().hex[:8]}"
    query = f"langchaincloudflareclientfixture {uuid.uuid4().hex}"
    item_id = ""

    client = CloudflareAISearchClient(
        account_id=_ACCOUNT_ID,
        api_token=_API_TOKEN,
        namespace=_NAMESPACE,
    )

    try:
        created = client.create_instance(instance_name)
        assert created["id"] == instance_name

        listed = client.list_instances(search=instance_name)
        assert any(instance.get("id") == instance_name for instance in listed)

        stats = client.stats(instance_name)
        assert isinstance(stats, dict)

        item = client.upload_item(
            f"{instance_name}.md",
            "\n".join(
                [
                    "# LangChain Cloudflare AI Search client fixture",
                    "",
                    f"{query} validates temporary instance lifecycle tests.",
                ]
            ),
            content_type="text/markdown",
            metadata={"suite": "langchain-cloudflare"},
            wait_for_completion=True,
            instance_name=instance_name,
        )
        item_id = item["id"]
        if item.get("status") != "completed":
            item = client.wait_for_item(item_id, instance_name=instance_name)
        assert item["status"] == "completed"

        items = client.list_items(instance_name)
        assert any(uploaded.get("id") == item_id for uploaded in items)

        result = client.search(
            query,
            instance_name=instance_name,
            ai_search_options={
                "retrieval": {
                    "max_num_results": 1,
                },
                "query_rewrite": {"enabled": False},
                "reranking": {"enabled": False},
            },
        )
        chunks = result.get("chunks") or []
        assert chunks
        assert query in chunks[0].get("text", "")
    finally:
        if item_id:
            client.delete_item(
                item_id,
                instance_name=instance_name,
                missing_ok=True,
            )
        client.delete_instance(instance_name, missing_ok=True)

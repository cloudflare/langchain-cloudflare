"""Unit tests for CloudflareAISearchClient (offline: no HTTP is issued)."""

import re
from typing import Any

import pytest
import requests

from langchain_cloudflare._errors import TokenErrors
from langchain_cloudflare.ai_search import CloudflareAISearchClient


class FakeResponse:
    """Small requests/httpx response stub."""

    def __init__(self, data: Any, status_code: int = 200):
        self._data = data
        self.status_code = status_code
        self.content = b"{}" if data is not None else b""

    def json(self) -> Any:
        return self._data

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            raise requests.HTTPError(response=self)


class RequestRecorder:
    """Capture requests.request calls and return a Cloudflare API envelope."""

    def __init__(self, result: Any):
        self.result = result
        self.calls: list[dict[str, Any]] = []

    def __call__(self, method: str, url: str, **kwargs: Any) -> FakeResponse:
        self.calls.append({"method": method, "url": url, **kwargs})
        return FakeResponse({"success": True, "result": self.result})


def _make_client(**overrides: Any) -> CloudflareAISearchClient:
    """Construct a client with valid dummy REST credentials."""
    params = {
        "account_id": "abc123",
        "api_token": "valid-token",
        "instance_name": "test-instance",
    }
    params.update(overrides)
    return CloudflareAISearchClient(**params)


# MARK: - REST Client Tests
class TestRESTClient:
    """Test REST credential handling, URLs, and request bodies."""

    def test_missing_account_id_raises(self) -> None:
        """Missing account_id should raise ValueError."""
        with pytest.raises(ValueError, match="account ID"):
            CloudflareAISearchClient(account_id="", api_token="tok")

    def test_missing_token_raises(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Missing token should raise the AI Search admin-token error."""
        for key in (
            "CF_AI_SEARCH_API_TOKEN",
            "TEST_CF_API_TOKEN",
            "CF_API_TOKEN",
            "CLOUDFLARE_API_TOKEN",
        ):
            monkeypatch.delenv(key, raising=False)

        with pytest.raises(
            ValueError,
            match=re.escape(str(TokenErrors.INSUFFICIENT_AI_SEARCH_ADMIN_TOKENS)),
        ):
            CloudflareAISearchClient(account_id="abc123", api_token="")

    def test_create_instance_default_namespace(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Default namespace uses the namespace-scoped collection endpoint."""
        recorder = RequestRecorder({"id": "docs"})
        monkeypatch.setattr(requests, "request", recorder)

        result = _make_client().create_instance(
            "docs",
            type="web-crawler",
            source="developers.cloudflare.com",
        )

        assert result == {"id": "docs"}
        assert recorder.calls[0]["method"] == "POST"
        assert recorder.calls[0]["url"] == (
            "https://api.cloudflare.com/client/v4/accounts/abc123/"
            "ai-search/namespaces/default/instances"
        )
        assert recorder.calls[0]["json"] == {
            "id": "docs",
            "type": "web-crawler",
            "source": "developers.cloudflare.com",
        }
        assert recorder.calls[0]["headers"] == {"Authorization": "Bearer valid-token"}

    def test_get_instance_namespace_url(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Non-default namespace uses the namespace-scoped endpoint."""
        recorder = RequestRecorder({"id": "docs"})
        monkeypatch.setattr(requests, "request", recorder)

        result = _make_client(namespace="tenant-a").get_instance("docs")

        assert result == {"id": "docs"}
        assert recorder.calls[0]["url"] == (
            "https://api.cloudflare.com/client/v4/accounts/abc123/"
            "ai-search/namespaces/tenant-a/instances/docs"
        )

    def test_list_instances_params(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """List instances sends pagination and search query params."""
        recorder = RequestRecorder([{"id": "docs"}])
        monkeypatch.setattr(requests, "request", recorder)

        result = _make_client().list_instances(page=2, per_page=10, search="docs")

        assert result == [{"id": "docs"}]
        assert recorder.calls[0]["method"] == "GET"
        assert recorder.calls[0]["params"] == {
            "page": 2,
            "per_page": 10,
            "search": "docs",
        }

    def test_upload_item_body(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Item upload uses multipart form fields and leaves Content-Type unset."""
        recorder = RequestRecorder({"id": "item-1", "key": "docs.md"})
        monkeypatch.setattr(requests, "request", recorder)

        result = _make_client().upload_item(
            "docs.md",
            "# Docs",
            content_type="text/markdown",
            metadata={"category": "docs"},
            wait_for_completion=True,
        )

        assert result["id"] == "item-1"
        call = recorder.calls[0]
        assert call["url"].endswith(
            "/ai-search/namespaces/default/instances/test-instance/items"
        )
        assert call["files"]["file"] == ("docs.md", b"# Docs", "text/markdown")
        assert call["data"] == {
            "metadata": '{"category": "docs"}',
            "wait_for_completion": "true",
        }
        assert "Content-Type" not in call["headers"]

    def test_search_body(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Search builds a raw query body with AI Search options."""
        recorder = RequestRecorder({"chunks": []})
        monkeypatch.setattr(requests, "request", recorder)

        _make_client().search(
            "hello",
            ai_search_options={"retrieval": {"max_num_results": 3}},
        )

        assert recorder.calls[0]["json"] == {
            "query": "hello",
            "ai_search_options": {"retrieval": {"max_num_results": 3}},
        }

    def test_search_rejects_query_and_messages(self) -> None:
        """Search should not accept both query shapes at once."""
        with pytest.raises(ValueError, match="either query or messages"):
            _make_client().search("hello", messages=[{"role": "user", "content": "hi"}])


# MARK: - Binding Test Fakes
class FakeItemHandle:
    """Fake item handle returned by items.get()."""

    def __init__(self, item_id: str):
        self.item_id = item_id

    async def info(self) -> dict[str, Any]:
        return {"id": self.item_id, "status": "completed"}


class FakeItems:
    """Fake AI Search items binding."""

    def __init__(self) -> None:
        self.deleted: list[str] = []
        self.uploads: list[tuple[Any, ...]] = []

    async def upload(self, *args: Any) -> dict[str, Any]:
        self.uploads.append(args)
        return {"id": "item-1", "key": args[0]}

    async def uploadAndPoll(self, *args: Any) -> dict[str, Any]:
        self.uploads.append(args)
        return {"id": "item-1", "key": args[0], "status": "completed"}

    async def list(self, params: dict[str, Any]) -> dict[str, Any]:
        return {"result": [{"id": "item-1", "status": params.get("status")}]}

    def get(self, item_id: str) -> FakeItemHandle:
        return FakeItemHandle(item_id)

    async def delete(self, item_id: str) -> None:
        self.deleted.append(item_id)


class FakeInstance:
    """Fake AI Search instance binding."""

    def __init__(self, instance_id: str):
        self.instance_id = instance_id
        self.items = FakeItems()
        self.updated: dict[str, Any] = {}

    async def info(self) -> dict[str, Any]:
        return {"id": self.instance_id, "status": "active"}

    async def stats(self) -> dict[str, Any]:
        return {"completed": 1, "queued": 0}

    async def update(self, payload: dict[str, Any]) -> dict[str, Any]:
        self.updated = payload
        return {"id": self.instance_id, **payload}

    async def search(self, payload: dict[str, Any]) -> dict[str, Any]:
        return {"chunks": [{"text": payload.get("query", "")}]}

    async def chatCompletions(self, payload: dict[str, Any]) -> dict[str, Any]:
        return {"messages": payload["messages"]}


class FakeNamespaceBinding:
    """Fake ai_search_namespaces binding."""

    def __init__(self) -> None:
        self.instances: dict[str, FakeInstance] = {}
        self.deleted: list[str] = []
        self.list_params: dict[str, Any] = {}

    def get(self, instance_name: str) -> FakeInstance:
        return self.instances.setdefault(instance_name, FakeInstance(instance_name))

    async def create(self, payload: dict[str, Any]) -> FakeInstance:
        instance = FakeInstance(payload["id"])
        self.instances[payload["id"]] = instance
        return instance

    async def list(self, params: dict[str, Any]) -> dict[str, Any]:
        self.list_params = params
        return {"result": [{"id": instance_id} for instance_id in self.instances]}

    async def delete(self, instance_name: str) -> None:
        self.deleted.append(instance_name)
        self.instances.pop(instance_name, None)


# MARK: - Binding Tests
class TestBindingClient:
    """Test async Worker binding behavior."""

    async def test_namespace_binding_lifecycle(self) -> None:
        """Namespace binding supports create/list/get/delete."""
        binding = FakeNamespaceBinding()
        client = CloudflareAISearchClient(binding=binding)

        created = await client.acreate_instance("docs", type="r2", source="bucket")
        listed = await client.alist_instances(search="docs")
        info = await client.aget_instance("docs")
        await client.adelete_instance("docs")

        assert created == {"id": "docs", "status": "active"}
        assert listed == [{"id": "docs"}]
        assert binding.list_params == {"search": "docs"}
        assert info == {"id": "docs", "status": "active"}
        assert binding.deleted == ["docs"]

    async def test_binding_item_methods(self) -> None:
        """Binding client can upload, list, get, and delete items."""
        binding = FakeNamespaceBinding()
        client = CloudflareAISearchClient(binding=binding)
        await client.acreate_instance("docs")

        uploaded = await client.aupload_item(
            "docs.md",
            "# Docs",
            instance_name="docs",
            wait_for_completion=True,
            metadata={"category": "docs"},
        )
        items = await client.alist_items("docs", status="completed")
        item = await client.aget_item("item-1", instance_name="docs")
        await client.adelete_item("item-1", instance_name="docs")

        assert uploaded["status"] == "completed"
        assert items == [{"id": "item-1", "status": "completed"}]
        assert item == {"id": "item-1", "status": "completed"}
        assert binding.get("docs").items.deleted == ["item-1"]

    async def test_binding_search_and_chat(self) -> None:
        """Binding client can run query and chat methods through an instance."""
        binding = FakeNamespaceBinding()
        client = CloudflareAISearchClient(binding=binding)
        await client.acreate_instance("docs")

        search = await client.asearch("hello", instance_name="docs")
        chat = await client.achat_completions(
            [{"role": "user", "content": "hello"}],
            instance_name="docs",
        )

        assert search == {"chunks": [{"text": "hello"}]}
        assert chat == {"messages": [{"role": "user", "content": "hello"}]}

    async def test_instance_binding_rejects_namespace_operations(self) -> None:
        """Instance-specific bindings cannot create/list/delete instances."""
        client = CloudflareAISearchClient(
            binding=FakeInstance("docs"),
            instance_name="docs",
        )

        with pytest.raises(NotImplementedError, match="ai_search_namespaces"):
            await client.acreate_instance("other")

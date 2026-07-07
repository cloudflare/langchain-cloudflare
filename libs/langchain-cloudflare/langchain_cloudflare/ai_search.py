"""Cloudflare AI Search administration client."""

# MARK: - Imports
from __future__ import annotations

import inspect
import json
import os
import time
from typing import Any, Dict, List, Optional

import requests
from langchain_core.utils import from_env
from pydantic import SecretStr

from ._errors import TokenErrors

# MARK: - Constants
DEFAULT_BASE_URL = "https://api.cloudflare.com/client/v4"
DEFAULT_NAMESPACE = "default"
DEFAULT_WAIT_SECONDS = 3
DEFAULT_TIMEOUT_SECONDS = 120
FAILED_ITEM_STATUSES = {"error"}
DONE_ITEM_STATUSES = {"completed"}


# MARK: - Helpers
def _drop_none(data: Dict[str, Any]) -> Dict[str, Any]:
    """Return a copy of ``data`` with all ``None``-valued keys removed."""
    return {key: value for key, value in data.items() if value is not None}


def _to_py(value: Any) -> Any:
    """Convert Pyodide JS proxies into plain Python objects when possible."""
    if hasattr(value, "to_py"):
        value = value.to_py()

    if isinstance(value, dict):
        return {key: _to_py(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_to_py(item) for item in value]
    return value


async def _maybe_await(value: Any) -> Any:
    """Await binding return values that are awaitable."""
    if inspect.isawaitable(value):
        return await value
    return value


def _check_api_success(data: Any) -> None:
    """Raise when a Cloudflare API envelope reports ``success: false``."""
    if isinstance(data, dict) and data.get("success") is False:
        errors = data.get("errors") or data
        raise RuntimeError(f"AI Search API request failed: {errors}")


def _extract_result(data: Any) -> Any:
    """Unwrap a Cloudflare API envelope if one is present."""
    data = _to_py(data)
    _check_api_success(data)
    if isinstance(data, dict) and "result" in data:
        return data["result"]
    return data


def _as_dict(data: Any) -> Dict[str, Any]:
    """Return an unwrapped API response as a dict."""
    result = _extract_result(data)
    if result is None:
        return {}
    if isinstance(result, dict):
        return result
    return {"value": result}


def _as_list(data: Any) -> List[Dict[str, Any]]:
    """Return an unwrapped API response as a list of dicts."""
    result = _extract_result(data)
    if isinstance(result, list):
        return [item for item in result if isinstance(item, dict)]
    return []


def _json_or_empty(response: Any) -> Dict[str, Any]:
    """Return a JSON response body, or ``{}`` for an empty response."""
    content = getattr(response, "content", b"")
    if not content:
        return {}
    try:
        data = response.json()
    except ValueError:
        return {}
    return data if isinstance(data, dict) else {"result": data}


# MARK: - CloudflareAISearchClient
class CloudflareAISearchClient:
    """Client for managing Cloudflare AI Search instances and uploaded items.

    Use this class for provisioning and administration: create or delete
    instances, upload built-in-storage items, inspect stats, and run raw search
    requests. For LangChain retrieval chains, use ``CloudflareAISearchRetriever``.

    The REST path uses an API token with ``AI Search:Edit`` and ``AI Search:Run``.
    In Python Workers, pass an ``ai_search_namespaces`` binding for namespace
    administration. Instance-specific ``ai_search`` bindings can call instance
    methods such as ``stats`` and ``items`` but cannot list, create, or delete
    instances.
    """

    # MARK: - Init
    def __init__(
        self,
        account_id: Optional[str] = None,
        api_token: Optional[str] = None,
        instance_name: Optional[str] = None,
        namespace: Optional[str] = None,
        base_url: str = DEFAULT_BASE_URL,
        binding: Any = None,
    ) -> None:
        """Initialize the AI Search client."""
        self.base_url = base_url.rstrip("/")
        self.binding = binding
        self.namespace = (
            namespace
            if namespace is not None
            else from_env("CF_AI_SEARCH_NAMESPACE", default=DEFAULT_NAMESPACE)()
        )
        self.instance_name = (
            instance_name
            if instance_name is not None
            else from_env("CF_AI_SEARCH_INSTANCE_NAME", default="")()
        )

        if account_id is None:
            account_id = from_env("CF_ACCOUNT_ID", default="")()
        self.account_id = account_id

        token = (
            api_token
            or os.environ.get("CF_AI_SEARCH_API_TOKEN")
            or os.environ.get("TEST_CF_API_TOKEN")
            or os.environ.get("CF_API_TOKEN")
            or os.environ.get("CLOUDFLARE_API_TOKEN")
        )
        self.api_token = SecretStr(token) if token else None

        if self.binding is not None:
            self.headers: Dict[str, str] = {}
            return

        if not self.account_id:
            raise ValueError(TokenErrors.NO_ACCOUNT_ID_SET)
        if not self.api_token or not self.api_token.get_secret_value():
            raise ValueError(TokenErrors.INSUFFICIENT_AI_SEARCH_ADMIN_TOKENS)

        self.headers = {
            "Authorization": f"Bearer {self.api_token.get_secret_value()}",
        }

    # MARK: - URL Builders
    @property
    def _ai_search_base_url(self) -> str:
        """Return the account-scoped AI Search base URL."""
        return f"{self.base_url}/accounts/{self.account_id}/ai-search"

    def _resolve_namespace(self, namespace: Optional[str] = None) -> str:
        """Resolve the namespace for an operation."""
        resolved = namespace if namespace is not None else self.namespace
        return resolved or DEFAULT_NAMESPACE

    def _resolve_instance_name(self, instance_name: Optional[str] = None) -> str:
        """Resolve the instance name for an operation."""
        resolved = instance_name or self.instance_name
        if not resolved:
            raise ValueError(TokenErrors.NO_AI_SEARCH_INSTANCE)
        return resolved

    def _instances_url(self, namespace: Optional[str] = None) -> str:
        """Build the REST URL for instance collection operations."""
        resolved_namespace = self._resolve_namespace(namespace)
        return f"{self._ai_search_base_url}/namespaces/{resolved_namespace}/instances"

    def _instance_url(
        self,
        instance_name: Optional[str] = None,
        namespace: Optional[str] = None,
    ) -> str:
        """Build the REST URL for an instance."""
        resolved_instance = self._resolve_instance_name(instance_name)
        return f"{self._instances_url(namespace)}/{resolved_instance}"

    def _items_url(
        self,
        instance_name: Optional[str] = None,
        namespace: Optional[str] = None,
    ) -> str:
        """Build the REST URL for an instance's items."""
        return f"{self._instance_url(instance_name, namespace)}/items"

    # MARK: - Request Helpers
    def _require_rest(self) -> None:
        """Raise if a synchronous REST method is called with a binding client."""
        if self.binding is not None:
            raise NotImplementedError(
                "AI Search bindings are async-only. Use the corresponding "
                "`a...` method when passing a Worker binding."
            )

    def _request_json(
        self,
        method: str,
        url: str,
        request_kwargs: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """Issue a REST request and return the JSON API body."""
        self._require_rest()
        request_kwargs = dict(request_kwargs or {})
        headers = dict(self.headers)
        headers.update(request_kwargs.pop("headers", {}))
        response = requests.request(
            method,
            url,
            headers=headers,
            **request_kwargs,
            **kwargs,
        )
        response.raise_for_status()
        data = _json_or_empty(response)
        _check_api_success(data)
        return data

    async def _arequest_json(
        self,
        method: str,
        url: str,
        request_kwargs: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """Issue an async REST request and return the JSON API body."""
        self._require_rest()
        import httpx

        request_kwargs = dict(request_kwargs or {})
        headers = dict(self.headers)
        headers.update(request_kwargs.pop("headers", {}))
        async with httpx.AsyncClient() as client:
            response = await client.request(
                method,
                url,
                headers=headers,
                **request_kwargs,
                **kwargs,
            )
        response.raise_for_status()
        data = _json_or_empty(response)
        _check_api_success(data)
        return data

    def _binding_instance(self, instance_name: Optional[str] = None) -> Any:
        """Return an AI Search instance handle from a Worker binding."""
        if self.binding is None:
            raise ValueError("An AI Search binding is required")

        resolved = instance_name or self.instance_name
        if hasattr(self.binding, "get"):
            return self.binding.get(self._resolve_instance_name(resolved))

        if resolved and self.instance_name and resolved != self.instance_name:
            raise ValueError(
                "Instance-specific AI Search bindings cannot switch instances. "
                "Pass an ai_search_namespaces binding for multi-instance access."
            )

        return self.binding

    async def _abinding_instance(self, instance_name: Optional[str] = None) -> Any:
        """Asynchronously return an AI Search instance handle from a binding."""
        if self.binding is None:
            raise ValueError("An AI Search binding is required")

        resolved = instance_name or self.instance_name
        if hasattr(self.binding, "get"):
            return await _maybe_await(
                self.binding.get(self._resolve_instance_name(resolved))
            )

        if resolved and self.instance_name and resolved != self.instance_name:
            raise ValueError(
                "Instance-specific AI Search bindings cannot switch instances. "
                "Pass an ai_search_namespaces binding for multi-instance access."
            )

        return self.binding

    def _require_namespace_binding(self, method_name: str) -> Any:
        """Return a namespace-binding method or raise a clear error."""
        method = getattr(self.binding, method_name, None)
        if method is None:
            raise NotImplementedError(
                "This operation requires an ai_search_namespaces binding. "
                "Instance-specific ai_search bindings do not support "
                f"`{method_name}`."
            )
        return method

    def _build_instance_body(
        self,
        instance_name: Optional[str],
        instance_config: Optional[Dict[str, Any]],
        extra_config: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Build an instance create/update payload."""
        body = dict(instance_config or {})
        body.update(extra_config)
        if instance_name is not None:
            body["id"] = instance_name
        if "id" not in body:
            body["id"] = self._resolve_instance_name(None)
        return body

    # MARK: - Instances
    def list_instances(
        self,
        *,
        namespace: Optional[str] = None,
        page: Optional[int] = None,
        per_page: Optional[int] = None,
        search: Optional[str] = None,
        order_by: Optional[str] = None,
        order_by_direction: Optional[str] = None,
        request_kwargs: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        """List AI Search instances."""
        params = _drop_none(
            {
                "page": page,
                "per_page": per_page,
                "search": search,
                "order_by": order_by,
                "order_by_direction": order_by_direction,
            }
        )
        data = self._request_json(
            "GET",
            self._instances_url(namespace),
            request_kwargs=request_kwargs,
            params=params or None,
        )
        return _as_list(data)

    async def alist_instances(
        self,
        *,
        namespace: Optional[str] = None,
        page: Optional[int] = None,
        per_page: Optional[int] = None,
        search: Optional[str] = None,
        order_by: Optional[str] = None,
        order_by_direction: Optional[str] = None,
        request_kwargs: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        """Asynchronously list AI Search instances."""
        if self.binding is not None:
            list_method = self._require_namespace_binding("list")
            params = _drop_none(
                {
                    "page": page,
                    "per_page": per_page,
                    "search": search,
                    "order_by": order_by,
                    "order_by_direction": order_by_direction,
                }
            )
            from .bindings import convert_payload_for_binding

            response = await _maybe_await(
                list_method(convert_payload_for_binding(params))
            )
            return _as_list(response)

        params = _drop_none(
            {
                "page": page,
                "per_page": per_page,
                "search": search,
                "order_by": order_by,
                "order_by_direction": order_by_direction,
            }
        )
        data = await self._arequest_json(
            "GET",
            self._instances_url(namespace),
            request_kwargs=request_kwargs,
            params=params or None,
        )
        return _as_list(data)

    def create_instance(
        self,
        instance_name: Optional[str] = None,
        *,
        namespace: Optional[str] = None,
        instance_config: Optional[Dict[str, Any]] = None,
        request_kwargs: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """Create an AI Search instance."""
        body = self._build_instance_body(instance_name, instance_config, kwargs)
        data = self._request_json(
            "POST",
            self._instances_url(namespace),
            request_kwargs=request_kwargs,
            json=body,
        )
        return _as_dict(data)

    async def acreate_instance(
        self,
        instance_name: Optional[str] = None,
        *,
        namespace: Optional[str] = None,
        instance_config: Optional[Dict[str, Any]] = None,
        request_kwargs: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """Asynchronously create an AI Search instance."""
        body = self._build_instance_body(instance_name, instance_config, kwargs)
        if self.binding is not None:
            create_method = self._require_namespace_binding("create")
            from .bindings import convert_payload_for_binding

            instance = await _maybe_await(
                create_method(convert_payload_for_binding(body))
            )
            if hasattr(instance, "info"):
                return _as_dict(await _maybe_await(instance.info()))
            return _as_dict(instance)

        data = await self._arequest_json(
            "POST",
            self._instances_url(namespace),
            request_kwargs=request_kwargs,
            json=body,
        )
        return _as_dict(data)

    def get_instance(
        self,
        instance_name: Optional[str] = None,
        *,
        namespace: Optional[str] = None,
        request_kwargs: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Read an AI Search instance."""
        data = self._request_json(
            "GET",
            self._instance_url(instance_name, namespace),
            request_kwargs=request_kwargs,
        )
        return _as_dict(data)

    async def aget_instance(
        self,
        instance_name: Optional[str] = None,
        *,
        namespace: Optional[str] = None,
        request_kwargs: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Asynchronously read an AI Search instance."""
        if self.binding is not None:
            instance = await self._abinding_instance(instance_name)
            return _as_dict(await _maybe_await(instance.info()))

        data = await self._arequest_json(
            "GET",
            self._instance_url(instance_name, namespace),
            request_kwargs=request_kwargs,
        )
        return _as_dict(data)

    def update_instance(
        self,
        instance_name: Optional[str] = None,
        *,
        namespace: Optional[str] = None,
        instance_config: Optional[Dict[str, Any]] = None,
        request_kwargs: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """Update an AI Search instance."""
        body = dict(instance_config or {})
        body.update(kwargs)
        data = self._request_json(
            "PUT",
            self._instance_url(instance_name, namespace),
            request_kwargs=request_kwargs,
            json=body,
        )
        return _as_dict(data)

    async def aupdate_instance(
        self,
        instance_name: Optional[str] = None,
        *,
        namespace: Optional[str] = None,
        instance_config: Optional[Dict[str, Any]] = None,
        request_kwargs: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """Asynchronously update an AI Search instance."""
        body = dict(instance_config or {})
        body.update(kwargs)
        if self.binding is not None:
            instance = await self._abinding_instance(instance_name)
            from .bindings import convert_payload_for_binding

            response = await _maybe_await(
                instance.update(convert_payload_for_binding(body))
            )
            return _as_dict(response)

        data = await self._arequest_json(
            "PUT",
            self._instance_url(instance_name, namespace),
            request_kwargs=request_kwargs,
            json=body,
        )
        return _as_dict(data)

    def delete_instance(
        self,
        instance_name: Optional[str] = None,
        *,
        namespace: Optional[str] = None,
        missing_ok: bool = False,
        request_kwargs: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Delete an AI Search instance."""
        try:
            data = self._request_json(
                "DELETE",
                self._instance_url(instance_name, namespace),
                request_kwargs=request_kwargs,
            )
        except requests.HTTPError as exc:
            if (
                missing_ok
                and exc.response is not None
                and exc.response.status_code == 404
            ):
                return {}
            raise
        return _as_dict(data)

    async def adelete_instance(
        self,
        instance_name: Optional[str] = None,
        *,
        namespace: Optional[str] = None,
        missing_ok: bool = False,
        request_kwargs: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Asynchronously delete an AI Search instance."""
        if self.binding is not None:
            delete_method = self._require_namespace_binding("delete")
            try:
                await _maybe_await(
                    delete_method(self._resolve_instance_name(instance_name))
                )
            except Exception:
                if missing_ok:
                    return {}
                raise
            return {}

        import httpx

        try:
            data = await self._arequest_json(
                "DELETE",
                self._instance_url(instance_name, namespace),
                request_kwargs=request_kwargs,
            )
        except httpx.HTTPStatusError as exc:
            if missing_ok and exc.response.status_code == 404:
                return {}
            raise
        return _as_dict(data)

    def stats(
        self,
        instance_name: Optional[str] = None,
        *,
        namespace: Optional[str] = None,
        request_kwargs: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Return AI Search instance indexing stats."""
        data = self._request_json(
            "GET",
            f"{self._instance_url(instance_name, namespace)}/stats",
            request_kwargs=request_kwargs,
        )
        return _as_dict(data)

    async def astats(
        self,
        instance_name: Optional[str] = None,
        *,
        namespace: Optional[str] = None,
        request_kwargs: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Asynchronously return AI Search instance indexing stats."""
        if self.binding is not None:
            instance = await self._abinding_instance(instance_name)
            return _as_dict(await _maybe_await(instance.stats()))

        data = await self._arequest_json(
            "GET",
            f"{self._instance_url(instance_name, namespace)}/stats",
            request_kwargs=request_kwargs,
        )
        return _as_dict(data)

    # MARK: - Items
    def _upload_data(
        self,
        *,
        metadata: Optional[Dict[str, Any]] = None,
        wait_for_completion: Optional[bool] = None,
    ) -> Dict[str, str]:
        """Build multipart form fields for item upload."""
        data: Dict[str, str] = {}
        if metadata is not None:
            data["metadata"] = json.dumps(metadata)
        if wait_for_completion is not None:
            data["wait_for_completion"] = str(wait_for_completion).lower()
        return data

    def upload_item(
        self,
        filename: str,
        content: Any,
        *,
        content_type: str = "text/plain",
        metadata: Optional[Dict[str, Any]] = None,
        wait_for_completion: Optional[bool] = None,
        instance_name: Optional[str] = None,
        namespace: Optional[str] = None,
        request_kwargs: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Upload a file to an AI Search instance's built-in storage."""
        upload_content = (
            content.encode("utf-8") if isinstance(content, str) else content
        )
        file_value = (filename, upload_content, content_type)
        data = self._request_json(
            "POST",
            self._items_url(instance_name, namespace),
            request_kwargs=request_kwargs,
            files={"file": file_value},
            data=self._upload_data(
                metadata=metadata,
                wait_for_completion=wait_for_completion,
            )
            or None,
        )
        return _as_dict(data)

    async def aupload_item(
        self,
        filename: str,
        content: Any,
        *,
        content_type: str = "text/plain",
        metadata: Optional[Dict[str, Any]] = None,
        wait_for_completion: Optional[bool] = None,
        poll_interval_ms: Optional[int] = None,
        timeout_ms: Optional[int] = None,
        instance_name: Optional[str] = None,
        namespace: Optional[str] = None,
        request_kwargs: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Asynchronously upload a file to an AI Search instance."""
        if self.binding is not None:
            instance = await self._abinding_instance(instance_name)
            options = _drop_none(
                {
                    "metadata": metadata,
                    "pollIntervalMs": poll_interval_ms,
                    "timeoutMs": timeout_ms,
                }
            )
            from .bindings import convert_payload_for_binding

            if wait_for_completion:
                upload_method = instance.items.uploadAndPoll
            else:
                upload_method = instance.items.upload

            if options:
                response = await _maybe_await(
                    upload_method(
                        filename, content, convert_payload_for_binding(options)
                    )
                )
            else:
                response = await _maybe_await(upload_method(filename, content))
            return _as_dict(response)

        upload_content = (
            content.encode("utf-8") if isinstance(content, str) else content
        )
        file_value = (filename, upload_content, content_type)
        data = await self._arequest_json(
            "POST",
            self._items_url(instance_name, namespace),
            request_kwargs=request_kwargs,
            files={"file": file_value},
            data=self._upload_data(
                metadata=metadata,
                wait_for_completion=wait_for_completion,
            )
            or None,
        )
        return _as_dict(data)

    def list_items(
        self,
        instance_name: Optional[str] = None,
        *,
        namespace: Optional[str] = None,
        page: Optional[int] = None,
        per_page: Optional[int] = None,
        status: Optional[str] = None,
        sort_by: Optional[str] = None,
        search: Optional[str] = None,
        source: Optional[str] = None,
        request_kwargs: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        """List uploaded items in an AI Search instance."""
        params = _drop_none(
            {
                "page": page,
                "per_page": per_page,
                "status": status,
                "sort_by": sort_by,
                "search": search,
                "source": source,
            }
        )
        data = self._request_json(
            "GET",
            self._items_url(instance_name, namespace),
            request_kwargs=request_kwargs,
            params=params or None,
        )
        return _as_list(data)

    async def alist_items(
        self,
        instance_name: Optional[str] = None,
        *,
        namespace: Optional[str] = None,
        page: Optional[int] = None,
        per_page: Optional[int] = None,
        status: Optional[str] = None,
        sort_by: Optional[str] = None,
        search: Optional[str] = None,
        source: Optional[str] = None,
        request_kwargs: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        """Asynchronously list uploaded items in an AI Search instance."""
        if self.binding is not None:
            instance = await self._abinding_instance(instance_name)
            params = _drop_none(
                {
                    "page": page,
                    "per_page": per_page,
                    "status": status,
                    "sort_by": sort_by,
                    "search": search,
                    "source": source,
                }
            )
            from .bindings import convert_payload_for_binding

            response = await _maybe_await(
                instance.items.list(convert_payload_for_binding(params))
            )
            return _as_list(response)

        params = _drop_none(
            {
                "page": page,
                "per_page": per_page,
                "status": status,
                "sort_by": sort_by,
                "search": search,
                "source": source,
            }
        )
        data = await self._arequest_json(
            "GET",
            self._items_url(instance_name, namespace),
            request_kwargs=request_kwargs,
            params=params or None,
        )
        return _as_list(data)

    def get_item(
        self,
        item_id: str,
        *,
        instance_name: Optional[str] = None,
        namespace: Optional[str] = None,
        request_kwargs: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Read item status and metadata."""
        data = self._request_json(
            "GET",
            f"{self._items_url(instance_name, namespace)}/{item_id}",
            request_kwargs=request_kwargs,
        )
        return _as_dict(data)

    async def aget_item(
        self,
        item_id: str,
        *,
        instance_name: Optional[str] = None,
        namespace: Optional[str] = None,
        request_kwargs: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Asynchronously read item status and metadata."""
        if self.binding is not None:
            instance = await self._abinding_instance(instance_name)
            item = await _maybe_await(instance.items.get(item_id))
            return _as_dict(await _maybe_await(item.info()))

        data = await self._arequest_json(
            "GET",
            f"{self._items_url(instance_name, namespace)}/{item_id}",
            request_kwargs=request_kwargs,
        )
        return _as_dict(data)

    def delete_item(
        self,
        item_id: str,
        *,
        instance_name: Optional[str] = None,
        namespace: Optional[str] = None,
        missing_ok: bool = False,
        request_kwargs: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Delete an uploaded item and its indexed chunks."""
        try:
            data = self._request_json(
                "DELETE",
                f"{self._items_url(instance_name, namespace)}/{item_id}",
                request_kwargs=request_kwargs,
            )
        except requests.HTTPError as exc:
            if (
                missing_ok
                and exc.response is not None
                and exc.response.status_code == 404
            ):
                return {}
            raise
        return _as_dict(data)

    async def adelete_item(
        self,
        item_id: str,
        *,
        instance_name: Optional[str] = None,
        namespace: Optional[str] = None,
        missing_ok: bool = False,
        request_kwargs: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Asynchronously delete an uploaded item and its indexed chunks."""
        if self.binding is not None:
            instance = await self._abinding_instance(instance_name)
            try:
                await _maybe_await(instance.items.delete(item_id))
            except Exception:
                if missing_ok:
                    return {}
                raise
            return {}

        import httpx

        try:
            data = await self._arequest_json(
                "DELETE",
                f"{self._items_url(instance_name, namespace)}/{item_id}",
                request_kwargs=request_kwargs,
            )
        except httpx.HTTPStatusError as exc:
            if missing_ok and exc.response.status_code == 404:
                return {}
            raise
        return _as_dict(data)

    def wait_for_item(
        self,
        item_id: str,
        *,
        instance_name: Optional[str] = None,
        namespace: Optional[str] = None,
        timeout_seconds: int = DEFAULT_TIMEOUT_SECONDS,
        poll_interval_seconds: int = DEFAULT_WAIT_SECONDS,
    ) -> Dict[str, Any]:
        """Wait until an uploaded item has finished indexing."""
        deadline = time.time() + timeout_seconds
        last_item: Dict[str, Any] = {}
        while time.time() < deadline:
            last_item = self.get_item(
                item_id,
                instance_name=instance_name,
                namespace=namespace,
            )
            status = last_item.get("status")
            if status in DONE_ITEM_STATUSES:
                return last_item
            if status in FAILED_ITEM_STATUSES:
                raise RuntimeError(f"AI Search item indexing failed: {last_item}")
            time.sleep(poll_interval_seconds)

        raise TimeoutError(
            f"AI Search item {item_id!r} did not finish indexing within "
            f"{timeout_seconds} seconds. Last item: {last_item}"
        )

    async def await_for_item(
        self,
        item_id: str,
        *,
        instance_name: Optional[str] = None,
        namespace: Optional[str] = None,
        timeout_seconds: int = DEFAULT_TIMEOUT_SECONDS,
        poll_interval_seconds: int = DEFAULT_WAIT_SECONDS,
    ) -> Dict[str, Any]:
        """Asynchronously wait until an uploaded item has finished indexing."""
        import asyncio

        deadline = time.time() + timeout_seconds
        last_item: Dict[str, Any] = {}
        while time.time() < deadline:
            last_item = await self.aget_item(
                item_id,
                instance_name=instance_name,
                namespace=namespace,
            )
            status = last_item.get("status")
            if status in DONE_ITEM_STATUSES:
                return last_item
            if status in FAILED_ITEM_STATUSES:
                raise RuntimeError(f"AI Search item indexing failed: {last_item}")
            await asyncio.sleep(poll_interval_seconds)

        raise TimeoutError(
            f"AI Search item {item_id!r} did not finish indexing within "
            f"{timeout_seconds} seconds. Last item: {last_item}"
        )

    # MARK: - Query
    def _build_search_body(
        self,
        query: Optional[str],
        messages: Optional[List[Dict[str, str]]],
        ai_search_options: Optional[Dict[str, Any]],
        extra_body: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Build an AI Search search request body."""
        if query and messages:
            raise ValueError("Provide either query or messages, not both")

        body = dict(extra_body)
        if query is not None:
            body["query"] = query
        elif messages is not None:
            body["messages"] = messages

        if ai_search_options is not None:
            body["ai_search_options"] = ai_search_options

        return body

    def search(
        self,
        query: Optional[str] = None,
        *,
        messages: Optional[List[Dict[str, str]]] = None,
        ai_search_options: Optional[Dict[str, Any]] = None,
        instance_name: Optional[str] = None,
        namespace: Optional[str] = None,
        request_kwargs: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """Run a raw AI Search query against an instance."""
        data = self._request_json(
            "POST",
            f"{self._instance_url(instance_name, namespace)}/search",
            request_kwargs=request_kwargs,
            json=self._build_search_body(query, messages, ai_search_options, kwargs),
        )
        return _as_dict(data)

    async def asearch(
        self,
        query: Optional[str] = None,
        *,
        messages: Optional[List[Dict[str, str]]] = None,
        ai_search_options: Optional[Dict[str, Any]] = None,
        instance_name: Optional[str] = None,
        namespace: Optional[str] = None,
        request_kwargs: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """Asynchronously run a raw AI Search query against an instance."""
        body = self._build_search_body(query, messages, ai_search_options, kwargs)
        if self.binding is not None:
            instance = await self._abinding_instance(instance_name)
            from .bindings import convert_aisearch_response, convert_payload_for_binding

            response = await _maybe_await(
                instance.search(convert_payload_for_binding(body))
            )
            return _as_dict(convert_aisearch_response(response))

        data = await self._arequest_json(
            "POST",
            f"{self._instance_url(instance_name, namespace)}/search",
            request_kwargs=request_kwargs,
            json=body,
        )
        return _as_dict(data)

    def chat_completions(
        self,
        messages: List[Dict[str, str]],
        *,
        instance_name: Optional[str] = None,
        namespace: Optional[str] = None,
        request_kwargs: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """Run an AI Search chat-completions request."""
        body = dict(kwargs)
        body["messages"] = messages
        data = self._request_json(
            "POST",
            f"{self._instance_url(instance_name, namespace)}/chat/completions",
            request_kwargs=request_kwargs,
            json=body,
        )
        return _as_dict(data)

    async def achat_completions(
        self,
        messages: List[Dict[str, str]],
        *,
        instance_name: Optional[str] = None,
        namespace: Optional[str] = None,
        request_kwargs: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """Asynchronously run an AI Search chat-completions request."""
        body = dict(kwargs)
        body["messages"] = messages
        if self.binding is not None:
            instance = await self._abinding_instance(instance_name)
            from .bindings import convert_payload_for_binding

            response = await _maybe_await(
                instance.chatCompletions(convert_payload_for_binding(body))
            )
            return _as_dict(response)

        data = await self._arequest_json(
            "POST",
            f"{self._instance_url(instance_name, namespace)}/chat/completions",
            request_kwargs=request_kwargs,
            json=body,
        )
        return _as_dict(data)

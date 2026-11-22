"""EmbeddingGemma 2 and legacy transport compatibility."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from langchain_cloudflare.embeddings import CloudflareWorkersAIEmbeddings


@pytest.fixture(params=[False, True], ids=["legacy", "gemma2"])
def case(request):
    gemma = request.param
    vectors = [[1.0, 2.0], [3.0, 4.0]]
    data = (
        [{"index": i, "embedding": v} for i, v in enumerate(vectors)]
        if gemma
        else vectors
    )
    model = "@cf/google/embeddinggemma-2" if gemma else "@cf/baai/bge-base-en-v1.5"
    emb = CloudflareWorkersAIEmbeddings(
        account_id="account",
        api_token="token",
        model_name=model,
        batch_size=1,
        dimensions=128 if gemma else None,
    )
    payloads = [
        {"input": [text], "dimensions": 128, "encoding_format": "float"}
        if gemma
        else {"text": [text]}
        for text in ["first text", "second"]
    ]
    return emb, vectors, data, payloads


def responses(data):
    return [
        MagicMock(json=lambda item=item: {"result": {"data": [item]}}) for item in data
    ]


def test_sync_documents_and_query(case):
    emb, vectors, data, payloads = case
    with patch(
        "langchain_cloudflare.embeddings.requests.post", side_effect=responses(data)
    ) as post:
        assert emb.embed_documents(["first\ntext", "second"]) == vectors
        assert [call.kwargs["json"] for call in post.call_args_list] == payloads
    with patch(
        "langchain_cloudflare.embeddings.requests.post", return_value=responses(data)[0]
    ) as post:
        assert emb.embed_query("first\ntext") == vectors[0]
        assert post.call_args.kwargs["json"] == payloads[0]


async def test_async_rest_documents_and_query(case):
    emb, vectors, data, payloads = case
    client = AsyncMock()
    client.post.side_effect = responses(data) + [responses(data)[0]]
    with patch("httpx.AsyncClient") as factory:
        factory.return_value.__aenter__.return_value = client
        assert await emb.aembed_documents(["first\ntext", "second"]) == vectors
        assert await emb.aembed_query("first\ntext") == vectors[0]
    assert [call.kwargs["json"] for call in client.post.call_args_list] == payloads + [
        payloads[0]
    ]


async def test_binding_documents_query_options_and_proxy(case):
    emb, vectors, data, payloads = case
    binding = AsyncMock()
    proxy = MagicMock()
    proxy.to_py.return_value = {"data": [data[0]]}
    binding.run.side_effect = [proxy, {"data": [data[1]]}, {"data": [data[0]]}]
    emb.binding = binding
    emb.reject_if_busy = True
    emb.ai_gateway = "gateway"
    with patch(
        "langchain_cloudflare.bindings.convert_payload_for_binding",
        side_effect=lambda x: x,
    ):
        assert await emb.aembed_documents(["first\ntext", "second"]) == vectors
        assert await emb.aembed_query("first\ntext") == vectors[0]
    for call, payload in zip(binding.run.call_args_list, payloads + [payloads[0]]):
        assert call.args == (
            emb.model_name,
            payload,
            {"gateway": {"id": "gateway"}, "rejectIfBusy": True},
        )


@pytest.mark.parametrize("binding", [None, AsyncMock()])
async def test_empty_documents_do_not_request(binding):
    emb = CloudflareWorkersAIEmbeddings(account_id="a", api_token="t", binding=binding)
    with patch("langchain_cloudflare.embeddings.requests.post") as post:
        assert emb.embed_documents([]) == []
        assert await emb.aembed_documents([]) == []
        post.assert_not_called()
    if binding is not None:
        binding.run.assert_not_called()


def test_gemma_defaults_and_prefixes_are_preserved():
    emb = CloudflareWorkersAIEmbeddings(
        binding=object(), model_name="@cf/google/embeddinggemma-2"
    )
    texts = ["task: search result | query: shoes", "title: none | text: shoes"]
    assert emb._embed_payload(texts) == {"input": texts, "encoding_format": "float"}
    emb.reject_if_busy = True
    assert emb._embed_payload(texts)["options"] == {"rejectIfBusy": True}


@pytest.mark.parametrize("dimensions", [0, -1])
def test_invalid_dimensions(dimensions):
    with pytest.raises(ValueError):
        CloudflareWorkersAIEmbeddings(binding=object(), dimensions=dimensions)


async def test_legacy_binding_list_without_options():
    binding = AsyncMock()
    binding.run.return_value = [[1.0, 2.0]]
    emb = CloudflareWorkersAIEmbeddings(binding=binding)
    with patch(
        "langchain_cloudflare.bindings.convert_payload_for_binding",
        side_effect=lambda x: x,
    ):
        assert await emb.aembed_query("query") == [1.0, 2.0]
    binding.run.assert_awaited_once_with(emb.model_name, {"text": ["query"]})


def test_rest_error_is_propagated(case):
    import requests

    emb, _, _, _ = case
    response = MagicMock()
    response.raise_for_status.side_effect = requests.HTTPError("capacity exceeded")
    with patch("langchain_cloudflare.embeddings.requests.post", return_value=response):
        with pytest.raises(requests.HTTPError, match="capacity exceeded"):
            emb.embed_query("query")


async def test_binding_error_is_propagated(case):
    emb, _, _, _ = case
    emb.binding = AsyncMock()
    emb.binding.run.side_effect = RuntimeError("capacity exceeded")
    with patch(
        "langchain_cloudflare.bindings.convert_payload_for_binding",
        side_effect=lambda x: x,
    ):
        with pytest.raises(RuntimeError, match="capacity exceeded"):
            await emb.aembed_query("query")


def test_gemma_multi_item_batch_preserves_order():
    emb = CloudflareWorkersAIEmbeddings(
        account_id="a", api_token="t", model_name="@cf/google/embeddinggemma-2"
    )
    data = [
        {"index": 0, "embedding": [1.0]},
        {"index": 1, "embedding": [2.0]},
    ]
    response = MagicMock()
    response.json.return_value = {"result": {"data": data}}
    with patch(
        "langchain_cloudflare.embeddings.requests.post", return_value=response
    ) as post:
        assert emb.embed_documents(["one", "two"]) == [[1.0], [2.0]]
        assert post.call_args.kwargs["json"]["input"] == ["one", "two"]


def test_legacy_dimensions_do_not_change_payload():
    emb = CloudflareWorkersAIEmbeddings(binding=object(), dimensions=256)
    assert emb._model_payload(["query"]) == {"text": ["query"]}

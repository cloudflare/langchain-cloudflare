"""Unit tests for langgraph_checkpoint_cloudflare_d1.utils."""

from langgraph_checkpoint_cloudflare_d1.utils import _metadata_predicate, search_where


def test_metadata_predicate_bool_true_encodes_as_one() -> None:
    # bool is a subclass of int in Python, so the int/str/float branch would
    # silently swallow a bool value unless it is checked first. A raw Python
    # `True`/`False` passed through as the SQL parameter would not match
    # SQLite's json_extract(), which returns integer 1/0 for JSON booleans.
    predicates, values = _metadata_predicate({"completed": True})
    assert values == [1]
    assert predicates == ["json_extract(CAST(metadata AS TEXT), '$.completed') = ?"]


def test_metadata_predicate_bool_false_encodes_as_zero() -> None:
    predicates, values = _metadata_predicate({"completed": False})
    assert values == [0]


def test_metadata_predicate_int_still_passes_through() -> None:
    predicates, values = _metadata_predicate({"step": 3})
    assert values == [3]


def test_search_where_applies_metadata_filter() -> None:
    where, params = search_where(None, {"source": "loop"})
    assert "json_extract" in where
    assert params == ["loop"]


def test_search_where_applies_before() -> None:
    config = {"configurable": {"thread_id": "t1"}}
    before = {"configurable": {"thread_id": "t1", "checkpoint_id": "c5"}}
    where, params = search_where(config, None, before)
    assert "checkpoint_id < ?" in where
    assert "c5" in params

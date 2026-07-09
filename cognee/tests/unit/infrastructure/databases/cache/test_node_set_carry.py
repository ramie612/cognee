"""Maniera patch coverage: node_set carried through the QA session cache.

Verifies the additive node_set field on SessionQAEntry round-trips through the
JSONB payload, and that the SQL adapter's dump/merge helpers propagate it and
never drop it on a partial update (the graph->session sync-back clobber lock).
"""

import pytest
from pydantic import ValidationError

from cognee.infrastructure.databases.cache.models import SessionQAEntry
from cognee.infrastructure.databases.cache.sql.SqlCacheAdapter import SqlCacheAdapter


def test_session_qa_entry_round_trips_node_set():
    entry = SessionQAEntry(
        time="2026-07-09T00:00:00", question="q", context="c", answer="a",
        node_set=["status", "client"],
    )
    dump = entry.model_dump()
    assert dump["node_set"] == ["status", "client"]
    restored = SessionQAEntry.model_validate(dump)
    assert restored.node_set == ["status", "client"]


def test_session_qa_entry_node_set_defaults_none():
    entry = SessionQAEntry(time="t", question="q", context="c", answer="a")
    assert entry.node_set is None
    assert entry.model_dump()["node_set"] is None


def test_old_payload_without_node_set_key_deserializes():
    """A pre-patch payload (no node_set key) validates with node_set=None."""
    old = {"time": "t", "question": "q", "context": "c", "answer": "a", "qa_id": "x"}
    assert SessionQAEntry.model_validate(old).node_set is None


def test_session_qa_entry_empty_node_set_normalizes_to_none():
    entry = SessionQAEntry(time="t", question="q", context="c", answer="a", node_set=[])
    assert entry.node_set is None


def test_session_qa_entry_node_set_rejects_non_list():
    with pytest.raises(ValidationError):
        SessionQAEntry(time="t", question="q", context="c", answer="a", node_set="status")


def test_session_qa_entry_node_set_rejects_non_str_items():
    with pytest.raises(ValidationError):
        SessionQAEntry(time="t", question="q", context="c", answer="a", node_set=[1, 2])


def test_build_qa_entry_dump_includes_node_set():
    dump = SqlCacheAdapter._build_qa_entry_dump("q", "c", "a", node_set=["topic"])
    assert dump["node_set"] == ["topic"]


def test_merge_entry_update_preserves_existing_node_set():
    """Clobber lock: a partial update that doesn't pass node_set keeps the stored one."""
    existing = {
        "time": "t", "question": "q", "context": "c", "answer": "a",
        "qa_id": "x", "node_set": ["keep-me"],
    }
    merged = SqlCacheAdapter._merge_entry_update(existing, feedback_score=5)
    assert merged["node_set"] == ["keep-me"]
    assert merged["feedback_score"] == 5


def test_merge_entry_update_sets_node_set_when_passed():
    existing = {"time": "t", "question": "q", "context": "c", "answer": "a", "qa_id": "x"}
    merged = SqlCacheAdapter._merge_entry_update(existing, node_set=["new"])
    assert merged["node_set"] == ["new"]


def test_qa_entry_dto_accepts_node_set():
    from cognee.memory.entries import QAEntry

    entry = QAEntry(question="q", answer="a", node_set=["client"])
    assert entry.node_set == ["client"]
    assert QAEntry(question="q", answer="a").node_set is None

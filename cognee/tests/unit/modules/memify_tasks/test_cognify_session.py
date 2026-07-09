import pytest
from unittest.mock import AsyncMock, call, patch

from cognee.tasks.memify.cognify_session import cognify_session
from cognee.exceptions import CogneeValidationError, CogneeSystemError


def _bundle(session_id, notes):
    return {"session_id": session_id, "notes": notes}


@pytest.mark.asyncio
async def test_cognify_session_bundle_per_note_add_single_cognify():
    """A per-session bundle is add()'d once per note (own node_set) then cognified once."""
    data = [
        _bundle(
            "s1",
            [
                {"text": "Question: q1\n\nAnswer: a1", "node_set": ["status", "user_sessions_from_cache"]},
                {"text": "Question: q2\n\nAnswer: a2", "node_set": ["decision", "user_sessions_from_cache"]},
            ],
        )
    ]
    with (
        patch("cognee.add", new_callable=AsyncMock) as mock_add,
        patch("cognee.cognify", new_callable=AsyncMock) as mock_cognify,
    ):
        await cognify_session(data, dataset_id="123", user="u1")

        assert mock_add.call_count == 2
        mock_add.assert_has_calls(
            [
                call("Question: q1\n\nAnswer: a1", dataset_id="123",
                     node_set=["status", "user_sessions_from_cache"], user="u1"),
                call("Question: q2\n\nAnswer: a2", dataset_id="123",
                     node_set=["decision", "user_sessions_from_cache"], user="u1"),
            ]
        )
        # exactly one cognify for the whole batch
        mock_cognify.assert_called_once_with(datasets=["123"], user="u1")


@pytest.mark.asyncio
async def test_cognify_session_single_bundle_not_wrapped_in_list():
    """A bare bundle dict (not list-wrapped) is also accepted."""
    data = _bundle("s1", [{"text": "note", "node_set": ["client"]}])
    with (
        patch("cognee.add", new_callable=AsyncMock) as mock_add,
        patch("cognee.cognify", new_callable=AsyncMock) as mock_cognify,
    ):
        await cognify_session(data, dataset_id="123")

        mock_add.assert_called_once_with("note", dataset_id="123", node_set=["client"], user=None)
        mock_cognify.assert_called_once_with(datasets=["123"], user=None)


@pytest.mark.asyncio
async def test_cognify_session_note_without_node_set_falls_back_to_default():
    """A note missing node_set gets the default session set."""
    data = [_bundle("s1", [{"text": "note"}])]
    with (
        patch("cognee.add", new_callable=AsyncMock) as mock_add,
        patch("cognee.cognify", new_callable=AsyncMock),
    ):
        await cognify_session(data, dataset_id="123")
        mock_add.assert_called_once_with(
            "note", dataset_id="123", node_set=["user_sessions_from_cache"], user=None
        )


@pytest.mark.asyncio
async def test_cognify_session_string_back_compat():
    """A raw string (pre-tagging shape) adds under the default set, then one cognify."""
    session_data = "Session ID: test\n\nQuestion: What is AI?\n\nAnswer: AI is artificial intelligence"
    with (
        patch("cognee.add", new_callable=AsyncMock) as mock_add,
        patch("cognee.cognify", new_callable=AsyncMock) as mock_cognify,
    ):
        await cognify_session(session_data, dataset_id="123")
        mock_add.assert_called_once_with(
            session_data, dataset_id="123", node_set=["user_sessions_from_cache"], user=None
        )
        mock_cognify.assert_called_once_with(datasets=["123"], user=None)


@pytest.mark.asyncio
async def test_cognify_session_list_of_strings_back_compat():
    """A list of raw strings (the batched pre-tagging shape) adds each under the default set."""
    data = ["one", "two"]
    with (
        patch("cognee.add", new_callable=AsyncMock) as mock_add,
        patch("cognee.cognify", new_callable=AsyncMock) as mock_cognify,
    ):
        await cognify_session(data, dataset_id="123")
        assert mock_add.call_count == 2
        mock_cognify.assert_called_once()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "data",
    [
        "",
        "   \n\t  ",
        None,
        [],
        _bundle("s1", []),
        _bundle("s1", [{"text": "   "}]),
        _bundle("s1", [{"text": ""}, {"node_set": ["x"]}]),
        [_bundle("s1", [])],
    ],
)
async def test_cognify_session_empty_variants_raise(data):
    """No usable notes (empty / whitespace-only / tagless-empty) raises the validation error."""
    with (
        patch("cognee.add", new_callable=AsyncMock) as mock_add,
        patch("cognee.cognify", new_callable=AsyncMock) as mock_cognify,
    ):
        with pytest.raises(CogneeValidationError) as exc_info:
            await cognify_session(data)
        assert "Session data cannot be empty" in str(exc_info.value)
        mock_add.assert_not_called()
        mock_cognify.assert_not_called()


@pytest.mark.asyncio
async def test_cognify_session_add_failure():
    """A cognee.add failure surfaces as CogneeSystemError."""
    data = [_bundle("s1", [{"text": "note", "node_set": ["x"]}])]
    with (
        patch("cognee.add", new_callable=AsyncMock) as mock_add,
        patch("cognee.cognify", new_callable=AsyncMock),
    ):
        mock_add.side_effect = Exception("Add operation failed")
        with pytest.raises(CogneeSystemError) as exc_info:
            await cognify_session(data)
        assert "Failed to cognify session data" in str(exc_info.value)
        assert "Add operation failed" in str(exc_info.value)


@pytest.mark.asyncio
async def test_cognify_session_cognify_failure():
    """A cognee.cognify failure surfaces as CogneeSystemError."""
    data = [_bundle("s1", [{"text": "note", "node_set": ["x"]}])]
    with (
        patch("cognee.add", new_callable=AsyncMock),
        patch("cognee.cognify", new_callable=AsyncMock) as mock_cognify,
    ):
        mock_cognify.side_effect = Exception("Cognify operation failed")
        with pytest.raises(CogneeSystemError) as exc_info:
            await cognify_session(data)
        assert "Failed to cognify session data" in str(exc_info.value)
        assert "Cognify operation failed" in str(exc_info.value)


@pytest.mark.asyncio
async def test_cognify_session_re_raises_validation_error():
    """CogneeValidationError is re-raised as-is (not wrapped)."""
    with pytest.raises(CogneeValidationError):
        await cognify_session("")


@pytest.mark.asyncio
async def test_cognify_session_with_special_characters():
    """Unicode / special characters in note text pass through unchanged."""
    text = "Question: What's special?™ © \n\nAnswer: Cognee is special!"
    data = [_bundle("s1", [{"text": text, "node_set": ["insights"]}])]
    with (
        patch("cognee.add", new_callable=AsyncMock) as mock_add,
        patch("cognee.cognify", new_callable=AsyncMock) as mock_cognify,
    ):
        await cognify_session(data, dataset_id="123")
        mock_add.assert_called_once_with(text, dataset_id="123", node_set=["insights"], user=None)
        mock_cognify.assert_called_once_with(datasets=["123"], user=None)

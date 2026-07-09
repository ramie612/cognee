import sys
import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from cognee.infrastructure.databases.cache.models import SessionQAEntry
from cognee.tasks.memify.extract_user_sessions import extract_user_sessions
from cognee.exceptions import CogneeSystemError
from cognee.modules.users.models import User

# Get the actual module object (not the function) for patching
extract_user_sessions_module = sys.modules["cognee.tasks.memify.extract_user_sessions"]


@pytest.fixture
def mock_user():
    """Create a mock user."""
    user = MagicMock(spec=User)
    user.id = "test-user-123"
    return user


@pytest.fixture
def mock_qa_data():
    """Create mock Q&A data as SessionQAEntry objects (same as SessionManager.get_session(formatted=False))."""
    return [
        SessionQAEntry(
            question="What is cognee?",
            context="context about cognee",
            answer="Cognee is a knowledge graph solution",
            time="2025-01-01T12:00:00",
        ),
        SessionQAEntry(
            question="How does it work?",
            context="how it works context",
            answer="It processes data and creates graphs",
            time="2025-01-01T12:05:00",
        ),
    ]


def _make_mock_session_manager(entries_or_empty, is_available: bool = True):
    """Create a mock SessionManager with get_session and is_available."""
    mock_sm = MagicMock()
    mock_sm.is_available = is_available
    mock_sm.get_session = AsyncMock(return_value=entries_or_empty)
    return mock_sm


@pytest.mark.asyncio
async def test_extract_user_sessions_success(mock_user, mock_qa_data):
    """Test successful extraction of sessions via SessionManager."""
    mock_session_manager = _make_mock_session_manager(mock_qa_data)

    with (
        patch.object(extract_user_sessions_module, "session_user") as mock_session_user,
        patch.object(
            extract_user_sessions_module,
            "get_session_manager",
            return_value=mock_session_manager,
        ),
    ):
        mock_session_user.get.return_value = mock_user

        sessions = []
        async for session in extract_user_sessions([{}], session_ids=["test_session"]):
            sessions.append(session)

        assert len(sessions) == 1
        bundle = sessions[0]
        assert bundle["session_id"] == "test_session"
        notes = bundle["notes"]
        assert len(notes) == 2
        assert notes[0]["text"] == (
            "Question: What is cognee?\n\nAnswer: Cognee is a knowledge graph solution"
        )
        assert notes[1]["text"] == (
            "Question: How does it work?\n\nAnswer: It processes data and creates graphs"
        )
        # untagged entries fall back to the default session node set
        assert notes[0]["node_set"] == ["user_sessions_from_cache"]
        assert notes[1]["node_set"] == ["user_sessions_from_cache"]
        mock_session_manager.get_session.assert_called_once_with(
            user_id="test-user-123",
            session_id="test_session",
            formatted=False,
        )


@pytest.mark.asyncio
async def test_extract_user_sessions_preserves_per_note_node_set(mock_user):
    """Each entry's node_set rides onto its note, unioned with the default set and sorted."""
    tagged = [
        SessionQAEntry(
            question="q1", context="", answer="a1", time="2025-01-01T12:00:00",
            node_set=["status", "client"],
        ),
        SessionQAEntry(
            question="q2", context="", answer="a2", time="2025-01-01T12:05:00",
            node_set=["decision"],
        ),
    ]
    mock_session_manager = _make_mock_session_manager(tagged)

    with (
        patch.object(extract_user_sessions_module, "session_user") as mock_session_user,
        patch.object(
            extract_user_sessions_module,
            "get_session_manager",
            return_value=mock_session_manager,
        ),
    ):
        mock_session_user.get.return_value = mock_user

        sessions = []
        async for session in extract_user_sessions([{}], session_ids=["s"]):
            sessions.append(session)

    notes = sessions[0]["notes"]
    # sorted() over the union with the default set
    assert notes[0]["node_set"] == ["client", "status", "user_sessions_from_cache"]
    assert notes[1]["node_set"] == ["decision", "user_sessions_from_cache"]


@pytest.mark.asyncio
async def test_extract_user_sessions_multiple_sessions(mock_user, mock_qa_data):
    """Test extraction of multiple sessions."""
    mock_session_manager = _make_mock_session_manager(mock_qa_data)

    with (
        patch.object(extract_user_sessions_module, "session_user") as mock_session_user,
        patch.object(
            extract_user_sessions_module,
            "get_session_manager",
            return_value=mock_session_manager,
        ),
    ):
        mock_session_user.get.return_value = mock_user

        sessions = []
        async for session in extract_user_sessions([{}], session_ids=["session1", "session2"]):
            sessions.append(session)

        assert len(sessions) == 2
        assert mock_session_manager.get_session.call_count == 2


@pytest.mark.asyncio
async def test_extract_user_sessions_no_data(mock_user, mock_qa_data):
    """Test extraction handles empty data parameter."""
    mock_session_manager = _make_mock_session_manager(mock_qa_data)

    with (
        patch.object(extract_user_sessions_module, "session_user") as mock_session_user,
        patch.object(
            extract_user_sessions_module,
            "get_session_manager",
            return_value=mock_session_manager,
        ),
    ):
        mock_session_user.get.return_value = mock_user

        sessions = []
        async for session in extract_user_sessions(None, session_ids=["test_session"]):
            sessions.append(session)

        assert len(sessions) == 1


@pytest.mark.asyncio
async def test_extract_user_sessions_no_session_ids(mock_user):
    """Test extraction handles no session IDs provided."""
    mock_session_manager = _make_mock_session_manager([])

    with (
        patch.object(extract_user_sessions_module, "session_user") as mock_session_user,
        patch.object(
            extract_user_sessions_module,
            "get_session_manager",
            return_value=mock_session_manager,
        ),
    ):
        mock_session_user.get.return_value = mock_user

        sessions = []
        async for session in extract_user_sessions([{}], session_ids=None):
            sessions.append(session)

        assert len(sessions) == 0
        mock_session_manager.get_session.assert_not_called()


@pytest.mark.asyncio
async def test_extract_user_sessions_empty_qa_data(mock_user):
    """Test extraction handles empty Q&A data (SessionManager returns empty list)."""
    mock_session_manager = _make_mock_session_manager([])

    with (
        patch.object(extract_user_sessions_module, "session_user") as mock_session_user,
        patch.object(
            extract_user_sessions_module,
            "get_session_manager",
            return_value=mock_session_manager,
        ),
    ):
        mock_session_user.get.return_value = mock_user

        sessions = []
        async for session in extract_user_sessions([{}], session_ids=["empty_session"]):
            sessions.append(session)

        assert len(sessions) == 0


@pytest.mark.asyncio
async def test_extract_user_sessions_session_manager_unavailable(mock_user):
    """Test extraction raises when SessionManager is not available."""
    mock_session_manager = _make_mock_session_manager("", is_available=False)

    with (
        patch.object(extract_user_sessions_module, "session_user") as mock_session_user,
        patch.object(
            extract_user_sessions_module,
            "get_session_manager",
            return_value=mock_session_manager,
        ),
    ):
        mock_session_user.get.return_value = mock_user

        with pytest.raises(CogneeSystemError) as exc_info:
            async for _ in extract_user_sessions([{}], session_ids=["test_session"]):
                pass

        assert "SessionManager not available" in str(exc_info.value)


@pytest.mark.asyncio
async def test_extract_user_sessions_session_manager_error_handling(mock_user, mock_qa_data):
    """Test extraction continues on SessionManager error for specific session."""
    mock_session_manager = _make_mock_session_manager(mock_qa_data)
    mock_session_manager.get_session.side_effect = [
        mock_qa_data,
        Exception("SessionManager error"),
        mock_qa_data,
    ]

    with (
        patch.object(extract_user_sessions_module, "session_user") as mock_session_user,
        patch.object(
            extract_user_sessions_module,
            "get_session_manager",
            return_value=mock_session_manager,
        ),
    ):
        mock_session_user.get.return_value = mock_user

        sessions = []
        async for session in extract_user_sessions(
            [{}], session_ids=["session1", "session2", "session3"]
        ):
            sessions.append(session)

        assert len(sessions) == 2

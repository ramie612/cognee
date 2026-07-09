from typing import Optional, List

from cognee.context_global_variables import session_user
from cognee.exceptions import CogneeSystemError
from cognee.infrastructure.session.get_session_manager import get_session_manager
from cognee.shared.logging_utils import get_logger
from cognee.modules.users.models import User

logger = get_logger("extract_user_sessions")


async def extract_user_sessions(
    data,
    session_ids: Optional[List[str]] = None,
):
    """
    Extract Q&A sessions for the current user via SessionManager.

    Retrieves all Q&A triplets from specified session IDs and yields them
    as formatted strings combining question, context, and answer. Session
    persistence relies on SessionManager; caching must be enabled for
    sessions to be available.

    Args:
        data: Data passed from memify. If empty dict ({}), no external data is provided.
        session_ids: Optional list of specific session IDs to extract.

    Yields:
        One dict bundle per session:
            {"session_id": str,
             "notes": [{"text": "Question: ...\n\nAnswer: ...",
                        "node_set": [<tag>, ..., "user_sessions_from_cache"]}, ...]}
        Each note carries its own NodeSet tags (the entry's node_set unioned with the
        default "user_sessions_from_cache" set) so cognify_session can materialize
        per-note graph tags in a single cognify batch.

    Raises:
        CogneeSystemError: If SessionManager is unavailable or extraction fails.
    """
    try:
        if not data or data == [{}]:
            logger.info("Fetching session metadata for current user")

        user: User = session_user.get()
        if not user:
            raise CogneeSystemError(message="No authenticated user found in context", log=False)

        user_id = str(user.id)

        session_manager = get_session_manager()
        if not session_manager.is_available:
            raise CogneeSystemError(
                message="SessionManager not available for session extraction, please enable caching in order to have sessions to save",
                log=False,
            )

        if session_ids:
            for session_id in session_ids:
                try:
                    qa_data = await session_manager.get_session(
                        user_id=user_id,
                        session_id=session_id,
                        formatted=False,
                    )
                    if qa_data:
                        logger.info(
                            f"Extracted session {session_id} via SessionManager with {len(qa_data)} Q&A pairs"
                        )
                        # Yield one bundle per session, preserving each note's own
                        # NodeSet tags (unioned with the default session set) so the
                        # per-note tags survive into the graph instead of being lost
                        # in a single concatenated, single-tagged document.
                        yield {
                            "session_id": session_id,
                            "notes": [
                                {
                                    "text": f"Question: {qa_pair.question}\n\nAnswer: {qa_pair.answer}",
                                    "node_set": sorted(
                                        {*(qa_pair.node_set or []), "user_sessions_from_cache"}
                                    ),
                                }
                                for qa_pair in qa_data
                            ],
                        }
                except Exception as e:
                    logger.warning(f"Failed to extract session {session_id}: {str(e)}")
                    continue
        else:
            logger.info(
                "No specific session_ids provided. Please specify which sessions to extract."
            )

    except CogneeSystemError:
        raise
    except Exception as e:
        logger.error(f"Error extracting user sessions: {str(e)}")
        raise CogneeSystemError(message=f"Failed to extract user sessions: {str(e)}", log=False)

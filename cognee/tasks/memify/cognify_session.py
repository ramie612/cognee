from typing import Optional
from uuid import UUID

import cognee

from cognee.exceptions import CogneeValidationError, CogneeSystemError
from cognee.shared.logging_utils import get_logger

logger = get_logger("cognify_session")

_DEFAULT_SESSION_NODE_SET = "user_sessions_from_cache"


def _iter_notes(data):
    """Yield (text, node_set) pairs from the extract_user_sessions output.

    The memify runner batches an extraction task's yields into a list, so `data`
    normally arrives as a list of per-session bundles
    ({"session_id", "notes": [{"text", "node_set"}, ...]}). For resilience this
    also accepts a single bundle, a raw string, or a list of raw strings (the
    pre-per-note-tagging shape); untagged notes fall back to the default session
    node set so their graph membership is unchanged.
    """
    items = data if isinstance(data, (list, tuple)) else [data]
    for item in items:
        if isinstance(item, dict) and "notes" in item:
            for note in item.get("notes") or []:
                text = (note or {}).get("text")
                if not text or not str(text).strip():
                    continue
                node_set = list((note or {}).get("node_set") or [_DEFAULT_SESSION_NODE_SET])
                yield str(text), node_set
        elif isinstance(item, str):
            if item.strip():
                yield item, [_DEFAULT_SESSION_NODE_SET]
        # anything else (None / unexpected shape) is skipped


async def cognify_session(data, dataset_id: Optional[UUID | str] = None, user=None) -> None:
    """
    Process and cognify session data into the knowledge graph.

    Adds each session note to cognee under its OWN NodeSet tags (falling back to
    the "user_sessions_from_cache" set for untagged notes), then triggers a single
    cognify run that extracts entities and relationships for every note added in
    this batch. N adds (no LLM) + 1 cognify (one LLM batch) preserves the original
    single-cognify-per-session cost while giving each note per-note graph tags.

    Args:
        data: A per-session bundle (or a list of bundles) produced by
            extract_user_sessions: {"session_id", "notes": [{"text", "node_set"}]}.
            A raw string / list of strings is also accepted for back-compat.
        dataset_id: Target dataset id.
        user: Authenticated user, threaded through for EBAC write access.

    Raises:
        CogneeValidationError: If no non-empty notes are provided.
        CogneeSystemError: If cognee operations fail.
    """
    try:
        notes = list(_iter_notes(data))
        if not notes:
            logger.warning("Empty session data provided to cognify_session task, skipping")
            raise CogneeValidationError(message="Session data cannot be empty", log=False)

        logger.info("Processing %d session note(s) for cognification", len(notes))

        for text, node_set in notes:
            await cognee.add(text, dataset_id=dataset_id, node_set=node_set, user=user)
        logger.debug("Added %d session note(s) to cognee with per-note node sets", len(notes))
        await cognee.cognify(datasets=[dataset_id], user=user)
        logger.info("Session data successfully cognified")

    except CogneeValidationError:
        raise
    except Exception as e:
        logger.error(f"Error cognifying session data: {str(e)}")
        raise CogneeSystemError(message=f"Failed to cognify session data: {str(e)}", log=False)

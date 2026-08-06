# Maniera cognee patches

This is Maniera's fork of [topoteretes/cognee](https://github.com/topoteretes/cognee).
The `maniera` branch carries a small set of patches on top of an upstream release
tag. It is deployed as an **overlay** onto the stock `cognee/cognee:<version>`
Docker image (the fork's pure-Python `cognee/` package is copied over the image's
installed package — same dependencies, patched source).

**Base ref:** `v1.2.2` (the exact tag the `cognee/cognee:1.2.2` image is built from;
verified by diffing the tag tree against the running container — the only delta is
the ebac commit below).

## Patch commits (on top of `v1.2.2`)

| Commit | Kind | Purpose | Files |
|---|---|---|---|
| `fix(ebac): thread user= through session/trace memify persistence` | bugfix | Under EBAC, `improve()`'s session/trace persistence dropped the authenticated user at two depths, so verbatim promotion of session Q&A + agent traces 403'd non-fatally and was silently lost. Threads `user=` through all six call sites (mirrors the working `apply_feedback_weights` pipeline). Formerly a build-time string-rewrite (`patch_improve_ebac.py`), now a proper commit. | `memify_pipelines/persist_sessions_in_knowledge_graph.py`, `memify_pipelines/persist_agent_trace_feedbacks_in_knowledge_graph.py`, `tasks/memify/cognify_session.py`, `tasks/memify/cognify_agent_trace_feedback.py` |
| `feat(session-cache): carry node_set through the QA session cache` | feature | Adds an optional `node_set` (list of NodeSet tag names) end-to-end through the fast session-cache write path so per-note graph tags survive the `remember() -> improve()` promotion. The tag rides the existing JSONB `payload` column (`SessionQAEntry.model_dump()`) — **no schema migration**; old rows deserialize with `node_set=None`. Fully additive (every new param defaults to `None`). | `infrastructure/databases/cache/models.py`, `memory/entries.py`, `infrastructure/databases/cache/cache_db_interface.py`, `.../cache/sql/SqlCacheAdapter.py`, `.../cache/fscache/FsCacheAdapter.py`, `.../cache/redis/RedisAdapter.py`, `.../cache/tapes/TapesCacheAdapter.py`, `infrastructure/session/session_manager.py`, `api/v1/remember/remember.py` |
| `feat(memify): materialize per-note NodeSet tags on session promotion` | feature | `extract_user_sessions` yields one per-session bundle (`{session_id, notes:[{text, node_set}]}`); `cognify_session` `add()`s each note under its own `node_set` then runs ONE `cognify()`. N adds (no LLM) + 1 cognify (one LLM batch) preserves the original single-cognify-per-session cost while giving each note per-note NodeSets. Composes with the ebac `user=` threading. | `tasks/memify/extract_user_sessions.py`, `tasks/memify/cognify_session.py` |
| `fix(improve): keep the caller's dataset reference on the session bridge` | bugfix | Sibling of the ebac bug above, same silent-loss shape. `improve()`'s `_bridge_sessions` / `_persist_session_traces` resolved the caller's dataset **UUID down to a NAME**, then handed the name to the three memify pipelines. Those re-resolve through `get_authorized_existing_datasets()`, which maps a NAME only against datasets the user OWNS (`get_dataset_ids` -> `get_datasets(user.id)`), so the caller's write GRANT was discarded: promoting session Q&A into a dataset you do not own failed the write check and the content was dropped. Non-fatal at every stage, so the API still answered `200 PipelineRunCompleted`. Passes the original reference through instead, and widens `dataset` to `Union[str, UUID]` on the three pipelines. Required for one shared team dataset that several principals write to. | `api/v1/improve/improve.py`, `memify_pipelines/apply_feedback_weights.py`, `memify_pipelines/persist_sessions_in_knowledge_graph.py`, `memify_pipelines/persist_agent_trace_feedbacks_in_knowledge_graph.py` |
| `fix(forget): enter the per-dataset database under its owner, not the caller` | bugfix | `forget()` opened the per-dataset database context with `set_database_global_context_variables(dataset_ref, user.id)`. That database is registered under the dataset's OWNER, so a caller holding only a `delete` grant tried to insert a SECOND `dataset_database` row for the same dataset and hit the primary key: `UniqueViolationError: duplicate key value violates unique constraint "dataset_database_pkey"`, surfaced as an opaque HTTP 500 "An error occurred during deletion". Deleting anything from a SHARED dataset was therefore impossible for everyone but its owner. Resolves the owner id (under the same caller-side delete-permission check) and enters the context with it; the delete helpers still authorize the caller, so no access widens. | `api/v1/forget/forget.py` |
| `test(maniera): align + extend coverage for the ebac and node_set patches` | tests | Aligns upstream unit tests to the patched call signatures (`user=` on cognify tasks; `node_set=` on the session-manager update assert; the memify bundle contract) and adds `tests/unit/infrastructure/databases/cache/test_node_set_carry.py`. | see the test commit |

## Build (overlay)

`/opt/maniera-cognee-v2/maniera-image/Dockerfile` on the VPS:

```dockerfile
FROM cognee/cognee:1.2.2
COPY cognee/ /app/cognee/            # a checkout of ramie612/cognee@maniera in the build context
RUN find /app/cognee -name __pycache__ -type d -prune -exec rm -rf {} + \
 && python -c "import cognee"        # fail the build loud if the overlay is broken
```

The build context is a `git clone`/worktree of `ramie612/cognee` on branch `maniera`.
Because the fork is based on the same release the base image ships, the overlay is
dependency-compatible (only Python source differs).

## Upgrade runbook (rebase onto a new cognee release)

```
git fetch upstream
git rebase v<new>          # conflicts surface ONLY on our ~12 touched files
# resolve, then:
python -m pytest tests/unit/modules/memify_tasks/test_cognify_session.py \
                 tests/unit/modules/memify_tasks/test_extract_user_sessions.py \
                 tests/unit/modules/memify_tasks/test_cognify_agent_trace_feedback.py \
                 tests/unit/infrastructure/session/test_session_manager.py \
                 tests/unit/infrastructure/databases/cache/
git push --force-with-lease origin maniera
```

Then bump the `FROM cognee/cognee:<new>` tag in the Dockerfile, rebuild, and do a
staged recreate of `maniera-cognee-v2` (back up + tag a rollback image first).

**If upstream fixes the ebac `user=` bug** on the new tag, drop the ebac commit during
the rebase (the rebase will show it as empty / already-applied).

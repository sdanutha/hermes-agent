"""Structured storage failure projection for session observers."""
import logging
import time
from fastapi import HTTPException
log = logging.getLogger(__name__)
_corrupt_store_warned_at = {}

_CORRUPT_STORE_WARN_INTERVAL_S = 300.0

CORRUPT_STORE_DETAIL = {
    "error": "state_db_corrupt",
    "message": "state.db corrupt — run `hermes doctor` (then `hermes doctor --fix` or `hermes sessions repair`).",
}

DELETED_WAL_DETAIL = {
    "error": "state_db_deleted_wal",
    "message": "another Hermes process still holds an old copy of the session database's write-ahead log — "
               "quit every Hermes process on this profile, run `hermes doctor` (it names the holders), "
               "then start Hermes again. Do not run `hermes doctor --fix` while they run.",
}

STATE_DB_REPLACED_DETAIL = {
    "error": "state_db_replaced",
    "message": "state.db was replaced while Hermes was running — stop Hermes, run `hermes doctor`, "
               "then start it again. Do not run `hermes doctor --fix`, which would repair the wrong file in place.",
}

_STORE_STATUS_DETAIL_BY_CAUSE = {"deleted_wal": DELETED_WAL_DETAIL, "replaced": STATE_DB_REPLACED_DETAIL}

def corrupt_store_status(db_path, exc) -> HTTPException:
    """The structured ``state_db_corrupt`` 503 for a malformed store, warning once per store per
    :data:`_CORRUPT_STORE_WARN_INTERVAL_S` (debug afterwards). Callers ``raise ... from exc``."""
    from hermes_state_errors import classify_persistence_error
    key, now = str(db_path), time.monotonic()
    last = _corrupt_store_warned_at.get(key)
    detail = _STORE_STATUS_DETAIL_BY_CAUSE.get(classify_persistence_error(exc), CORRUPT_STORE_DETAIL)
    if last is None or now - last >= _CORRUPT_STORE_WARN_INTERVAL_S:
        _corrupt_store_warned_at[key] = now
        log.warning("state.db at %s is unreadable (%s); dashboard reads return a status payload until it is "
                    "repaired — run `hermes doctor`", db_path, exc)
    else:
        log.debug("state.db at %s still has error: %s", db_path, exc)
    return HTTPException(status_code=503, detail={**detail, "path": key})

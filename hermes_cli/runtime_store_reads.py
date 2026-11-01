"""Owner-safe session store reads; never initialize a database from an observer."""
from pathlib import Path
import logging
_log = logging.getLogger(__name__)

def _session_db_read_probe_statements() -> tuple:
    """Probe the declared schema without reconciling it on a browsing request."""
    from hermes_state_schema import schema_read_probe_statements

    return schema_read_probe_statements()

def _open_session_db_at_path(db_path: Path, *, read_only: bool):
    """Browsing never initializes or repairs the owner's session store.

    Owner startup reconciles schema. The one write a browse may do: a store that lags
    SCHEMA_SQL (missing table/column, not corruption) with NO gateway owning it gets the
    missing declared tables/columns once, in one short locked transaction (#140315); an
    owned store is refused by name instead. Explicit mutation APIs retain their writable
    acquisition until their separate owner-RPC migration.
    """
    import sqlite3

    from fastapi import HTTPException
    from hermes_state import SessionDB
    from hermes_state_errors import is_transient_sqlite_error
    from hermes_state_registry import acquire

    if not read_only:
        return acquire(db_path)

    try:
        initialized = db_path.stat().st_size > 0
    except FileNotFoundError:
        initialized = False
    if not initialized:
        raise HTTPException(
            status_code=503,
            detail="Session store is not initialized. Start this profile's gateway to initialize it.")

    from hermes_state_reconcile import (
        DeclarativeDriftError, is_declarative_drift_error, reconcile_drift_for_browse,
    )

    def _open_probed():
        db = SessionDB(db_path=db_path, read_only=True)
        try:
            conn = getattr(db, "_conn", None)
            if conn is not None:
                for statement in _session_db_read_probe_statements():
                    conn.execute(statement).fetchone()
            return db
        except BaseException:
            db.close()
            raise

    try:
        try:
            return _open_probed()
        except sqlite3.OperationalError as exc:
            # A missing declared table/column is upgrade drift, not corruption (#140315): catch
            # an unowned store up once under its runtime lock, refuse by name when a gateway
            # owns it. Never destructive: additive DDL only, no repair from browse.
            if not is_declarative_drift_error(exc):
                raise
            try:
                reconcile_drift_for_browse(db_path)
            except DeclarativeDriftError as drift:
                _log.warning("session store at %s is behind SCHEMA_SQL: %s", db_path, drift)
                raise HTTPException(status_code=503, detail=str(drift)) from exc
            return _open_probed()
    except (sqlite3.DatabaseError, UnicodeDecodeError) as exc:
        from hermes_state_errors import is_malformed_db_error
        if isinstance(exc, sqlite3.DatabaseError) and is_malformed_db_error(exc):
            # Same structured 503 the analytics routers publish (``corrupt_store_as_status``): a
            # corrupt image names the repair (`hermes doctor`), never a generic "schema unavailable".
            from hermes_cli.runtime_store_errors import corrupt_store_status
            raise corrupt_store_status(db_path, exc) from exc
        if isinstance(exc, sqlite3.OperationalError) and is_transient_sqlite_error(exc):
            detail = "Session store is busy (disk I/O or lock). Retry; the list was not cleared."
        else:
            detail = (
                "Session store schema is unavailable or corrupt. Start or restart this profile's "
                "gateway to reconcile it; if it persists, run `hermes doctor` for diagnosis. "
                "Browsing does not repair the store.")
        raise HTTPException(status_code=503, detail=detail) from exc

"""File reads for session history with credential-file denial."""
from __future__ import annotations
import os
import stat
from pathlib import Path
from typing import Optional
import urllib.parse
import urllib.request
from fastapi import HTTPException

def _resolve_fs_candidate(raw: str, *, cwd: str | None = None) -> Path:
    candidate = Path(raw).expanduser()
    if not candidate.is_absolute():
        base = Path(cwd).expanduser() if cwd is not None else Path.cwd()
        if not base.is_absolute():
            raise HTTPException(status_code=400, detail="Session working directory is unavailable")
        candidate = base / candidate
    return candidate.resolve(strict=False)

def _fs_path(raw_path: str, *, cwd: str | None = None, decode_fallback: bool = True) -> Path:
    raw = str(raw_path or "").strip()
    if not raw:
        raise HTTPException(status_code=400, detail="Path is required")
    if "\0" in raw:
        raise HTTPException(status_code=400, detail="Invalid path")
    try:
        if raw.lower().startswith("file:"):
            parsed = urllib.parse.urlparse(raw)
            uri_path = parsed.path
            if parsed.netloc and parsed.netloc.lower() != "localhost":
                if os.name != "nt":
                    raise ValueError
                uri_path = f"//{parsed.netloc}{uri_path}"
            raw = urllib.request.url2pathname(uri_path)
        elif os.name == "nt":
            # MEDIA links can reuse Git Bash paths; native Path would read /c/ as C:\c\.
            from tools.environments.local import _msys_to_windows_path

            raw = _msys_to_windows_path(raw)
        candidate = _resolve_fs_candidate(raw, cwd=cwd)
        # A remote client hop may percent-encode a path on top of HTTP's own
        # decoding, so a non-ASCII name can arrive as a literal "%E5%8D%8A..."
        # string that stats as missing (issue #103425). The verbatim path wins
        # whenever it exists, so filenames that genuinely contain "%XX" keep
        # resolving as-is; the unquoted form only rescues the lookup.
        if decode_fallback and "%" in raw and not candidate.exists():
            decoded = _resolve_fs_candidate(urllib.parse.unquote(raw), cwd=cwd)
            if decoded.exists():
                candidate = decoded
        return candidate
    except (OSError, RuntimeError, ValueError):
        raise HTTPException(status_code=400, detail="Invalid path")


_SENSITIVE_MANAGED_FILE_BASENAMES = frozenset({
    "auth.json", "auth.lock", "credentials", "config.yaml", ".anthropic_oauth.json",
    "google_token.json", "google_oauth_pending.json", "google_oauth.json",
    "webhook_subscriptions.json", "bws_cache.json", "bws_cache.enc.json",
    ".git-credentials",  # git's credential-store cache (file_safety blocks it too)
})

_SENSITIVE_MANAGED_DIR_NAMES = frozenset({"mcp-tokens", "pairing"})

def _is_sensitive_filename(name: str) -> bool:
    """Basename denylist: ``.env`` / ``.env.<suffix>`` / ``.envrc`` plus the
    credential-store basenames. Case-insensitive so ``.ENV`` / ``Auth.JSON``
    on case-insensitive mounts can't slip past. Basename-only — call sites use
    :func:`_is_sensitive_path`, which adds the credential-directory check."""
    lowered = name.lower()
    if lowered == ".env" or lowered.startswith(".env.") or lowered == ".envrc":
        return True
    return lowered in _SENSITIVE_MANAGED_FILE_BASENAMES

def _is_sensitive_path(path: Path) -> bool:
    """True when the basename is sensitive OR any path component (case-
    insensitive) is a credential directory. Read-side guard (list/read/
    download); the write endpoints are a separate threat class.

    Read-side only: this guards list/read/download (the #57505 exfil surface). The write endpoints
    (upload/mkdir/delete) are a separate threat class handled by the write-path checks; extending this guard
    to them is out of scope for this fix.
    """
    if _is_sensitive_filename(path.name):
        return True
    return any(part.lower() in _SENSITIVE_MANAGED_DIR_NAMES for part in path.parts)

def _fs_regular_file(path: Path, profile_home: Optional[Path] = None) -> tuple[Path, os.stat_result]:
    target = _fs_path(str(path))
    try:
        st = target.stat()
    except (FileNotFoundError, NotADirectoryError):
        raise HTTPException(status_code=404, detail="File not found")
    except PermissionError:
        raise HTTPException(status_code=403, detail="File is not readable")
    except OSError as exc:
        raise HTTPException(status_code=400, detail=str(exc) or "Invalid path")
    if stat.S_ISDIR(st.st_mode):
        raise HTTPException(status_code=400, detail="Path points to a directory")
    if not stat.S_ISREG(st.st_mode):
        raise HTTPException(status_code=400, detail="Only regular files can be read")
    if _is_sensitive_path(target):
        raise HTTPException(status_code=403, detail="Access to sensitive files is not allowed")
    return target, st

_FS_DATA_URL_MAX_BYTES = 16 * 1024 * 1024

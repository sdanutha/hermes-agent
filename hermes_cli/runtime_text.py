"""Text helpers for headless management commands."""
from agent.i18n import t


def _cprint(*args, **kwargs):
    print(*args, **kwargs)


def _summarize_paths(paths, limit=5):
    more = _t("shared.more_suffix", count=len(paths)-limit) if len(paths)>limit else ""
    return ", ".join(paths[:limit])+more


def _pr(*lines: str) -> None:
    """print() each line."""
    for line in lines:
        print(line)


def _t(key: str, **kwargs) -> str:
    """Catalog text for this module: ``cli.commands.<key>`` in the active language."""
    return t(f"cli.commands.{key}", **kwargs)


def _tn(key: str, count: int, **kwargs) -> str:
    """Plural-aware ``_t``: ``<key>_one`` when ``count == 1``, else ``<key>_other``."""
    return _t(f"{key}_{'one' if count == 1 else 'other'}", count=count, **kwargs)


def _gt(key: str, **kwargs) -> str:
    """A ``gateway.<key>`` catalog entry the CLI shares verbatim with the messaging gateway."""
    return t(f"gateway.{key}", **kwargs)


def _lines(text: str, pad: str = "  ") -> list:
    """Each line of a (possibly multi-line) catalog value prefixed with ``pad`` (blank lines kept)."""
    return [f"{pad}{line}" if line else "" for line in text.splitlines()]


def _rollback_kept(result: dict) -> None:
    """The kept / not-removed path lines of a restore result (in-process CLI and gateway client alike)."""
    if skipped := result.get("skipped_user_edits"):
        _pr(*_lines(_gt("rollback.kept_user_edits", files=_summarize_paths(skipped))))
    if oversize := result.get("skipped_oversize"):
        print(f"  {_gt('rollback.kept_oversize', files=_summarize_paths(oversize))}")
    if failed := result.get("failed_deletes"):
        print(f"  {_gt('rollback.failed_deletes', files=_summarize_paths(failed))}")


def print_session_diff(cli, result: dict, stat_only: bool) -> None:
    """Render a ``CheckpointManager.session_diff`` result (``/diff session``): this process's agent's,
    or the gateway session's ``rollback.session_diff`` (``diff_lines``: its uncut body's line count)."""
    if not result.get("success"):
        return print(f"  {result.get('error') or _t('shared.could_not_generate_diff')}")
    stat, diff = result.get("stat", ""), result.get("diff", "")
    if result.get("empty") or (not stat and not diff):
        return print(f"  {_t('diff.no_session_changes')}")
    if stat:
        cli._print_diff_text(f"\n{stat}")
    if (diff or result.get("diff_lines")) and not stat_only:  # a body cut to nothing still names its lines
        cli._print_diff_body(diff, _t("diff.stat_hint_session"), total=int(result.get("diff_lines") or 0))


class _SeededQueryMessage:
    __slots__ = ("admission_id", "images", "text")

    def __init__(self, text, images=None, admission_id=None):
        self.text = text or ""
        self.images = list(images or [])
        self.admission_id = admission_id

    def __str__(self):
        return self.text

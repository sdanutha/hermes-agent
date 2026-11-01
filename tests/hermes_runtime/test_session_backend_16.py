

import json






import subprocess


import sys


import threading


import time


import types


from datetime import datetime


from pathlib import Path


from unittest.mock import patch


import pytest


from hermes_constants import reset_hermes_home_override, set_hermes_home_override








from hermes_runtime import server


from hermes_runtime.transport import bind_transport, reset_transport


def _dispatch_sync(req: dict, transport=None) -> dict | None:
    """Run one RPC to completion synchronously, regardless of pool routing.

    Voice RPCs (voice.toggle/record/tts) are in ``_LONG_HANDLERS`` so they run
    on the RPC pool and ``server.dispatch`` returns None — the pool worker
    writes the response via the bound transport. These tests exercise the
    handler's business logic, not the routing, so they drive the handler
    inline while preserving the transport-binding semantics ``dispatch``
    applies around a real request.
    """
    token = bind_transport(transport)
    try:
        return server.handle_request(req)
    finally:
        reset_transport(token)


@pytest.fixture(autouse=True)
def _neuter_agent_prewarm_timer(request, monkeypatch):
    """Stub the deferred agent pre-warm timer for every test in this module.

    ``session.create`` and non-eager ``session.resume`` fire a 50 ms
    background ``threading.Timer`` (``_schedule_agent_build``) that calls
    whatever ``server._make_agent`` is patched in AT FIRE TIME. Left live,
    a timer armed by one test outlives it and lands in the NEXT test's
    ``_make_agent`` mock, racily corrupting its captured state (the
    ``'tip' == 'cont_tip'`` flakes in the session_resume tests). Tests that
    exercise the deferred build itself opt back in with
    ``@pytest.mark.real_agent_prewarm``.
    """
    if request.node.get_closest_marker("real_agent_prewarm"):
        yield
        return
    monkeypatch.setattr(server, "_schedule_agent_build", lambda *a, **k: None)
    yield


@pytest.fixture(autouse=True)
def _reap_leaked_notification_pollers():
    """Stop and join notification pollers leaked by each test.

    session.init/create paths start a per-session poller daemon thread. A
    poller left running by one test steals-and-requeues events off the
    PROCESS-GLOBAL process_registry.completion_queue while a later test is
    asserting on it — the root cause of the flaky
    test_run_prompt_submit_requeues_all_unstarted_notifications_with_real_threading
    (two CI hits on unrelated PRs, Aug 2026). Set every registered poller's
    stop event (the loop wakes at least every 0.5s), then join with ONE
    small shared budget — never per-thread — so teardown stays O(seconds)
    for the whole file even when many tests leaked pollers.
    """
    yield
    pollers = [
        (stop, thread)
        for stop, thread in list(server._notification_pollers)
        if thread.is_alive()
    ]
    for stop, _thread in pollers:
        stop.set()
    deadline = time.time() + 3.0
    for _stop, thread in pollers:
        remaining = deadline - time.time()
        if remaining <= 0:
            break
        thread.join(timeout=remaining)
    server._notification_pollers[:] = [
        (stop, thread)
        for stop, thread in server._notification_pollers
        if thread.is_alive()
    ]


def _write_profile_cfg(home: Path, cwd: str | None) -> Path:
    import hermes_yaml as yaml

    home.mkdir(parents=True, exist_ok=True)
    cfg = {"terminal": {"cwd": cwd}} if cwd is not None else {}
    (home / "config.yaml").write_text(yaml.safe_dump(cfg), encoding="utf-8")
    return home


class _ChunkyStdout:
    def __init__(self):
        self.parts: list[str] = []

    def write(self, text: str) -> int:
        for ch in text:
            self.parts.append(ch)
            time.sleep(0)
        return len(text)

    def flush(self) -> None:
        return None


class _BrokenStdout:
    def write(self, text: str) -> int:
        raise BrokenPipeError

    def flush(self) -> None:
        return None


def _two_repo_project_skill_sessions(tmp_path, monkeypatch) -> tuple[Path, Path]:
    """Two trusted repos (``alpha-skill`` / ``beta-skill``) bound to sessions ``sid-a`` / ``sid-b``, in a
    launch shape whose process cwd and TERMINAL_CWD both point at a non-project dir."""
    from agent import skill_commands
    from agent import skill_utils
    from tools import skills_tool

    def repo(name: str, skill: str) -> Path:
        r = tmp_path / name
        (r / ".git").mkdir(parents=True)
        (r / ".hermes" / "skills" / skill).mkdir(parents=True)
        (r / ".hermes" / "skills" / skill / "SKILL.md").write_text(
            f"---\nname: {skill}\ndescription: from {name}\n---\n\n# {skill}\n\nBODY OF {skill.upper()}\n")
        return r

    repo_a, repo_b = repo("proj-a", "alpha-skill"), repo("proj-b", "beta-skill")
    home = tmp_path / "hermes-home"
    (home / "skills").mkdir(parents=True)
    (home / "config.yaml").write_text(
        f"skills:\n  external_dirs: []\n  trusted_project_dirs: ['{repo_a}', '{repo_b}']\n")
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(skills_tool, "SKILLS_DIR", home / "skills")
    monkeypatch.setattr(skill_utils, "_skills_cfg", lambda: {
        "external_dirs": [], "trusted_project_dirs": [str(repo_a), str(repo_b)]})
    skill_utils._external_dirs_cache_clear()
    monkeypatch.setattr(skill_commands, "_skill_commands_by_key", {})
    # Launch shape: process cwd and TERMINAL_CWD both point at a non-project dir (the resolved placeholder).
    elsewhere = tmp_path / "home-dir"
    elsewhere.mkdir()
    monkeypatch.chdir(elsewhere)
    monkeypatch.setenv("TERMINAL_CWD", str(elsewhere))
    monkeypatch.setattr(server, "_sessions", {
        "sid-a": {"session_key": "key-a", "cwd": str(repo_a)},
        "sid-b": {"session_key": "key-b", "cwd": str(repo_b)}})
    return repo_a, repo_b


def _sync_test_session(**extra):
    session = {
        "agent": types.SimpleNamespace(model="old/model"),
        "session_key": "session-key",
    }
    session.update(extra)
    return session


def _patch_config_model(monkeypatch, model, provider=""):
    monkeypatch.delenv("HERMES_MODEL", raising=False)
    monkeypatch.delenv("HERMES_INFERENCE_MODEL", raising=False)
    cfg_model = {"default": model}
    if provider:
        cfg_model["provider"] = provider
    monkeypatch.setattr(server, "_load_cfg", lambda: {"model": cfg_model})


def _capture_make_agent_kwargs(monkeypatch) -> dict:
    """Stub AIAgent so ``server._make_agent`` records the kwargs it was built with."""
    captured = {}

    def fake_agent(**kwargs):
        captured.update(kwargs)
        return types.SimpleNamespace(model=kwargs.get("model"))

    _setup_make_agent_mocks(monkeypatch, {})
    monkeypatch.setattr("run_agent.AIAgent", fake_agent)
    return captured


class _LoginSocket:
    """A live WS peer carrying the identity the upgrade auth minted."""

    def __init__(self, user_id="alice", provider="basic"):
        self.auth_identity = {"provider": provider, "user_id": user_id}
        self._closed = False

    def write(self, frame):
        return True


def _login_session(monkeypatch, sid, key, transport, **extra) -> dict:
    """A record the way session.create stamps it: the login lives on the record, not the transport slot."""
    record = {"session_key": key, "transport": transport,
              "auth_user_id": server._transport_auth_user_id(transport), **extra}
    monkeypatch.setitem(server._sessions, sid, record)
    return record


def _session(agent=None, **extra):
    return {
        "agent": agent if agent is not None else types.SimpleNamespace(),
        "session_key": "session-key",
        "history": [],
        "history_lock": threading.Lock(),
        "history_version": 0,
        "running": False,
        "attached_images": [],
        "image_counter": 0,
        "cols": 80,
        "slash_worker": None,
        "show_reasoning": False,
        "tool_progress_mode": "all",
        **extra,
    }


def _mk_merge_fake_db(verbatim, repaired, replaced):
    """SessionDB stand-in answering exactly like the real store: the repaired view
    (what `_load_durable_truncation_history` historically loaded) has the user;user
    run merged away, the un-repaired view keeps every physical row."""

    class _FakeDB:
        def get_messages_as_conversation(
            self,
            key,
            include_ancestors=False,
            repair_alternation=False,
            include_row_ids=False,
            **_kwargs,
        ) -> list:
            import copy as _copy

            assert key == "session-key"
            assert include_row_ids is True
            return _copy.deepcopy(repaired if repair_alternation else verbatim)

        def replace_messages(
            self, key, messages, active_only=False, archive_dropped=False, **_kwargs
        ):
            replaced.append((key, list(messages)))

    return _FakeDB()


class _StopAfterOneNotificationPoll:
    def __init__(self):
        self._checks = 0

    def is_set(self):
        self._checks += 1
        return self._checks > 1


def _configure_immediate_prompt_run(
    monkeypatch, tmp_path, *, immediate_threads=True
):
    class _ImmediateThread:
        def __init__(self, target=None, daemon=None, **_kwargs):
            self._target = target

        def start(self):
            if self._target is not None:
                self._target()

        def is_alive(self):
            return False

    if immediate_threads:
        monkeypatch.setattr(server.threading, "Thread", _ImmediateThread)
    monkeypatch.setattr(server, "_emit", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(server, "make_stream_renderer", lambda _cols: None)
    monkeypatch.setattr(server, "render_message", lambda _raw, _cols: None)
    monkeypatch.setattr(server, "_wire_callbacks", lambda _sid: None)
    monkeypatch.setattr(server, "_sync_agent_model_with_config", lambda *_args: None)
    monkeypatch.setattr(server, "_session_cwd", lambda _session: str(tmp_path))
    monkeypatch.setattr(server, "_register_session_cwd", lambda _session: None)
    monkeypatch.setattr(server, "_set_session_context", lambda *_args, **_kwargs: [])
    monkeypatch.setattr(server, "_clear_session_context", lambda _tokens: None)
    monkeypatch.setattr(server, "_session_info", lambda *_args: {})
    monkeypatch.setattr(server, "_get_usage", lambda _agent: {})
    monkeypatch.setattr(
        server, "_sync_session_key_after_compress", lambda *_args, **_kwargs: None
    )
    monkeypatch.setattr(server, "_drain_queued_prompt", lambda *_args: False)
    monkeypatch.setattr(server, "_voice_tts_enabled", lambda: False)
    monkeypatch.setattr(server, "_get_db", lambda: None)


class _RecordingAgent:
    model = "test-model"
    provider = "test-provider"

    def __init__(self, turns):
        self._turns = turns

    def clear_interrupt(self):
        return None

    def run_conversation(self, prompt, conversation_history=None, stream_callback=None, **_kwargs):
        self._turns.append(prompt)
        return {"final_response": "", "messages": []}


_SLASH_FILLER_COUNT = 60


def _slash_skill_fixtures(monkeypatch):
    """Stub a skill install big enough that a flat cap would truncate it."""
    filler = {f"/filler-{i:03d}": 0 for i in range(_SLASH_FILLER_COUNT)}
    usage = {"work": 297, "research": 84, "clean": 12}

    monkeypatch.setattr(
        "hermes_runtime.command_discovery._skill_usage_lookup",
        lambda: (
            lambda name: usage.get(name, 0),
            lambda name: "bundled" if name.startswith("unused-") else "local",
        ),
    )
    monkeypatch.setattr(
        "agent.skill_commands.get_skill_commands",
        lambda: {
            "/work": {"description": "Fresh worktree"},
            "/research": {"description": "Look it up"},
            "/clean": {"description": "Polish the diff"},
            "/unused-bundled": {"description": "Shipped, never opened"},
            **{cmd: {"description": "Filler"} for cmd in filler},
        },
    )
    monkeypatch.setattr("agent.skill_bundles.get_skill_bundles", dict)


def _slash_completions(text: str) -> list[dict]:
    resp = server.handle_request(
        {"id": "1", "method": "complete.slash", "params": {"text": text}}
    )
    return resp["result"]["items"]


class _RecordingSteerAgent:
    def __init__(self):
        self.steered = []

    def steer(self, text):
        self.steered.append(text)
        return True


def _open_request(sid, method="clarify"):
    from hermes_runtime import server_requests
    req = server_requests.ServerRequest(sid, method, {})
    with server_requests._lock:
        server_requests._open[req.id] = req
    return req


def _wake_lease_double():
    """Owner-gated pause/resume stand-in: only the lease holder can re-arm."""
    from tools import wake_word

    state = {"lease": None, "paused": False, "resumed": []}

    def pause_listening(*, owner):
        if state["lease"] is not owner:
            return False
        state["paused"] = True
        return True

    def resume_listening(*, owner):
        if state["lease"] is not owner:
            return False
        state["paused"] = False
        state["resumed"].append(owner)
        return True

    return wake_word, state, pause_listening, resume_listening


_PARTIAL_FAKE_HISTORY = [
    {"role": "user", "content": "msg1"},
    {"role": "assistant", "content": "resp1"},
    {"role": "user", "content": "msg2"},
    {"role": "assistant", "content": "resp2"},
    {"role": "user", "content": "keep this"},
    {"role": "assistant", "content": "keep this too"},
]


_PARTIAL_COMPRESSED_HEAD = [
    {"role": "user", "content": "[summary]"},
    {"role": "assistant", "content": "ok"},
]


def _partial_compress_agent(compress_context_calls):
    """Agent stub whose _compress_context records (history, focus_topic)."""
    agent = types.SimpleNamespace(
        _cached_system_prompt=None,
        tools=None,
        session_id="s1",
        context_compressor=None,  # keep _get_usage on the simple path
    )

    def _fake_compress_context(history, sys, approx_tokens=0, focus_topic=None, **kw):
        compress_context_calls.append((list(history), focus_topic))
        return list(_PARTIAL_COMPRESSED_HEAD), {}

    agent._compress_context = _fake_compress_context
    return agent


class _ImmediateThread:
    """Runs the target callable synchronously so assertions can follow."""

    def __init__(self, target=None, daemon=None, **_thread_options):
        self._target = target

    def start(self):
        self._target()


def _setup_make_agent_mocks(monkeypatch, cfg):
    monkeypatch.setattr(server, "_load_cfg", lambda: cfg)
    monkeypatch.setattr(
        server, "_resolve_startup_runtime", lambda: ("test-model", None)
    )
    monkeypatch.setattr(
        "hermes_cli.runtime_provider.resolve_runtime_provider",
        lambda requested=None, target_model=None: {
            "provider": None,
            "base_url": None,
            "api_key": None,
            "api_mode": None,
            "command": None,
            "args": None,
            "credential_pool": None,
        },
    )
    monkeypatch.setattr(server, "_load_tool_progress_mode", lambda: "off")
    monkeypatch.setattr(server, "_load_reasoning_config", lambda model="": None)
    monkeypatch.setattr(server, "_load_service_tier", lambda: None)
    monkeypatch.setattr(server, "_load_enabled_toolsets", lambda *_a, **_kw: None)
    monkeypatch.setattr(server, "_get_db", lambda: None)
    monkeypatch.setattr(server, "_agent_cbs", lambda sid: {})


class _FakeAgentForBackground:
    base_url = None
    api_key = None
    provider = None
    api_mode = None
    acp_command = None
    acp_args = None
    model = "test-model"
    enabled_toolsets = None
    ephemeral_system_prompt = None
    providers_allowed = None
    providers_ignored = None
    providers_order = None
    provider_sort = None
    provider_require_parameters = False
    provider_data_collection = None
    reasoning_config = None
    service_tier = None
    request_overrides = {}
    _fallback_model = None


_PNG_1X1_B64 = (
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNk"
    "+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg=="
)


def _attach_bytes_cli(monkeypatch):
    fake_cli = types.ModuleType("cli")
    fake_cli._IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".gif", ".webp", ".bmp"}
    monkeypatch.setitem(sys.modules, "cli", fake_cli)


def _idle_evictable_session(now):
    """A session that satisfies every eviction precondition."""
    ready = threading.Event()
    ready.set()
    old = now - 10 * 3600  # well past the 6h TTL
    return {
        "running": False,
        "agent_ready": ready,
        "transport": server._detached_ws_transport,  # dead/detached
        "last_active": old,
        "created_at": old,
    }


def _periodic_trim_calls(monkeypatch):
    """Stub the reaper's side effects and capture trim_memory calls (delayed import → patch the module attr)."""
    from hermes_cli import mem_trim

    calls = []
    monkeypatch.setattr(server, "_session_pending_kind", lambda sid: "")
    monkeypatch.setattr(server, "_close_session_by_id", lambda *a, **k: None)
    monkeypatch.setattr(server, "_enforce_session_cap", lambda: None)
    monkeypatch.setattr(server, "_reclaim_orphaned_leases", lambda: None)
    monkeypatch.setattr(mem_trim, "trim_memory", lambda **kw: calls.append(kw.get("reason", "")) or True)
    return calls


class _BillingHeaders:
    def __init__(self, values):
        self._values = values

    def get(self, key):
        return self._values.get(key)


def _sub_rpc(method, params):
    # These RPCs are in _LONG_HANDLERS (pool-routed → dispatch returns None and the
    # worker writes via the transport), so drive the inline handler directly.
    return server.handle_request({"id": "1", "method": method, "params": params})["result"]


class _BareAgent:
    """Agent stub with no compressor — exercises the active_subagents path
    independent of the `if comp:` context-percent block."""

    model = "x"


def _fake_tts_modules(monkeypatch, *, requirements=True, playback_stops=None, listen=None, transcribe=None):
    """Install lightweight tools.tts_tool / tools.voice_mode fakes."""
    started = {}

    def fake_stream(text_queue, stop, done, **_kw):
        started["queue"] = text_queue
        stop.wait(5)
        done.set()

    def default_listen(should_stop, capture=False, on_trigger=None, **_kw):
        return None if capture else False

    def default_fd_listen(should_stop, is_playing=None, on_trigger=None, **_kw):
        return None

    monkeypatch.setitem(
        sys.modules,
        "tools.tts_tool",
        types.SimpleNamespace(
            check_tts_requirements=lambda: requirements,
            _get_provider=lambda cfg: "edge",
            _load_tts_config=dict,
            get_env_value=lambda key, default="": default,
        ),
    )
    monkeypatch.setitem(
        sys.modules,
        "tools.tts_tool_speaker",
        types.SimpleNamespace(stream_tts_to_speaker=fake_stream),
    )
    monkeypatch.setitem(
        sys.modules,
        "tools.voice_mode",
        types.SimpleNamespace(
            stop_playback=lambda: (playback_stops.append(True) if playback_stops is not None else None),
            listen_for_speech=listen or default_listen,
            full_duplex_listen=listen or default_fd_listen,
            is_audio_output_active=lambda: False,
            transcribe_recording=transcribe or (lambda path, model=None: {"success": True, "transcript": ""}),
        ),
    )
    # Fresh listener slot per test — the arm is idempotent per process.
    monkeypatch.setattr(server, "_fd_listener_active", False)
    return started


def _capture_server_request(monkeypatch, result):
    """Stub the server-request send and capture (method, sid, params, timeout)."""
    from hermes_runtime import server_requests
    captured = {}

    def fake_send(method, sid, params, *, timeout, qids=None):
        captured.update(method=method, sid=sid, params=params, timeout=timeout, qids=qids)
        return result

    monkeypatch.setattr(server_requests, "send", fake_send)
    return captured


BRANCH_REASONING = "the parent's chain of thought"


BRANCH_REASONING_CONTENT = "the parent's reasoning content"


BRANCH_REASONING_DETAILS = [
    {"type": "reasoning.text", "text": "keep the parent's plan", "format": "unknown"}
]


BRANCH_CODEX_REASONING_ITEMS = [
    {"id": "rs_1", "type": "reasoning", "encrypted_content": "opaque-blob"}
]


BRANCH_CODEX_MESSAGE_ITEMS = [
    {
        "id": "msg_1",
        "type": "message",
        "role": "assistant",
        "content": [{"type": "output_text", "text": "done"}],
    }
]


def _branch_history():
    return [
        {"role": "user", "content": "hello"},
        {
            "role": "assistant",
            "content": "done",
            "reasoning": BRANCH_REASONING,
            "reasoning_content": BRANCH_REASONING_CONTENT,
            "reasoning_details": BRANCH_REASONING_DETAILS,
            "codex_reasoning_items": BRANCH_CODEX_REASONING_ITEMS,
            "codex_message_items": BRANCH_CODEX_MESSAGE_ITEMS,
        },
        # Timeline marker: rides as role=user but must keep its tag through
        # the branch copy, or it re-enters the truncate ordinal address space
        # as a phantom user turn after a restart (#82756).
        {
            "role": "user",
            "content": "[System: personality changed]",
            "display_kind": "personality_switch",
        },
    ]


def _branched_marker(db, session_key):
    return next(
        (
            m
            for m in db.get_messages_as_conversation(session_key)
            if m.get("display_kind") == "personality_switch"
        ),
        None,
    )


def _branched_assistant(db, session_key):
    return next(
        m
        for m in db.get_messages_as_conversation(session_key)
        if m["role"] == "assistant"
    )


def _join_turn_thread(sess, timeout=10.0):
    """Join the real turn ``prompt.submit`` started, so nothing it does races the
    assertions or leaks into the next test. The submit's dispatch thread hands off to a
    ``prompt-turn-*`` worker that replaces ``_run_thread``: follow the handle until it
    stops changing."""
    deadline = time.monotonic() + timeout
    while isinstance(run_thread := sess.get("_run_thread"), threading.Thread):
        run_thread.join(timeout=max(0.0, deadline - time.monotonic()))
        assert not run_thread.is_alive(), "prompt.submit turn thread did not finish"
        if sess.get("_run_thread") is run_thread:
            return


def test_verification_status_returns_recorded_evidence(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_VERIFY_ON_STOP", "1")  # ledger is inert when the guard is off
    profile_home = tmp_path / "profiles" / "verify"
    profile_home.mkdir(parents=True)
    monkeypatch.setattr(server, "_profile_home", lambda p: profile_home if p == "verify" else None)
    token = set_hermes_home_override(profile_home)
    project = tmp_path / "project"
    project.mkdir()
    (project / ".git").mkdir()
    (project / "package.json").write_text(
        json.dumps({"scripts": {"test": "vitest"}}),
        encoding="utf-8",
    )
    (project / "pnpm-lock.yaml").write_text("", encoding="utf-8")
    try:
        from agent.verification_evidence import record_terminal_result

        record_terminal_result(
            command="pnpm run test",
            cwd=project,
            session_id="sid",
            exit_code=0,
            output="green",
        )

        resp = server.handle_request(
            {
                "id": "1",
                "method": "verification.status",
                "params": {"cwd": str(project), "session_id": "sid", "profile": "verify"},
            }
        )
    finally:
        reset_hermes_home_override(token)

    verification = resp["result"]["verification"]
    assert verification["status"] == "passed"
    assert verification["evidence"]["canonical_command"] == "pnpm run test"
    assert verification["evidence"]["scope"] == "full"


def test_verification_status_outside_workspace_is_not_applicable(monkeypatch, tmp_path):
    monkeypatch.setenv("HERMES_VERIFY_ON_STOP", "1")  # ledger is inert when the guard is off
    # A cwd with no project facts (outside any code workspace) must report
    # not_applicable. Force the "no facts" precondition rather than relying on
    # tmp_path's ancestors being pristine — a stray marker file in a shared
    # tmp-root ancestor (e.g. /tmp/package.json left by another tool) would
    # otherwise make _marker_root() resolve tmp_path as a workspace and flip
    # the status to "unverified".
    from agent import coding_context

    monkeypatch.setattr(coding_context, "project_facts_for", lambda _cwd=None: None)

    home = tmp_path / ".hermes"
    home.mkdir()
    token = set_hermes_home_override(home)
    try:
        resp = server.handle_request(
            {
                "id": "1",
                "method": "verification.status",
                "params": {"cwd": str(tmp_path), "session_id": "sid"},
            }
        )
    finally:
        reset_hermes_home_override(token)

    assert resp["result"]["verification"]["status"] == "not_applicable"


def test_config_get_indicator_normalizes_casing_and_whitespace(monkeypatch):
    """Hand-edited config.yaml stays consistent with what the TUI shows.

    Frontend's `normalizeIndicatorStyle` lowercases + trims, so config.get
    must do the same — otherwise `/indicator` prints 'EMOJI ' while the
    UI is actually rendering the kaomoji default."""
    monkeypatch.setattr(
        server, "_load_cfg", lambda: {"display": {"tui_status_indicator": " EMOJI "}}
    )
    resp = server.handle_request(
        {"id": "1", "method": "config.get", "params": {"key": "indicator"}}
    )
    assert resp["result"] == {"value": "emoji"}


def test_config_get_indicator_falls_back_to_default_for_unknown(monkeypatch):
    """An unknown value in config.yaml falls back to the same default
    the frontend uses (`_INDICATOR_DEFAULT`)."""
    monkeypatch.setattr(
        server, "_load_cfg", lambda: {"display": {"tui_status_indicator": "rainbow"}}
    )
    resp = server.handle_request(
        {"id": "1", "method": "config.get", "params": {"key": "indicator"}}
    )
    assert resp["result"] == {"value": "kaomoji"}


def test_config_set_indicator_accepts_known_value(monkeypatch):
    written: dict = {}
    monkeypatch.setattr(
        server,
        "_write_config_key",
        lambda k, v: written.update({k: v}),
    )
    resp = server.handle_request(
        {
            "id": "1",
            "method": "config.set",
            "params": {"key": "indicator", "value": "EMOJI"},
        }
    )
    assert resp["result"] == {"key": "indicator", "value": "emoji"}
    assert written == {"display.tui_status_indicator": "emoji"}


def test_reload_env_rpc_surfaces_errors(monkeypatch):
    def _broken():
        raise RuntimeError("env path locked")

    fake = types.SimpleNamespace(reload_env=_broken)
    with patch.dict(sys.modules, {"hermes_cli.config": fake}):
        resp = server.handle_request({"id": "1", "method": "reload.env", "params": {}})

    assert "error" in resp
    assert "env path locked" in resp["error"]["message"]


def test_make_agent_reads_nested_max_turns(monkeypatch):
    _setup_make_agent_mocks(monkeypatch, {"agent": {"max_turns": 200}})

    with patch("run_agent.AIAgent") as mock_agent:
        server._make_agent("sid1", "key1")

    assert mock_agent.call_args.kwargs["max_iterations"] == 200


def test_make_agent_nested_max_turns_takes_priority(monkeypatch):
    _setup_make_agent_mocks(
        monkeypatch, {"agent": {"max_turns": 400}, "max_turns": 100}
    )

    with patch("run_agent.AIAgent") as mock_agent:
        server._make_agent("sid1", "key1")

    assert mock_agent.call_args.kwargs["max_iterations"] == 400


def test_make_agent_uses_session_runtime_overrides(monkeypatch):
    _setup_make_agent_mocks(monkeypatch, {})
    resolved = {}

    def fake_resolve_runtime_provider(requested=None, target_model=None):
        resolved["requested"] = requested
        resolved["target_model"] = target_model
        return {
            "provider": requested,
            "base_url": None,
            "api_key": None,
            "api_mode": None,
            "command": None,
            "args": None,
            "credential_pool": None,
        }

    monkeypatch.setattr(
        "hermes_cli.runtime_provider.resolve_runtime_provider",
        fake_resolve_runtime_provider,
    )

    with patch("run_agent.AIAgent") as mock_agent:
        server._make_agent(
            "sid1",
            "key1",
            model_override="gpt-5.4",
            provider_override="openai-codex",
            reasoning_config_override={"enabled": True, "effort": "high"},
            service_tier_override="priority",
        )

    assert resolved == {"requested": "openai-codex", "target_model": "gpt-5.4"}
    assert mock_agent.call_args.kwargs["model"] == "gpt-5.4"
    assert mock_agent.call_args.kwargs["provider"] == "openai-codex"
    assert mock_agent.call_args.kwargs["reasoning_config"] == {"enabled": True, "effort": "high"}
    assert mock_agent.call_args.kwargs["service_tier"] == "priority"


def test_make_agent_handles_null_agent_config(monkeypatch):
    _setup_make_agent_mocks(monkeypatch, {"agent": None, "max_turns": 80})

    with patch("run_agent.AIAgent") as mock_agent:
        server._make_agent("sid1", "key1")

    assert mock_agent.call_args.kwargs["max_iterations"] == 80


def test_background_agent_kwargs_reads_nested_max_turns(monkeypatch):
    monkeypatch.setattr(server, "_load_cfg", lambda: {"agent": {"max_turns": 300}})

    kwargs = server._background_agent_kwargs(_FakeAgentForBackground(), "task_1")

    assert kwargs["max_iterations"] == 300


def test_background_agent_kwargs_falls_back_to_root_max_turns(monkeypatch):
    monkeypatch.setattr(server, "_load_cfg", lambda: {"max_turns": 50})

    kwargs = server._background_agent_kwargs(_FakeAgentForBackground(), "task_1")

    assert kwargs["max_iterations"] == 50


def test_background_agent_kwargs_handles_null_agent_config(monkeypatch):
    monkeypatch.setattr(server, "_load_cfg", lambda: {"agent": None, "max_turns": 40})

    kwargs = server._background_agent_kwargs(_FakeAgentForBackground(), "task_1")

    assert kwargs["max_iterations"] == 40


def test_notification_poller_delivers_completion(monkeypatch):
    """Poller picks up completion events and triggers agent turns."""
    import queue as _queue_mod

    from tools.process_registry import process_registry

    turns = []
    emitted = []

    class _Agent:
        def run_conversation(self, prompt, conversation_history=None, stream_callback=None, **_kwargs):
            turns.append(prompt)
            return {
                "final_response": "ok",
                "messages": [{"role": "assistant", "content": "ok"}],
            }

    class _ImmediateThread:
        def __init__(self, target=None, daemon=None, **_thread_options):
            self._target = target
        def start(self):
            self._target()

    sess = _session(agent=_Agent())
    server._sessions["sid_poll"] = sess
    monkeypatch.setattr(server.threading, "Thread", _ImmediateThread)
    monkeypatch.setattr(server, "_emit", lambda *a, **kw: emitted.append(a))
    monkeypatch.setattr(server, "make_stream_renderer", lambda cols: None)
    monkeypatch.setattr(server, "render_message", lambda raw, cols: None)

    # Isolate the completion queue for the duration of this test. The poller
    # reads process_registry.completion_queue by attribute at runtime; the
    # event below carries no session_key, so any *other* poller (a leaked
    # daemon thread from another test, or a concurrent one in the same xdist
    # worker) is allowed to dequeue and dispatch it to its own session — whose
    # agent may be a fixture double without run_conversation. A fresh Queue
    # here fully isolates this test; monkeypatch restores the original on
    # teardown. (Same pattern as test_notification_poller_requeues_when_busy.)
    isolated_queue: _queue_mod.Queue = _queue_mod.Queue()
    monkeypatch.setattr(process_registry, "completion_queue", isolated_queue)
    process_registry._completion_consumed.discard("proc_poller_test")

    stop = threading.Event()

    # Put event on queue, then immediately signal stop so the poller
    # runs exactly one iteration.
    isolated_queue.put({
        "type": "completion",
        "session_id": "proc_poller_test",
        "command": "echo hello",
        "exit_code": 0,
        "output": "hello",
    })
    stop.set()

    try:
        server._notification_poller_loop(stop, "sid_poll", sess)

        # Should have emitted a status.update with kind=process
        status_calls = [a for a in emitted if a[0] == "status.update"]
        assert len(status_calls) >= 1
        assert status_calls[0][2]["kind"] == "process"

        # Should have triggered an agent turn
        assert len(turns) == 1
        assert "proc_poller_test" in turns[0]
    finally:
        server._sessions.pop("sid_poll", None)
        while not process_registry.completion_queue.empty():
            process_registry.completion_queue.get_nowait()


def test_notification_poller_skips_consumed(monkeypatch):
    """Already-consumed completions are not dispatched by the poller."""
    import queue as _queue_mod

    from tools.process_registry import process_registry

    turns = []

    class _Agent:
        def run_conversation(self, prompt, conversation_history=None, stream_callback=None, **_kwargs):
            turns.append(prompt)
            return {"final_response": "ok", "messages": []}

    class _ImmediateThread:
        def __init__(self, target=None, daemon=None, **_thread_options):
            self._target = target
        def start(self):
            self._target()

    sess = _session(agent=_Agent())
    server._sessions["sid_skip"] = sess
    monkeypatch.setattr(server.threading, "Thread", _ImmediateThread)
    monkeypatch.setattr(server, "_emit", lambda *a, **kw: None)
    monkeypatch.setattr(server, "make_stream_renderer", lambda cols: None)
    monkeypatch.setattr(server, "render_message", lambda raw, cols: None)

    # Isolate the completion queue so a concurrent/leaked poller in the same
    # xdist worker can't dequeue this session_key-less event before our poller
    # does. monkeypatch restores the shared singleton on teardown. (Same
    # pattern as test_notification_poller_requeues_when_busy.)
    isolated_queue: _queue_mod.Queue = _queue_mod.Queue()
    monkeypatch.setattr(process_registry, "completion_queue", isolated_queue)

    process_registry._completion_consumed.add("proc_already_done")
    isolated_queue.put({
        "type": "completion",
        "session_id": "proc_already_done",
        "command": "echo x",
        "exit_code": 0,
        "output": "x",
    })

    stop = threading.Event()
    stop.set()

    try:
        server._notification_poller_loop(stop, "sid_skip", sess)
        assert len(turns) == 0
    finally:
        server._sessions.pop("sid_skip", None)
        process_registry._completion_consumed.discard("proc_already_done")
        while not process_registry.completion_queue.empty():
            process_registry.completion_queue.get_nowait()


def test_notification_poller_starts_no_turn_after_stop(monkeypatch):
    """After the user pressed Stop, a completion that arrives must not start an
    automatic model turn; it waits (requeued) until the user submits again."""
    import queue as _queue_mod

    from tools.process_registry import process_registry

    started = []
    sess = _session(running=False, _turn_cancel_requested=True)
    server._sessions["sid_stopped"] = sess
    monkeypatch.setattr(server, "_emit", lambda *a, **kw: None)
    monkeypatch.setattr(server, "_run_prompt_submit", lambda *a, **kw: started.append(a) or True)
    isolated_queue: _queue_mod.Queue = _queue_mod.Queue()
    monkeypatch.setattr(process_registry, "completion_queue", isolated_queue)
    process_registry._completion_consumed.discard("proc_after_stop")
    isolated_queue.put({
        "type": "completion", "session_id": "proc_after_stop", "command": "make build",
        "exit_code": 0, "output": "ok",
    })
    stop = threading.Event()
    stop.set()

    try:
        server._notification_poller_loop(stop, "sid_stopped", sess)

        assert started == []
        assert sess["running"] is False
        assert isolated_queue.get_nowait()["session_id"] == "proc_after_stop"
    finally:
        server._sessions.pop("sid_stopped", None)


def test_goal_continuation_starts_no_turn_after_stop(monkeypatch):
    """A Stop that lands after the model answered but before the post-turn follow-ups
    must not let an active /goal chain its continuation turn."""
    started = []
    sess = _session(running=False, _turn_cancel_requested=True)
    monkeypatch.setattr(server, "_emit", lambda *a, **kw: None)
    monkeypatch.setattr(server, "_run_prompt_submit", lambda *a, **kw: started.append(a) or True)

    server._run_post_turn_followups("r", "sid_goal_stopped", sess, {}, "keep going")

    assert started == []
    assert sess["running"] is False


def test_notification_poller_requeues_when_busy(monkeypatch):
    """When the agent is busy, the poller requeues the event."""
    import queue as _queue_mod

    from tools.process_registry import process_registry

    emitted = []

    sess = _session(running=True)  # agent is busy
    server._sessions["sid_busy"] = sess
    monkeypatch.setattr(server, "_emit", lambda *a, **kw: emitted.append(a))

    # Isolate the completion queue for the duration of this test. The poller
    # reads process_registry.completion_queue by attribute at runtime, so a
    # fresh Queue here means no concurrently-running test in the same xdist
    # worker can put/get on the shared singleton mid-run and drain the event
    # we expect to be requeued. monkeypatch restores the original on teardown.
    isolated_queue: _queue_mod.Queue = _queue_mod.Queue()
    monkeypatch.setattr(process_registry, "completion_queue", isolated_queue)
    process_registry._completion_consumed.discard("proc_busy_test")

    evt = {
        "type": "completion",
        "session_id": "proc_busy_test",
        "command": "make build",
        "exit_code": 0,
        "output": "ok",
    }
    isolated_queue.put(evt)

    stop = threading.Event()
    stop.set()

    try:
        server._notification_poller_loop(stop, "sid_busy", sess)

        # Status update was emitted (user sees it)
        status_calls = [a for a in emitted if a[0] == "status.update"]
        assert len(status_calls) == 1

        # Event was requeued (agent was busy, no turn triggered)
        assert not isolated_queue.empty()
        requeued = isolated_queue.get_nowait()
        assert requeued["session_id"] == "proc_busy_test"
    finally:
        server._sessions.pop("sid_busy", None)
        while not process_registry.completion_queue.empty():
            process_registry.completion_queue.get_nowait()


def test_session_save_writes_under_hermes_home_with_system_prompt(monkeypatch, tmp_path):
    """TUI /save (session.save RPC) must snapshot under the Hermes profile
    home — not the project/workspace CWD — and include the system prompt,
    mirroring the classic CLI /save and the dashboard save export.

    Regression: the gateway handler wrote ``hermes_conversation_*.json`` to
    ``os.path.abspath(...)`` (the workspace CWD) and only exported ``model``
    and ``messages``, so ``system_prompt`` was missing.
    """
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))

    # Run from a different CWD to prove the snapshot does NOT leak there.
    work = tmp_path / "workspace"
    work.mkdir()
    monkeypatch.chdir(work)

    sid = "save-sid"
    agent = types.SimpleNamespace(
        model="hermes-test",
        session_id="20260101_120000_abc123",
        session_start=datetime(2026, 1, 1, 12, 0, 0),
        _cached_system_prompt="You are Hermes.",
    )
    history = [
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": "hello"},
    ]
    server._sessions[sid] = {
        "agent": agent,
        "session_key": "save-key",
        "history": history,
        "history_lock": threading.Lock(),
        "created_at": 1735732800.0,
    }
    try:
        resp = server._methods["session.save"]("1", {"session_id": sid})
    finally:
        server._sessions.pop(sid, None)

    assert "result" in resp, resp
    saved_file = Path(resp["result"]["file"])

    # Must NOT leak into the workspace/project CWD.
    assert not list(work.glob("hermes_conversation_*.json"))

    saved_dir = home / "sessions" / "saved"
    assert saved_file.parent == saved_dir
    assert saved_file.exists()

    payload = json.loads(saved_file.read_text(encoding="utf-8-sig"))
    assert payload["model"] == "hermes-test"
    assert payload["id"] == "20260101_120000_abc123"  # importable: import_sessions keys on raw["id"]
    assert payload["started_at"] == datetime(2026, 1, 1, 12, 0, 0).timestamp()
    assert payload["system_prompt"] == "You are Hermes."
    assert payload["messages"] == history


def test_session_save_lands_in_the_sessions_own_profile_a_b_a(monkeypatch, tmp_path):
    """/save writes under the SESSION's profile home, launch -> secondary -> launch under multiplexing:
    the RPC runs unscoped, so get_hermes_home() alone names the launch profile (#125241)."""
    from agent.secret_scope import set_multiplex_active
    launch_home = tmp_path / ".hermes"
    work_home = launch_home / "profiles" / "s6probe-work"
    work_home.mkdir(parents=True)
    monkeypatch.setenv("HERMES_HOME", str(launch_home))
    set_multiplex_active(True)
    parents = []
    try:
        for i, home in enumerate((None, work_home, None)):
            sid = f"save-profile-sid-{i}"
            server._sessions[sid] = {
                "agent": types.SimpleNamespace(model="hermes-test", session_id="s1", session_start=None,
                                               _cached_system_prompt=""),
                "session_key": sid, "profile_home": str(home) if home else None,
                "history": [{"role": "user", "content": "hi"}], "history_lock": threading.Lock(),
            }
            try:
                resp = server._methods["session.save"]("1", {"session_id": sid})
            finally:
                server._sessions.pop(sid, None)
            assert "result" in resp, resp
            parents.append(Path(resp["result"]["file"]).parent)
    finally:
        set_multiplex_active(False)

    saved = launch_home / "sessions" / "saved"
    assert parents == [saved, work_home / "sessions" / "saved", saved]
    assert len(list((work_home / "sessions" / "saved").glob("hermes_conversation_*.json"))) == 1


def test_session_save_proxies_to_compute_host_history(monkeypatch):
    """Isolated turns own history in the host; /save must not export the stale parent mirror."""
    sid = "save-host-sid"
    server._sessions[sid] = _session(agent=None, _compute_host_active=True)
    calls = []

    def send_control(control_sid, **kwargs):
        calls.append((control_sid, kwargs))
        return {"type": "control.ack", "result": {"file": "/tmp/host-save.json"}}

    monkeypatch.setattr(server, "_session_uses_compute_host", lambda _session: True)
    monkeypatch.setattr(server, "_send_compute_host_control", send_control)
    try:
        resp = server._methods["session.save"]("1", {"session_id": sid})
    finally:
        server._sessions.pop(sid, None)

    assert resp["result"] == {"file": "/tmp/host-save.json"}
    assert calls == [(sid, {"route_name": "session.save", "wait": True})]


def test_notification_event_dedup_key_preserves_distinct_watch_matches():
    """Watch-match identity includes match content, not just session/type."""
    base = {
        "type": "watch_match",
        "session_id": "proc_watch",
        "command": "tail -f app.log",
        "pattern": "READY",
        "output": "READY on port 8000",
        "suppressed": 0,
    }

    identical = dict(base)
    distinct_output = {**base, "output": "READY on port 9000"}
    distinct_pattern = {**base, "pattern": "MIGRATION_DONE"}

    base_key = server._notification_event_dedup_key(base)
    assert server._notification_event_dedup_key(identical) == base_key
    assert server._notification_event_dedup_key(distinct_output) != base_key
    assert server._notification_event_dedup_key(distinct_pattern) != base_key


def test_notification_poller_emits_distinct_watch_matches_once(monkeypatch):
    """Distinct watch matches from one process emit; exact replay is deduped."""
    import queue as _queue_mod

    from tools.process_registry import process_registry

    turns = []
    emitted = []

    def _fake_run_prompt_submit(rid, sid, session, text, **_kwargs):
        turns.append(text)
        with session["history_lock"]:
            session["running"] = False

    sess = _session()
    server._sessions["sid_watch_dedup"] = sess
    monkeypatch.setattr(server, "_emit", lambda *a, **kw: emitted.append(a))
    monkeypatch.setattr(server, "_run_prompt_submit", _fake_run_prompt_submit)

    isolated_queue: _queue_mod.Queue = _queue_mod.Queue()
    monkeypatch.setattr(process_registry, "completion_queue", isolated_queue)

    base = {
        "type": "watch_match",
        "session_id": "proc_watch_dedup",
        "command": "tail -f app.log",
        "pattern": "READY",
        "output": "READY on port 8000",
        "suppressed": 0,
    }
    isolated_queue.put(base)
    isolated_queue.put({**base, "output": "READY on port 9000"})
    isolated_queue.put(dict(base))

    stop = threading.Event()
    stop.set()

    try:
        server._notification_poller_loop(stop, "sid_watch_dedup", sess)
        status_calls = [a for a in emitted if a[0] == "status.update"]
        assert len(status_calls) == 2
        status_text = "\n".join(call[2]["text"] for call in status_calls)
        assert "READY on port 8000" in status_text
        assert "READY on port 9000" in status_text
        assert len(turns) == 3
    finally:
        server._sessions.pop("sid_watch_dedup", None)
        while not process_registry.completion_queue.empty():
            process_registry.completion_queue.get_nowait()


def test_notification_event_dedup_key_keeps_completions_one_shot():
    """Completion identity remains process-session scoped to avoid floods."""
    first = {
        "type": "completion",
        "session_id": "proc_done",
        "command": "make build",
        "exit_code": 0,
        "output": "first output",
    }
    replay = {
        "type": "completion",
        "session_id": "proc_done",
        "command": "make build --again",
        "exit_code": 1,
        "output": "different output should not change completion key",
    }

    assert server._notification_event_dedup_key(first) == server._notification_event_dedup_key(
        replay
    )


def test_image_attach_bytes_writes_to_gateway_dir(monkeypatch, tmp_path):
    """Remote client uploads base64 bytes; gateway writes them to its own disk."""
    _attach_bytes_cli(monkeypatch)
    monkeypatch.setattr(server, "_hermes_home", tmp_path)
    server._sessions["abx"] = _session()

    resp = server.handle_request(
        {
            "id": "1",
            "method": "image.attach_bytes",
            "params": {
                "session_id": "abx",
                "content_base64": _PNG_1X1_B64,
                "filename": "shot.png",
            },
        }
    )

    res = resp["result"]
    assert res["attached"] is True
    written = Path(res["path"])
    assert written.is_file()
    assert written.parent == tmp_path / "images"
    assert written.read_bytes().startswith(b"\x89PNG")
    assert len(server._sessions["abx"]["attached_images"]) == 1
    assert res["bytes"] > 0


def test_image_attach_bytes_accepts_data_url_prefix(monkeypatch, tmp_path):
    _attach_bytes_cli(monkeypatch)
    monkeypatch.setattr(server, "_hermes_home", tmp_path)
    server._sessions["abx2"] = _session()

    resp = server.handle_request(
        {
            "id": "1",
            "method": "image.attach_bytes",
            "params": {
                "session_id": "abx2",
                "content_base64": f"data:image/png;base64,{_PNG_1X1_B64}",
            },
        }
    )
    assert resp["result"]["attached"] is True


def test_image_attach_bytes_data_alias_and_magic_sniff(monkeypatch, tmp_path):
    """Older desktop builds send `data` (not content_base64); ext sniffed from bytes."""
    _attach_bytes_cli(monkeypatch)
    monkeypatch.setattr(server, "_hermes_home", tmp_path)
    server._sessions["abx3"] = _session()

    resp = server.handle_request(
        {
            "id": "1",
            "method": "image.attach_bytes",
            "params": {"session_id": "abx3", "data": _PNG_1X1_B64},
        }
    )
    res = resp["result"]
    assert res["attached"] is True
    assert Path(res["path"]).suffix == ".png"  # sniffed from magic bytes


def test_image_attach_bytes_rejects_invalid_base64(monkeypatch, tmp_path):
    _attach_bytes_cli(monkeypatch)
    monkeypatch.setattr(server, "_hermes_home", tmp_path)
    server._sessions["abx4"] = _session()

    resp = server.handle_request(
        {
            "id": "1",
            "method": "image.attach_bytes",
            "params": {"session_id": "abx4", "content_base64": "!!!not base64!!!"},
        }
    )
    assert "error" in resp
    assert resp["error"]["code"] == 4017


def test_image_attach_bytes_rejects_oversize(monkeypatch, tmp_path):
    import base64 as _b64

    _attach_bytes_cli(monkeypatch)
    monkeypatch.setattr(server, "_hermes_home", tmp_path)
    monkeypatch.setattr(server, "_ATTACH_BYTES_MAX_BYTES", 10)
    server._sessions["abx5"] = _session()

    big = _b64.b64encode(b"\x89PNG\r\n\x1a\n" + b"0" * 100).decode("ascii")
    resp = server.handle_request(
        {
            "id": "1",
            "method": "image.attach_bytes",
            "params": {"session_id": "abx5", "content_base64": big},
        }
    )
    assert "error" in resp
    assert resp["error"]["code"] == 4018


def test_image_attach_bytes_rejects_unsupported_extension(monkeypatch, tmp_path):
    _attach_bytes_cli(monkeypatch)
    monkeypatch.setattr(server, "_hermes_home", tmp_path)
    server._sessions["abx6"] = _session()

    # filename hint forces a non-image extension; magic sniff is bypassed by hint
    resp = server.handle_request(
        {
            "id": "1",
            "method": "image.attach_bytes",
            "params": {
                "session_id": "abx6",
                "content_base64": _PNG_1X1_B64,
                "filename": "evil.exe",
            },
        }
    )
    assert "error" in resp
    assert resp["error"]["code"] == 4016


def test_pdf_attach_requires_poppler(monkeypatch, tmp_path):
    """Without pdftoppm on PATH, pdf.attach returns a clear 5028."""
    _attach_bytes_cli(monkeypatch)
    monkeypatch.setattr(server, "_hermes_home", tmp_path)
    monkeypatch.setattr("shutil.which", lambda _name: None)
    server._sessions["pdf1"] = _session()

    resp = server.handle_request(
        {
            "id": "1",
            "method": "pdf.attach",
            "params": {"session_id": "pdf1", "content_base64": "JVBERi0xLjQK"},
        }
    )
    assert "error" in resp
    assert resp["error"]["code"] == 5028


def test_pdf_attach_rejects_non_pdf_bytes(monkeypatch, tmp_path):
    import base64 as _b64

    _attach_bytes_cli(monkeypatch)
    monkeypatch.setattr(server, "_hermes_home", tmp_path)
    monkeypatch.setattr("shutil.which", lambda _name: "/usr/bin/pdftoppm")
    server._sessions["pdf2"] = _session()

    not_pdf = _b64.b64encode(b"this is not a pdf").decode("ascii")
    resp = server.handle_request(
        {
            "id": "1",
            "method": "pdf.attach",
            "params": {"session_id": "pdf2", "content_base64": not_pdf},
        }
    )
    assert "error" in resp
    assert resp["error"]["code"] == 4017


def test_pdf_attach_requires_path_or_bytes(monkeypatch, tmp_path):
    _attach_bytes_cli(monkeypatch)
    monkeypatch.setattr(server, "_hermes_home", tmp_path)
    monkeypatch.setattr("shutil.which", lambda _name: "/usr/bin/pdftoppm")
    server._sessions["pdf3"] = _session()

    resp = server.handle_request(
        {"id": "1", "method": "pdf.attach", "params": {"session_id": "pdf3"}}
    )
    assert "error" in resp
    assert resp["error"]["code"] == 4015


def test_decode_attach_base64_helper():
    import base64 as _b64

    raw = _b64.b64encode(b"hello").decode("ascii")
    assert server._decode_attach_base64(raw, mime_prefix="image/") == b"hello"
    assert (
        server._decode_attach_base64(f"data:image/png;base64,{raw}", mime_prefix="image/")
        == b"hello"
    )
    # whitespace inside payload is tolerated
    assert server._decode_attach_base64(raw[:4] + "\n" + raw[4:], mime_prefix="image/") == b"hello"
    assert server._decode_attach_base64("@@@", mime_prefix="image/") is None


def test_sniff_image_ext_magic_and_filename():
    assert server._sniff_image_ext(b"\x89PNG\r\n\x1a\n") == ".png"
    assert server._sniff_image_ext(b"\xff\xd8\xff\xe0") == ".jpg"
    assert server._sniff_image_ext(b"GIF89a....") == ".gif"
    assert server._sniff_image_ext(b"RIFF1234WEBPxxxx") == ".webp"
    assert server._sniff_image_ext(b"BM......") == ".bmp"
    assert server._sniff_image_ext(b"unknown") == ".png"  # fallback
    # filename hint wins over magic bytes
    assert server._sniff_image_ext(b"\x89PNG", "photo.jpeg") == ".jpeg"


def test_slash_worker_close_reaps_zombie_and_closes_fds():
    """A hung worker is SIGKILLed, the zombie reaped, all pipes closed — once."""
    calls = {k: 0 for k in ("terminate", "kill", "wait", "stdin", "stdout", "stderr")}

    class FakeStream:
        def __init__(self, name):
            self.name = name

        def close(self):
            calls[self.name] += 1

    class FakeProc:
        stdin, stdout, stderr = (FakeStream(n) for n in ("stdin", "stdout", "stderr"))

        def poll(self):
            return None  # always alive -> forces terminate then kill

        def terminate(self):
            calls["terminate"] += 1

        def kill(self):
            calls["kill"] += 1

        def wait(self, timeout=None):
            calls["wait"] += 1
            raise subprocess.TimeoutExpired(cmd="x", timeout=timeout)

    worker = object.__new__(server._SlashWorker)
    worker.proc = FakeProc()

    worker.close()
    worker.close()  # idempotent

    assert calls["terminate"] == 1
    assert calls["kill"] == 1
    assert calls["wait"] >= 2  # reaped after both terminate and kill
    assert calls["stdin"] == calls["stdout"] == calls["stderr"] == 1


def test_close_session_by_id_is_idempotent_and_full(monkeypatch):
    """One call tears the session down fully; a second is a no-op."""
    calls = {"worker": 0, "agent": 0, "unreg": 0, "finalize": 0}

    class W:
        def close(self):
            calls["worker"] += 1

    class A:
        def close(self):
            calls["agent"] += 1

    def _fake_finalize(s, end_reason="tui_close"):
        # Real _finalize_session is the single chokepoint that closes the
        # slash-worker; mirror that here so the test exercises the actual
        # teardown contract (worker close lives in finalize, not the caller).
        calls["finalize"] += 1
        w = s.get("slash_worker")
        if w:
            w.close()

    monkeypatch.setattr(server, "_finalize_session", _fake_finalize)
    monkeypatch.setattr(
        "tools.approval.unregister_gateway_notify",
        lambda key: calls.__setitem__("unreg", calls["unreg"] + 1), raising=False,
    )
    server._sessions["sid-1"] = {"session_key": "k1", "agent": A(), "slash_worker": W()}

    assert server._close_session_by_id("sid-1", end_reason="ws_disconnect") is True
    assert server._close_session_by_id("sid-1", end_reason="ws_disconnect") is False
    assert calls == {"worker": 1, "agent": 1, "unreg": 1, "finalize": 1}
    assert "sid-1" not in server._sessions


def test_attach_worker_closes_orphan_when_session_already_torn_down():
    """A worker built after its session was reaped must be closed, not orphaned."""
    closed = []

    class W:
        def close(self):
            closed.append(True)

    server._sessions.pop("gone", None)
    detached = {"session_key": "k"}  # not in _sessions -> already torn down
    server._attach_worker("gone", detached, W())

    assert closed == [True]
    assert "slash_worker" not in detached
    assert "gone" not in server._sessions


def test_attach_worker_stores_worker_on_live_session():
    class W:
        def close(self):
            raise AssertionError("must not close a worker for a live session")

    live = {"session_key": "k"}
    server._sessions["live"] = live
    worker = W()
    try:
        server._attach_worker("live", live, worker)
        assert live["slash_worker"] is worker
    finally:
        server._sessions.pop("live", None)


def test_restart_slash_worker_closes_orphan_when_session_reaped(monkeypatch):
    """Post-turn restart of a session reaped mid-flight (e.g. close_on_disconnect
    fired while `running` flipped false) must close both the stale worker and
    the fresh replacement, not orphan either."""
    closed = []

    class _FakeWorker:
        def __init__(self, *a, **k):
            pass

        def close(self):
            closed.append(True)

    monkeypatch.setattr(server, "_SlashWorker", _FakeWorker)
    server._sessions.pop("reaped", None)
    # not in _sessions -> torn down concurrently; carries a live worker so the
    # restart path actually runs (a workerless session is a restart no-op now)
    reaped = {"session_key": "k", "slash_worker": _FakeWorker()}
    server._restart_slash_worker("reaped", reaped)

    # stale worker closed by the restart, fresh worker closed by _attach_worker
    # (sid no longer maps to this session)
    assert closed == [True, True]
    assert "reaped" not in server._sessions


def test_restart_slash_worker_stores_on_live_session(monkeypatch):
    class _FakeWorker:
        def __init__(self, *a, **k):
            pass

        def close(self):
            pass

    monkeypatch.setattr(server, "_SlashWorker", _FakeWorker)
    old_worker = _FakeWorker()
    live = {"session_key": "k", "slash_worker": old_worker}
    server._sessions["live-restart"] = live
    try:
        server._restart_slash_worker("live-restart", live)
        assert isinstance(live["slash_worker"], _FakeWorker)
        assert live["slash_worker"] is not old_worker
    finally:
        server._sessions.pop("live-restart", None)


def test_restart_slash_worker_noop_without_worker(monkeypatch):
    """A session that never spawned a worker (slash.exec not used yet) must
    stay workerless across a restart — spawning here would fork the per-worker
    stdio MCP fleet for sessions that never run worker-routed commands."""
    spawned = []

    class _FakeWorker:
        def __init__(self, *a, **k):
            spawned.append(True)

        def close(self):
            pass

    monkeypatch.setattr(server, "_SlashWorker", _FakeWorker)
    live = {"session_key": "k", "slash_worker": None}
    server._sessions["lazy-noop"] = live
    try:
        server._restart_slash_worker("lazy-noop", live)
        assert spawned == []
        assert live["slash_worker"] is None
    finally:
        server._sessions.pop("lazy-noop", None)


def test_slash_exec_concurrent_first_use_spawns_single_worker(monkeypatch):
    """With eager pre-warm removed, slash.exec is the only spawn path — two
    concurrent worker-routed commands on a fresh session must not each fork a
    full MCP-fleet worker. The per-session spawn lock serializes first use.

    The probe is an unregistered command so it is not answered live, pending-input,
    bundle, skill, or plugin — it falls through to the worker spawn (the path under
    test). ``/context`` was the original probe but is now answered in-process for a
    local session (the #93280 fix), so it no longer reaches the worker."""
    import time as _time

    spawned = []
    barrier = threading.Barrier(2, timeout=5)

    class _SlowWorker:
        def __init__(self, *a, **k):
            spawned.append(self)
            _time.sleep(0.05)  # widen the None-observation window

        def run(self, cmd):
            return f"ran {cmd}"

        def close(self):
            pass

    monkeypatch.setattr(server, "_SlashWorker", _SlowWorker)
    monkeypatch.setattr(server, "_mirror_slash_side_effects", lambda *a, **k: None)
    session = _session(slash_worker=None)
    server._sessions["race-spawn"] = session

    results = []

    def _exec(n):
        barrier.wait()
        resp = server.handle_request(
            {
                "id": str(n),
                "method": "slash.exec",
                "params": {"command": "/frobnicate-probe", "session_id": "race-spawn"},
            }
        )
        results.append(resp)

    try:
        threads = [threading.Thread(target=_exec, args=(i,)) for i in range(2)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=10)
        assert len(spawned) == 1, (
            f"concurrent slash.exec spawned {len(spawned)} workers — first-use "
            f"spawn must be serialized per session"
        )
        assert session["slash_worker"] is spawned[0]
        assert all("result" in r for r in results), results
    finally:
        server._sessions.pop("race-spawn", None)


def test_close_sessions_for_transport_closes_flagged_repoints_rest(monkeypatch):
    seen = []
    monkeypatch.setattr(
        server,
        "_teardown_popped_session",
        lambda session, *, end_reason: seen.append((session["_sid"], end_reason)) or True,
    )
    # Detached session "b" would schedule a real grace-reap threading.Timer that
    # outlives the test; grace=0 short-circuits it so no thread lingers.
    monkeypatch.setattr(server, "_WS_ORPHAN_REAP_GRACE_S", 0)
    transport = object()  # the disconnecting transport
    server._sessions.clear()
    server._sessions["a"] = {"transport": transport, "close_on_disconnect": True}
    server._sessions["b"] = {"transport": transport, "close_on_disconnect": False}
    try:
        server._close_sessions_for_transport(transport, end_reason="ws_disconnect")
        assert seen == [("a", "ws_disconnect")]  # only the flagged one closed
        assert server._sessions["b"]["transport"] is server._detached_ws_transport  # re-pointed
    finally:
        server._sessions.clear()


@pytest.mark.parametrize("close_on_disconnect", [True, False])
def test_close_sessions_for_transport_skips_session_rebound_before_claim(
    monkeypatch, close_on_disconnect
):
    """A resume between snapshot and claim keeps either session type alive."""
    reaps = []
    teardowns = []
    monkeypatch.setattr(
        server, "_schedule_ws_orphan_reap", lambda sid: reaps.append(sid)
    )
    monkeypatch.setattr(
        server,
        "_teardown_popped_session",
        lambda session, *, end_reason: teardowns.append((session, end_reason)) or True,
    )
    old_transport = object()  # the disconnecting transport
    new_transport = object()  # live rebind target (no _closed attr → alive)
    session = {"transport": old_transport, "close_on_disconnect": close_on_disconnect}
    original_sessions_lock = server._sessions_lock
    rebound = threading.Event()

    class _SnapshotInterlock:
        """Rebind in a second thread immediately after the ownership snapshot."""

        def __init__(self):
            self._snapshot_released = False

        def __enter__(self):
            original_sessions_lock.acquire()
            return self

        def __exit__(self, exc_type, exc, traceback):
            original_sessions_lock.release()
            if not self._snapshot_released:
                self._snapshot_released = True

                def _resume_rebind():
                    with server._session_resume_lock:
                        session["transport"] = new_transport
                    rebound.set()

                thread = threading.Thread(target=_resume_rebind)
                thread.start()
                assert rebound.wait(timeout=1)
                thread.join(timeout=1)
            return False

    monkeypatch.setattr(server, "_sessions_lock", _SnapshotInterlock())
    server._sessions.clear()
    server._sessions["rebound"] = session
    try:
        reaped, detached = server._close_sessions_for_transport(old_transport)
        assert reaped == 0
        assert detached == 0
        assert server._sessions["rebound"] is session
        assert session["transport"] is new_transport
        assert teardowns == []
        assert reaps == []
    finally:
        server._sessions.clear()


def test_session_create_records_close_on_disconnect_flag(monkeypatch):
    monkeypatch.setattr(server, "_start_agent_build", lambda sid, session: None)
    server._sessions.clear()
    try:
        on = server.handle_request(
            {"id": "1", "method": "session.create", "params": {"close_on_disconnect": True}}
        )["result"]["session_id"]
        off = server.handle_request(
            {"id": "2", "method": "session.create", "params": {}}
        )["result"]["session_id"]
        assert server._sessions[on]["close_on_disconnect"]
        assert not server._sessions[off]["close_on_disconnect"]
    finally:
        server._sessions.clear()



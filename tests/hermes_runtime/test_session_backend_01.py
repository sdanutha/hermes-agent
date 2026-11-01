

import json


import logging


import os




import sys


import threading


import time


import types




from pathlib import Path




import pytest


from hermes_constants import reset_hermes_home_override, set_hermes_home_override


from hermes_cli.active_sessions import active_session_registry_snapshot






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


def test_session_slot_is_claimed_on_first_turn_not_on_create(monkeypatch, tmp_path):
    home = tmp_path / ".hermes"
    home.mkdir()
    (home / "config.yaml").write_text("max_concurrent_sessions: 1\n", encoding="utf-8")
    token = set_hermes_home_override(home)

    def _clear_server_sessions():
        for session in list(server._sessions.values()):
            server._teardown_session(session)
        server._sessions.clear()

    try:
        server._cfg_cache = None
        server._cfg_sig = None
        server._cfg_path = None
        _clear_server_sessions()
        monkeypatch.setattr(server, "_start_agent_build", lambda *args, **kwargs: None)
        monkeypatch.setattr(server, "_completion_cwd", lambda params=None: str(tmp_path))

        # Opening a chat must NOT take a slot. Every tile paint and every
        # background reconnect-resume calls session.create, and an unprompted
        # draft has no DB row and is filtered out of the sidebar — so a slot
        # held here is invisible to the user while still starving the other
        # surfaces that share this cap.
        first = server._methods["session.create"]("r1", {"cols": 80})
        second = server._methods["session.create"]("r2", {"cols": 80})
        assert "result" in first and "result" in second
        sid = first["result"]["session_id"]
        other = second["result"]["session_id"]
        assert active_session_registry_snapshot() == []

        # The first turn is what claims the slot, and is re-entrant.
        assert server._ensure_active_session_slot(sid, server._sessions[sid]) is None
        assert server._ensure_active_session_slot(sid, server._sessions[sid]) is None
        assert len(active_session_registry_snapshot()) == 1

        blocked = server._ensure_active_session_slot(other, server._sessions[other])
        assert blocked  # refusal reason returned; exact wording not pinned

        closed = server._methods["session.close"]("r3", {"session_id": sid})
        assert closed["result"]["closed"] is True
        assert active_session_registry_snapshot() == []

        assert server._ensure_active_session_slot(other, server._sessions[other]) is None
    finally:
        _clear_server_sessions()
        server._cfg_cache = None
        server._cfg_sig = None
        server._cfg_path = None
        reset_hermes_home_override(token)


def test_session_context_uses_session_cwd(monkeypatch, tmp_path):
    """Desktop/TUI sessions must pin the agent cwd per session.

    The gateway process itself is often launched from apps/desktop in dev, so
    falling back to os.getcwd() makes agents answer from the desktop app folder
    even when the sidebar/session cwd is a real project.
    """
    from agent.runtime_cwd import resolve_agent_cwd

    sid = "cwd-sid"
    session_key = "cwd-key"
    project = tmp_path / "project"
    project.mkdir()
    (project / ".git").mkdir()
    launcher = tmp_path / "apps" / "desktop"
    launcher.mkdir(parents=True)

    server._sessions[sid] = {"session_key": session_key, "cwd": str(project)}
    monkeypatch.delenv("TERMINAL_CWD", raising=False)
    monkeypatch.chdir(launcher)

    tokens = server._set_session_context(session_key)
    try:
        assert resolve_agent_cwd() == project
    finally:
        server._clear_session_context(tokens)
        server._sessions.pop(sid, None)


def test_handoff_fail_marks_only_inflight_rows(monkeypatch):
    class DbContext:
        def __init__(self, db):
            self.db = db

        def __enter__(self):
            return self.db

        def __exit__(self, *_args):
            return False

    class FakeDb:
        def __init__(self, state):
            self.state = state
            self.failed_with = None

        def get_handoff_state(self, _key):
            return {"state": self.state, "platform": "telegram", "error": None}

        def fail_handoff(self, _key, error):
            self.failed_with = error
            self.state = "failed"

    sid = "rt-handoff"
    server._sessions[sid] = {"session_key": "stored-handoff"}
    try:
        pending = FakeDb("pending")
        monkeypatch.setattr(server, "_session_db", lambda _session: DbContext(pending))
        result = server._methods["handoff.fail"]("r1", {"session_id": sid, "error": "timed out"})
        assert result["result"] == {"failed": True, "state": "failed"}
        assert pending.failed_with == "timed out"

        completed = FakeDb("completed")
        monkeypatch.setattr(server, "_session_db", lambda _session: DbContext(completed))
        result = server._methods["handoff.fail"]("r2", {"session_id": sid, "error": "late timeout"})
        assert result["result"] == {"failed": False, "state": "completed"}
        assert completed.failed_with is None
    finally:
        server._sessions.pop(sid, None)


def test_dashboard_process_isolation_config_coerces_raw_values():
    cfg = {
        "dashboard": {
            "turn_isolation": "yes",
            "compute_host_heartbeat_secs": "30",
            "compute_host_respawn_max": "0",
        }
    }

    assert server._load_dashboard_process_isolation_config(cfg) == {
        "turn_isolation": True,
        "compute_host_heartbeat_secs": 30,
        "compute_host_respawn_max": 0,
    }

    malformed = {"dashboard": "enabled"}
    assert server._load_dashboard_process_isolation_config(malformed) == {
        "turn_isolation": False,
        "compute_host_heartbeat_secs": 15,
        "compute_host_respawn_max": 3,
    }


def test_prompt_submit_dispatches_to_compute_host_when_turn_isolation_enabled(monkeypatch):
    class FakeSupervisor:
        def __init__(self):
            self.frames = []
            self.callback = None

        def submit_turn(self, frame, *, on_complete=None):
            self.frames.append(frame)
            self.callback = on_complete
            return frame["request_id"]

    fake_supervisor = FakeSupervisor()
    seed_history = [{"role": "user", "content": "previous"}]
    server._sessions["iso-sid"] = _session(history=list(seed_history))
    server._sessions["iso-sid"]["agent"] = None
    server._sessions["iso-sid"]["agent_ready"] = threading.Event()
    parent_writes = {"ensure_session": 0, "persist_seed": 0}
    monkeypatch.setattr(
        server,
        "_load_cfg",
        lambda: {"dashboard": {"turn_isolation": True}},
    )
    monkeypatch.setattr(
        server,
        "_ensure_session_db_row",
        lambda _session: parent_writes.__setitem__(
            "ensure_session", parent_writes["ensure_session"] + 1
        ),
    )
    monkeypatch.setattr(
        server,
        "_persist_branch_seed",
        lambda _session: parent_writes.__setitem__(
            "persist_seed", parent_writes["persist_seed"] + 1
        ),
    )
    monkeypatch.setattr(server, "_get_compute_host_supervisor", lambda _cfg=None: fake_supervisor)

    try:
        resp = server.handle_request(
            {
                "id": "submit",
                "method": "prompt.submit",
                "params": {"session_id": "iso-sid", "text": "hello"},
            }
        )
        assert resp["result"] == {"status": "streaming", "turn_isolation": True}
        assert fake_supervisor.frames[0]["type"] == "turn.start"
        assert fake_supervisor.frames[0]["sid"] == "iso-sid"
        assert fake_supervisor.frames[0]["text"] == "hello"
        assert fake_supervisor.frames[0]["history"] == seed_history
        assert server._sessions["iso-sid"]["history"] == seed_history
        assert parent_writes == {"ensure_session": 0, "persist_seed": 0}
        assert server._sessions["iso-sid"]["running"] is True

        fake_supervisor.callback(
            {
                "type": "turn.end",
                "sid": "iso-sid",
                "request_id": "submit",
                "history_version": 1,
            }
        )
        assert server._sessions["iso-sid"]["running"] is False
        assert server._sessions["iso-sid"]["history_version"] == 1
    finally:
        server._sessions.pop("iso-sid", None)


def test_compute_host_explicit_images_do_not_clear_later_attachment(monkeypatch):
    class _Supervisor:
        def submit_turn(self, _frame, *, on_complete=None):
            session["attached_images"].append("/tmp/c.png")

    session = _session(attached_images=[])
    monkeypatch.setattr(server, "_get_compute_host_supervisor", lambda _cfg=None: _Supervisor())

    response = server._submit_prompt_to_compute_host(
        "r1", "sid", session, "B", image_paths=["/tmp/b.png"]
    )

    assert response["result"]["status"] == "streaming"
    assert session["attached_images"] == ["/tmp/c.png"]


def test_prompt_submit_unknown_session_logs_warning(caplog):
    """A submit against a reaped runtime id must leave a diagnosable trace.

    Regression for #90428: messages sent into a session whose in-memory
    runtime was detached on WS disconnect and orphan-reaped vanished
    silently — the 4001 was never logged, so "request arrived and was
    rejected" was indistinguishable from "request never arrived".
    """
    for session in list(server._sessions.values()):
        server._teardown_session(session)
    server._sessions.clear()

    with caplog.at_level(logging.WARNING, logger="hermes_runtime.server"):
        resp = _dispatch_sync(
            {
                "id": "r1",
                "method": "prompt.submit",
                "params": {"session_id": "gone-sid", "text": "hello"},
            }
        )

    assert resp["id"] == "r1"
    assert resp["error"]["code"] == 4001
    # The runtime id and method name must be in the warning so a client looping
    # on a stale runtime id is attributable from logs (exact wording not pinned).
    assert any(
        rec.levelno >= logging.WARNING
        and "gone-sid" in rec.getMessage()
        and "prompt.submit" in rec.getMessage()
        for rec in caplog.records
    )


def test_prompt_submit_fails_open_inline_when_compute_host_dispatch_breaks(monkeypatch):
    class _BrokenSupervisor:
        def submit_turn(self, frame, *, on_complete=None):
            if on_complete is not None:
                on_complete(
                    {
                        "type": "turn.error",
                        "request_id": frame["request_id"],
                        "reason": "send_failed",
                        "message": "broken pipe",
                    }
                )
            raise BrokenPipeError("broken pipe")

    class _ImmediateThread:
        def __init__(self, target=None, **_kwargs):
            self._target = target

        def start(self):
            assert self._target is not None
            self._target()

    session = _session(agent=None, agent_ready=threading.Event())
    server._sessions["iso-fallback"] = session
    inline_calls = []
    monkeypatch.setattr(server, "_load_cfg", lambda: {"dashboard": {"turn_isolation": True}})
    monkeypatch.setattr(server, "_get_compute_host_supervisor", lambda _cfg=None: _BrokenSupervisor())
    monkeypatch.setattr(server, "_ensure_session_db_row", lambda _session: None)
    monkeypatch.setattr(server, "_persist_branch_seed", lambda _session: None)
    monkeypatch.setattr(server, "_start_agent_build", lambda _sid, _session: None)
    monkeypatch.setattr(server, "_wait_agent", lambda _session, _rid: None)
    # The deferred inline-fallback thread now waits via the patient variant.
    monkeypatch.setattr(server, "_wait_agent_for_prompt", lambda _session, _rid, _sid: None)
    monkeypatch.setattr(
        server,
        "_run_prompt_submit",
        lambda rid, sid, _session, text, **_kwargs: inline_calls.append((rid, sid, text)),
    )
    monkeypatch.setattr(server.threading, "Thread", _ImmediateThread)

    try:
        resp = server.handle_request(
            {
                "id": "fallback-turn",
                "method": "prompt.submit",
                "params": {"session_id": "iso-fallback", "text": "hello"},
            }
        )
    finally:
        server._sessions.pop("iso-fallback", None)

    assert resp == {
        "jsonrpc": "2.0",
        "id": "fallback-turn",
        "result": {"status": "streaming"},
    }
    assert inline_calls == [("fallback-turn", "iso-fallback", "hello")]
    assert session.get("_compute_host_active") is not True


def test_compute_host_turn_end_updates_metadata_mirror(monkeypatch):
    # _session_info embeds get_update_result(), whose value flips whenever the
    # background update-check thread happens to finish. This test compares two
    # snapshots taken at different times, so pin the value to keep it
    # deterministic regardless of how long the preceding tests ran.
    import hermes_cli.banner as _banner

    monkeypatch.setattr(_banner, "get_update_result", lambda timeout=0.5: None)
    session = _session(
        agent=None,
        agent_ready=threading.Event(),
        history=[{"role": "user", "content": "serving process must not read this"}],
        _compute_host_active=True,
    )
    server._sessions["iso-sid"] = session
    emitted = []
    monkeypatch.setattr(server, "_emit", lambda event, sid, payload=None: emitted.append((event, sid, payload)))

    try:
        server._on_compute_host_turn_done(
            "turn-1",
            "iso-sid",
            session,
            {
                "type": "turn.end",
                "sid": "iso-sid",
                "request_id": "turn-1",
                "session_key": "rotated-session-key",
                "history_version": 4,
                "message_count": 3,
                "session_info": {
                    "model": "host-model",
                    "provider": "host-provider",
                    "system_prompt": "host system prompt",
                    "tools": {"core": ["terminal"]},
                    "usage": {"total": 140, "context_used": 80, "context_max": 1000},
                },
            },
        )

        assert session["session_key"] == "rotated-session-key"
        assert session["history_version"] == 4
        assert session["_metadata_mirror"]["model"] == "host-model"
        info = server._session_info(None, session)
        assert info["model"] == "host-model"
        assert info["provider"] == "host-provider"
        assert info["system_prompt"] == "host system prompt"
        assert info["tools"] == {"core": ["terminal"]}
        assert info["usage"]["total"] == 140
        assert "credential_warning" not in info
        assert emitted[-1] == ("session.info", "iso-sid", info)
    finally:
        server._sessions.pop("iso-sid", None)


def test_compute_host_open_request_survives_activation_and_proxies_locks_and_responses(monkeypatch):
    """A host-owned server request (batch clarify) is mirrored by the parent so `open_requests` replays it;
    `clarify.lock` and the client's response frame are relayed to the child that owns the wait."""
    class _Supervisor:
        def __init__(self):
            self.responses = []

        def respond(self, sid, params, *, timeout=15.0):
            self.responses.append((sid, dict(params), timeout))
            lock = params.get("lock") or {}
            remaining = ["q1"] if lock.get("question_id") == "q0" else []
            return {"type": "respond.ack", "response": {"result": {"status": "ok", "remaining": remaining}}}

    sid = "host-clarify"
    supervisor = _Supervisor()
    session = _session(agent=None, agent_ready=threading.Event(), _compute_host_active=True)
    server._sessions[sid] = session
    monkeypatch.setattr(server, "_load_cfg", lambda: {"dashboard": {"turn_isolation": True}})
    monkeypatch.setattr(server, "_get_compute_host_supervisor", lambda _cfg=None: supervisor)
    monkeypatch.setattr(server, "write_json", lambda _message: True)

    questions = [{"qid": "q0", "question": "First?", "choices": ["a"]}, {"qid": "q1", "question": "Second?", "choices": ["b"]}]
    try:
        server._relay_compute_host_rpc({"jsonrpc": "2.0", "id": "srq-host", "method": "clarify",
                                        "params": {"session_id": sid, "questions": questions}})

        activated = server._live_session_payload(sid, session)
        assert activated["open_requests"] == [{"id": "srq-host", "method": "clarify",
                                               "params": {"session_id": sid, "questions": questions}}]

        response = server.handle_request({"id": "lock-q0", "method": "clarify.lock",
                                          "params": {"request_id": "srq-host", "question_id": "q0", "answer": "a"}})
        assert response["result"] == {"status": "ok", "remaining": ["q1"]}
        assert supervisor.responses == [(sid, {"lock": {"request_id": "srq-host", "question_id": "q0", "answer": "a"}}, 15.0)]
        assert server._live_session_payload(sid, session)["open_requests"][0]["params"]["answers"] == {"q0": "a"}

        # The client's response frame (cancel-all) is relayed to the child and clears the mirror.
        assert server.dispatch({"jsonrpc": "2.0", "id": "srq-host", "result": {}}) is None
        assert supervisor.responses[-1] == (sid, {"frame": {"jsonrpc": "2.0", "id": "srq-host", "result": {}}}, 15.0)
        assert "open_requests" not in server._live_session_payload(sid, session)
    finally:
        server._sessions.pop(sid, None)


def test_compute_host_interrupt_forwards_when_parent_running_mirror_is_stale(monkeypatch):
    """The host, not the parent's mirrored running flag, owns interruption."""
    interrupted = []

    class _Supervisor:
        def interrupt(self, sid, *, request_id=None):
            interrupted.append((sid, request_id))

    sid = "host-stale-running"
    server._sessions[sid] = _session(
        agent=None,
        agent_ready=threading.Event(),
        _compute_host_active=True,
        running=False,
    )
    monkeypatch.setattr(server, "_load_cfg", lambda: {"dashboard": {"turn_isolation": True}})
    monkeypatch.setattr(server, "_get_compute_host_supervisor", lambda _cfg=None: _Supervisor())

    try:
        response = server.handle_request(
            {"id": "interrupt", "method": "session.interrupt", "params": {"session_id": sid}}
        )
        assert response["result"] == {"status": "interrupted", "turn_isolation": True}
        assert interrupted == [(sid, "interrupt-interrupt")]
    finally:
        server._sessions.pop(sid, None)


def test_compute_host_interrupt_skips_lazy_session_with_no_hosted_turn(monkeypatch):
    """A lazy session that never submitted a hosted turn must not spawn a host.

    ``HostSupervisor.interrupt()`` calls ``start()``, so forwarding the
    interrupt unconditionally would launch a compute-host child just to
    deliver an interrupt for a session with no work in it.
    """
    class _Supervisor:
        def interrupt(self, sid, *, request_id=None):  # pragma: no cover - must not run
            raise AssertionError("interrupt must not be forwarded for idle lazy sessions")

    sid = "lazy-idle"
    session = _session(
        agent_ready=threading.Event(),
        running=False,
    )
    session["agent"] = None  # _session() substitutes a namespace for None
    server._sessions[sid] = session
    monkeypatch.setattr(server, "_load_cfg", lambda: {"dashboard": {"turn_isolation": True}})
    monkeypatch.setattr(server, "_get_compute_host_supervisor", lambda _cfg=None: _Supervisor())

    try:
        response = server.handle_request(
            {"id": "interrupt", "method": "session.interrupt", "params": {"session_id": sid}}
        )
        assert response["result"] == {"status": "interrupted", "turn_isolation": True}
    finally:
        server._sessions.pop(sid, None)


def test_slash_exec_compress_flag_on_applies_host_control_mirror(monkeypatch):
    class _ExplodingWorker:
        def __init__(self, *args, **kwargs):
            raise AssertionError("slash worker should not run for isolated /compress")

    class _FakeSupervisor:
        def __init__(self):
            self.controls = []

        def control(self, sid, *, route_name, payload=None, wait=True, timeout=30.0, on_late_ack=None):
            self.controls.append((sid, route_name, dict(payload or {}), wait))
            return {
                "type": "control.ack",
                "sid": sid,
                "request_id": (payload or {}).get("request_id", "control-1"),
                "route_name": route_name,
                "output": "Compressed 4 → 2 messages",
                "session_key": "host-rotated-key",
                "history_version": 9,
                "message_count": 2,
                "session_info": {
                    "model": "host-model",
                    "provider": "host-provider",
                    "usage": {"total": 42},
                },
            }

    fake = _FakeSupervisor()
    session = _session(agent=None, agent_ready=threading.Event(), _compute_host_active=True)
    server._sessions["sid"] = session
    monkeypatch.setattr(server, "_load_cfg", lambda: {"dashboard": {"turn_isolation": True}})
    monkeypatch.setattr(server, "_get_compute_host_supervisor", lambda _cfg=None: fake)
    monkeypatch.setattr(server, "_SlashWorker", _ExplodingWorker)
    monkeypatch.setattr(server, "_compress_session_history", lambda *a, **k: (_ for _ in ()).throw(AssertionError("parent compressed")))
    monkeypatch.setattr(server, "_sync_session_key_after_compress", lambda *a, **k: (_ for _ in ()).throw(AssertionError("parent identity guard ran")))

    try:
        resp = server.handle_request(
            {
                "id": "1",
                "method": "slash.exec",
                "params": {"command": "compress focus", "session_id": "sid"},
            }
        )
    finally:
        server._sessions.pop("sid", None)

    assert resp["result"]["output"] == "Compressed 4 → 2 messages"
    assert fake.controls[0][1] == "slash.compress"
    assert fake.controls[0][2]["command"] == "/compress focus"
    assert session["session_key"] == "host-rotated-key"
    assert session["history_version"] == 9
    assert server._session_info(None, session)["model"] == "host-model"


def test_prompt_submit_golden_transcript_matches_flag_off_and_on(monkeypatch):
    class _ImmediateThread:
        def __init__(self, target=None, daemon=None, **_kwargs):
            self._target = target

        def start(self):
            assert self._target is not None
            self._target()

    class _Agent:
        model = "gold-model"
        provider = "gold-provider"
        session_id = "session-key"
        session_input_tokens = 10
        session_output_tokens = 5
        session_prompt_tokens = 10
        session_completion_tokens = 5
        session_total_tokens = 15
        session_api_calls = 1
        context_compressor = None

        def clear_interrupt(self):
            return None

        def run_conversation(self, prompt, conversation_history=None, stream_callback=None, **_kwargs):
            if stream_callback is not None:
                stream_callback("hi")
            return {
                "final_response": "hi",
                "messages": [
                    {"role": "user", "content": prompt},
                    {"role": "assistant", "content": "hi"},
                ],
            }

    fixed_info = {"model": "gold-model", "provider": "gold-provider", "usage": {"total": 15}}
    usage = server._get_usage(_Agent())
    monkeypatch.setattr(server.threading, "Thread", _ImmediateThread)
    monkeypatch.setattr(server, "_ensure_session_db_row", lambda _session: None)
    monkeypatch.setattr(server, "_persist_branch_seed", lambda _session: None)
    monkeypatch.setattr(server, "_session_info", lambda _agent, _session=None: dict(fixed_info))
    monkeypatch.setattr(server, "make_stream_renderer", lambda _cols: None)
    monkeypatch.setattr(server, "render_message", lambda _raw, _cols: None)
    fake_title = types.ModuleType("agent.title_generator")
    fake_title.maybe_auto_title = lambda *args, **kwargs: None
    monkeypatch.setitem(sys.modules, "agent.title_generator", fake_title)

    def run_flag_off():
        events = []
        monkeypatch.setattr(server, "_emit", lambda event, sid, payload=None: events.append((event, sid, payload)))
        monkeypatch.setattr(server, "_load_cfg", lambda: {"dashboard": {"turn_isolation": False}})
        server._sessions["sid"] = _session(
            agent=_Agent(), model_override={"model": "gold-model", "provider": "gold-provider"}
        )
        try:
            response = server.handle_request(
                {"id": "turn-1", "method": "prompt.submit", "params": {"session_id": "sid", "text": "hello"}}
            )
            assert response["result"]["status"] == "streaming"
            return events
        finally:
            server._sessions.pop("sid", None)

    def run_flag_on():
        events = []
        monkeypatch.setattr(server, "_emit", lambda event, sid, payload=None: events.append((event, sid, payload)))
        monkeypatch.setattr(server, "_load_cfg", lambda: {"dashboard": {"turn_isolation": True}})

        class _FakeSupervisor:
            def submit_turn(self, frame, *, on_complete=None):
                sid = frame["sid"]
                server._emit("message.start", sid)
                server._emit("message.delta", sid, {"text": "hi"})
                server._emit("message.complete", sid, {"text": "hi", "usage": usage, "status": "complete"})
                server._emit("session.info", sid, dict(fixed_info))
                if on_complete is not None:
                    on_complete(
                        {
                            "type": "turn.end",
                            "sid": sid,
                            "request_id": frame["request_id"],
                            "session_key": "session-key",
                            "history_version": 1,
                            "message_count": 2,
                            "session_info": dict(fixed_info),
                            "session_info_emitted": True,
                        }
                    )
                return frame["request_id"]

        monkeypatch.setattr(server, "_get_compute_host_supervisor", lambda _cfg=None: _FakeSupervisor())
        session = _session(
            agent=None,
            agent_ready=threading.Event(),
            _compute_host_active=True,
            model_override={"model": "gold-model", "provider": "gold-provider"},
        )
        session["agent"] = None
        server._sessions["sid"] = session
        try:
            response = server.handle_request(
                {"id": "turn-1", "method": "prompt.submit", "params": {"session_id": "sid", "text": "hello"}}
            )
            assert response["result"]["status"] == "streaming"
            return events
        finally:
            server._sessions.pop("sid", None)

    assert run_flag_on() == run_flag_off()


def test_session_context_explicit_cwd_for_ephemeral_task(monkeypatch, tmp_path):
    """Background/preview tasks use ephemeral ids absent from `_sessions`, so the
    parent workspace is passed explicitly; it must pin instead of clearing back
    to the gateway launch dir."""
    from agent.runtime_cwd import resolve_agent_cwd

    project = tmp_path / "project"
    project.mkdir()
    launcher = tmp_path / "apps" / "desktop"
    launcher.mkdir(parents=True)

    monkeypatch.delenv("TERMINAL_CWD", raising=False)
    monkeypatch.chdir(launcher)

    tokens = server._set_session_context("bg_deadbe", cwd=str(project))
    try:
        assert resolve_agent_cwd() == project
    finally:
        server._clear_session_context(tokens)


def test_profile_scoped_mcp_discovery_uses_target_home(monkeypatch, tmp_path):
    """MCP discovery must start under the selected profile's HERMES_HOME."""
    from hermes_cli import mcp_startup
    from hermes_constants import get_hermes_home
    from hermes_runtime import entry

    profile_home = tmp_path / "profiles" / "sheepyr"
    profile_home.mkdir(parents=True)

    (profile_home / "config.yaml").write_text(
        "mcp_servers:\n"
        "  bluesky_sheepyr:\n"
        "    command: test-command\n",
        encoding="utf-8",
    )

    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "default"))
    token = set_hermes_home_override(str(profile_home))

    seen = []

    monkeypatch.setattr(mcp_startup, "_mcp_discovery_started", set())
    monkeypatch.setattr(mcp_startup, "_mcp_discovery_thread", {})
    # ensure_mcp_discovery_started flips this module global; monkeypatch it so
    # the enablement doesn't leak into sibling tests in this file.
    monkeypatch.setattr(entry, "_mcp_discovery_enabled", False)
    monkeypatch.setattr(
        mcp_startup,
        "_discover_mcp_tools_without_interactive_oauth",
        lambda: seen.append(str(get_hermes_home())),
    )

    try:
        entry.ensure_mcp_discovery_started()
        thread = mcp_startup._current_home_thread()
        assert thread is not None
        thread.join(timeout=2)
    finally:
        reset_hermes_home_override(token)

    assert seen == [str(profile_home)]


def test_profile_scoped_agent_build_starts_mcp_discovery_in_profile_home(
    monkeypatch, tmp_path
):
    """Agent construction must start MCP discovery under the selected profile."""
    import threading
    import uuid

    from hermes_constants import get_hermes_home

    profile_home = tmp_path / "profiles" / "sheepyr"
    profile_home.mkdir(parents=True)

    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "default"))

    seen = []
    built = threading.Event()

    monkeypatch.setattr(
        server,
        "_make_agent",
        lambda *args, **kwargs: built.set()
        or type("Agent", (), {"model": "test"})(),
    )
    monkeypatch.setattr(
        "hermes_runtime.entry.ensure_mcp_discovery_started",
        lambda: seen.append(str(get_hermes_home())),
    )
    monkeypatch.setattr(server, "_wire_callbacks", lambda _sid: None)
    monkeypatch.setattr(server, "_SlashWorker", lambda *args: None)
    monkeypatch.setattr(server, "_attach_worker", lambda *args: None)
    monkeypatch.setattr(server, "_config_model_target", lambda: ("", ""))
    # CI runs this huge file serially under load; a prior session's _build can
    # still be finishing (session.info emit) when the next test starts, so a
    # 2s Event wait flakes. Unique sid + longer bound; still fail closed.
    monkeypatch.setattr(server, "_start_notification_poller", lambda *a, **k: None)
    monkeypatch.setattr(server, "_schedule_mcp_late_refresh", lambda *a, **k: None)
    monkeypatch.setattr(server, "_emit", lambda *a, **k: None)

    ready = threading.Event()
    sid = f"test-sid-{uuid.uuid4().hex[:8]}"
    session = {
        "agent_ready": ready,
        "session_key": f"test-key-{uuid.uuid4().hex[:8]}",
        "profile_home": str(profile_home),
    }

    server._sessions[sid] = session
    try:
        server._start_agent_build(sid, session)
        assert built.wait(timeout=15), "agent build thread never called _make_agent"
        assert ready.wait(timeout=5), "agent_ready never set after build"
    finally:
        server._sessions.pop(sid, None)

    assert seen == [str(profile_home)]


def test_profile_scoped_agent_build_installs_secret_scope(monkeypatch, tmp_path):
    """Agent construction must install the selected profile's secret scope.

    Without it, get_secret() falls through to process os.environ, so a session
    "switched" to profile X resolves credentials from the LAUNCH profile's
    .env (#67605 item 2).
    """
    import threading
    import uuid

    from agent.secret_scope import current_secret_scope

    profile_home = tmp_path / "profiles" / "grace"
    profile_home.mkdir(parents=True)
    (profile_home / ".env").write_text(
        "PROXMOX_TOKEN=grace-secret\n", encoding="utf-8"
    )

    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "default"))

    scopes = []
    built = threading.Event()

    def _fake_make_agent(*args, **kwargs):
        scope = current_secret_scope()
        scopes.append(dict(scope) if scope else None)
        built.set()
        return type("Agent", (), {"model": "test"})()

    monkeypatch.setattr(server, "_make_agent", _fake_make_agent)
    monkeypatch.setattr(
        "hermes_runtime.entry.ensure_mcp_discovery_started", lambda: None
    )
    monkeypatch.setattr(server, "_wire_callbacks", lambda _sid: None)
    monkeypatch.setattr(server, "_SlashWorker", lambda *args: None)
    monkeypatch.setattr(server, "_attach_worker", lambda *args: None)
    monkeypatch.setattr(server, "_config_model_target", lambda: ("", ""))
    # Same CI flake class as the MCP profile-home test: bound wait + less work
    # on the build thread (no poller / late MCP refresh / session.info emit).
    monkeypatch.setattr(server, "_start_notification_poller", lambda *a, **k: None)
    monkeypatch.setattr(server, "_schedule_mcp_late_refresh", lambda *a, **k: None)
    monkeypatch.setattr(server, "_emit", lambda *a, **k: None)

    ready = threading.Event()
    sid = f"test-secret-sid-{uuid.uuid4().hex[:8]}"
    session = {
        "agent_ready": ready,
        "session_key": f"test-secret-key-{uuid.uuid4().hex[:8]}",
        "profile_home": str(profile_home),
    }

    server._sessions[sid] = session
    try:
        server._start_agent_build(sid, session)
        assert built.wait(timeout=15), "agent build thread never called _make_agent"
        assert ready.wait(timeout=5), "agent_ready never set after build"
    finally:
        server._sessions.pop(sid, None)

    assert scopes == [{"PROXMOX_TOKEN": "grace-secret"}]


def test_profile_configured_cwd_reads_target_profile(tmp_path):
    """A profile's own terminal.cwd is read from its config.yaml."""
    project = tmp_path / "proj"
    project.mkdir()
    home = _write_profile_cfg(tmp_path / "home", str(project))
    assert server._profile_configured_cwd(home) == str(project)


def test_profile_configured_cwd_skips_placeholders_and_missing(tmp_path):
    """Placeholder values, missing config, and bad paths fall through to None."""
    assert server._profile_configured_cwd(None) is None
    assert server._profile_configured_cwd(tmp_path / "nope") is None
    for placeholder in (".", "auto", "cwd", ""):
        home = _write_profile_cfg(tmp_path / placeholder.strip("."), placeholder)
        assert server._profile_configured_cwd(home) is None
    home = _write_profile_cfg(tmp_path / "ghost", str(tmp_path / "does-not-exist"))
    assert server._profile_configured_cwd(home) is None


def test_completion_cwd_prefers_profile_over_stale_env(monkeypatch, tmp_path):
    """Issue #40334: a new session bound to another profile must use THAT
    profile's terminal.cwd, not the launch profile's stale TERMINAL_CWD."""
    profile_b = tmp_path / "ef-design"
    profile_b.mkdir()
    home = _write_profile_cfg(tmp_path / "home-b", str(profile_b))
    stale = tmp_path / "mahjong"
    stale.mkdir()

    monkeypatch.setenv("TERMINAL_CWD", str(stale))
    monkeypatch.setattr(server, "_hermes_home", tmp_path / "launch-home")
    monkeypatch.setattr(server, "_profile_home", lambda name: home if name else None)

    assert server._completion_cwd({"profile": "ef-design"}) == str(profile_b)
    # No profile and no launch config → fallback to the launch env var.
    assert server._completion_cwd({}) == str(stale)


def test_completion_cwd_prefers_launch_config_over_stale_env(monkeypatch, tmp_path):
    """Dashboard /chat's launch-profile in-memory gateway must honor config.

    The embedded Node TUI child gets TERMINAL_CWD from the dashboard PTY bridge,
    but the default-profile chat attaches to the dashboard process's already
    running in-memory gateway. That process may not have TERMINAL_CWD in its own
    environment (or has a stale one), so config.yaml is read directly and wins
    over the process env before falling back to the launch directory.
    """
    configured = tmp_path / "omni"
    configured.mkdir()
    stale = tmp_path / "hermes-agent"
    stale.mkdir()
    launch_home = _write_profile_cfg(tmp_path / "launch-home", str(configured))

    monkeypatch.setenv("TERMINAL_CWD", str(stale))
    monkeypatch.setattr(server, "_hermes_home", launch_home)
    monkeypatch.setattr(server, "_profile_home", lambda _name: None)

    assert server._completion_cwd({}) == str(configured)


def test_default_session_cwd_prefers_launch_config(monkeypatch, tmp_path):
    """A freshly created / resumed session with no explicit cwd lands in the
    configured terminal.cwd, not os.getcwd(), even when the in-memory gateway
    process env carries a stale TERMINAL_CWD."""
    configured = tmp_path / "workspace"
    configured.mkdir()
    stale = tmp_path / "launch-dir"
    stale.mkdir()
    launch_home = _write_profile_cfg(tmp_path / "launch-home", str(configured))

    monkeypatch.setenv("TERMINAL_CWD", str(stale))
    monkeypatch.setattr(server, "_hermes_home", launch_home)

    assert server._default_session_cwd() == str(configured)

    # No launch config → fall back to the process env var.
    (launch_home / "config.yaml").write_text("{}", encoding="utf-8")
    assert server._default_session_cwd() == str(stale)


def test_completion_cwd_explicit_cwd_wins_over_profile(monkeypatch, tmp_path):
    """An explicit client-provided cwd still beats the profile config."""
    explicit = tmp_path / "explicit"
    explicit.mkdir()
    profile_b = tmp_path / "configured"
    profile_b.mkdir()
    home = _write_profile_cfg(tmp_path / "home-c", str(profile_b))

    monkeypatch.setattr(server, "_profile_home", lambda name: home if name else None)
    result = server._completion_cwd(
        {"cwd": str(explicit), "cwd_explicit": True, "profile": "ef-design"}
    )
    assert result == str(explicit)


def test_completion_cwd_profile_overrides_inherited_workspace(monkeypatch, tmp_path):
    """Issue #52589: the desktop seeds a new chat's cwd from its app-global workspace
    (the launch profile's configured directory) — that inherited default must NOT
    override the target profile's own ``terminal.cwd``."""
    launch_ws = tmp_path / "workspace"
    launch_ws.mkdir()
    profile_ws = tmp_path / "products"
    profile_ws.mkdir()
    home = _write_profile_cfg(tmp_path / "home-dev", str(profile_ws))

    monkeypatch.setattr(server, "_profile_home", lambda name: home if name else None)
    # No cwd_explicit: the client cwd is the inherited app-global workspace.
    assert (
        server._completion_cwd({"profile": "dev", "cwd": str(launch_ws)}) == str(profile_ws)
    )


def test_completion_cwd_explicit_pick_wins_over_profile(monkeypatch, tmp_path):
    """Issue #52589 regression guard: a deliberate workspace pick (``cwd_explicit``)
    still beats the profile's configured ``terminal.cwd``."""
    explicit = tmp_path / "explicit-lane"
    explicit.mkdir()
    profile_ws = tmp_path / "products"
    profile_ws.mkdir()
    home = _write_profile_cfg(tmp_path / "home-dev", str(profile_ws))

    monkeypatch.setattr(server, "_profile_home", lambda name: home if name else None)
    assert (
        server._completion_cwd(
            {"profile": "dev", "cwd": str(explicit), "cwd_explicit": True}
        )
        == str(explicit)
    )


def test_completion_cwd_inherited_workspace_without_profile_cfg_kept(monkeypatch, tmp_path):
    """An inherited workspace stays when the target profile has NO configured
    terminal.cwd — the profile override only applies when one exists (#52589)."""
    launch_ws = tmp_path / "workspace"
    launch_ws.mkdir()
    no_cfg_home = tmp_path / "home-plain"
    no_cfg_home.mkdir()
    (no_cfg_home / "config.yaml").write_text("{}", encoding="utf-8")

    monkeypatch.setattr(server, "_profile_home", lambda name: no_cfg_home if name else None)
    assert server._completion_cwd({"profile": "plain", "cwd": str(launch_ws)}) == str(launch_ws)


def test_terminal_task_cwd_local_backend_uses_session_cwd(monkeypatch, tmp_path):
    """A local terminal backend must keep host-validated session cwd behaviour."""
    project = tmp_path / "project"
    project.mkdir()
    monkeypatch.setenv("TERMINAL_ENV", "local")
    monkeypatch.delenv("TERMINAL_CWD", raising=False)

    assert server._terminal_task_cwd({"cwd": str(project)}) == str(project)


def test_terminal_task_cwd_ssh_uses_remote_path_unvalidated(monkeypatch):
    """SSH (non-local) backend: the configured remote cwd is used verbatim even
    though it does not exist on the local host. This is the jonbohz fix — host
    `isdir()` validation would otherwise discard the remote path and fall back
    to os.getcwd(), running commands against the wrong machine."""
    remote = "/home/jonboh/workspace/proj"  # does not exist on this host
    assert not os.path.isdir(remote)
    monkeypatch.setenv("TERMINAL_ENV", "ssh")
    monkeypatch.setenv("TERMINAL_CWD", remote)

    assert server._terminal_task_cwd({"cwd": "/some/host/dir"}) == remote


def test_terminal_task_cwd_ssh_falls_back_to_config(monkeypatch):
    """When TERMINAL_CWD is unset, the SSH path reads terminal.cwd from config."""
    remote = "/home/jonboh/workspace/from-config"
    monkeypatch.setenv("TERMINAL_ENV", "ssh")
    monkeypatch.delenv("TERMINAL_CWD", raising=False)
    monkeypatch.setattr(server, "_load_cfg", lambda: {"terminal": {"cwd": remote}})

    assert server._terminal_task_cwd({"cwd": "/some/host/dir"}) == remote


def test_terminal_task_cwd_ssh_sentinel_cwd_uses_remote_home(monkeypatch):
    """An SSH placeholder must not register the TUI host's session cwd."""
    monkeypatch.setenv("TERMINAL_ENV", "ssh")
    monkeypatch.setenv("TERMINAL_CWD", "auto")
    monkeypatch.setattr(server, "_load_cfg", lambda: {"terminal": {"cwd": "."}})

    assert server._terminal_task_cwd({"cwd": "/host/session/dir"}) == "~"


def test_write_json_serializes_concurrent_writes(monkeypatch):
    """Assert StdioTransport holds _stdout_lock across the full stream.write.

    The old char-by-char sleep mock made this test take long enough that
    leftover background write_json calls from earlier cases in this file
    could append an extra line (intermittent ``assert 9 == 8`` on CI/main).
    Match the WS concurrent-send check: count in-flight writes, and only
    assert on frames that carry this test's marker payload.
    """
    marker = "x" * 24
    active = 0
    max_active = 0
    gate = threading.Lock()
    frames: list[str] = []

    class RecordingStdout:
        def write(self, text: str) -> int:
            nonlocal active, max_active
            with gate:
                active += 1
                max_active = max(max_active, active)
            try:
                # Release the GIL while "in write" so a missing outer lock
                # would let another thread bump max_active above 1.
                time.sleep(0.01)
                frames.append(text)
            finally:
                with gate:
                    active -= 1
            return len(text)

        def flush(self) -> None:
            return None

    monkeypatch.setattr(server, "_real_stdout", RecordingStdout())

    barrier = threading.Barrier(8)

    def _worker(seq: int) -> None:
        barrier.wait(timeout=5)
        server.write_json({"seq": seq, "text": marker})

    threads = [threading.Thread(target=_worker, args=(i,)) for i in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=10)
        assert not t.is_alive()

    assert max_active == 1

    ours = []
    for frame in frames:
        assert frame.endswith("\n"), frame
        obj = json.loads(frame)
        if obj.get("text") == marker and "seq" in obj:
            ours.append(obj)

    assert {obj["seq"] for obj in ours} == set(range(8))
    assert len(ours) == 8


def test_write_json_returns_false_on_broken_pipe(monkeypatch):
    monkeypatch.setattr(server, "_real_stdout", _BrokenStdout())

    assert server.write_json({"ok": True}) is False


def test_write_json_unserializable_payload_becomes_error_frame(monkeypatch, caplog):
    """The stdio twin of the WS guard (#92506): an unserializable result must reach the Ink TUI /
    stdio bridge as a JSON-RPC error frame with the original id plus a log line, not kill the pool
    worker silently while the client waits forever."""
    import datetime
    import logging

    out = _ChunkyStdout()
    monkeypatch.setattr(server, "_real_stdout", out)
    with caplog.at_level(logging.ERROR, logger="hermes_runtime.transport"):
        assert server.write_json({"jsonrpc": "2.0", "id": "profiles",
                                  "result": {"created": datetime.datetime(2026, 8, 22)}}) is True
    frame = json.loads("".join(out.parts))
    assert frame["id"] == "profiles" and frame["error"]["code"] == -32603
    assert "datetime" in frame["error"]["message"]
    assert any(rec.levelno >= logging.ERROR for rec in caplog.records)


def test_write_json_drops_detached_ws_frames(monkeypatch):
    out = _ChunkyStdout()
    monkeypatch.setattr(server, "_real_stdout", out)
    server._sessions["detached-sid"] = {"transport": server._detached_ws_transport}
    try:
        assert server.write_json({
            "jsonrpc": "2.0",
            "method": "event",
            "params": {"session_id": "detached-sid", "type": "message.delta"},
        }) is False
        assert out.parts == []
    finally:
        server._sessions.pop("detached-sid", None)


@pytest.mark.parametrize("count", [0, 3])
def test_get_usage_projects_live_compression_count(count):
    agent = types.SimpleNamespace(
        context_compressor=types.SimpleNamespace(compression_count=count),
        model="test-model",
    )

    assert server._get_usage(agent)["compressions"] == count


def test_usage_ticker_emits_wrapped_usage_payload(monkeypatch):
    # The live ticker must nest the snapshot under a "usage" key, matching the
    # message.complete / session.info payloads the desktop & TUI handlers read
    # as payload.usage. Emitting the bare _get_usage() dict (payload.input/total
    # …) silently drops every live tick on the client side.
    events: list[tuple[str, str, dict]] = []
    monkeypatch.setattr(
        server, "_emit", lambda event_type, sid, payload: events.append((event_type, sid, payload))
    )
    snapshot = {"input": 1200, "total": 1280}
    monkeypatch.setattr(server, "_get_usage", lambda agent: dict(snapshot))

    stop, thread = server._start_usage_ticker("sess-1", object(), interval=0.01)
    # The dedup baseline is sampled synchronously inside _start_usage_ticker,
    # so this mutation is guaranteed to read as the first counter movement.
    snapshot["total"] = 2400
    try:
        deadline = time.time() + 1.0
        while not events and time.time() < deadline:
            time.sleep(0.01)
    finally:
        stop.set()
        thread.join(timeout=2.0)

    assert events, "ticker never emitted"
    event_type, sid, payload = events[0]
    assert event_type == "session.usage"
    assert sid == "sess-1"
    assert payload == {"usage": {"input": 1200, "total": 2400}}













import sys


import threading


import time


import types




from pathlib import Path




import pytest










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


def test_session_create_records_source(monkeypatch):
    monkeypatch.setattr(server, "_start_agent_build", lambda sid, session: None)
    server._sessions.clear()
    try:
        sid = server.handle_request(
            {"id": "1", "method": "session.create", "params": {"source": "tool"}}
        )["result"]["session_id"]
        assert server._sessions[sid]["source"] == "tool"
    finally:
        server._sessions.clear()


def test_shutdown_sessions_closes_every_session_via_helper(monkeypatch):
    seen = []
    monkeypatch.setattr(
        server, "_close_session_by_id",
        lambda sid, *, end_reason: seen.append((sid, end_reason)),
    )
    server._sessions.clear()
    server._sessions["a"] = {}
    server._sessions["b"] = {}
    try:
        server._shutdown_sessions()
        assert sorted(sid for sid, _ in seen) == ["a", "b"]
        assert {reason for _, reason in seen} == {"tui_shutdown"}
    finally:
        server._sessions.clear()


def test_session_is_evictable_when_idle_dead_and_quiescent(monkeypatch):
    monkeypatch.setattr(server, "_session_pending_kind", lambda sid: "")
    now = time.time()
    assert server._session_is_evictable("s", _idle_evictable_session(now), now) is True


def test_session_not_evictable_violating_each_exemption(monkeypatch):
    monkeypatch.setattr(server, "_session_pending_kind", lambda sid: "")
    now = time.time()
    live_transport = type("T", (), {"_closed": False})()

    running = _idle_evictable_session(now) | {"running": True}
    assert server._session_is_evictable("s", running, now) is False

    starting = _idle_evictable_session(now)
    starting["agent_ready"] = threading.Event()  # not set -> still starting
    assert server._session_is_evictable("s", starting, now) is False

    on_socket = _idle_evictable_session(now) | {"transport": live_transport}
    assert server._session_is_evictable("s", on_socket, now) is False

    recent = _idle_evictable_session(now) | {"last_active": now}
    assert server._session_is_evictable("s", recent, now) is False

    young = _idle_evictable_session(now) | {"created_at": now}
    assert server._session_is_evictable("s", young, now) is False

    # Pending input request, even when everything else looks idle.
    monkeypatch.setattr(server, "_session_pending_kind", lambda sid: "input")
    assert server._session_is_evictable("s", _idle_evictable_session(now), now) is False


def test_reap_idle_sessions_closes_only_evictable(monkeypatch):
    closed = []
    monkeypatch.setattr(server, "_session_pending_kind", lambda sid: "")
    monkeypatch.setattr(
        server, "_close_session_by_id",
        lambda sid, *, end_reason, predicate=None: closed.append((sid, end_reason)),
    )
    now = time.time()
    server._sessions.clear()
    server._sessions["stale"] = _idle_evictable_session(now)
    server._sessions["fresh"] = _idle_evictable_session(now) | {"last_active": now}
    try:
        server._reap_idle_sessions()
        assert closed == [("stale", "idle_timeout")]
    finally:
        server._sessions.clear()


def test_periodic_trim_deferred_while_a_session_is_busy_or_attached(monkeypatch):
    """The gen-2 collect + malloc_trim stalls the loop for its whole duration (#58576): it must not run
    while any session is mid-turn or still holds a live client, whatever the other sessions look like."""
    calls = _periodic_trim_calls(monkeypatch)
    now = time.time()
    live = types.SimpleNamespace(_closed=False)
    for busy in ({"running": True}, {"transport": live}):
        server._sessions.clear()
        server._sessions["idle"] = _idle_evictable_session(now)
        server._sessions["busy"] = _idle_evictable_session(now) | busy
        try:
            server._reap_idle_sessions()
            assert calls == [], busy
        finally:
            server._sessions.clear()


def test_periodic_trim_runs_once_every_session_is_quiescent(monkeypatch):
    """Every quiescent scan trims, even with no victims: no sessions at all, or only recent (not yet
    TTL-evictable) sessions that are detached and idle."""
    calls = _periodic_trim_calls(monkeypatch)
    now = time.time()
    for sessions in ({}, {"parked": _idle_evictable_session(now) | {"last_active": now, "created_at": now}}):
        calls.clear()
        server._sessions.clear()
        server._sessions.update(sessions)
        try:
            server._reap_idle_sessions()
            assert len(calls) == 1, sessions
        finally:
            server._sessions.clear()


def test_turn_completion_trim_skips_while_another_session_is_running(monkeypatch):
    """Only OTHER sessions gate the post-turn trim: a sole session trims at every turn end; a second in-flight turn defers it (#58576)."""
    calls = _periodic_trim_calls(monkeypatch)
    monkeypatch.setattr(server, "_clear_session_context", lambda tokens: None)
    now = time.time()
    own = _idle_evictable_session(now) | {"running": True, "transport": types.SimpleNamespace(_closed=False)}
    server._sessions.clear()
    server._sessions["own"] = own
    try:
        server._post_turn_housekeeping("own", own, server._TurnRun(agent=None, one_turn_restore=None, terminal_callback=None, receipt_committed=True))
        assert len(calls) == 1

        calls.clear()
        server._sessions["other"] = _idle_evictable_session(now) | {"running": True}
        server._post_turn_housekeeping("own", own, server._TurnRun(agent=None, one_turn_restore=None, terminal_callback=None, receipt_committed=True))
        assert calls == []
    finally:
        server._sessions.clear()


def test_reap_idle_sessions_logs_trim_failure(monkeypatch, caplog):
    from hermes_cli import mem_trim

    _periodic_trim_calls(monkeypatch)
    monkeypatch.setattr(mem_trim, "trim_memory", lambda **_kw: (_ for _ in ()).throw(RuntimeError("boom")))
    server._sessions.clear()
    try:
        with caplog.at_level("DEBUG", logger="hermes_runtime.server"):
            server._reap_idle_sessions()
        assert "boom" in caplog.text
    finally:
        server._sessions.clear()


def test_ttl_reaper_spares_session_with_active_delegation(monkeypatch):
    from tools import async_delegation

    closed = []
    delegation_id = "deleg_ttl_reaper_test"
    monkeypatch.setattr(server, "_session_pending_kind", lambda sid: "")
    monkeypatch.setattr(server, "_enforce_session_cap", lambda: None)
    monkeypatch.setattr(server, "_reclaim_orphaned_leases", lambda: None)
    monkeypatch.setattr(
        server,
        "_close_session_by_id",
        lambda sid, *, end_reason, predicate=None: closed.append((sid, end_reason)),
    )
    now = time.time()
    server._sessions.clear()
    server._sessions["delegating-ttl"] = _idle_evictable_session(now)
    with async_delegation._records_lock:
        async_delegation._records[delegation_id] = {
            "status": "running",
            "origin_ui_session_id": "delegating-ttl",
        }

    try:
        server._reap_idle_sessions()
        assert closed == []
    finally:
        server._sessions.clear()
        with async_delegation._records_lock:
            async_delegation._records.pop(delegation_id, None)


def test_lru_reaper_spares_active_delegation_and_evicts_idle_peer(monkeypatch):
    from tools import async_delegation

    closed = []
    delegation_id = "deleg_lru_reaper_test"
    monkeypatch.setattr(server, "_session_pending_kind", lambda sid: "")
    monkeypatch.setattr(server, "_max_live_sessions", lambda: 1)
    monkeypatch.setattr(
        server,
        "_close_session_by_id",
        lambda sid, *, end_reason, predicate=None: closed.append((sid, end_reason)),
    )
    now = time.time()
    server._sessions.clear()
    server._sessions["delegating-lru"] = _idle_evictable_session(now) | {
        "last_active": now - 20 * 3600
    }
    server._sessions["idle-peer"] = _idle_evictable_session(now)
    with async_delegation._records_lock:
        async_delegation._records[delegation_id] = {
            "status": "running",
            "origin_ui_session_id": "delegating-lru",
        }

    try:
        server._enforce_session_cap()
        assert closed == [("idle-peer", "lru_evict")]
    finally:
        server._sessions.clear()
        with async_delegation._records_lock:
            async_delegation._records.pop(delegation_id, None)


def test_ttl_reaper_revalidates_session_before_teardown(monkeypatch):
    closed = []
    calls = {"count": 0}
    live_transport = type("T", (), {"_closed": False})()
    original_is_evictable = server._session_is_evictable
    monkeypatch.setattr(server, "_session_pending_kind", lambda sid: "")
    monkeypatch.setattr(server, "_enforce_session_cap", lambda: None)
    monkeypatch.setattr(server, "_reclaim_orphaned_leases", lambda: None)
    monkeypatch.setattr(
        server,
        "_teardown_session",
        lambda session, *, end_reason: closed.append((session, end_reason)),
    )

    def _reattach_after_scan(sid, session, now):
        calls["count"] += 1
        evictable = original_is_evictable(sid, session, now)
        if calls["count"] == 1:
            session["transport"] = live_transport
        return evictable

    monkeypatch.setattr(server, "_session_is_evictable", _reattach_after_scan)
    now = time.time()
    server._sessions.clear()
    server._sessions["ttl-race"] = _idle_evictable_session(now)

    try:
        server._reap_idle_sessions()
        assert server._sessions["ttl-race"]["transport"] is live_transport
        assert calls["count"] == 2
        assert closed == []
    finally:
        server._sessions.clear()


def test_lru_reaper_revalidates_and_tries_next_candidate(monkeypatch):
    closed = []
    live_transport = type("T", (), {"_closed": False})()
    original_is_evictable = server._session_is_lru_evictable
    monkeypatch.setattr(server, "_session_pending_kind", lambda sid: "")
    monkeypatch.setattr(server, "_max_live_sessions", lambda: 1)
    monkeypatch.setattr(
        server,
        "_teardown_session",
        lambda session, *, end_reason: closed.append((session, end_reason)),
    )

    def _reattach_oldest_after_scan(sid, session):
        evictable = original_is_evictable(sid, session)
        if sid == "lru-race" and session.get("transport") is server._detached_ws_transport:
            session["transport"] = live_transport
        return evictable

    monkeypatch.setattr(server, "_session_is_lru_evictable", _reattach_oldest_after_scan)
    now = time.time()
    server._sessions.clear()
    server._sessions["lru-race"] = _idle_evictable_session(now) | {
        "last_active": now - 20 * 3600
    }
    server._sessions["idle-peer"] = _idle_evictable_session(now)

    try:
        server._enforce_session_cap()
        assert server._sessions["lru-race"]["transport"] is live_transport
        assert "idle-peer" not in server._sessions
        assert [reason for _session, reason in closed] == ["lru_evict"]
    finally:
        server._sessions.clear()


def test_session_create_records_ui_model_as_session_override(monkeypatch):
    """The desktop composer owns its model as plain UI state and ships it on
    session.create. The gateway must record it as a PER-SESSION override (built
    into the agent), never a global config write — picking a model for a new chat
    must not mutate the profile default.
    """
    monkeypatch.setattr(server, "_enable_gateway_prompts", lambda: None)
    # Don't run the real deferred build in this storage-focused test.
    monkeypatch.setattr(server, "_start_agent_build", lambda *a, **k: None)
    try:
        resp = server._methods["session.create"](
            "r1",
            {
                "cols": 80,
                "model": "claude-sonnet-4.6",
                "provider": "anthropic",
                "reasoning_effort": "high",
                "fast": True,
            },
        )
        sid = resp["result"]["session_id"]
        sess = server._sessions[sid]
        assert sess["model_override"] == {"model": "claude-sonnet-4.6", "provider": "anthropic"}
        assert sess["create_reasoning_override"] is not None
        assert sess["create_service_tier_override"] == "priority"
        # The immediate response reflects the override (not the global default) so
        # the client never clobbers its sticky pick before the build lands.
        assert resp["result"]["info"]["model"] == "claude-sonnet-4.6"
        assert resp["result"]["info"]["provider"] == "anthropic"

        # Explicit false is not the same as omission: it must suppress a Fast
        # profile default for this session's first request.
        normal = server._methods["session.create"](
            "r2", {"cols": 80, "fast": False}
        )
        normal_sess = server._sessions[normal["result"]["session_id"]]
        assert normal_sess["create_service_tier_override"] == ""

        # No knobs → no overrides; the session builds from the profile default.
        plain = server._methods["session.create"]("r3", {"cols": 80})
        plain_sess = server._sessions[plain["result"]["session_id"]]
        assert plain_sess["model_override"] is None
        assert plain_sess["create_reasoning_override"] is None
        assert plain_sess["create_service_tier_override"] is None
    finally:
        server._sessions.clear()


@pytest.mark.parametrize("service_tier_override", ["priority", ""])
def test_start_agent_build_passes_session_model_override(
    monkeypatch, service_tier_override
):
    """A model staged on the session (e.g. by session.create from the desktop
    composer) must reach _make_agent so the first build runs on it directly —
    no global config, no build-then-switch.
    """
    captured = {}

    class FakeWorker:
        def __init__(self, *_a, **_k):
            pass

        def close(self):
            pass

    def fake_make_agent(sid, key, session_id=None, session_db=None, **kwargs):
        captured.update(kwargs)
        return types.SimpleNamespace(model="claude-sonnet-4.6")

    monkeypatch.setattr(
        server, "_set_session_context", lambda target, cwd=None: []
    )
    monkeypatch.setattr(server, "_clear_session_context", lambda tokens: None)
    monkeypatch.setattr(server, "_make_agent", fake_make_agent)
    monkeypatch.setattr(server, "_SlashWorker", FakeWorker)
    monkeypatch.setattr(server, "_attach_worker", lambda *a, **k: None)
    monkeypatch.setattr(server, "_wire_callbacks", lambda _sid: None)
    monkeypatch.setattr(server, "_emit", lambda *a, **k: None)
    monkeypatch.setattr(server, "_session_info", lambda *a, **k: {})
    monkeypatch.setattr(server, "_start_notification_poller", lambda *a, **k: None)
    monkeypatch.setattr(server, "_notify_session_boundary", lambda *a, **k: None)
    monkeypatch.setattr(server, "_probe_config_health", lambda *_a: None)

    sid = "build-sid"
    override = {"model": "claude-sonnet-4.6", "provider": "anthropic"}
    reasoning = {"enabled": True, "effort": "high"}
    session = {
        "agent": None,
        "agent_ready": threading.Event(),
        "session_key": "k1",
        "profile_home": None,
        "model_override": override,
        "create_reasoning_override": reasoning,
        "create_service_tier_override": service_tier_override,
    }
    server._sessions[sid] = session
    try:
        server._start_agent_build(sid, session)
        assert session["agent_ready"].wait(timeout=3), "agent build did not finish"
        assert captured.get("model_override") == override
        assert captured.get("reasoning_config_override") == reasoning
        assert captured.get("service_tier_override") == service_tier_override
        assert session["agent"].model == "claude-sonnet-4.6"
    finally:
        server._sessions.clear()


def test_reset_session_agent_clears_session_overrides(monkeypatch):
    """/new is a full conversation boundary: session-scoped /model, /reasoning,
    and /fast overrides do NOT carry into the fresh agent — it re-derives
    everything from config.yaml (#48055, #23131)."""
    captured = {}
    new_agent = types.SimpleNamespace(model="openai/gpt-5.4", service_tier="")
    session = _session(
        agent=types.SimpleNamespace(
            model="openai/gpt-5.4",
            reasoning_config={"enabled": True, "effort": "high"},
            service_tier="",
        ),
        model_override={"model": "openai/gpt-5.4"},
        create_reasoning_override={"enabled": True, "effort": "high"},
        create_service_tier_override="",
    )

    def make_agent(*_args, **kwargs):
        captured.update(kwargs)
        return new_agent

    monkeypatch.setattr(server, "_set_session_context", lambda _key: [])
    monkeypatch.setattr(server, "_clear_session_context", lambda _tokens: None)
    monkeypatch.setattr(server, "_make_agent", make_agent)
    monkeypatch.setattr(server, "_config_model_target", lambda: ("", ""))
    monkeypatch.setattr(server, "_load_show_reasoning", lambda: True)
    monkeypatch.setattr(server, "_load_tool_progress_mode", lambda: "all")
    monkeypatch.setattr(server, "_session_info", lambda *_args: {})
    monkeypatch.setattr(server, "_emit", lambda *_args: None)
    monkeypatch.setattr(server, "_restart_slash_worker", lambda *_args: None)

    server._reset_session_agent("sid", session)

    # No session overrides forwarded — fresh agent builds from config.
    assert "model_override" not in captured
    assert "reasoning_config_override" not in captured
    assert "service_tier_override" not in captured
    # And the session pins are gone so a later rebuild can't resurrect them.
    assert "model_override" not in session
    assert "create_reasoning_override" not in session
    assert "create_service_tier_override" not in session
    assert session["agent"] is new_agent


@pytest.mark.parametrize(
    "card,expected",
    [
        ("canonical", {"kind": "canonical"}),
        (
            "distinct",
            {
                "kind": "distinct",
                "payment_method_id": "pm_auto",
                "brand": None,
                "last4": None,
            },
        ),
        ("none", {"kind": "none"}),
    ],
)
def test_billing_state_serializes_auto_reload_card_union(monkeypatch, card, expected):
    from agent.billing_view import AutoReload, AutoReloadCard, BillingState

    monkeypatch.setattr(server, "_usage_payload", lambda state: {"available": False})
    auto_reload_card = AutoReloadCard(
        kind=card,
        payment_method_id="pm_auto" if card == "distinct" else None,
    )
    state = BillingState(
        logged_in=True,
        auto_reload=AutoReload(enabled=True, card=auto_reload_card),
    )

    result = server._serialize_billing_state(state)

    assert result["auto_reload"]["card"] == expected


def test_billing_state_serializes_server_plan_capability(monkeypatch):
    from agent.billing_view import BillingState

    monkeypatch.setattr(server, "_usage_payload", lambda state: {"available": False})
    state = BillingState(
        logged_in=True,
        role="MEMBER",
        can_change_plan_raw=True,
    )

    result = server._serialize_billing_state(state)

    assert result["is_admin"] is False
    assert result["can_change_plan"] is True


@pytest.mark.parametrize(
    "status,error,retry_after",
    [
        (503, "stripe_unavailable", 75),
        (429, "upgrade_cap_exceeded", None),
        (429, "rate_limited", None),
    ],
)
def test_billing_error_serialization_preserves_server_code(
    status, error, retry_after
):
    import hermes_cli.nous_billing as nb

    headers = _BillingHeaders({"Retry-After": str(retry_after)}) if retry_after else None
    with pytest.raises(nb.BillingTransient) as ei:
        nb._raise_for_error(status, {"error": error}, headers)

    result = server._serialize_billing_error(ei.value)

    assert result["error"] == error
    assert ei.value.error == error
    assert result["retry_after"] == retry_after


def test_billing_rate_limit_without_error_defaults_wire_code():
    import hermes_cli.nous_billing as nb

    exc = nb.BillingRateLimited("slow down", status=429, retry_after=10)

    result = server._serialize_billing_error(exc)

    assert result["error"] == "rate_limited"


def test_subscription_preview_serializes_quote(monkeypatch):
    import hermes_cli.nous_billing as nb

    monkeypatch.setattr(
        nb,
        "post_subscription_preview",
        lambda subscription_type_id: {
            "effect": "charge_now",
            "reason": None,
            "currentTierId": "plus",
            "currentTierName": "Plus",
            "targetTierId": "ultra",
            "targetTierName": "Ultra",
            "monthlyCreditsDelta": "6000",
            "amountDueNowCents": 1234,
            "effectiveAt": None,
        },
    )
    res = _sub_rpc("subscription.preview", {"subscription_type_id": "ultra"})
    assert res["ok"] is True
    assert res["effect"] == "charge_now"
    assert res["amount_due_now_cents"] == 1234
    assert res["target_tier_name"] == "Ultra"
    assert res["monthly_credits_delta"] == "6000"


def test_subscription_preview_requires_tier():
    res = _sub_rpc("subscription.preview", {})
    assert res["ok"] is False
    assert res["error"] == "invalid_request"


def test_subscription_preview_scope_error_maps_to_step_up(monkeypatch):
    import hermes_cli.nous_billing as nb

    def _raise(subscription_type_id):
        raise nb.BillingScopeRequired("billing:manage required")

    monkeypatch.setattr(nb, "post_subscription_preview", _raise)
    res = _sub_rpc("subscription.preview", {"subscription_type_id": "ultra"})
    assert res["ok"] is False
    assert res["error"] == "insufficient_scope"


def test_subscription_change_cancellation(monkeypatch):
    import hermes_cli.nous_billing as nb

    seen = {}

    def _put(*, subscription_type_id=None, cancel=False):
        seen["tier"] = subscription_type_id
        seen["cancel"] = cancel
        return {"rail": "stripe", "cancelAtPeriodEnd": True, "message": "Scheduled to cancel."}

    monkeypatch.setattr(nb, "put_subscription_pending_change", _put)
    res = _sub_rpc("subscription.change", {"cancel": True})
    assert res["ok"] is True
    assert seen == {"tier": None, "cancel": True}
    assert res["message"] == "Scheduled to cancel."


def test_subscription_change_tier_downgrade(monkeypatch):
    import hermes_cli.nous_billing as nb

    seen = {}

    def _put(*, subscription_type_id=None, cancel=False):
        seen["tier"] = subscription_type_id
        seen["cancel"] = cancel
        return {"rail": "stripe", "changeType": "downgrade", "targetTierName": "Plus", "message": "Scheduled."}

    monkeypatch.setattr(nb, "put_subscription_pending_change", _put)
    res = _sub_rpc("subscription.change", {"subscription_type_id": "plus"})
    assert res["ok"] is True
    assert seen == {"tier": "plus", "cancel": False}


def test_subscription_change_requires_tier_or_cancel():
    res = _sub_rpc("subscription.change", {})
    assert res["ok"] is False
    assert res["error"] == "invalid_request"


def test_subscription_upgrade_echoes_status_and_idempotency(monkeypatch):
    import hermes_cli.nous_billing as nb

    seen = {}

    def _upgrade(*, subscription_type_id, idempotency_key):
        seen["key"] = idempotency_key
        return {"status": "upgraded", "targetTierId": "ultra", "targetTierName": "Ultra"}

    monkeypatch.setattr(nb, "post_subscription_upgrade", _upgrade)
    res = _sub_rpc("subscription.upgrade", {"subscription_type_id": "ultra", "idempotency_key": "k-1"})
    assert res["ok"] is True
    assert res["status"] == "upgraded"
    assert res["target_tier_name"] == "Ultra"
    assert res["idempotency_key"] == "k-1"
    assert seen["key"] == "k-1"


def test_subscription_upgrade_requires_action_surfaces_recovery(monkeypatch):
    import hermes_cli.nous_billing as nb

    monkeypatch.setattr(
        nb,
        "post_subscription_upgrade",
        lambda *, subscription_type_id, idempotency_key: {
            "status": "requires_action",
            "reason": "authentication_required",
            "recoveryUrl": "https://portal.example/subscription?org_id=o",
        },
    )
    res = _sub_rpc("subscription.upgrade", {"subscription_type_id": "ultra"})
    # The RPC succeeds; the CHARGE needs 3DS → status + recovery_url for the portal.
    assert res["ok"] is True
    assert res["status"] == "requires_action"
    assert res["recovery_url"].startswith("https://portal.example")
    assert res["idempotency_key"]  # minted when the caller omits one


def test_get_usage_perf_readouts_present():
    """cache_hit_pct / avg_latency_s / avg_tps mirror the classic CLI bar."""
    from collections import deque

    class _PerfAgent:
        model = "x"
        session_prompt_tokens = 27_873
        session_cache_read_tokens = 24_369
        _api_latency_history = deque([2.1, 4.3], maxlen=10)
        _api_output_history = deque([130, 190], maxlen=10)

    usage = server._get_usage(_PerfAgent())
    assert usage["cache_hit_pct"] == 87
    assert usage["avg_latency_s"] == 3.2
    assert usage["avg_tps"] == 50.0  # true throughput sum(out)/sum(lat), not mean of ratios


def test_get_usage_perf_readouts_omitted_without_data():
    """Zero cache reads / empty history omit the keys — never fabricate 0s."""

    class _ColdAgent:
        model = "x"
        session_prompt_tokens = 100
        session_cache_read_tokens = 0

    usage = server._get_usage(_ColdAgent())
    assert "cache_hit_pct" not in usage
    assert "avg_latency_s" not in usage
    assert "avg_tps" not in usage


def test_get_usage_perf_readouts_guard_negative_latency():
    """Odd provider timings (negative durations seen in logs) are dropped."""
    from collections import deque

    class _WeirdAgent:
        model = "x"
        _api_latency_history = deque([-0.8], maxlen=10)
        _api_output_history = deque([100], maxlen=10)

    usage = server._get_usage(_WeirdAgent())
    assert "avg_latency_s" not in usage
    assert "avg_tps" not in usage


def test_get_usage_includes_active_subagents(monkeypatch):
    import tools.async_delegation as ad_mod
    monkeypatch.setattr(ad_mod, "active_count", lambda: 4)
    usage = server._get_usage(_BareAgent())
    assert usage["active_subagents"] == 4


def test_get_usage_safe_when_active_count_raises(monkeypatch):
    """A raising active_count() must not break the usage payload."""
    import tools.async_delegation as ad_mod

    def _boom():
        raise RuntimeError("boom")

    monkeypatch.setattr(ad_mod, "active_count", _boom)
    usage = server._get_usage(_BareAgent())
    # Field omitted, but the rest of the payload is intact.
    assert "active_subagents" not in usage
    assert usage["model"] == "x"


class TestResolveRuntimeWithFallback:
    """Tests for _resolve_runtime_with_fallback(): init-time provider
    fallback when the primary provider raises AuthError."""

    def test_primary_success_returns_runtime(self, monkeypatch):
        """When primary resolve succeeds, return its result directly."""
        expected = {"provider": "openai", "api_key": "tok"}
        monkeypatch.setattr(
            "hermes_cli.runtime_provider.resolve_runtime_provider",
            lambda **kw: expected,
        )
        resolution = server._resolve_runtime_with_fallback(
            {"requested": "openai"}
        )
        assert resolution.runtime == expected
        assert resolution.selected_model is None
        assert resolution.used_fallback is False

    def test_auth_error_tries_fallback_chain(self, monkeypatch):
        """On AuthError from primary, walk fallback_providers chain."""
        from hermes_cli.auth import AuthError

        fallback_runtime = {"provider": "deepseek", "api_key": "fb-tok"}

        def fake_resolve(**kwargs):
            if kwargs.get("requested") == "openai-codex":
                raise AuthError("No Codex credentials stored")
            return fallback_runtime

        monkeypatch.setattr(
            "hermes_cli.runtime_provider.resolve_runtime_provider",
            fake_resolve,
        )
        monkeypatch.setattr(
            server,
            "_load_fallback_model",
            lambda: [{"provider": "deepseek", "model": "deepseek-v4-pro"}],
        )
        resolution = server._resolve_runtime_with_fallback(
            {"requested": "openai-codex"},
        )
        assert resolution.runtime == fallback_runtime
        assert resolution.selected_model == "deepseek-v4-pro"
        assert resolution.used_fallback is True


    def test_auth_error_skips_provider_only_fallback(self, monkeypatch):
        """Auth fallback requires one complete provider/model pair."""
        from hermes_cli.auth import AuthError

        requested = []
        fallback_runtime = {"provider": "openrouter", "api_key": "fb-tok"}

        def fake_resolve(**kwargs):
            requested.append(kwargs.get("requested"))
            if kwargs.get("requested") == "openai-codex":
                raise AuthError("No Codex credentials stored")
            return fallback_runtime

        monkeypatch.setattr(
            "hermes_cli.runtime_provider.resolve_runtime_provider",
            fake_resolve,
        )
        monkeypatch.setattr(
            server,
            "_load_fallback_model",
            lambda: [
                {"provider": "anthropic"},
                {"provider": "openrouter", "model": "z-ai/glm-5.2"},
            ],
        )

        resolution = server._resolve_runtime_with_fallback(
            {"requested": "openai-codex"}
        )

        assert requested == ["openai-codex", "openrouter"]
        assert resolution.runtime == fallback_runtime
        assert resolution.selected_model == "z-ai/glm-5.2"
        assert resolution.used_fallback is True

    def test_fallback_entry_key_env_resolves_api_key(self, monkeypatch):
        """A fallback entry naming its key via key_env passes the resolved
        env value as explicit_api_key (#43861, @VrtxOmega)."""
        from hermes_cli.auth import AuthError

        monkeypatch.setenv("FB_TEST_KEY", "env-resolved-key")
        captured = {}
        fallback_runtime = {"provider": "openrouter", "api_key": "x"}

        def fake_resolve(**kwargs):
            if kwargs.get("requested") == "openai-codex":
                raise AuthError("No Codex credentials stored")
            captured.update(kwargs)
            return fallback_runtime

        monkeypatch.setattr(
            "hermes_cli.runtime_provider.resolve_runtime_provider",
            fake_resolve,
        )
        monkeypatch.setattr(
            server,
            "_load_fallback_model",
            lambda: [
                {
                    "provider": "openrouter",
                    "model": "z-ai/glm-5.2",
                    "key_env": "FB_TEST_KEY",
                }
            ],
        )
        resolution = server._resolve_runtime_with_fallback(
            {"requested": "openai-codex"}
        )
        assert resolution.used_fallback is True
        assert captured.get("explicit_api_key") == "env-resolved-key"

    def test_auth_error_all_fallbacks_fail_raises(self, monkeypatch):
        """When all fallbacks also fail, re-raise the original AuthError."""
        from hermes_cli.auth import AuthError

        def fake_resolve(**kwargs):
            raise AuthError("No credentials for " + str(kwargs.get("requested")))

        monkeypatch.setattr(
            "hermes_cli.runtime_provider.resolve_runtime_provider",
            fake_resolve,
        )
        monkeypatch.setattr(
            server,
            "_load_fallback_model",
            lambda: [{"provider": "deepseek", "model": "deepseek-v4-pro"}],
        )
        import pytest

        with pytest.raises(AuthError, match="No credentials for openai-codex"):
            server._resolve_runtime_with_fallback(
                {"requested": "openai-codex"},
            )

    def test_auth_error_skips_non_dict_entries(self, monkeypatch):
        """Fallback chain entries that are not dicts are skipped."""
        from hermes_cli.auth import AuthError

        fallback_runtime = {"provider": "anthropic", "api_key": "ant-tok"}

        def fake_resolve(**kwargs):
            if kwargs.get("requested") == "openai-codex":
                raise AuthError("No Codex credentials stored")
            return fallback_runtime

        monkeypatch.setattr(
            "hermes_cli.runtime_provider.resolve_runtime_provider",
            fake_resolve,
        )
        monkeypatch.setattr(
            server,
            "_load_fallback_model",
            lambda: [
                "invalid-string-entry",
                {"provider": "anthropic", "model": "claude-sonnet-4-6"},
            ],
        )
        resolution = server._resolve_runtime_with_fallback(
            {"requested": "openai-codex"},
        )
        assert resolution.runtime == fallback_runtime
        assert resolution.selected_model == "claude-sonnet-4-6"
        assert resolution.used_fallback is True

    def test_make_agent_uses_fallback_on_auth_error(self, monkeypatch):
        """Integration: _make_agent falls back to configured fallback
        provider when the primary provider raises AuthError."""
        import types

        from hermes_cli.auth import AuthError

        captured = {}
        fallback_runtime = {
            "provider": "deepseek",
            "api_key": "fb-tok",
            "base_url": "https://fallback.invalid/v1",
        }

        def fake_resolve(**kwargs):
            if kwargs.get("requested") in (None, "openai-codex"):
                raise AuthError("No Codex credentials stored")
            return fallback_runtime

        def fake_agent(**kwargs):
            captured.update(kwargs)
            return types.SimpleNamespace(model=kwargs.get("model"))

        monkeypatch.delenv("HERMES_MODEL", raising=False)
        monkeypatch.delenv("HERMES_INFERENCE_MODEL", raising=False)
        monkeypatch.delenv("HERMES_TUI_PROVIDER", raising=False)
        monkeypatch.setattr(
            server,
            "_load_cfg",
            lambda: {
                "model": {"default": "gpt-5.5", "provider": "openai-codex"},
                "fallback_providers": [
                    {"provider": "deepseek", "model": "deepseek-v4-pro"},
                ],
            },
        )
        monkeypatch.setattr(
            "hermes_cli.runtime_provider.resolve_runtime_provider",
            fake_resolve,
        )
        monkeypatch.setattr("run_agent.AIAgent", fake_agent)
        monkeypatch.setattr(server, "_load_enabled_toolsets", lambda *_a, **_kw: ["file"])
        monkeypatch.setattr(server, "_get_db", lambda: None)

        agent = server._make_agent(
            "sid",
            "session-key",
            model_override={
                "model": "gpt-5.5",
                "provider": "openai-codex",
                "base_url": "https://chatgpt.com/backend-api/codex",
                "api_key": "stale-codex-token",
            },
        )

        assert agent.model == "deepseek-v4-pro"
        assert captured["provider"] == "deepseek"
        assert captured["base_url"] == "https://fallback.invalid/v1"
        assert captured["api_key"] == "fb-tok"


def test_get_usage_does_not_substitute_cumulative_total_for_context_used():
    """An external context engine that does not report last_prompt_tokens must
    not have the cumulative lifetime session_total_tokens shown as its current
    context occupancy — that substitution produced impossible 1.9m/120k (100%)
    status-bar readings (#50421). With no real current occupancy known,
    context_used/percent stay unset rather than wrong."""
    agent = types.SimpleNamespace(
        model="test-model",
        session_total_tokens=1_900_000,
        context_compressor=types.SimpleNamespace(
            last_prompt_tokens=0,
            context_length=120_000,
            compression_count=0,
        ),
    )
    usage = server._get_usage(agent)
    assert usage.get("context_used") != 1_900_000
    assert "context_used" not in usage
    assert "context_percent" not in usage


def test_get_usage_reports_real_current_occupancy():
    """When the compressor reports a real current prompt size, context_used is
    that value (not the cumulative total) and the percent is sane."""
    agent = types.SimpleNamespace(
        model="test-model",
        session_total_tokens=1_900_000,
        context_compressor=types.SimpleNamespace(
            last_prompt_tokens=60_000,
            context_length=120_000,
            compression_count=2,
        ),
    )
    usage = server._get_usage(agent)
    assert usage["context_used"] == 60_000
    assert usage["context_max"] == 120_000
    assert usage["context_percent"] == 50


def test_get_usage_clamps_post_compression_sentinel():
    """Right after a compression, last_prompt_tokens is the -1 sentinel
    (conversation_compression sets it until the next real usage report). It is
    truthy, so `or 0` doesn't neutralize it — the guard must clamp <0 to 0 so
    the transitional turn emits no gauge instead of leaking context_used=-1."""
    agent = types.SimpleNamespace(
        model="test-model",
        session_total_tokens=4_000_000,
        context_compressor=types.SimpleNamespace(
            last_prompt_tokens=-1,
            context_length=1_048_576,
            compression_count=6,
        ),
    )
    usage = server._get_usage(agent)
    assert "context_used" not in usage
    assert "context_percent" not in usage


def test_tts_stream_begin_requires_voice_tts(monkeypatch):
    monkeypatch.setenv("HERMES_VOICE_TTS", "0")
    assert server._tts_stream_begin() is None


def test_tts_stream_begin_requires_working_provider(monkeypatch):
    monkeypatch.setenv("HERMES_VOICE_TTS", "1")
    _fake_tts_modules(monkeypatch, requirements=False)
    assert server._tts_stream_begin() is None


def test_tts_stream_begin_and_stop_lifecycle(monkeypatch):
    """begin() spawns the consumer; stop() cuts it and clears the slot."""
    monkeypatch.setenv("HERMES_VOICE_TTS", "1")
    monkeypatch.setenv("HERMES_VOICE", "0")  # no barge-in monitor (no mic)
    playback_stops: list = []
    started = _fake_tts_modules(monkeypatch, playback_stops=playback_stops)

    text_queue = server._tts_stream_begin()
    assert text_queue is not None
    assert started["queue"] is text_queue

    with server._tts_stream_lock:
        state = server._tts_stream_state
    assert state is not None and not state["stop"].is_set()

    server._tts_stream_stop()
    assert state["stop"].is_set()
    assert playback_stops == [True]
    with server._tts_stream_lock:
        assert server._tts_stream_state is None


def test_tts_stream_begin_barges_in_on_previous_pipeline(monkeypatch):
    """A new turn's pipeline stops the previous turn's speech (one speaker)."""
    monkeypatch.setenv("HERMES_VOICE_TTS", "1")
    monkeypatch.setenv("HERMES_VOICE", "0")
    _fake_tts_modules(monkeypatch)

    server._tts_stream_begin()
    with server._tts_stream_lock:
        first = server._tts_stream_state
    server._tts_stream_begin()
    assert first is not None and first["stop"].is_set()
    server._tts_stream_stop()


def test_tts_stream_stop_latches_interruption_for_next_turn(monkeypatch):
    """Cutting live speech (interrupt / typing barge) marks the latch the next
    turn's model note consumes; a mode change (user_barge=False) does not."""
    import tools.tts_streaming as ts

    ts._interrupted_at = None
    monkeypatch.setenv("HERMES_VOICE_TTS", "1")
    monkeypatch.setenv("HERMES_VOICE", "0")
    _fake_tts_modules(monkeypatch)

    server._tts_stream_begin()
    server._tts_stream_stop()  # default: user barge
    assert ts.take_speech_interrupted() is True

    server._tts_stream_begin()
    server._tts_stream_stop(user_barge=False)  # /voice off
    assert ts.take_speech_interrupted() is False


def test_tts_stream_stop_after_natural_finish_does_not_latch(monkeypatch):
    """Speech that already finished (done set) isn't an interruption."""
    import tools.tts_streaming as ts

    ts._interrupted_at = None
    monkeypatch.setenv("HERMES_VOICE_TTS", "1")
    monkeypatch.setenv("HERMES_VOICE", "0")
    _fake_tts_modules(monkeypatch)

    server._tts_stream_begin()
    with server._tts_stream_lock:
        server._tts_stream_state["done"].set()
    server._tts_stream_stop()
    assert ts.take_speech_interrupted() is False


def test_tts_stream_vad_barge_in_cuts_pipeline_and_submits_capture(monkeypatch, tmp_path):
    """User speech during playback cuts TTS at the moment of detection
    (voice.interrupted), then the captured interruption is transcribed and
    emitted as voice.transcript so the TUI submits it — complete from its
    first syllable, no re-record round trip. The cut also latches the
    speech-interrupted note for the next turn."""
    import tools.tts_streaming as ts

    ts._interrupted_at = None
    monkeypatch.setenv("HERMES_VOICE_TTS", "1")
    monkeypatch.setenv("HERMES_VOICE", "1")
    monkeypatch.setattr(server, "_load_cfg", lambda: {"voice": {"barge_in": True}})
    events: list = []
    monkeypatch.setattr(
        server, "_voice_emit", lambda event, payload=None: events.append((event, payload))
    )

    wav = tmp_path / "barge.wav"
    wav.write_bytes(b"RIFF")

    def fake_listen(should_stop, is_playing=None, on_trigger=None, **_kw):
        on_trigger("playback")  # playback cut happens at detection
        return str(wav)

    _fake_tts_modules(
        monkeypatch,
        listen=fake_listen,
        transcribe=lambda path, model=None: {"success": True, "transcript": "stop, actually—"},
    )

    server._tts_stream_begin()
    deadline = time.monotonic() + 5.0
    while time.monotonic() < deadline and wav.exists():
        time.sleep(0.01)  # unlink (finally) runs after the transcript emit
    assert ("voice.interrupted", None) in events
    assert ("voice.transcript", {"text": "stop, actually—"}) in events
    assert not wav.exists()  # capture temp file cleaned up
    assert ts.take_speech_interrupted() is True  # VAD cut latches the model note
    server._tts_stream_stop()


def test_full_duplex_generation_phase_interrupts_running_turn(monkeypatch, tmp_path):
    """Speech DURING LLM generation (no TTS audio yet) must interrupt the
    in-flight agent turn via the same seam session.interrupt uses, and the
    captured interjection is emitted as voice.transcript. This is the
    half-duplex gap: previously no listener existed until playback started."""
    import tools.tts_streaming as ts

    ts._interrupted_at = None
    monkeypatch.setenv("HERMES_VOICE", "1")
    monkeypatch.setenv("HERMES_VOICE_TTS", "0")
    monkeypatch.setattr(server, "_load_cfg", lambda: {"voice": {"barge_in": True}})
    events: list = []
    monkeypatch.setattr(
        server, "_voice_emit", lambda event, payload=None: events.append((event, payload))
    )

    wav = tmp_path / "interject.wav"
    wav.write_bytes(b"RIFF")

    interrupted = threading.Event()
    fake_agent = types.SimpleNamespace(interrupt=lambda: interrupted.set())
    fake_session = {"running": True, "agent": fake_agent}
    monkeypatch.setattr(server, "_sessions", {"sid-fd": fake_session})

    def fake_listen(should_stop, is_playing=None, on_trigger=None, **_kw):
        assert is_playing is not None and is_playing() is False  # generation phase
        on_trigger("generation")
        return str(wav)

    _fake_tts_modules(
        monkeypatch,
        listen=fake_listen,
        transcribe=lambda path, model=None: {"success": True, "transcript": "wait, try another way"},
    )

    server._arm_full_duplex_listener()

    deadline = time.monotonic() + 5.0
    while time.monotonic() < deadline and wav.exists():
        time.sleep(0.01)
    assert interrupted.is_set()  # the running turn was interrupted
    assert ("voice.interrupted", None) in events
    assert ("voice.transcript", {"text": "wait, try another way"}) in events
    assert not wav.exists()



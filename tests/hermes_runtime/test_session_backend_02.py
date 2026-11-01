





import os




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


def test_usage_ticker_skips_unchanged_snapshots(monkeypatch):
    # A single long API call leaves the token counters frozen for many
    # intervals; the ticker must emit nothing at all (the client already has
    # the turn-start values from the previous message.complete / session.info).
    # Only a changed snapshot emits.
    events: list[dict] = []
    monkeypatch.setattr(
        server, "_emit", lambda event_type, sid, payload: events.append(payload)
    )
    snapshot = {"input": 1200, "total": 1280}
    monkeypatch.setattr(server, "_get_usage", lambda agent: dict(snapshot))

    stop, thread = server._start_usage_ticker("sess-1", object(), interval=0.01)
    try:
        # ~15 ticks with counters frozen at the turn-start baseline: zero frames.
        time.sleep(0.15)
        assert events == []

        # Counters move → the next tick emits the new snapshot.
        snapshot["total"] = 2400
        deadline = time.time() + 1.0
        while not events and time.time() < deadline:
            time.sleep(0.01)
    finally:
        stop.set()
        thread.join(timeout=2.0)

    assert events == [{"usage": {"input": 1200, "total": 2400}}]


def test_usage_ticker_baseline_sampled_before_thread_start(monkeypatch):
    """The dedup baseline must be sampled synchronously in _start_usage_ticker,
    not inside the ticker thread: a late-scheduled thread would otherwise seed
    itself with counters the turn's first API call already bumped, absorbing
    that first growth so it never emits."""
    events: list[dict] = []
    monkeypatch.setattr(
        server, "_emit", lambda event_type, sid, payload: events.append(payload)
    )
    snapshot = {"input": 1200, "total": 1280}
    monkeypatch.setattr(server, "_get_usage", lambda agent: dict(snapshot))

    class _SlowStartThread(threading.Thread):
        def start(self):
            # Deterministic stand-in for a scheduler delay: the turn's first
            # API call bumps the counters before the ticker thread ever runs.
            snapshot["total"] = 2400
            super().start()

    monkeypatch.setattr(server, "_RealThread", _SlowStartThread)

    stop, thread = server._start_usage_ticker("sess-1", object(), interval=0.01)
    try:
        deadline = time.time() + 1.0
        while not events and time.time() < deadline:
            time.sleep(0.01)
    finally:
        stop.set()
        thread.join(timeout=2.0)

    # An in-thread seed would have read 2400 as the baseline and stayed
    # silent; the synchronous seed (1280) sees it as the first growth.
    assert events == [{"usage": {"input": 1200, "total": 2400}}]


def test_usage_ticker_stop_join_prevents_late_ticks(monkeypatch):
    """The stop sequence (set + join) must guarantee no session.usage after it
    returns: a tick captured mid-turn but emitted after message.complete would
    roll the client's final usage back to a stale snapshot (clients merge
    payload.usage unconditionally)."""
    events: list[str] = []
    monkeypatch.setattr(
        server, "_emit", lambda event_type, sid, payload: events.append(event_type)
    )

    in_snapshot = threading.Event()
    release = threading.Event()
    calls = {"n": 0}

    def _blocking_get_usage(agent):
        calls["n"] += 1
        if calls["n"] == 1:
            return {"total": 0}  # dedup seed
        # First real tick: hold it mid-snapshot so the stop lands while the
        # iteration is already past the stop.wait() gate.
        in_snapshot.set()
        release.wait(2.0)
        return {"total": 999}

    monkeypatch.setattr(server, "_get_usage", _blocking_get_usage)

    stop, thread = server._start_usage_ticker("sess-1", object(), interval=0.01)
    assert in_snapshot.wait(2.0), "ticker never reached a snapshot"

    # Turn ends while the tick is mid-snapshot: run the exact stop sequence
    # _run_prompt_submit uses, then emit message.complete.
    stop.set()
    release.set()
    thread.join(timeout=2.0)
    assert not thread.is_alive(), "ticker thread survived the stop sequence"
    server._emit("message.complete", "sess-1", {})

    # The in-flight tick was dropped (stop re-checked before emit), so nothing
    # can land after — let alone overwrite — the final usage.
    assert "session.usage" not in events
    assert events[-1] == "message.complete"


def test_run_prompt_submit_never_ticks_after_message_complete(monkeypatch):
    """End-to-end ordering through _run_prompt_submit: live session.usage ticks
    happen strictly before message.complete, never after it."""
    events: list[str] = []
    tick_seen = threading.Event()

    def _record_emit(event_type, sid, payload=None):
        events.append(event_type)
        if event_type == "session.usage":
            tick_seen.set()

    monkeypatch.setattr(server, "_emit", _record_emit)

    counter = {"n": 0}

    def _moving_usage(agent):
        counter["n"] += 1
        return {"total": counter["n"]}  # moves every sample → every tick emits

    monkeypatch.setattr(server, "_get_usage", _moving_usage)
    real_ticker = server._start_usage_ticker
    monkeypatch.setattr(
        server,
        "_start_usage_ticker",
        lambda sid, agent, interval=1.0: real_ticker(sid, agent, interval=0.01),
    )
    monkeypatch.setattr(server.threading, "Thread", _ImmediateThread)
    monkeypatch.setattr(server, "make_stream_renderer", lambda cols: None)
    monkeypatch.setattr(server, "render_message", lambda raw, cols: None)
    monkeypatch.setattr(server, "_get_db", lambda: None)

    class _Agent:
        def run_conversation(
            self,
            prompt,
            conversation_history=None,
            stream_callback=None,
            persist_user_message=None,
        ):
            # Hold the turn open until at least one live tick has fired.
            assert tick_seen.wait(5.0), "no live tick during the turn"
            return {"final_response": "done", "messages": [], "completed": True}

    server._sessions["sid"] = _session(agent=_Agent())
    try:
        server.handle_request(
            {
                "id": "1",
                "method": "prompt.submit",
                "params": {"session_id": "sid", "text": "hello"},
            }
        )
    finally:
        server._sessions.pop("sid", None)

    assert "session.usage" in events
    assert "message.complete" in events
    last_tick = max(i for i, e in enumerate(events) if e == "session.usage")
    assert last_tick < events.index("message.complete")


def test_usage_ticker_unbounded_join_waits_out_blocked_emit(monkeypatch):
    """A tick stalled inside _emit (a transport write can block up to
    _WS_WRITE_TIMEOUT_S = 10s on a stalled event loop) must be waited out by
    the stop sequence, not abandoned: stop.set() + an unbounded join may only
    return after the in-flight emit has fully flushed, so nothing can land
    after message.complete."""
    order: list[str] = []
    in_emit = threading.Event()
    release = threading.Event()

    def _stalled_emit(event_type, sid, payload):
        in_emit.set()
        release.wait(10.0)  # the stalled transport write
        order.append(event_type)

    monkeypatch.setattr(server, "_emit", _stalled_emit)

    counter = {"n": 0}

    def _moving_usage(agent):
        counter["n"] += 1
        return {"total": counter["n"]}  # moves every sample → a tick emits

    monkeypatch.setattr(server, "_get_usage", _moving_usage)

    stop, thread = server._start_usage_ticker("sess-1", object(), interval=0.01)
    assert in_emit.wait(2.0), "no tick got in flight"

    # Run the exact stop sequence _run_prompt_submit uses, on a side thread so
    # the test can observe whether it returns while the emit is still stuck.
    stopped = threading.Event()

    def _stop_sequence():
        stop.set()
        thread.join()
        stopped.set()

    stopper = threading.Thread(target=_stop_sequence, daemon=True)
    stopper.start()

    # While the tick is stalled in the transport write, the stop sequence must
    # NOT complete — a timed join returning here is exactly the bug: the
    # caller would proceed to message.complete with the tick still pending.
    assert not stopped.wait(0.2), "stop sequence returned with the tick still in flight"

    release.set()
    assert stopped.wait(2.0), "stop sequence never completed after the emit flushed"
    stopper.join(timeout=2.0)

    # The flushed tick strictly precedes anything the caller emits afterwards.
    order.append("message.complete")
    assert order == ["session.usage", "message.complete"]


def test_tui_verbose_tool_details_fail_closed_when_redaction_fails(monkeypatch):
    redact_module = types.ModuleType("agent.redact")

    def fail_redaction(*_args, **_kwargs):
        raise RuntimeError("redaction unavailable")

    redact_module.redact_sensitive_text = fail_redaction
    monkeypatch.setitem(sys.modules, "agent.redact", redact_module)

    assert server._redact_tui_verbose_text("api_key=secret") == ""
    assert server._tool_args_text({"api_key": "secret"}) == ""
    assert server._tool_result_text("token=secret") == ""


def test_tui_verbose_tool_details_are_capped_before_emit(monkeypatch):
    monkeypatch.setattr(server, "_TUI_VERBOSE_TEXT_MAX_CHARS", 12)
    monkeypatch.setattr(server, "_TUI_VERBOSE_TEXT_MAX_LINES", 2)

    capped = server._cap_tui_verbose_text("one\ntwo\nthree\nfour")

    assert capped.endswith("three\nfour")
    assert "one" not in capped


def test_tui_verbose_default_cap_stays_small(monkeypatch):
    # Regression guard for #34095: the verbose tool text shipped to the TUI is
    # rendered into a persisted, expanded-by-default trail block for the whole
    # session. Raising this cap back toward the old 16KB re-introduces the Ink
    # render-tree blowup that silently OOM-killed the TUI. Keep it small.
    huge = "x" * 40_000
    capped = server._cap_tui_verbose_text(huge)

    assert len(capped) < 2_000


def test_tui_verbose_tool_events_omit_details_when_redaction_fails(monkeypatch):
    redact_module = types.ModuleType("agent.redact")

    def fail_redaction(*_args, **_kwargs):
        raise RuntimeError("redaction unavailable")

    redact_module.redact_sensitive_text = fail_redaction
    monkeypatch.setitem(sys.modules, "agent.redact", redact_module)

    events: list[tuple[str, str, dict]] = []
    monkeypatch.setattr(
        server, "_emit", lambda event_type, sid, payload: events.append((event_type, sid, payload))
    )
    monkeypatch.setitem(
        server._sessions,
        "redaction-test",
        {"tool_progress_mode": "verbose", "tool_started_at": {}},
    )

    server._on_tool_start("redaction-test", "tool-1", "terminal", {"command": "pwd"})
    server._on_tool_complete("redaction-test", "tool-1", "terminal", {"command": "pwd"}, "done")

    assert events[0][0] == "tool.start"
    assert events[1][0] == "tool.complete"
    assert "args_text" not in events[0][2]
    assert "result_text" not in events[1][2]


def test_tui_tool_output_risk_event_exposes_metadata_without_raw_output(monkeypatch):
    events: list[tuple[str, str, dict]] = []
    monkeypatch.setattr(
        server, "_emit", lambda event_type, sid, payload: events.append((event_type, sid, payload))
    )
    monkeypatch.setitem(
        server._sessions,
        "risk-test",
        {"tool_progress_mode": "all"},
    )

    server._on_tool_progress(
        "risk-test",
        "tool.output_risk",
        "web_extract",
        tool_call_id="tool-1",
        risk_metadata={
            "risk": "high",
            "findings": ["prompt_injection"],
            "redacted": False,
        },
    )

    assert events == [(
        "tool.output_risk",
        "risk-test",
        {
            "tool_id": "tool-1",
            "name": "web_extract",
            "risk": "high",
            "findings": ["prompt_injection"],
            "redacted": False,
        },
    )]
    assert "result" not in events[0][2]


def test_tui_clarify_lifecycle_events_emit_when_tool_progress_off(monkeypatch):
    events: list[tuple[str, str, dict]] = []
    monkeypatch.setattr(
        server, "_emit", lambda event_type, sid, payload: events.append((event_type, sid, payload))
    )
    monkeypatch.setitem(
        server._sessions,
        "clarify-off-test",
        {"tool_progress_mode": "off", "tool_started_at": {}},
    )

    args = {"questions": [{"question": "Pick one", "choices": ["A", "B"]}]}
    result = '{"responses":[{"question":"Pick one","choices_offered":["A","B"],"status":"answered","user_response":"A"}],"outcome":"submitted"}'

    server._on_tool_start("clarify-off-test", "tool-clarify", "clarify", args)
    server._on_tool_complete("clarify-off-test", "tool-clarify", "clarify", args, result)

    assert [event[0] for event in events] == ["tool.start", "tool.complete"]
    assert events[0][2]["name"] == "clarify"
    assert events[0][2]["tool_id"] == "tool-clarify"
    assert events[1][2]["result"]["responses"][0]["user_response"] == "A"


def test_tui_non_interactive_tool_lifecycle_stays_hidden_when_tool_progress_off(monkeypatch):
    events: list[tuple[str, str, dict]] = []
    monkeypatch.setattr(
        server, "_emit", lambda event_type, sid, payload: events.append((event_type, sid, payload))
    )
    monkeypatch.setitem(
        server._sessions,
        "terminal-off-test",
        {"tool_progress_mode": "off", "tool_started_at": {}},
    )

    server._on_tool_start("terminal-off-test", "tool-1", "terminal", {"command": "pwd"})
    server._on_tool_complete("terminal-off-test", "tool-1", "terminal", {"command": "pwd"}, "done")

    assert events == []


def test_dispatch_rejects_non_object_request():
    resp = server.dispatch([])

    assert resp["id"] is None
    assert resp["error"]["code"] == -32600


def test_dispatch_rejects_non_object_params():
    resp = server.dispatch({"id": "1", "method": "session.create", "params": []})

    assert resp["id"] == "1"
    assert resp["error"]["code"] == -32602


def test_system_battery_fails_open(monkeypatch):
    def boom():
        raise RuntimeError("no battery subsystem")

    monkeypatch.setitem(
        sys.modules,
        "agent.battery",
        types.SimpleNamespace(read_battery=boom, battery_category=lambda _s: "dim"),
    )

    resp = server.dispatch({"id": "b2", "method": "system.battery", "params": {}})

    assert resp["result"]["available"] is False
    assert resp["result"]["percent"] is None


def test_config_set_battery_toggles_and_persists(monkeypatch):
    writes: dict[str, object] = {}
    monkeypatch.setattr(server, "_load_cfg", lambda: {"display": {"battery": False}})
    monkeypatch.setattr(
        server, "_write_config_key", lambda k, v: writes.__setitem__(k, v)
    )

    resp = server.dispatch(
        {"id": "c1", "method": "config.set", "params": {"key": "battery", "value": ""}}
    )

    assert resp["result"] == {"key": "battery", "value": "on"}
    assert writes == {"display.battery": True}


def test_config_set_battery_explicit_off(monkeypatch):
    writes: dict[str, object] = {}
    monkeypatch.setattr(server, "_load_cfg", lambda: {"display": {"battery": True}})
    monkeypatch.setattr(
        server, "_write_config_key", lambda k, v: writes.__setitem__(k, v)
    )

    resp = server.dispatch(
        {
            "id": "c2",
            "method": "config.set",
            "params": {"key": "battery", "value": "off"},
        }
    )

    assert resp["result"] == {"key": "battery", "value": "off"}
    assert writes == {"display.battery": False}


def test_voice_toggle_returns_configured_record_key(monkeypatch):
    monkeypatch.setattr(
        server,
        "_load_cfg",
        lambda: {"voice": {"record_key": "ctrl+o"}},
    )
    monkeypatch.setitem(
        sys.modules,
        "tools.voice_mode",
        types.SimpleNamespace(
            check_voice_requirements=lambda: {"available": True, "details": ""}
        ),
    )
    # ``voice.toggle`` action=on mutates ``os.environ["HERMES_VOICE"]``
    # directly (CLI parity, runtime-only flag). Take monkeypatch
    # ownership of the var so the change is reverted at teardown and
    # later tests don't inherit a stale ON state (Copilot round-5
    # review on #19835).
    monkeypatch.setenv("HERMES_VOICE", "0")

    on_resp = _dispatch_sync(
        {"id": "voice-on", "method": "voice.toggle", "params": {"action": "on"}}
    )
    status_resp = _dispatch_sync(
        {"id": "voice-status", "method": "voice.toggle", "params": {"action": "status"}}
    )

    assert on_resp["result"]["record_key"] == "ctrl+o"
    assert status_resp["result"]["record_key"] == "ctrl+o"


def test_voice_toggle_handles_non_dict_voice_cfg(monkeypatch):
    """Round-3 Copilot review regression on #19835.

    ``_load_cfg()`` is raw ``yaml.safe_load()`` output — a hand-edited
    ``voice: true`` / ``voice: cmd+b`` / ``voice: null`` leaves ``voice``
    as a bool/str/None, not a dict. Previously ``.get("record_key")``
    on a non-dict broke every ``voice.toggle`` branch. Now it falls
    back to the documented default.
    """
    monkeypatch.setitem(
        sys.modules,
        "tools.voice_mode",
        types.SimpleNamespace(
            check_voice_requirements=lambda: {"available": True, "details": ""}
        ),
    )

    for bad in (True, "cmd+b", None, 42, ["ctrl+b"]):
        monkeypatch.setattr(server, "_load_cfg", lambda b=bad: {"voice": b})

        status_resp = _dispatch_sync(
            {
                "id": "voice-status",
                "method": "voice.toggle",
                "params": {"action": "status"},
            }
        )

        assert (
            status_resp["result"]["record_key"] == "ctrl+b"
        ), f"voice.record_key fell back to default for voice={bad!r}"

    # Round-4 follow-up: the YAML root itself may be a non-dict. A
    # hand-edit that collapses config.yaml to a scalar / list would
    # otherwise crash ``.get("voice")`` before the inner isinstance
    # guard gets a chance to run.
    for bad_root in (True, None, [], "ctrl+b", 42):
        monkeypatch.setattr(server, "_load_cfg", lambda r=bad_root: r)

        status_resp = _dispatch_sync(
            {
                "id": "voice-status-root",
                "method": "voice.toggle",
                "params": {"action": "status"},
            }
        )

        assert (
            status_resp["result"]["record_key"] == "ctrl+b"
        ), f"voice.record_key fell back to default for root={bad_root!r}"


def test_voice_record_start_handles_non_dict_voice_cfg(monkeypatch):
    """Round-7 Copilot review regression on #19835.

    The ``voice.record`` start path previously read
    ``_load_cfg().get("voice", {}).get(...)`` without any shape checks.
    When ``voice`` is a non-dict (bool/scalar/list) ``get`` raises
    AttributeError and the handler returns 5025 instead of falling
    back to the VAD defaults. Now it uses ``_voice_cfg_dict()`` and
    non-numeric silence values are coerced to the documented defaults.
    """
    captured: dict = {}

    def fake_start_continuous(**kwargs):
        captured.update(kwargs)

    monkeypatch.setitem(
        sys.modules,
        "hermes_cli.voice",
        types.SimpleNamespace(
            start_continuous=fake_start_continuous, stop_continuous=lambda: None
        ),
    )
    monkeypatch.setenv("HERMES_VOICE", "1")

    for bad in (True, "cmd+b", None, 42, ["ctrl+b"], {"silence_threshold": "loud"}):
        captured.clear()
        monkeypatch.setattr(server, "_load_cfg", lambda b=bad: {"voice": b})

        resp = _dispatch_sync(
            {
                "id": "voice-record",
                "method": "voice.record",
                "params": {"action": "start"},
            }
        )

        assert (
            "result" in resp
        ), f"voice.record raised for voice={bad!r}: {resp.get('error')}"
        assert resp["result"]["status"] == "recording"
        assert captured["silence_threshold"] == 200
        assert captured["silence_duration"] == 3.0
        assert captured["auto_restart"] is False


    # Round-12 Copilot review regression on #19835: ``bool`` is a subclass
    # of ``int``, so the naive ``isinstance(threshold, (int, float))``
    # guard would forward ``silence_threshold: true`` as ``1`` instead
    # of falling back to the documented 200 default.
    for bad_bool_cfg in (
        {"silence_threshold": True, "silence_duration": False},
        {"silence_threshold": False},
        {"silence_duration": True},
    ):
        captured.clear()
        monkeypatch.setattr(server, "_load_cfg", lambda c=bad_bool_cfg: {"voice": c})

        resp = _dispatch_sync(
            {
                "id": "voice-record-bool",
                "method": "voice.record",
                "params": {"action": "start"},
            }
        )

        assert "result" in resp, f"voice.record raised for bool cfg={bad_bool_cfg!r}"
        assert (
            captured["silence_threshold"] == 200
        ), f"bool silence_threshold leaked through for {bad_bool_cfg!r}"
        assert (
            captured["silence_duration"] == 3.0
        ), f"bool silence_duration leaked through for {bad_bool_cfg!r}"
        assert captured["auto_restart"] is False


def test_prompt_submit_typed_stop_phrase_ends_voice_chat(monkeypatch):
    """Typed bare stop phrase during an active voice chat is consumed at the
    prompt.submit choke point: voice mode flips off, a distinct
    voice.transcript {stop_phrase, typed} event fires, and NO turn starts.
    """
    calls = {"stop_continuous": 0}
    emitted = []
    monkeypatch.setattr(
        server, "_emit", lambda event, sid, payload=None: emitted.append((event, payload))
    )
    monkeypatch.setitem(
        sys.modules,
        "tools.voice_mode",
        types.SimpleNamespace(
            is_voice_stop_phrase=lambda t: t.strip().lower().strip(".!?") == "stop"
        ),
    )
    monkeypatch.setitem(
        sys.modules,
        "hermes_cli.voice",
        types.SimpleNamespace(
            stop_continuous=lambda force_transcribe=False: calls.__setitem__(
                "stop_continuous", calls["stop_continuous"] + 1
            )
        ),
    )
    monkeypatch.setattr(server, "_tts_stream_stop", lambda user_barge=False: None)
    monkeypatch.setenv("HERMES_VOICE", "1")
    monkeypatch.setenv("HERMES_VOICE_TTS", "1")

    resp = server.dispatch(
        {
            "id": "typed-stop",
            "method": "prompt.submit",
            "params": {"session_id": "any-sid", "text": "Stop."},
        }
    )

    assert resp["result"] == {"voice_stopped": True}
    assert os.environ["HERMES_VOICE"] == "0"
    assert os.environ["HERMES_VOICE_TTS"] == "0"
    assert calls["stop_continuous"] == 1
    assert ("voice.transcript", {"stop_phrase": True, "typed": True}) in emitted


def test_prompt_submit_typed_stop_passes_through_when_voice_off(monkeypatch):
    """Outside a voice chat, typed "stop" is a normal message — the stop
    matcher must not even be consulted (guard is on voice mode)."""
    monkeypatch.setitem(
        sys.modules,
        "tools.voice_mode",
        types.SimpleNamespace(
            is_voice_stop_phrase=lambda t: (_ for _ in ()).throw(
                AssertionError("stop matcher must not run when voice is off")
            )
        ),
    )
    monkeypatch.setenv("HERMES_VOICE", "0")

    resp = server.dispatch(
        {
            "id": "typed-stop-off",
            "method": "prompt.submit",
            "params": {"session_id": "missing-sid", "text": "stop"},
        }
    )

    # The submit proceeds into normal handling (here: unknown session error),
    # NOT the voice_stopped consumption path.
    assert resp.get("result") != {"voice_stopped": True}


def test_wake_owner_is_sticky_and_routes_detection_to_first_transport(monkeypatch):
    from tools import wake_word

    state = {"owner": None, "callback": None, "paused": False}
    voice_callbacks = {}

    def start_listening(callback, *, owner, config, external_audio=False):
        if state["owner"] is not None and state["owner"] is not owner:
            raise wake_word.WakeWordInUse
        state.update(
            owner=owner,
            callback=callback,
            paused=False,
            external_audio=bool(external_audio),
        )

    def pause_listening(*, owner):
        if state["owner"] is not owner:
            return False
        state["paused"] = True
        return True

    def stop_listening(*, owner):
        if state["owner"] is not owner:
            return False
        state.update(owner=None, callback=None, paused=False)
        return True

    def resume_listening(*, owner):
        if state["owner"] is not owner:
            return False
        state["paused"] = False
        return True

    def start_continuous(**callbacks):
        voice_callbacks.update(callbacks)
        return True

    monkeypatch.setattr(wake_word, "load_wake_word_config", lambda: {
        "enabled": True,
        "phrase": "hey hermes",
        "surface": "auto",
        "start_new_session": True,
    })
    monkeypatch.setattr(wake_word, "check_wake_word_requirements", lambda _cfg: {
        "available": True,
        "phrase": "hey hermes",
        "provider": "test",
        "hint": "",
    })
    monkeypatch.setattr(wake_word, "start_listening", start_listening)
    monkeypatch.setattr(wake_word, "pause_listening", pause_listening)
    monkeypatch.setattr(wake_word, "stop_listening", stop_listening)
    monkeypatch.setattr(wake_word, "owns_listener", lambda owner: state["owner"] is owner)
    monkeypatch.setattr(
        wake_word,
        "is_listening",
        lambda: state["owner"] is not None and not state["paused"],
    )
    monkeypatch.setattr(
        wake_word,
        "resume_listening",
        resume_listening,
    )
    monkeypatch.setitem(
        sys.modules,
        "hermes_cli.voice",
        types.SimpleNamespace(
            start_continuous=start_continuous,
            stop_continuous=lambda **_kwargs: None,
        ),
    )
    monkeypatch.setenv("HERMES_VOICE", "1")

    first = types.SimpleNamespace(_closed=False)
    second = types.SimpleNamespace(_closed=False)
    emitted = []
    monkeypatch.setattr(
        server,
        "_emit",
        lambda event, sid, payload: emitted.append(
            (event, sid, payload, server.current_transport())
        ),
    )
    server._wake_owner_transport = None
    server._wake_owner_surface = ""
    try:
        started = _dispatch_sync({
            "id": "wake-1",
            "method": "wake.start",
            "params": {"surface": "gui", "session_id": "first-session"},
        }, transport=first)
        denied = _dispatch_sync({
            "id": "wake-2",
            "method": "wake.start",
            "params": {"surface": "tui", "session_id": "second-session"},
        }, transport=second)
        denied_stop = server.dispatch({
            "id": "wake-stop-2",
            "method": "wake.stop",
            "params": {},
        }, transport=second)
        denied_voice_stop = _dispatch_sync({
            "id": "voice-stop-2",
            "method": "voice.record",
            "params": {"action": "stop"},
        }, transport=second)

        assert started["result"]["started"] is True
        assert denied["result"] == {
            "started": False,
            "reason": "owned",
            "owner_surface": "gui",
        }
        assert denied_stop["result"] == {
            "stopped": False,
            "reason": "not_owner",
            "disabled_persisted": False,
        }
        assert denied_voice_stop["result"] == {
            "status": "busy",
            "reason": "wake_owned",
        }

        state["callback"]()
        assert emitted == [(
            "wake.detected",
            "first-session",
            {"phrase": "hey hermes", "profile": None, "start_new_session": True},
            first,
        )]
        assert state["paused"] is True

        voice_started = _dispatch_sync({
            "id": "voice-start-1",
            "method": "voice.record",
            "params": {"action": "start", "session_id": "first-session"},
        }, transport=first)
        assert voice_started["result"]["status"] == "recording"
        voice_callbacks["on_status"]("idle")
        assert state["paused"] is False

        stopped = server.dispatch({
            "id": "wake-stop-1",
            "method": "wake.stop",
            "params": {},
        }, transport=first)
        assert stopped["result"] == {
            "stopped": True,
            "reason": None,
            "disabled_persisted": False,
        }

        reclaimed = _dispatch_sync({
            "id": "wake-reclaim-2",
            "method": "wake.start",
            "params": {"surface": "tui", "session_id": "second-session"},
        }, transport=second)
        assert reclaimed["result"]["started"] is True
        assert state["owner"] is second

        state["callback"]()
        assert emitted[-1] == (
            "wake.detected",
            "second-session",
            {"phrase": "hey hermes", "profile": None, "start_new_session": True},
            second,
        )

        stopped_again = server.dispatch({
            "id": "wake-stop-2-after-reclaim",
            "method": "wake.stop",
            "params": {},
        }, transport=second)
        assert stopped_again["result"] == {
            "stopped": True,
            "reason": None,
            "disabled_persisted": False,
        }
    finally:
        server._wake_owner_transport = None
        server._wake_owner_surface = ""


def test_wake_toggle_persists_enabled_flag_only_on_explicit_gesture(monkeypatch):
    """The ear toggle / /wake on|off write wake_word.enabled; auto-arm never does."""
    from tools import wake_word

    config = {"enabled": False, "phrase": "hey hermes", "surface": "auto",
              "start_new_session": True}
    persisted = []

    def fake_persist(enabled):
        persisted.append(enabled)
        config["enabled"] = enabled
        return True

    monkeypatch.setattr(server, "_persist_wake_enabled", fake_persist)
    monkeypatch.setattr(wake_word, "load_wake_word_config", lambda: dict(config))
    monkeypatch.setattr(wake_word, "check_wake_word_requirements", lambda _cfg: {
        "available": True,
        "phrase": "hey hermes",
        "provider": "test",
        "hint": "",
    })
    listener = {"owner": None}
    monkeypatch.setattr(
        wake_word, "start_listening",
        lambda callback, *, owner, config, external_audio=False: listener.update(
            owner=owner, external_audio=bool(external_audio)
        ),
    )
    monkeypatch.setattr(
        wake_word, "stop_listening",
        lambda *, owner: listener["owner"] is owner and not listener.update(owner=None),
    )
    monkeypatch.setattr(wake_word, "owns_listener", lambda owner: listener["owner"] is owner)

    transport = types.SimpleNamespace(_closed=False)
    server._wake_owner_transport = None
    server._wake_owner_surface = ""
    try:
        # Passive auto-arm (no persist): refused, config untouched.
        passive = _dispatch_sync({
            "id": "wake-passive",
            "method": "wake.start",
            "params": {"surface": "gui"},
        }, transport=transport)
        assert passive["result"] == {"started": False, "reason": "disabled"}
        assert persisted == []

        # Explicit gesture: enables in config AND arms.
        clicked = _dispatch_sync({
            "id": "wake-click",
            "method": "wake.start",
            "params": {"surface": "gui", "persist": True},
        }, transport=transport)
        assert clicked["result"]["started"] is True
        assert clicked["result"]["enabled_persisted"] is True
        assert persisted == [True]

        # Explicit stop: disables in config.
        stopped = server.dispatch({
            "id": "wake-click-off",
            "method": "wake.stop",
            "params": {"persist": True},
        }, transport=transport)
        assert stopped["result"]["stopped"] is True
        assert stopped["result"]["disabled_persisted"] is True
        assert persisted == [True, False]

        # persist does NOT override an explicit surface scoping.
        config.update(enabled=True, surface="tui")
        scoped = _dispatch_sync({
            "id": "wake-scoped",
            "method": "wake.start",
            "params": {"surface": "gui", "persist": True},
        }, transport=transport)
        assert scoped["result"] == {"started": False, "reason": "disabled_for_surface"}
        assert persisted == [True, False]
    finally:
        server._wake_owner_transport = None
        server._wake_owner_surface = ""


def test_voice_record_start_forwards_max_recording_seconds(monkeypatch):
    """voice.max_recording_seconds must reach start_continuous from the TUI.

    The CLI wiring alone doesn't cover TUI recordings: the gateway forwards
    recorder params explicitly, so a missing kwarg here silently leaves the
    cap dead in the TUI while CLI tests stay green. Semantics mirror the
    silence params: non-numeric / bool / missing falls back to the documented
    120 default, an explicit numeric value <= 0 disables the cap.
    """
    captured: dict = {}

    def fake_start_continuous(**kwargs):
        captured.update(kwargs)

    monkeypatch.setitem(
        sys.modules,
        "hermes_cli.voice",
        types.SimpleNamespace(
            start_continuous=fake_start_continuous, stop_continuous=lambda: None
        ),
    )
    monkeypatch.setenv("HERMES_VOICE", "1")

    for cfg, expected in (
        ({"max_recording_seconds": 45}, 45),        # explicit cap forwarded as-is
        ({"max_recording_seconds": 0}, 0.0),        # explicit 0 = disabled
        ({"max_recording_seconds": -5}, 0.0),       # negative = disabled
        ({}, 120.0),                                # missing = documented default
        ({"max_recording_seconds": True}, 120.0),   # bool must not become 1s cap
        ({"max_recording_seconds": "long"}, 120.0), # garbage = documented default
    ):
        captured.clear()
        monkeypatch.setattr(server, "_load_cfg", lambda c=cfg: {"voice": c})

        resp = _dispatch_sync(
            {
                "id": "voice-record-cap",
                "method": "voice.record",
                "params": {"action": "start"},
            }
        )

        assert "result" in resp, f"voice.record raised for cfg={cfg!r}: {resp.get('error')}"
        assert resp["result"]["status"] == "recording"
        assert (
            captured["max_recording_seconds"] == expected
        ), f"cfg={cfg!r} forwarded {captured.get('max_recording_seconds')!r}, expected {expected!r}"


def test_voice_record_start_cuts_inflight_tts(monkeypatch):
    """PTT barge-in (#40010): arming the mic cuts in-flight streaming TTS.

    The CLI record-key handler already cuts TTS before starting a capture;
    the gateway's ``voice.record`` start path did not, so a TUI/Desktop user
    pressing push-to-talk mid-reply kept hearing the agent talk over them.
    The cut must happen before ``start_continuous`` arms the mic, and must
    latch as a user barge (the next turn's model note), not a mode change.
    """
    order: list[str] = []
    tts_calls: list[bool] = []

    def fake_start_continuous(**_kwargs):
        order.append("start_continuous")
        return True

    monkeypatch.setitem(
        sys.modules,
        "hermes_cli.voice",
        types.SimpleNamespace(
            start_continuous=fake_start_continuous, stop_continuous=lambda **_kwargs: None
        ),
    )
    monkeypatch.setattr(server, "_load_cfg", lambda: {"voice": {}})
    monkeypatch.setattr(
        server, "_tts_stream_stop", lambda user_barge=True: tts_calls.append(user_barge)
    )
    monkeypatch.setenv("HERMES_VOICE", "1")

    resp = _dispatch_sync(
        {
            "id": "voice-record-tts-cut",
            "method": "voice.record",
            "params": {"action": "start"},
        }
    )

    assert resp is not None and "result" in resp, f"voice.record raised: {resp and resp.get('error')}"
    assert tts_calls == [True], "PTT start must cut TTS as a user barge"
    assert order == ["start_continuous"], "start_continuous must still arm the mic"


def test_voice_record_stop_forces_transcription(monkeypatch):
    captured: dict = {}

    def fake_stop_continuous(**kwargs):
        captured.update(kwargs)

    monkeypatch.setitem(
        sys.modules,
        "hermes_cli.voice",
        types.SimpleNamespace(
            start_continuous=lambda **_kwargs: None,
            stop_continuous=fake_stop_continuous,
        ),
    )

    resp = _dispatch_sync(
        {
            "id": "voice-record-stop",
            "method": "voice.record",
            "params": {"action": "stop"},
        }
    )

    assert resp["result"]["status"] == "stopped"
    assert captured["force_transcribe"] is True


def test_voice_record_stop_updates_event_session_id(monkeypatch):
    monkeypatch.setitem(
        sys.modules,
        "hermes_cli.voice",
        types.SimpleNamespace(
            start_continuous=lambda **_kwargs: True,
            stop_continuous=lambda **_kwargs: None,
        ),
    )
    monkeypatch.setattr(server, "_voice_event_sid", "old-session")

    resp = _dispatch_sync(
        {
            "id": "voice-record-stop-session",
            "method": "voice.record",
            "params": {"action": "stop", "session_id": "new-session"},
        }
    )

    assert resp["result"]["status"] == "stopped"
    assert server._voice_event_sid == "new-session"


def test_voice_record_start_reports_busy_when_stop_is_in_progress(monkeypatch):
    monkeypatch.setitem(
        sys.modules,
        "hermes_cli.voice",
        types.SimpleNamespace(
            start_continuous=lambda **_kwargs: False,
            stop_continuous=lambda **_kwargs: None,
        ),
    )
    monkeypatch.setenv("HERMES_VOICE", "1")
    monkeypatch.setattr(server, "_load_cfg", lambda: {"voice": {}})

    resp = _dispatch_sync(
        {
            "id": "voice-record-busy",
            "method": "voice.record",
            "params": {"action": "start"},
        }
    )

    assert resp["result"]["status"] == "busy"


def test_voice_toggle_tts_branch_also_carries_record_key(monkeypatch):
    """Round-2 Copilot review regression on #19835.

    The ``tts`` branch used to omit ``record_key`` from its response, so a
    TUI client would parse ``r.record_key ?? 'ctrl+b'`` and reset a
    custom binding to the default on every TTS toggle. Every branch of
    ``voice.toggle`` now carries the configured key so frontend state
    stays authoritative.
    """
    monkeypatch.setattr(
        server,
        "_load_cfg",
        lambda: {"voice": {"record_key": "ctrl+space"}},
    )
    monkeypatch.setitem(
        sys.modules,
        "tools.voice_mode",
        types.SimpleNamespace(
            check_voice_requirements=lambda: {"available": True, "details": ""}
        ),
    )
    monkeypatch.setenv("HERMES_VOICE", "1")
    # setenv (not delenv) — the handler writes HERMES_VOICE_TTS directly, and
    # delenv on an absent var registers no teardown, leaking TTS=1 into every
    # later test in the file (which now spins up the streaming TTS pipeline).
    monkeypatch.setenv("HERMES_VOICE_TTS", "0")

    tts_resp = _dispatch_sync(
        {"id": "voice-tts", "method": "voice.toggle", "params": {"action": "tts"}}
    )

    assert tts_resp["result"]["record_key"] == "ctrl+space"
    assert tts_resp["result"]["tts"] is True


def test_load_enabled_toolsets_prefers_tui_env(monkeypatch):
    monkeypatch.setenv("HERMES_TUI_TOOLSETS", "web, terminal, ,memory")

    assert server._load_enabled_toolsets() == ["web", "terminal", "memory"]


def test_load_enabled_toolsets_filters_invalid_tui_env(monkeypatch, capsys):
    monkeypatch.setenv("HERMES_TUI_TOOLSETS", "web, nope")
    monkeypatch.setitem(
        sys.modules,
        "hermes_cli.plugins",
        types.SimpleNamespace(discover_plugins=lambda: None),
    )

    assert server._load_enabled_toolsets() == ["web"]
    assert "nope" in capsys.readouterr().err


def test_load_enabled_toolsets_accepts_plugin_env_after_discovery(monkeypatch):
    monkeypatch.setenv("HERMES_TUI_TOOLSETS", "plugin_demo")

    import toolsets

    discovered = {"ready": False}
    original_validate = toolsets.validate_toolset

    def fake_validate(name):
        return name == "plugin_demo" and discovered["ready"] or original_validate(name)

    monkeypatch.setattr(toolsets, "validate_toolset", fake_validate)
    monkeypatch.setitem(
        sys.modules,
        "hermes_cli.plugins",
        types.SimpleNamespace(
            discover_plugins=lambda: discovered.update({"ready": True})
        ),
    )

    assert server._load_enabled_toolsets() == ["plugin_demo"]


def test_load_enabled_toolsets_folds_project_into_focus_posture(monkeypatch):
    # Focus-mode coding posture returns before the config fallback, but it's
    # still a GUI-only resolver — `project` must come along so the desktop keeps
    # the project tools while sitting in a repo.
    monkeypatch.delenv("HERMES_TUI_TOOLSETS", raising=False)

    import agent.coding_context as cc

    monkeypatch.setattr(cc, "coding_selection", lambda **_: ["coding", "figma"])

    assert server._load_enabled_toolsets("tui") == ["coding", "figma", "project"]



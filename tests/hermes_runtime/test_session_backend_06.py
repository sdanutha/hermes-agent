import contextlib










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


def test_ws_orphan_reap_disabled_when_grace_zero(monkeypatch):
    """Grace=0 disables the reaper entirely (pre-fix park-forever behaviour)."""
    fired = {"timer": False}

    class _Timer:
        def __init__(self, *a, **k):
            fired["timer"] = True

        def start(self):
            pass

    monkeypatch.setattr(server, "_WS_ORPHAN_REAP_GRACE_S", 0.0)
    monkeypatch.setattr(server.threading, "Timer", _Timer)
    server._schedule_ws_orphan_reap("any-sid")
    assert fired["timer"] is False


def test_ws_orphan_reap_defers_running_turn_with_fresh_activity(monkeypatch):
    """#98028/#100325: a client-absent turn whose activity clock is fresh is
    NOT interrupted — it keeps running detached and the reaper re-polls at the
    grace interval. Once the clock goes stale the wedged-turn interrupt fires,
    and after the turn settles the session is reaped as before."""
    callbacks = []
    delays = []
    interrupted = []
    torn_down = []

    class _Timer:
        def __init__(self, delay, callback):
            delays.append(delay)
            callbacks.append(callback)
            self.daemon = False

        def start(self):
            return None

    activity = {"seconds_since_activity": 1.0}
    agent = types.SimpleNamespace(
        get_activity_summary=lambda: dict(activity),
        interrupt=lambda message=None: interrupted.append("interrupted"),
    )

    class _DeadThread:
        def is_alive(self):
            return False

    session = _session(
        agent=agent,
        transport=server._detached_ws_transport,
        running=True,
        _run_thread=_DeadThread(),
    )
    server._sessions["fresh-sid"] = session
    monkeypatch.setattr(server, "_WS_ORPHAN_REAP_GRACE_S", 0.01)
    monkeypatch.setattr(server, "_WS_ORPHAN_ACTIVITY_STALE_S", 300.0)
    monkeypatch.setattr(server.threading, "Timer", _Timer)
    monkeypatch.setattr(server, "_load_cfg", dict)
    monkeypatch.setattr(
        server,
        "_teardown_popped_session",
        lambda claimed, *, end_reason: torn_down.append((claimed, end_reason)) or True,
    )

    try:
        server._schedule_ws_orphan_reap("fresh-sid")

        # Two grace cycles with fresh activity: no interrupt, reschedule at
        # the GRACE interval (not the 1s interrupt-settle poll).
        for _ in range(2):
            callbacks.pop(0)()
            assert interrupted == []
            assert not session.get("_client_gone_interrupt_requested")
            assert delays[-1] == server._WS_ORPHAN_REAP_GRACE_S
            assert "fresh-sid" in server._sessions

        # Activity goes stale (turn wedged) -> interrupt fires on next poll.
        activity["seconds_since_activity"] = 301.0
        callbacks.pop(0)()
        assert interrupted == ["interrupted"]
        assert session["_client_gone_interrupt_requested"] is True

        # Turn settles -> reap proceeds exactly as today.
        session["running"] = False
        callbacks.pop(0)()
        assert "fresh-sid" not in server._sessions
        assert torn_down == [(session, "ws_orphan_reap")]
    finally:
        server._sessions.pop("fresh-sid", None)


def test_ws_orphan_activity_gate_zero_restores_interrupt_at_grace(monkeypatch):
    """ws_orphan_activity_stale_s=0 opts out: fresh activity no longer defers
    the client-gone interrupt (pre-#98028 behaviour)."""
    callbacks = []
    interrupted = []

    class _Timer:
        def __init__(self, _delay, callback):
            callbacks.append(callback)
            self.daemon = False

        def start(self):
            return None

    class _LiveThread:
        def is_alive(self):
            return True

    agent = types.SimpleNamespace(
        get_activity_summary=lambda: {"seconds_since_activity": 0.5},
        interrupt=lambda message=None: interrupted.append("interrupted"),
    )
    session = _session(
        agent=agent,
        transport=server._detached_ws_transport,
        running=True,
        _run_thread=_LiveThread(),
    )
    server._sessions["optout-sid"] = session
    monkeypatch.setattr(server, "_WS_ORPHAN_REAP_GRACE_S", 0.01)
    monkeypatch.setattr(server, "_WS_ORPHAN_ACTIVITY_STALE_S", 0.0)
    monkeypatch.setattr(server.threading, "Timer", _Timer)
    monkeypatch.setattr(server, "_load_cfg", dict)

    try:
        server._schedule_ws_orphan_reap("optout-sid")
        callbacks.pop(0)()
        assert interrupted == ["interrupted"]
        assert session["_client_gone_interrupt_requested"] is True
    finally:
        server._sessions.pop("optout-sid", None)


def test_ws_orphan_activity_gate_unreadable_summary_stays_eligible(monkeypatch):
    """A broken/opaque activity summary must fail CLOSED (not fresh): the
    wedged-turn interrupt-at-grace safety net is preserved."""

    def _boom():
        raise RuntimeError("summary unavailable")

    agent = types.SimpleNamespace(get_activity_summary=_boom)
    monkeypatch.setattr(server, "_WS_ORPHAN_ACTIVITY_STALE_S", 300.0)
    assert server._ws_orphan_turn_activity_is_fresh({"agent": agent}) is False
    # No agent / no summary method: same conservative answer.
    assert server._ws_orphan_turn_activity_is_fresh({"agent": None}) is False
    assert (
        server._ws_orphan_turn_activity_is_fresh(
            {"agent": types.SimpleNamespace()}
        )
        is False
    )
    # Never-stamped clock (None) is not fresh either.
    agent2 = types.SimpleNamespace(
        get_activity_summary=lambda: {"seconds_since_activity": None}
    )
    assert server._ws_orphan_turn_activity_is_fresh({"agent": agent2}) is False


def test_init_session_fires_reset_hook(monkeypatch):
    hooks = []

    class _FakeWorker:
        def __init__(self, key, model, profile_home=None):
            self.key = key

        def close(self):
            return None

    monkeypatch.setattr(server, "_SlashWorker", _FakeWorker)
    monkeypatch.setattr(server, "_wire_callbacks", lambda _sid: None)
    monkeypatch.setattr(server, "_emit", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        server,
        "_notify_session_boundary",
        lambda event, session_id, *_args: hooks.append((event, session_id)),
    )

    import tools.approval as _approval

    monkeypatch.setattr(_approval, "register_gateway_notify", lambda key, cb: None)
    monkeypatch.setattr(_approval, "load_permanent_allowlist", lambda: None)

    sid = "sid"
    try:
        server._init_session(
            sid,
            "session-key",
            types.SimpleNamespace(model="x"),
            history=[],
            cols=80,
        )
        assert ("on_session_reset", "session-key") in hooks
    finally:
        server._sessions.pop(sid, None)


def test_session_title_creates_row_and_sets_immediately_when_not_ready(monkeypatch):
    """An explicit /title before the first message must persist NOW, not queue.

    Regression: the desktop deferred the DB row to the first prompt, so a
    /title typed before any message only stashed ``pending_title`` and relied
    on a post-turn apply block. When that turn never landed under the session
    key, the title was silently lost and the sidebar fell back to the message
    preview. The handler now creates the row up front (mirroring the messaging
    gateway) so an explicit /title takes effect immediately.
    """
    state = {"row": None, "title": None, "ensured": False}

    class _FakeDB:
        def get_session_title(self, _key):
            return state["title"]

        def get_session(self, _key):
            return state["row"]

        def set_session_title(self, _key, title):
            # Mirrors SessionDB: UPDATE affects 0 rows until the row exists.
            if state["row"] is None:
                return False
            state["title"] = title
            return True

    fake_db = _FakeDB()

    def _fake_ensure_row(_session):
        # The real _ensure_session_db_row does an INSERT OR IGNORE.
        state["ensured"] = True
        state["row"] = {"id": "session-key", "title": None}


    @contextlib.contextmanager
    def _fake_session_db(_session):
        yield fake_db

    server._sessions["sid"] = _session(pending_title=None)
    monkeypatch.setattr(server, "_get_db", lambda: fake_db)
    monkeypatch.setattr(server, "_ensure_session_db_row", _fake_ensure_row)
    monkeypatch.setattr(server, "_session_db", _fake_session_db)
    try:
        set_resp = server.handle_request(
            {
                "id": "1",
                "method": "session.title",
                "params": {"session_id": "sid", "title": "my-custom-name"},
            }
        )

        # No longer queued — the row is created and the title set immediately.
        assert set_resp["result"]["pending"] is False
        assert set_resp["result"]["title"] == "my-custom-name"
        assert state["ensured"] is True, "the row must be created up front"
        assert state["title"] == "my-custom-name"
        assert server._sessions["sid"]["pending_title"] is None

        # A subsequent read reflects the persisted title.
        get_resp = server.handle_request(
            {"id": "2", "method": "session.title", "params": {"session_id": "sid"}}
        )
        assert get_resp["result"]["title"] == "my-custom-name"
    finally:
        server._sessions.pop("sid", None)


def test_session_title_falls_back_to_queue_when_row_create_fails(monkeypatch):
    """If row creation can't take (DB down / racing writer), keep the queue.

    The post-turn apply block is still the recovery path, so a /title that
    can't persist up front must not be dropped — it falls back to
    ``pending_title`` exactly as before.
    """

    class _FakeDB:
        def get_session_title(self, _key):
            return None

        def get_session(self, _key):
            return None

        def set_session_title(self, _key, _title):
            return False

    fake_db = _FakeDB()

    def _fake_ensure_row(_session):
        # Simulate a persist that didn't take — row still absent.
        pass


    @contextlib.contextmanager
    def _fake_session_db(_session):
        yield fake_db

    server._sessions["sid"] = _session(pending_title=None)
    monkeypatch.setattr(server, "_get_db", lambda: fake_db)
    monkeypatch.setattr(server, "_ensure_session_db_row", _fake_ensure_row)
    monkeypatch.setattr(server, "_session_db", _fake_session_db)
    try:
        set_resp = server.handle_request(
            {
                "id": "1",
                "method": "session.title",
                "params": {"session_id": "sid", "title": "queued title"},
            }
        )

        assert set_resp["result"]["pending"] is True
        assert set_resp["result"]["title"] == "queued title"
        assert server._sessions["sid"]["pending_title"] == "queued title"

        get_resp = server.handle_request(
            {"id": "2", "method": "session.title", "params": {"session_id": "sid"}}
        )
        assert get_resp["result"]["title"] == "queued title"
    finally:
        server._sessions.pop("sid", None)


def test_notification_event_routing_by_session_key(monkeypatch):
    """Background-process events surface only in the session that owns them."""
    mine = _session(session_key="mine")
    other = _session(session_key="other")
    monkeypatch.setattr(server, "_sessions", {"a": mine, "b": other})

    # My own event → handle it.
    assert server._notification_event_belongs_elsewhere("a", mine, {"session_key": "mine"}) is False
    # Global/system event with no owner → handle it.
    assert server._notification_event_belongs_elsewhere("a", mine, {"session_key": ""}) is False
    assert server._notification_event_belongs_elsewhere("a", mine, {}) is False
    # Owned by another *live* session → defer to that session's poller.
    assert server._notification_event_belongs_elsewhere("a", mine, {"session_key": "other"}) is True
    # Owner is gone (not in _sessions) → handle as fallback so it isn't lost.
    assert server._notification_event_belongs_elsewhere("a", mine, {"session_key": "ghost"}) is False


def test_async_delegation_event_prefers_origin_ui_session(monkeypatch):
    """Detached subagent completions return to the commissioning TUI tab.

    Regression: when the durable session key was stale/orphaned, whichever
    desktop poller woke first could consume the async result and inject it into
    an unrelated session.
    """
    mine = _session(session_key="current-key")
    other = _session(session_key="unrelated-key")
    monkeypatch.setattr(server, "_sessions", {"origin-sid": mine, "other-sid": other})
    monkeypatch.setattr(server, "_get_db", lambda: None)
    evt = {
        "type": "async_delegation",
        "session_key": "stale-or-rotated-key",
        "origin_ui_session_id": "origin-sid",
    }

    assert server._notification_event_belongs_elsewhere("other-sid", other, evt) is True
    assert server._notification_event_belongs_elsewhere("origin-sid", mine, evt) is False


def test_notification_event_follows_compression_continuation(monkeypatch):
    """Events keyed to a compressed parent route to the live continuation."""
    old_parent = _session(session_key="old-parent")
    live_tip = _session(session_key="new-tip")
    monkeypatch.setattr(server, "_sessions", {"old-sid": old_parent, "tip-sid": live_tip})

    class _DB:
        def resolve_resume_session_id(self, session_id):
            return "new-tip" if session_id == "old-parent" else session_id

    monkeypatch.setattr(server, "_get_db", lambda: _DB())
    evt = {"type": "async_delegation", "session_key": "old-parent"}

    assert server._notification_event_belongs_elsewhere("old-sid", old_parent, evt) is True
    assert server._notification_event_belongs_elsewhere("tip-sid", live_tip, evt) is False
    # A third session must leave it alone for the continuation's poller.
    third = _session(session_key="third")
    monkeypatch.setattr(
        server,
        "_sessions",
        {"old-sid": old_parent, "tip-sid": live_tip, "third-sid": third},
    )
    assert server._notification_event_belongs_elsewhere("third-sid", third, evt) is True


def test_finalized_origin_ui_session_falls_back_to_live_continuation(monkeypatch):
    """A closed origin tab must not steal its resumed continuation's result."""
    finalized_origin = _session(session_key="old-parent", _finalized=True)
    live_tip = _session(session_key="new-tip")
    monkeypatch.setattr(
        server,
        "_sessions",
        {"origin-sid": finalized_origin, "tip-sid": live_tip},
    )

    class _DB:
        def resolve_resume_session_id(self, session_id):
            return "new-tip" if session_id == "old-parent" else session_id

    monkeypatch.setattr(server, "_get_db", lambda: _DB())
    evt = {
        "type": "async_delegation",
        "session_key": "old-parent",
        "origin_ui_session_id": "origin-sid",
    }

    assert server._notification_event_belongs_elsewhere("origin-sid", finalized_origin, evt) is True
    assert server._notification_event_belongs_elsewhere("tip-sid", live_tip, evt) is False


def test_prompt_submit_rejects_negative_truncate_ordinal(monkeypatch):
    """A negative truncate_before_user_ordinal must be rejected, not honoured.

    The handler validates the upper bound (`ordinal >= len(user_indices)`) but a
    negative ordinal would otherwise slip through and hit Python negative
    indexing: `user_indices[-1]` selects the LAST user turn, truncating history
    to everything before it and persisting that loss via replace_messages — an
    unrecoverable overwrite of the session DB. Reject it on the safe 4018 path
    and leave the in-memory history and the DB untouched.
    """
    replaced = []

    class _FakeDB:
        def replace_messages(
            self,
            key,
            messages,
            active_only=False,
            archive_dropped=False,
            reject_active_turn_lease=False,
        ):
            replaced.append((key, list(messages)))

    history = [
        {"role": "user", "content": "first"},
        {"role": "assistant", "content": "ok"},
        {"role": "user", "content": "second"},
        {"role": "assistant", "content": "done"},
    ]
    server._sessions["trunc-sid"] = _session(history=list(history))
    monkeypatch.setattr(server, "_get_db", lambda: _FakeDB())
    # If the guard ever lets a negative ordinal through, these would run and the
    # session would be marked busy; failing here makes that regression loud.
    monkeypatch.setattr(
        server, "_start_agent_build", lambda *a, **k: pytest.fail("must not start a turn")
    )
    monkeypatch.setattr(
        server, "_start_inflight_turn", lambda *a, **k: pytest.fail("must not start a turn")
    )

    try:
        resp = server.handle_request(
            {
                "id": "1",
                "method": "prompt.submit",
                "params": {
                    "session_id": "trunc-sid",
                    "text": "next",
                    "truncate_before_user_ordinal": -1,
                    "confirm_truncate": True,
                },
            }
        )
        assert resp["error"]["code"] == 4018
        # History and the DB are left exactly as they were — no silent loss.
        assert server._sessions["trunc-sid"]["history"] == history
        assert server._sessions["trunc-sid"]["running"] is False
        assert replaced == []
    finally:
        server._sessions.pop("trunc-sid", None)


def test_prompt_submit_refuses_boolean_ordinal(monkeypatch):
    """A JSON `true` ordinal must return 4004, not coerce to turn 1.

    bool is an int subclass, so `int(True) == 1`: a client bug that sends
    `truncate_before_user_ordinal: true` with confirm_truncate would aim a
    confirmed rewind at the SECOND user turn and hard-truncate everything
    after the first — the same silent-loss class as #82756.
    """
    history = [
        {"role": "user", "content": "first"},
        {"role": "assistant", "content": "reply 1"},
        {"role": "user", "content": "second"},
        {"role": "assistant", "content": "reply 2"},
    ]
    server._sessions["bool-trunc-sid"] = _session(history=list(history))
    monkeypatch.setattr(
        server, "_start_agent_build", lambda *a, **k: pytest.fail("must not start a turn")
    )
    monkeypatch.setattr(
        server, "_start_inflight_turn", lambda *a, **k: pytest.fail("must not start a turn")
    )

    try:
        resp = server.handle_request(
            {
                "id": "1",
                "method": "prompt.submit",
                "params": {
                    "session_id": "bool-trunc-sid",
                    "text": "new turn",
                    "truncate_before_user_ordinal": True,
                    "confirm_truncate": True,
                },
            }
        )
        assert resp.get("error") is not None
        assert resp["error"]["code"] == 4004
        assert server._sessions["bool-trunc-sid"]["history"] == history
    finally:
        server._sessions.pop("bool-trunc-sid", None)


def test_prompt_submit_refuses_confirm_truncate_without_target(monkeypatch):
    """confirm_truncate with no ordinal is leaked rewind state — fail fast.

    The desktop auto-attaches confirm_truncate whenever it builds truncation
    params (#82756); a bare flag on an ordinary submit means the client's
    composer state is corrupted. Refusing loudly surfaces the client bug
    instead of quietly ignoring the flag.
    """
    history = [
        {"role": "user", "content": "first"},
        {"role": "assistant", "content": "reply 1"},
    ]
    server._sessions["bare-confirm-sid"] = _session(history=list(history))
    monkeypatch.setattr(
        server, "_start_agent_build", lambda *a, **k: pytest.fail("must not start a turn")
    )
    monkeypatch.setattr(
        server, "_start_inflight_turn", lambda *a, **k: pytest.fail("must not start a turn")
    )

    try:
        resp = server.handle_request(
            {
                "id": "1",
                "method": "prompt.submit",
                "params": {
                    "session_id": "bare-confirm-sid",
                    "text": "new turn",
                    "confirm_truncate": True,
                },
            }
        )
        assert resp.get("error") is not None
        assert resp["error"]["code"] == 4004
        assert server._sessions["bare-confirm-sid"]["history"] == history
    finally:
        server._sessions.pop("bare-confirm-sid", None)


def test_prompt_submit_refuses_unconfirmed_nonempty_truncation(monkeypatch):
    """An ordinal without confirm_truncate must not drop the session tail.

    #80763: a desktop client carried a leftover truncate_before_user_ordinal
    into an ORDINARY submit. The request was indistinguishable from a real
    rewind — in-range ordinal, non-empty result — so the empty-truncation guard
    never fired and replace_messages() DELETEd 244 durable rows (296 -> 52).
    Intent has to be stated: refuse on 4029 and leave memory and DB untouched.
    """
    replaced = []

    class _FakeDB:
        def replace_messages(
            self,
            key,
            messages,
            active_only=False,
            archive_dropped=False,
            reject_active_turn_lease=False,
        ):
            replaced.append((key, list(messages)))

    history = [
        {"role": "user", "content": "first"},
        {"role": "assistant", "content": "ok"},
        {"role": "user", "content": "second"},
        {"role": "assistant", "content": "done"},
        {"role": "user", "content": "third"},
        {"role": "assistant", "content": "sure"},
    ]
    server._sessions["unconfirmed-trunc-sid"] = _session(history=list(history))
    monkeypatch.setattr(server, "_get_db", lambda: _FakeDB())
    monkeypatch.setattr(
        server, "_start_agent_build", lambda *a, **k: pytest.fail("must not start a turn")
    )
    monkeypatch.setattr(
        server, "_start_inflight_turn", lambda *a, **k: pytest.fail("must not start a turn")
    )

    def _submit(**extra):
        return server.handle_request(
            {
                "id": "1",
                "method": "prompt.submit",
                "params": {
                    "session_id": "unconfirmed-trunc-sid",
                    "text": "an ordinary typed message",
                    "truncate_before_user_ordinal": 2,
                    **extra,
                },
            }
        )

    try:
        resp = _submit()
        assert resp["error"]["code"] == 4029
        assert "confirm_truncate" in resp["error"]["message"]
        # Explicit falsey values must not satisfy the opt-in either.
        for falsey in (False, 0, "", "false", "no"):
            assert _submit(confirm_truncate=falsey)["error"]["code"] == 4029, falsey
        # confirm_empty_truncate is a different gate — it must not stand in for
        # rewind intent on a cut that leaves the transcript non-empty.
        assert _submit(confirm_empty_truncate=True)["error"]["code"] == 4029
        session = server._sessions["unconfirmed-trunc-sid"]
        assert session["history"] == history
        assert session["history_version"] == 0
        assert session["running"] is False
        assert replaced == []
    finally:
        server._sessions.pop("unconfirmed-trunc-sid", None)


def test_prompt_submit_truncates_by_message_id(monkeypatch):
    """#82756: truncate_before_message_id resolves target message and cuts history accurately."""
    replaced = []

    class _FakeDB:
        def replace_messages(
            self,
            key,
            messages,
            active_only=False,
            archive_dropped=False,
            reject_active_turn_lease=False,
        ):
            replaced.append((key, list(messages)))

    history = [
        {"id": "msg-1", "role": "user", "content": "first"},
        {"role": "assistant", "content": "reply 1"},
        {"id": "msg-2", "role": "user", "content": "second"},
        {"role": "assistant", "content": "reply 2"},
    ]
    server._sessions["msg-id-trunc-sid"] = _session(history=list(history))
    monkeypatch.setattr(server, "_get_db", lambda: _FakeDB())
    monkeypatch.setattr(
        server, "_start_agent_build", lambda *a, **k: None
    )
    monkeypatch.setattr(
        server, "_start_inflight_turn", lambda *a, **k: None
    )

    try:
        resp = server.handle_request(
            {
                "id": "1",
                "method": "prompt.submit",
                "params": {
                    "session_id": "msg-id-trunc-sid",
                    "text": "new turn",
                    "truncate_before_message_id": "msg-2",
                    "confirm_truncate": True,
                },
            }
        )
        assert resp.get("result") is not None
        assert len(replaced) == 1
        assert replaced[0][1] == history[:2]
    finally:
        server._sessions.pop("msg-id-trunc-sid", None)


def test_prompt_submit_truncation_falls_back_to_sid_when_session_key_null(monkeypatch):
    """#81904: a NULL session_key must not FK-fail the truncation persist.

    CLI-origin sessions resumed in the Desktop have no session_key; the
    truncation path used to call replace_messages(None, ...), whose reinsert
    violated the messages.session_id FK ("FOREIGN KEY constraint failed" →
    "Restore failed"). The persist must key off the session id instead —
    for CLI-origin rows the durable sessions.id IS the requested sid.
    """
    replaced = []

    class _FakeDB:
        def replace_messages(
            self,
            key,
            messages,
            active_only=False,
            archive_dropped=False,
            reject_active_turn_lease=False,
        ):
            replaced.append((key, list(messages)))

    history = [
        {"_row_id": 101, "role": "user", "content": "first"},
        {"_row_id": 102, "role": "assistant", "content": "reply 1"},
        {"_row_id": 103, "role": "user", "content": "second"},
        {"_row_id": 104, "role": "assistant", "content": "reply 2"},
    ]
    server._sessions["null-key-trunc-sid"] = _session(
        history=list(history), session_key=None
    )
    monkeypatch.setattr(server, "_get_db", lambda: _FakeDB())
    monkeypatch.setattr(server, "_start_agent_build", lambda *a, **k: None)
    monkeypatch.setattr(server, "_start_inflight_turn", lambda *a, **k: None)
    # This invariant ends at durable truncation; no model turn is requested by
    # the fixture. Keep its asynchronous continuation inside this test's scope.
    monkeypatch.setattr(server, "_run_prompt_submit", lambda *a, **k: None)

    try:
        resp = server.handle_request(
            {
                "id": "1",
                "method": "prompt.submit",
                "params": {
                    "session_id": "null-key-trunc-sid",
                    "text": "new turn",
                    "truncate_before_row_id": 103,
                    "truncate_before_user_ordinal": 1,
                    "confirm_truncate": True,
                },
            }
        )
        assert resp.get("result") is not None
        assert len(replaced) == 1
        # Keyed by the session id, never None (the FK-violating value).
        assert replaced[0][0] == "null-key-trunc-sid"
        assert replaced[0][1] == history[:2]
    finally:
        # The prompt worker resolves server bindings when it runs. Reap it before
        # monkeypatch teardown or it can consume the next test's notification.
        session = server._sessions.get("null-key-trunc-sid")
        worker = session.get("_run_thread") if session else None
        if worker is not None:
            worker.join(timeout=5)
            assert not worker.is_alive()
        server._sessions.pop("null-key-trunc-sid", None)


def test_prompt_submit_refuses_ordinal_and_message_id_mismatch(monkeypatch):
    """#82756: A mismatch between truncate_before_user_ordinal and truncate_before_message_id must return 4030."""
    history = [
        {"id": "msg-1", "role": "user", "content": "first"},
        {"role": "assistant", "content": "reply 1"},
        {"id": "msg-2", "role": "user", "content": "second"},
        {"role": "assistant", "content": "reply 2"},
    ]
    server._sessions["mismatch-trunc-sid"] = _session(history=list(history))
    monkeypatch.setattr(
        server, "_start_agent_build", lambda *a, **k: pytest.fail("must not start a turn")
    )
    monkeypatch.setattr(
        server, "_start_inflight_turn", lambda *a, **k: pytest.fail("must not start a turn")
    )

    try:
        resp = server.handle_request(
            {
                "id": "1",
                "method": "prompt.submit",
                "params": {
                    "session_id": "mismatch-trunc-sid",
                    "text": "new turn",
                    "truncate_before_message_id": "msg-2",  # ordinal index 1
                    "truncate_before_user_ordinal": 0,      # mismatch (stale 0)
                    "confirm_truncate": True,
                },
            }
        )
        assert resp.get("error") is not None
        assert resp["error"]["code"] == 4030
    finally:
        server._sessions.pop("mismatch-trunc-sid", None)


def test_prompt_submit_refuses_ordinal_only_when_history_has_row_ids(monkeypatch):
    """A durable session must not trust an ordinal without a row-id target."""
    replaced = []

    class _FakeDB:
        def replace_messages(
            self,
            key,
            messages,
            active_only=False,
            archive_dropped=False,
            reject_active_turn_lease=False,
        ):
            replaced.append((key, list(messages)))

    history = [
        {"_row_id": 101, "role": "user", "content": "first"},
        {"_row_id": 102, "role": "assistant", "content": "reply 1"},
        {"_row_id": 103, "role": "user", "content": "second"},
        {"_row_id": 104, "role": "assistant", "content": "reply 2"},
    ]
    server._sessions["ordinal-only-durable-sid"] = _session(history=list(history))
    monkeypatch.setattr(server, "_get_db", lambda: _FakeDB())
    monkeypatch.setattr(
        server, "_start_agent_build", lambda *a, **k: pytest.fail("must not start a turn")
    )
    monkeypatch.setattr(
        server, "_start_inflight_turn", lambda *a, **k: pytest.fail("must not start a turn")
    )

    try:
        resp = server.handle_request(
            {
                "id": "1",
                "method": "prompt.submit",
                "params": {
                    "session_id": "ordinal-only-durable-sid",
                    "text": "retry",
                    "truncate_before_user_ordinal": 1,
                    "confirm_truncate": True,
                },
            }
        )

        assert resp["error"]["code"] == 4004
        assert "truncate_before_row_id" in resp["error"]["message"]
        assert server._sessions["ordinal-only-durable-sid"]["history"] == history
        assert server._sessions["ordinal-only-durable-sid"]["running"] is False
        assert replaced == []
    finally:
        server._sessions.pop("ordinal-only-durable-sid", None)


def test_prompt_submit_refuses_ordinal_only_when_durable_history_is_unstamped(monkeypatch):
    """Durability comes from state.db, not optional stamps on the live copy."""
    replaced = []
    history = [
        {"role": "user", "content": "first"},
        {"role": "assistant", "content": "reply 1"},
        {"role": "user", "content": "second"},
        {"role": "assistant", "content": "reply 2"},
    ]

    class _FakeDB:
        def get_messages_as_conversation(self, key, **kwargs):
            assert key == "session-key"
            assert kwargs["include_row_ids"] is True
            return [dict(message, _row_id=100 + index) for index, message in enumerate(history)]

        def replace_messages(self, key, messages, active_only=False, archive_dropped=False):
            replaced.append((key, list(messages)))

    server._sessions["unstamped-durable-sid"] = _session(history=list(history))
    monkeypatch.setattr(server, "_get_db", lambda: _FakeDB())
    monkeypatch.setattr(
        server, "_start_agent_build", lambda *a, **k: pytest.fail("must not start a turn")
    )
    monkeypatch.setattr(
        server, "_start_inflight_turn", lambda *a, **k: pytest.fail("must not start a turn")
    )

    try:
        resp = server.handle_request(
            {
                "id": "1",
                "method": "prompt.submit",
                "params": {
                    "session_id": "unstamped-durable-sid",
                    "text": "retry",
                    "truncate_before_user_ordinal": 1,
                    "confirm_truncate": True,
                },
            }
        )

        assert resp["error"]["code"] == 4004
        assert "truncate_before_row_id" in resp["error"]["message"]
        assert server._sessions["unstamped-durable-sid"]["history"] == history
        assert replaced == []
    finally:
        server._sessions.pop("unstamped-durable-sid", None)


def test_prompt_submit_truncates_by_row_id(monkeypatch):
    """#82959: prompt.submit with truncate_before_row_id must cut at the target row id."""
    replaced = []

    class _FakeDB:
        def replace_messages(
            self,
            key,
            messages,
            active_only=False,
            archive_dropped=False,
            reject_active_turn_lease=False,
        ):
            replaced.append((key, list(messages)))

    history = [
        {"_row_id": 101, "role": "user", "content": "first"},
        {"_row_id": 102, "role": "assistant", "content": "reply 1"},
        {"_row_id": 103, "role": "user", "content": "second"},
        {"_row_id": 104, "role": "assistant", "content": "reply 2"},
    ]
    sess = _session(history=list(history))
    server._sessions["row-id-trunc-sid"] = sess
    monkeypatch.setattr(server, "_get_db", lambda: _FakeDB())
    started = []
    monkeypatch.setattr(
        server, "_start_agent_build", lambda *a, **k: started.append(k)
    )

    try:
        resp = server.handle_request(
            {
                "id": "1",
                "method": "prompt.submit",
                "params": {
                    "session_id": "row-id-trunc-sid",
                    "text": "new turn",
                    "truncate_before_row_id": 103,
                    "truncate_before_user_ordinal": 1,
                    "confirm_truncate": True,
                },
            }
        )
        assert resp.get("error") is None
        assert len(sess["history"]) == 2
        assert sess["history"][-1]["content"] == "reply 1"
        assert len(replaced) == 1
        assert replaced[0][1] == history[:2]
    finally:
        server._sessions.pop("row-id-trunc-sid", None)


def test_prompt_submit_truncates_by_string_row_id(monkeypatch):
    """#82959: String row IDs in history match correctly against integer truncate_before_row_id."""
    replaced = []

    class _FakeDB:
        def replace_messages(
            self,
            key,
            messages,
            active_only=False,
            archive_dropped=False,
            reject_active_turn_lease=False,
        ):
            replaced.append((key, list(messages)))

    history = [
        {"_row_id": "101", "role": "user", "content": "first"},
        {"_row_id": "102", "role": "assistant", "content": "reply 1"},
        {"_row_id": "103", "role": "user", "content": "second"},
        {"_row_id": "104", "role": "assistant", "content": "reply 2"},
    ]
    sess = _session(history=list(history))
    server._sessions["str-row-id-trunc-sid"] = sess
    monkeypatch.setattr(server, "_get_db", lambda: _FakeDB())
    monkeypatch.setattr(server, "_start_agent_build", lambda *a, **k: None)

    try:
        resp = server.handle_request(
            {
                "id": "1",
                "method": "prompt.submit",
                "params": {
                    "session_id": "str-row-id-trunc-sid",
                    "text": "new turn",
                    "truncate_before_row_id": 103,
                    "confirm_truncate": True,
                },
            }
        )
        assert resp.get("error") is None
        assert len(sess["history"]) == 2
    finally:
        server._sessions.pop("str-row-id-trunc-sid", None)


def test_prompt_submit_refuses_ordinal_and_row_id_mismatch(monkeypatch):
    """#82959: A mismatch between truncate_before_user_ordinal and truncate_before_row_id must return 4030."""
    history = [
        {"_row_id": 201, "role": "user", "content": "first"},
        {"_row_id": 202, "role": "assistant", "content": "reply 1"},
        {"_row_id": 203, "role": "user", "content": "second"},
        {"_row_id": 204, "role": "assistant", "content": "reply 2"},
    ]
    server._sessions["row-mismatch-sid"] = _session(history=list(history))
    monkeypatch.setattr(
        server, "_start_agent_build", lambda *a, **k: pytest.fail("must not start a turn")
    )

    try:
        resp = server.handle_request(
            {
                "id": "1",
                "method": "prompt.submit",
                "params": {
                    "session_id": "row-mismatch-sid",
                    "text": "new turn",
                    "truncate_before_row_id": 203,  # user turn ordinal 1
                    "truncate_before_user_ordinal": 0,  # mismatch (stale 0)
                    "confirm_truncate": True,
                },
            }
        )
        assert resp.get("error") is not None
        assert resp["error"]["code"] == 4030
    finally:
        server._sessions.pop("row-mismatch-sid", None)


def test_prompt_submit_refuses_boolean_row_id(monkeypatch):
    """Boolean truncate_before_row_id must return 4004."""
    history = [
        {"_row_id": 301, "role": "user", "content": "first"},
        {"_row_id": 302, "role": "assistant", "content": "reply 1"},
    ]
    server._sessions["bool-row-sid"] = _session(history=list(history))
    try:
        resp = server.handle_request(
            {
                "id": "1",
                "method": "prompt.submit",
                "params": {
                    "session_id": "bool-row-sid",
                    "text": "new turn",
                    "truncate_before_row_id": True,
                    "confirm_truncate": True,
                },
            }
        )
        assert resp.get("error") is not None
        assert resp["error"]["code"] == 4004
    finally:
        server._sessions.pop("bool-row-sid", None)


def test_prompt_submit_row_id_not_found(monkeypatch):
    """Unknown truncate_before_row_id must return 4018."""
    history = [
        {"_row_id": 401, "role": "user", "content": "first"},
    ]
    server._sessions["missing-row-sid"] = _session(history=list(history))
    try:
        resp = server.handle_request(
            {
                "id": "1",
                "method": "prompt.submit",
                "params": {
                    "session_id": "missing-row-sid",
                    "text": "new turn",
                    "truncate_before_row_id": 999,
                    "confirm_truncate": True,
                },
            }
        )
        assert resp.get("error") is not None
        assert resp["error"]["code"] == 4018
    finally:
        server._sessions.pop("missing-row-sid", None)


def test_prompt_submit_resolves_row_id_absorbed_into_marker_merge(monkeypatch):
    """#94486: after a mid-session model switch, the marker is persisted as
    role=user (#48338) and the next real user row merges into it under
    repair. The merged row must stay addressable (plain row's _row_id kept,
    display classification dropped) so prompt.submit with
    truncate_before_row_id resolves instead of failing closed with the
    user's input silently dropped.
    """
    import copy

    from agent.agent_runtime_helpers import repair_message_sequence

    verbatim = [
        {"_row_id": 12131, "role": "user", "content": "first"},
        {"_row_id": 12133, "role": "assistant", "content": "reply 1"},
        {
            "_row_id": 12134,
            "role": "user",
            "display_kind": "model_switch",
            "content": "[System: The active model for this chat has changed to ds4.]",
        },
        {"_row_id": 12135, "role": "user", "content": "the dropped prompt"},
    ]
    repaired = copy.deepcopy(verbatim)
    assert repair_message_sequence(None, repaired) == 1
    # The merged pair survives as ONE addressable user row carrying the
    # plain row's durable id — this is the property the gateway relies on.
    assert len(repaired) == 3
    assert repaired[2]["_row_id"] == 12135
    assert not repaired[2].get("display_kind")

    replaced = []

    class _FakeDB:
        def get_messages_as_conversation(
            self, key, include_ancestors=False, repair_alternation=False,
            include_row_ids=False, **_kwargs
        ):
            assert key == "session-key"
            assert include_row_ids is True
            return copy.deepcopy(repaired if repair_alternation else verbatim)

        def replace_messages(self, key, messages, active_only=False, archive_dropped=False, **_kwargs):
            replaced.append((key, list(messages)))


    @contextlib.contextmanager
    def _fake_session_db(_session):
        yield _FakeDB()

    # Live history as a RUNNING session holds it after a turn rewrite: the
    # repaired, merged shape with row-id stamps still attached (the live
    # dicts carry them), so the in-memory lookup resolves directly.
    live = copy.deepcopy(repaired)
    assert live[2]["_row_id"] == 12135
    server._sessions["merged-marker-sid"] = _session(history=list(live))
    monkeypatch.setattr(server, "_session_db", _fake_session_db)
    monkeypatch.setattr(server, "_get_db", lambda: _FakeDB())
    monkeypatch.setattr(server, "_start_agent_build", lambda *a, **k: None)
    monkeypatch.setattr(server, "_start_inflight_turn", lambda *a, **k: None)

    try:
        resp = server.handle_request(
            {
                "id": "1",
                "method": "prompt.submit",
                "params": {
                    "session_id": "merged-marker-sid",
                    "text": "new turn",
                    "truncate_before_row_id": 12135,
                    "confirm_truncate": True,
                },
            }
        )
        assert resp.get("error") is None, resp.get("error")
        # The cut lands before the merged marker+prompt pair: everything the
        # pair would have buried is kept, the pair itself is replaced by the
        # freshly submitted turn. (Rows carry _row_id stamps because the
        # durable fallback healed the live list's missing stamps first.)
        assert len(replaced) == 1
        assert replaced[0][0] == "session-key"
        assert replaced[0][1] == [
            {"role": "user", "content": "first", "_row_id": 12131},
            {"role": "assistant", "content": "reply 1", "_row_id": 12133},
        ]
        assert len(server._sessions["merged-marker-sid"]["history"]) == 2
    finally:
        server._sessions.pop("merged-marker-sid", None)



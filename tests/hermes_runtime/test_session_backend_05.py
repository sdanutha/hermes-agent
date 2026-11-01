









import sys


import threading


import time


import types




from pathlib import Path




import pytest








from tools import async_delegation as ad


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


def test_background_agent_kwargs_falls_back_to_config_disabled_toolsets(monkeypatch):
    agent = types.SimpleNamespace(
        model="gpt-5.5",
        provider="anthropic",
        _fallback_chain=[],
    )
    monkeypatch.setattr(server, "_load_cfg", lambda: {"max_turns": 25})
    monkeypatch.setattr(server, "_load_enabled_toolsets", lambda *_a, **_kw: ["file"])
    monkeypatch.setattr(server, "_load_disabled_toolsets", lambda: ["browser"])
    monkeypatch.setattr(server, "_get_db", lambda: None)

    kwargs = server._background_agent_kwargs(agent, "task-id")

    assert kwargs["disabled_toolsets"] == ["browser"]


def test_startup_runtime_resolves_short_alias_without_network(monkeypatch):
    monkeypatch.setenv("HERMES_MODEL", "sonnet")
    monkeypatch.delenv("HERMES_TUI_PROVIDER", raising=False)
    monkeypatch.delenv("HERMES_INFERENCE_PROVIDER", raising=False)
    monkeypatch.setattr(server, "_load_cfg", lambda: {"model": {"provider": "auto"}})
    monkeypatch.setattr(
        "hermes_cli.models.fetch_openrouter_models",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("network lookup should not run")
        ),
    )

    model, provider = server._resolve_startup_runtime()

    assert provider == "anthropic"
    assert model.startswith("claude-sonnet")


def test_startup_runtime_does_not_call_network_detector(monkeypatch):
    monkeypatch.setenv("HERMES_MODEL", "sonnet")
    monkeypatch.delenv("HERMES_TUI_PROVIDER", raising=False)
    monkeypatch.delenv("HERMES_INFERENCE_PROVIDER", raising=False)
    monkeypatch.setattr(server, "_load_cfg", lambda: {"model": {"provider": "auto"}})
    monkeypatch.setattr(
        "hermes_cli.models.detect_provider_for_model",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("network detector called")
        ),
    )

    model, provider = server._resolve_startup_runtime()

    assert model
    assert provider in {None, "anthropic"}


def test_session_close_commits_memory_and_fires_finalize_hook(monkeypatch):
    calls = {"hooks": []}

    agent = types.SimpleNamespace(session_id="session-key")
    agent.commit_memory_session = lambda history: calls.setdefault("history", history)
    server._sessions["sid"] = _session(
        agent=agent, history=[{"role": "user", "content": "hello"}]
    )
    monkeypatch.setattr(
        server,
        "_notify_session_boundary",
        lambda event, session_id, *_args: calls["hooks"].append((event, session_id)),
    )

    try:
        resp = server.handle_request(
            {"id": "1", "method": "session.close", "params": {"session_id": "sid"}}
        )
        assert resp["result"]["closed"] is True
        assert calls["history"] == [{"role": "user", "content": "hello"}]
        assert ("on_session_finalize", "session-key") in calls["hooks"]
    finally:
        server._sessions.pop("sid", None)


def test_session_close_releases_resume_lock_before_slow_teardown(monkeypatch):
    """One slow session finalizer must not stall unrelated session.resume RPCs."""
    teardown_started = threading.Event()
    release_teardown = threading.Event()
    response = {}

    def _slow_teardown(_session, *, end_reason="tui_close"):
        assert end_reason == "tui_close"
        teardown_started.set()
        assert release_teardown.wait(timeout=2.0)

    monkeypatch.setattr(server, "_teardown_session", _slow_teardown)
    server._sessions["slow-close"] = _session()

    def _close():
        response.update(
            server.handle_request(
                {
                    "id": "close",
                    "method": "session.close",
                    "params": {"session_id": "slow-close"},
                }
            )
        )

    thread = threading.Thread(target=_close)
    thread.start()
    acquired = False
    try:
        assert teardown_started.wait(timeout=1.0)
        assert "slow-close" not in server._sessions
        acquired = server._session_resume_lock.acquire(timeout=0.2)
        assert acquired, "slow teardown kept the global resume lock held"
    finally:
        if acquired:
            server._session_resume_lock.release()
        release_teardown.set()
        thread.join(timeout=2.0)
        server._sessions.pop("slow-close", None)

    assert not thread.is_alive()
    assert response["result"] == {"closed": True, "messages": []}


def test_session_close_settles_active_turn_before_teardown(monkeypatch):
    """Close must not tear down agent resources while their turn is unwinding."""
    turn_started = threading.Event()
    release_turn = threading.Event()
    teardown_started = threading.Event()
    response = {}

    def _turn():
        turn_started.set()
        assert release_turn.wait(timeout=2.0)

    def _teardown(_session, *, end_reason="tui_close"):
        if end_reason == "tui_close":
            teardown_started.set()

    session = _session()
    run_thread = threading.Thread(target=_turn)
    session["_run_thread"] = run_thread
    server._sessions["settle-close"] = session
    monkeypatch.setattr(server, "_teardown_session", _teardown)
    monkeypatch.setattr(
        server, "_TURN_SETTLE_BEFORE_CLOSE_SECONDS", 1.0, raising=False
    )

    close_thread = threading.Thread(
        target=lambda: response.update(
            server.handle_request(
                {
                    "id": "close",
                    "method": "session.close",
                    "params": {"session_id": "settle-close"},
                }
            )
        )
    )
    run_thread.start()
    close_thread.start()
    try:
        assert turn_started.wait(timeout=1.0)
        assert not teardown_started.wait(timeout=0.1)
        release_turn.set()
        close_thread.join(timeout=2.0)
    finally:
        release_turn.set()
        run_thread.join(timeout=2.0)
        close_thread.join(timeout=2.0)
        server._sessions.pop("settle-close", None)

    assert not close_thread.is_alive()
    assert teardown_started.is_set()
    assert response["result"] == {"closed": True, "messages": []}


def test_ws_orphan_reap_interrupts_isolated_turn_then_reaps(monkeypatch):
    callbacks = []
    interrupted = []
    torn_down = []

    class _Timer:
        def __init__(self, _delay, callback):
            callbacks.append(callback)
            self.daemon = False

        def start(self):
            return None

    class _Supervisor:
        def interrupt(self, sid, *, request_id=None):
            interrupted.append((sid, request_id))

    session = _session(
        agent=None,
        agent_ready=threading.Event(),
        transport=server._detached_ws_transport,
        running=True,
        _compute_host_active=True,
        history=[{"role": "assistant", "content": "partial"}],
        queued_prompt={"text": "must not run"},
    )
    server._sessions["isolated-sid"] = session
    monkeypatch.setattr(server, "_WS_ORPHAN_REAP_GRACE_S", 0.01)
    monkeypatch.setattr(server.threading, "Timer", _Timer)
    monkeypatch.setattr(
        server, "_load_cfg", lambda: {"dashboard": {"turn_isolation": True}}
    )
    monkeypatch.setattr(
        server, "_get_compute_host_supervisor", lambda _cfg=None: _Supervisor()
    )
    monkeypatch.setattr(
        server,
        "_teardown_popped_session",
        lambda claimed, *, end_reason: torn_down.append((claimed, end_reason)) or True,
    )

    try:
        server._schedule_ws_orphan_reap("isolated-sid")
        callbacks.pop(0)()

        assert interrupted == [("isolated-sid", "client-gone-isolated-sid")]
        assert session["_turn_cancel_requested"] is True
        assert session["queued_prompt"] is None
        assert session["history"] == [{"role": "assistant", "content": "partial"}]
        assert len(callbacks) == 1

        callbacks.pop(0)()

        assert interrupted == [("isolated-sid", "client-gone-isolated-sid")]
        assert len(callbacks) == 1

        session["running"] = False
        callbacks.pop(0)()

        assert "isolated-sid" not in server._sessions
        assert torn_down == [(session, "ws_orphan_reap")]
    finally:
        server._sessions.pop("isolated-sid", None)


def test_ws_orphan_reap_spares_turn_reattached_within_grace(monkeypatch):
    callbacks = []
    interrupted = []

    class _Timer:
        def __init__(self, _delay, callback):
            callbacks.append(callback)

        def start(self):
            return None

    class _LiveThread:
        def is_alive(self):
            return True

    class _LiveTransport:
        def write(self, *_args, **_kwargs):
            return True

    disconnecting_transport = _LiveTransport()
    session = _session(
        agent=types.SimpleNamespace(
            interrupt=lambda: interrupted.append("interrupted")
        ),
        transport=disconnecting_transport,
        running=True,
        _run_thread=_LiveThread(),
    )
    server._sessions["reattached-sid"] = session
    monkeypatch.setattr(server, "_WS_ORPHAN_REAP_GRACE_S", 0.01)
    monkeypatch.setattr(server.threading, "Timer", _Timer)
    monkeypatch.setattr(server, "_load_cfg", dict)

    try:
        server._close_sessions_for_transport(disconnecting_transport)
        assert session["transport"] is server._detached_ws_transport

        session["transport"] = _LiveTransport()
        callbacks.pop(0)()

        assert interrupted == []
        assert "reattached-sid" in server._sessions
        assert callbacks == []
    finally:
        server._sessions.pop("reattached-sid", None)


def test_session_resume_does_not_rebind_after_client_gone_interrupt_claim(monkeypatch):
    class _DB:
        def get_session(self, session_id):
            assert session_id == "stored-sid"
            return {"id": session_id, "cwd": "/tmp"}

        def resolve_resume_session_id(self, session_id):
            return session_id

    live_transport = object()
    session = _session(
        session_key="stored-sid",
        transport=server._detached_ws_transport,
        running=True,
        _client_gone_interrupt_requested=True,
    )
    server._sessions["live-sid"] = session
    monkeypatch.setattr(server, "_get_db", lambda: _DB())
    monkeypatch.setattr(server, "current_transport", lambda: live_transport)

    try:
        response = server.handle_request(
            {
                "id": "resume-after-claim",
                "method": "session.resume",
                "params": {"session_id": "stored-sid"},
            }
        )

        assert response is not None
        assert response["error"]["code"] == 4009
        assert session["transport"] is server._detached_ws_transport
    finally:
        server._sessions.pop("live-sid", None)


def test_ws_orphan_reap_defers_running_turn_for_active_delegation(monkeypatch):
    callbacks = []
    interrupted = []
    delegation_active = iter((True, False, False))

    class _Timer:
        def __init__(self, _delay, callback):
            callbacks.append(callback)

        def start(self):
            return None

    class _LiveThread:
        def is_alive(self):
            return True

    def _interrupt():
        interrupted.append("interrupted")
        session["running"] = False

    session = _session(
        agent=types.SimpleNamespace(interrupt=_interrupt),
        transport=server._detached_ws_transport,
        running=True,
        _run_thread=_LiveThread(),
    )
    server._sessions["delegating-turn"] = session
    monkeypatch.setattr(server, "_WS_ORPHAN_REAP_GRACE_S", 0.01)
    monkeypatch.setattr(server.threading, "Timer", _Timer)
    monkeypatch.setattr(server, "_load_cfg", dict)
    monkeypatch.setattr(
        server,
        "_session_has_active_delegations",
        lambda *_args, **_kwargs: next(delegation_active),
    )
    monkeypatch.setattr(server, "_teardown_popped_session", lambda *_args, **_kwargs: True)

    try:
        server._schedule_ws_orphan_reap("delegating-turn")
        callbacks.pop(0)()

        assert interrupted == []
        assert len(callbacks) == 1

        callbacks.pop(0)()

        assert interrupted == ["interrupted"]
        assert len(callbacks) == 1

        callbacks.pop(0)()
        assert "delegating-turn" not in server._sessions
    finally:
        server._sessions.pop("delegating-turn", None)


def test_ws_orphan_reap_interrupts_in_process_turn(monkeypatch):
    callbacks = []
    interrupted = []

    class _Timer:
        def __init__(self, _delay, callback):
            callbacks.append(callback)

        def start(self):
            return None

    class _LiveThread:
        def is_alive(self):
            return True

    def _interrupt():
        interrupted.append("interrupted")
        session["running"] = False

    session = _session(
        agent=types.SimpleNamespace(interrupt=_interrupt),
        transport=server._detached_ws_transport,
        running=True,
        _run_thread=_LiveThread(),
    )
    server._sessions["inline-sid"] = session
    monkeypatch.setattr(server, "_WS_ORPHAN_REAP_GRACE_S", 0.01)
    monkeypatch.setattr(server.threading, "Timer", _Timer)
    monkeypatch.setattr(server, "_load_cfg", dict)

    try:
        server._schedule_ws_orphan_reap("inline-sid")
        callbacks.pop(0)()

        assert interrupted == ["interrupted"]
        assert session["_turn_cancel_requested"] is True
        assert len(callbacks) == 1
    finally:
        server._sessions.pop("inline-sid", None)


def test_ws_disconnect_running_sidecar_still_closes_without_orphan_timer(monkeypatch):
    closed = []
    scheduled = []
    transport = object()
    server._sessions["sidecar-sid"] = _session(
        transport=transport,
        running=True,
        close_on_disconnect=True,
    )
    monkeypatch.setattr(
        server,
        "_teardown_popped_session",
        lambda session, *, end_reason: closed.append((session["_sid"], end_reason)) or True,
    )
    monkeypatch.setattr(
        server, "_schedule_ws_orphan_reap", lambda sid: scheduled.append(sid)
    )

    try:
        reaped, detached = server._close_sessions_for_transport(transport)

        assert (reaped, detached) == (1, 0)
        assert closed == [("sidecar-sid", "ws_disconnect")]
        assert scheduled == []
    finally:
        server._sessions.pop("sidecar-sid", None)


def test_ws_orphan_reap_releases_resume_lock_before_slow_teardown(monkeypatch):
    """Grace reaping claims under the lock but finalizes after releasing it."""
    scheduled = {}
    teardown_started = threading.Event()
    release_teardown = threading.Event()

    class _Timer:
        def __init__(self, _delay, callback):
            scheduled["callback"] = callback

        def start(self):
            return None

    def _slow_teardown(_session, *, end_reason="tui_close"):
        assert end_reason == "ws_orphan_reap"
        teardown_started.set()
        assert release_teardown.wait(timeout=10.0)

    monkeypatch.setattr(server, "_WS_ORPHAN_REAP_GRACE_S", 0.01)
    monkeypatch.setattr(server.threading, "Timer", _Timer)
    monkeypatch.setattr(server, "_teardown_session", _slow_teardown)
    # The reap's delegation check reads the session row from the launch store; open that handle
    # before the timer fires so the wait below measures lock release, not a cold state.db open.
    server._get_db()
    server._sessions["slow-orphan"] = _session(
        transport=server._detached_ws_transport,
        running=False,
    )

    server._schedule_ws_orphan_reap("slow-orphan")
    thread = threading.Thread(target=scheduled["callback"])
    thread.start()
    acquired = False
    try:
        assert teardown_started.wait(timeout=10.0)
        assert "slow-orphan" not in server._sessions
        acquired = server._session_resume_lock.acquire(timeout=2.0)
        assert acquired, "orphan teardown kept the global resume lock held"
    finally:
        if acquired:
            server._session_resume_lock.release()
        release_teardown.set()
        thread.join(timeout=10.0)
        server._sessions.pop("slow-orphan", None)

    assert not thread.is_alive()


def test_ws_orphan_reap_reschedules_while_mid_turn_then_reaps(monkeypatch):
    """A detached session that is still running must keep the reap timer (#85578)."""
    callbacks = []
    torn_down = []

    class _Timer:
        def __init__(self, _delay, callback):
            callbacks.append(callback)
            self.daemon = False

        def start(self):
            return None

    monkeypatch.setattr(server, "_WS_ORPHAN_REAP_GRACE_S", 0.01)
    monkeypatch.setattr(server.threading, "Timer", _Timer)
    monkeypatch.setattr(
        server,
        "_teardown_session",
        lambda session, *, end_reason="tui_close": torn_down.append(
            (session, end_reason)
        ),
    )
    live = _session(
        transport=server._detached_ws_transport,
        running=True,
    )
    server._sessions["midturn-sid"] = live

    try:
        server._schedule_ws_orphan_reap("midturn-sid")
        callbacks.pop(0)()

        assert "midturn-sid" in server._sessions
        assert len(callbacks) == 1
        assert torn_down == []

        live["running"] = False
        callbacks.pop(0)()

        assert "midturn-sid" not in server._sessions
        assert len(torn_down) == 1
        assert torn_down[0][1] == "ws_orphan_reap"
    finally:
        server._sessions.pop("midturn-sid", None)


def test_ws_orphan_reap_waits_for_active_delegation_then_reaps(monkeypatch):
    from tools import async_delegation

    callbacks = []
    torn_down = []
    delegation_id = "deleg_ws_orphan_reap_test"

    class _Timer:
        def __init__(self, _delay, callback):
            callbacks.append(callback)
            self.daemon = False

        def start(self):
            return None

    monkeypatch.setattr(server, "_WS_ORPHAN_REAP_GRACE_S", 0.01)
    monkeypatch.setattr(server.threading, "Timer", _Timer)
    monkeypatch.setattr(
        server,
        "_teardown_session",
        lambda session, *, end_reason="tui_close": torn_down.append(
            (session, end_reason)
        ),
    )
    server._sessions["delegating-sid"] = _session(
        transport=server._detached_ws_transport,
        running=False,
    )
    with async_delegation._records_lock:
        async_delegation._records[delegation_id] = {
            "status": "running",
            "origin_ui_session_id": "delegating-sid",
        }

    try:
        server._schedule_ws_orphan_reap("delegating-sid")
        callbacks.pop(0)()

        assert "delegating-sid" in server._sessions
        assert len(callbacks) == 1
        assert torn_down == []

        with async_delegation._records_lock:
            async_delegation._records[delegation_id]["status"] = "completed"
        callbacks.pop(0)()

        assert "delegating-sid" not in server._sessions
        assert len(torn_down) == 1
        assert torn_down[0][1] == "ws_orphan_reap"
    finally:
        server._sessions.pop("delegating-sid", None)
        with async_delegation._records_lock:
            async_delegation._records.pop(delegation_id, None)


def test_ws_orphan_reap_retries_when_delegation_lookup_fails(monkeypatch):
    from tools import async_delegation

    callbacks = []
    torn_down = []

    class _Timer:
        def __init__(self, _delay, callback):
            callbacks.append(callback)
            self.daemon = False

        def start(self):
            return None

    def _raise_lookup_error(*_args, **_kwargs):
        raise RuntimeError("delegation registry unavailable")

    monkeypatch.setattr(server, "_WS_ORPHAN_REAP_GRACE_S", 0.01)
    monkeypatch.setattr(server.threading, "Timer", _Timer)
    monkeypatch.setattr(
        async_delegation, "has_live_for_session", _raise_lookup_error
    )
    monkeypatch.setattr(
        server,
        "_teardown_session",
        lambda session, *, end_reason="tui_close": torn_down.append(
            (session, end_reason)
        ),
    )
    server._sessions["lookup-error-sid"] = _session(
        transport=server._detached_ws_transport,
        running=False,
    )

    try:
        server._schedule_ws_orphan_reap("lookup-error-sid")
        callbacks.pop(0)()

        assert "lookup-error-sid" in server._sessions
        assert len(callbacks) == 1
        assert torn_down == []
    finally:
        server._sessions.pop("lookup-error-sid", None)


def test_finalize_session_closes_slash_worker(monkeypatch):
    """_finalize_session closes the slash_worker subprocess itself.

    Regression for #38095: the worker cleanup used to live only in the
    callers (_teardown_session / _shutdown_sessions), so any code path that
    finalized a session without going through them leaked the worker. Folding
    close() into the single _finalized-guarded chokepoint makes the cleanup
    defense-in-depth and idempotent.
    """
    closed = {"count": 0}

    class _FakeWorker:
        def close(self):
            closed["count"] += 1

    monkeypatch.setattr(server, "_notify_session_boundary", lambda *a, **k: None)
    monkeypatch.setattr(server, "_get_db", lambda: None)

    session = _session(slash_worker=_FakeWorker())

    server._finalize_session(session)
    assert closed["count"] == 1
    assert session.get("_finalized") is True

    # Idempotent: a second finalize (or a follow-up teardown) must not
    # re-close the worker — the _finalized guard short-circuits.
    server._finalize_session(session)
    server._teardown_session(session)
    assert closed["count"] == 1


def test_close_transport_rebinds_session_to_remaining_viewer(monkeypatch):
    """Closing a pop-out window's transport must leave the session with the
    still-open window instead of stranding it on the drop sentinel (#83716).

    The rebind #83716 added is gone; multi-client fan-out subsumes it. Both
    windows are attached to the slot at once, so the pop-out is a fan-out peer
    rather than a viewer waiting to be promoted, and closing it detaches that
    peer while retaining the surviving ordered mailbox. This pins the same
    guarantee through the mechanism that replaced the rebind: the session is
    not parked, not reaped, not handed to the orphan reaper, and the surviving
    window keeps receiving frames.
    """
    reap_calls = []
    monkeypatch.setattr(server, "_schedule_ws_orphan_reap", lambda sid: reap_calls.append(sid))

    class _LiveTransport:
        def __init__(self):
            self.frames = []
            self.received = threading.Event()

        def write(self, obj=None, *a, **k):
            self.frames.append(obj)
            self.received.set()
            return True

    main = _LiveTransport()
    popout = _LiveTransport()
    session = _session(transport=None, running=False)
    # Build the state the way production does: every window that resumes goes
    # through _live_session_payload, which attaches it into the slot and then
    # stamps it into the viewers registry.
    server._attach_session_transport(session, main)
    server._attach_session_transport(session, popout)
    session["viewers"] = {main: 100.0, popout: 200.0}
    server._sessions["multi-sid"] = session
    assert isinstance(session["transport"], server.FanoutTransport)

    try:
        reaped, detached = server._close_sessions_for_transport(popout)

        assert reaped == 0 and detached == 0
        assert server._session_transport_contains(session, main)
        assert not server._session_transport_contains(session, popout)
        assert "multi-sid" not in reap_calls
        assert server._ws_session_is_orphaned(session) is False

        # And it is still a working stream, not just a surviving reference.
        server._emit("message.delta", "multi-sid", {"text": "still here"})
        assert main.received.wait(timeout=5)
        assert [(f.get("params") or {}).get("type") for f in main.frames] == [
            "message.delta"
        ]
        assert popout.frames == []
    finally:
        # The fake slot must not outlive the test: _sessions is module state and
        # later sweeps would walk it.
        server._sessions.pop("multi-sid", None)


def test_close_transport_detaches_when_no_viewers_remain(monkeypatch):
    """The last viewer closing still lands the session on the drop sentinel
    and schedules the grace reap (unchanged single-window behavior)."""
    reap_calls = []
    monkeypatch.setattr(server, "_schedule_ws_orphan_reap", lambda sid: reap_calls.append(sid))

    class _LiveTransport:
        def write(self, *a, **k):
            return True

    only = _LiveTransport()
    session = _session(transport=only, running=False)
    session["viewers"] = {only: 100.0}
    server._sessions["solo-sid"] = session

    reaped, detached = server._close_sessions_for_transport(only)

    assert reaped == 0 and detached == 1
    assert session["transport"] is server._detached_ws_transport
    assert reap_calls == ["solo-sid"]


def test_close_transport_skips_dead_remaining_viewers(monkeypatch):
    """A viewer whose socket is already dead must not hold the session open.

    #83716's rebind refused to hand the session to a dead viewer; fan-out
    membership keeps that filter through _transport_is_live_peer, which is what
    decides whether anything survives the departing client. Both windows are
    ATTACHED here, which is the state production builds — a viewer that was
    never attached leaves the slot single-client and exercises the ordinary park
    path instead of this one.
    """
    reap_calls = []
    monkeypatch.setattr(server, "_schedule_ws_orphan_reap", lambda sid: reap_calls.append(sid))

    class _LiveTransport:
        def write(self, *a, **k):
            return True

    dead = _LiveTransport()
    popout = _LiveTransport()
    session = _session(transport=None, running=False)
    server._attach_session_transport(session, dead)
    server._attach_session_transport(session, popout)
    session["viewers"] = {dead: 100.0, popout: 200.0}
    assert isinstance(session["transport"], server.FanoutTransport)
    # The socket goes away without a disconnect reaching the gateway; the latch
    # _transport_is_dead reads is the only trace it leaves behind.
    dead._closed = True
    server._sessions["dead-viewer-sid"] = session

    try:
        reaped, detached = server._close_sessions_for_transport(popout)

        assert reaped == 0 and detached == 1
        assert session["transport"] is server._detached_ws_transport
        assert reap_calls == ["dead-viewer-sid"]
    finally:
        server._sessions.pop("dead-viewer-sid", None)


def test_live_session_payload_registers_transport_as_viewer():
    """Resume/activate through _live_session_payload must register the caller
    as a viewer so the disconnect path has something to re-bind to (#83716)."""
    class _LiveTransport:
        def write(self, *a, **k):
            return True

    t = _LiveTransport()
    session = _session(transport=server._detached_ws_transport, running=False)
    server._live_session_payload("viewer-sid", session, transport=t)

    assert session["transport"] is t
    assert t in session.get("viewers", {})


def test_ws_orphan_reap_spares_reattached_session(monkeypatch):
    """A session that rebinds a live transport is NOT considered orphaned."""

    class _LiveTransport:
        def write(self, *a, **k):
            return True

    # Reattached: transport is a live (non-stdio) transport.
    reattached = _session(transport=_LiveTransport(), running=False)
    assert server._ws_session_is_orphaned(reattached) is False

    # Mid-turn sessions are also spared even if detached.
    mid_turn = _session(transport=server._detached_ws_transport, running=True)
    assert server._ws_session_is_orphaned(mid_turn) is False

    # Already finalized sessions are spared (idempotency).
    done = _session(
        transport=server._detached_ws_transport,
        running=False,
        _finalized=True,
    )
    assert server._ws_session_is_orphaned(done) is False


def test_resume_rebind_cancels_pending_ws_orphan_reap(monkeypatch):
    """Re-binding a live transport via _live_session_payload must cancel the
    pending ws-orphan reap Timer (storm killer, part 1)."""
    cancelled = []

    class _Timer:
        def __init__(self, _delay, fn):
            self.fn = fn

        def start(self):
            return None

        def cancel(self):
            cancelled.append(self)

    class _LiveTransport:
        def write(self, *a, **k):
            return True

    monkeypatch.setattr(server, "_WS_ORPHAN_REAP_GRACE_S", 0.01)
    monkeypatch.setattr(server.threading, "Timer", _Timer)
    session = _session(transport=server._detached_ws_transport, running=False)
    server._sessions["cancel-sid"] = session

    try:
        server._schedule_ws_orphan_reap("cancel-sid")
        assert "cancel-sid" in server._pending_ws_reaps

        server._live_session_payload("cancel-sid", session, transport=_LiveTransport())

        assert "cancel-sid" not in server._pending_ws_reaps
        assert len(cancelled) == 1
    finally:
        server._sessions.pop("cancel-sid", None)
        server._pending_ws_reaps.pop("cancel-sid", None)


def test_claim_or_reuse_live_winner_cancels_pending_reap(monkeypatch):
    """The winner's pending reap is cancelled only once guarded reuse is accepted."""
    cancelled = []

    class _Timer:
        def __init__(self, _delay, fn):
            self.fn = fn

        def start(self):
            return None

        def cancel(self):
            cancelled.append(self)

    monkeypatch.setattr(server, "_WS_ORPHAN_REAP_GRACE_S", 0.01)
    monkeypatch.setattr(server.threading, "Timer", _Timer)
    agent = types.SimpleNamespace(session_id="stored-claim")
    winner = _session(
        agent=agent,
        session_key="stored-claim",
        transport=server._detached_ws_transport,
        running=False,
    )
    server._sessions["winner-sid"] = winner

    try:
        server._schedule_ws_orphan_reap("winner-sid")
        assert "winner-sid" in server._pending_ws_reaps

        live = server._claim_or_reuse_live(
            "fresh-sid", "stored-claim", _session(), None
        )

        assert live == ("winner-sid", winner)
        assert "winner-sid" in server._pending_ws_reaps
        assert cancelled == []
        assert winner["transport"] is server._detached_ws_transport

        transport = object()
        monkeypatch.setattr(server, "current_transport", lambda: transport)
        ctx = server._Resume(1, {"omit_messages": True}, "stored-claim")
        response = server._resume_reuse_live(ctx, *live)

        assert response["result"]["session_id"] == "winner-sid"
        assert winner["transport"] is transport
        assert "winner-sid" not in server._pending_ws_reaps
        assert len(cancelled) == 1
    finally:
        server._sessions.pop("winner-sid", None)
        server._sessions.pop("fresh-sid", None)
        server._pending_ws_reaps.pop("winner-sid", None)


def test_superseded_runtime_finalized_without_reclaimed_broadcast(monkeypatch):
    """When a resume mints a fresh runtime for a stored id whose prior runtime
    is sentinel-parked, the old record is finalized quietly with end_reason
    superseded_by_resume — no session.reclaimed broadcast, and its pending
    reap Timer is cancelled (storm killer, part 2)."""
    cancelled = []
    broadcasts = []
    torn_down = []

    class _Timer:
        def __init__(self, _delay, fn):
            self.fn = fn

        def start(self):
            return None

        def cancel(self):
            cancelled.append(self)

    monkeypatch.setattr(server, "_WS_ORPHAN_REAP_GRACE_S", 0.01)
    monkeypatch.setattr(server.threading, "Timer", _Timer)
    monkeypatch.setattr(
        server,
        "_broadcast_global_event",
        lambda event, payload=None: broadcasts.append((event, payload)),
    )
    monkeypatch.setattr(
        server,
        "_teardown_popped_session",
        lambda popped, *, end_reason: torn_down.append((popped, end_reason)) or True,
    )
    monkeypatch.setattr(server, "_register_session_cwd", lambda _s: None)

    old_agent = types.SimpleNamespace(session_id="stored-super")
    old = _session(
        agent=old_agent,
        session_key="stored-super",
        transport=server._detached_ws_transport,
        running=False,
    )
    server._sessions["old-sid"] = old
    fresh = _session(session_key="stored-super")

    try:
        server._schedule_ws_orphan_reap("old-sid")
        assert "old-sid" in server._pending_ws_reaps

        # The old runtime looks live to _find_live_session_by_key, so make it
        # invisible the way a real mint path would (its client is gone and the
        # resume slow path only mints after the fast path found no live match:
        # mark it finalized-for-lookup via a different stored key is wrong —
        # instead simulate the mint race by removing it from lookup).
        old["_finalized"] = False
        monkeypatch.setattr(server, "_find_live_session_by_key", lambda _k, *_a: None)

        result = server._claim_or_reuse_live("new-sid", "stored-super", fresh, None)

        assert result is None
        assert server._sessions.get("new-sid") is fresh
        assert "old-sid" not in server._sessions
        assert "old-sid" not in server._pending_ws_reaps
        assert len(cancelled) == 1
        assert torn_down == [(old, "superseded_by_resume")]
        assert broadcasts == []  # no session.reclaimed storm
    finally:
        server._sessions.pop("old-sid", None)
        server._sessions.pop("new-sid", None)
        server._pending_ws_reaps.pop("old-sid", None)
        server._pending_ws_reaps.pop("new-sid", None)


def test_lazy_unpersisted_resume_rebinds_transport_and_cancels_reap(monkeypatch):
    """The lazy/unpersisted resume branch (no state.db row yet — every fresh
    Bot Chat) must ALSO rebind the transport and cancel the pending reap when
    it hands back a sentinel-parked live record. Found by live WS E2E after
    the #93361 merge: the unit-covered paths (_live_session_payload,
    _reuse_live_response, _claim_or_reuse_live) all cancelled, but this branch
    returned the record while leaving it on the drop sentinel with the reap
    Timer armed — the storm survived for unpersisted sessions (storm killer,
    part 3)."""
    cancelled = []

    class _Timer:
        def __init__(self, _delay, fn):
            self.fn = fn

        def start(self):
            return None

        def cancel(self):
            cancelled.append(self)

    class _LiveTransport:
        def write(self, *a, **k):
            return True

    live_transport = _LiveTransport()
    monkeypatch.setattr(server, "_WS_ORPHAN_REAP_GRACE_S", 0.01)
    monkeypatch.setattr(server.threading, "Timer", _Timer)
    monkeypatch.setattr(server, "current_transport", lambda: live_transport)
    # get_session/get_session_by_title miss -> forces the unpersisted branch
    monkeypatch.setattr(
        server,
        "_get_db",
        lambda: types.SimpleNamespace(
            get_session=lambda _t: None,
            get_session_by_title=lambda _t: None,
        ),
    )

    session = _session(
        session_key="stored-lazy",
        transport=server._detached_ws_transport,
        running=False,
        history=[],
        profile_home=None,
    )
    server._sessions["lazy-sid"] = session

    try:
        server._schedule_ws_orphan_reap("lazy-sid")
        assert "lazy-sid" in server._pending_ws_reaps

        resp = _dispatch_sync(
            {
                "id": "lz1",
                "method": "session.resume",
                "params": {"session_id": "stored-lazy", "omit_messages": True},
            },
            transport=live_transport,
        )

        assert resp is not None and resp["result"]["session_id"] == "lazy-sid"
        assert session["transport"] is live_transport
        assert "lazy-sid" not in server._pending_ws_reaps
        assert len(cancelled) == 1
    finally:
        server._sessions.pop("lazy-sid", None)
        server._pending_ws_reaps.pop("lazy-sid", None)


def test_ws_orphan_reap_still_fires_when_never_resumed(monkeypatch):
    """Nobody re-resumes: the reap fires normally and unregisters its Timer."""
    callbacks = []
    torn_down = []

    class _Timer:
        def __init__(self, _delay, fn):
            callbacks.append(fn)

        def start(self):
            return None

        def cancel(self):
            return None

    monkeypatch.setattr(server, "_WS_ORPHAN_REAP_GRACE_S", 0.01)
    monkeypatch.setattr(server.threading, "Timer", _Timer)
    monkeypatch.setattr(
        server,
        "_teardown_popped_session",
        lambda popped, *, end_reason: torn_down.append((popped, end_reason)) or True,
    )
    session = _session(transport=server._detached_ws_transport, running=False)
    server._sessions["lonely-sid"] = session

    try:
        server._schedule_ws_orphan_reap("lonely-sid")
        assert "lonely-sid" in server._pending_ws_reaps
        callbacks.pop(0)()

        assert "lonely-sid" not in server._sessions
        assert "lonely-sid" not in server._pending_ws_reaps
        assert torn_down == [(session, "ws_orphan_reap")]
    finally:
        server._sessions.pop("lonely-sid", None)
        server._pending_ws_reaps.pop("lonely-sid", None)


def test_ws_orphan_reap_spares_detached_session_with_running_async_delegation(monkeypatch):
    """A detached desktop session with live background delegation is parked.

    Regression for Desktop session switches / transient WS detaches: the parent
    turn is idle, but a background delegate_task still owns the session's
    return address. Reaping immediately interrupts the child and turns its
    completion into an unowned orphan.
    """
    timers = []
    closed = []

    class _Timer:
        def __init__(self, _delay, fn):
            self.fn = fn
            timers.append(self)

        def start(self):
            return None

    class _DB:
        def get_session(self, _session_id):
            return {"id": "sess_bg", "source": "desktop"}

    monkeypatch.setattr(server, "_WS_ORPHAN_REAP_GRACE_S", 0.01)
    monkeypatch.setattr(server.threading, "Timer", _Timer)
    monkeypatch.setattr(server, "_get_db", lambda: _DB())
    monkeypatch.setattr(
        server,
        "_teardown_popped_session",
        lambda session, *, end_reason="tui_close": (
            closed.append((session["_sid"], end_reason)) if session is not None else None
        ),
    )

    server._sessions["bg-sid"] = _session(
        transport=server._detached_ws_transport,
        running=False,
        session_key="sess_bg",
    )
    ad._reset_for_tests()
    try:
        with ad._records_lock:
            ad._records["deleg_bg"] = {
                "delegation_id": "deleg_bg",
                "status": "running",
                "session_key": "sess_bg",
                "origin_ui_session_id": "bg-sid",
                "interrupt_fn": lambda: None,
            }

        server._schedule_ws_orphan_reap("bg-sid")
        assert len(timers) == 1

        timers.pop(0).fn()

        assert closed == []
        assert "bg-sid" in server._sessions
        assert len(timers) == 1

        with ad._records_lock:
            ad._records["deleg_bg"]["status"] = "finalizing"
            ad._records["deleg_bg"]["interrupt_fn"] = None

        timers.pop(0).fn()

        assert closed == []
        assert "bg-sid" in server._sessions
        assert len(timers) == 1

        with ad._records_lock:
            ad._records["deleg_bg"]["status"] = "completed"

        timers.pop(0).fn()

        assert closed == [("bg-sid", "ws_orphan_reap")]
    finally:
        ad._reset_for_tests()
        server._sessions.pop("bg-sid", None)



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


@pytest.mark.real_agent_prewarm
def test_session_create_no_race_keeps_worker_alive(monkeypatch):
    """Regression guard: when session.close does NOT race, the build
    thread must install the notify normally and leave it alone (no
    over-eager cleanup) — and must not pre-warm a slash worker (spawn
    is on-demand in slash.exec)."""
    closed_workers: list[str] = []
    unregistered_keys: list[str] = []

    class _FakeWorker:
        def __init__(self, key, model, profile_home=None):
            self.key = key

        def close(self):
            closed_workers.append(self.key)

    class _FakeAgent:
        def __init__(self):
            self.model = "x"
            self.provider = "openrouter"
            self.base_url = ""
            self.api_key = ""

    monkeypatch.setattr(server, "_make_agent", lambda sid, key, session_db=None, **_kwargs: _FakeAgent())
    monkeypatch.setattr(server, "_SlashWorker", _FakeWorker)
    monkeypatch.setattr(
        server,
        "_get_db",
        lambda: types.SimpleNamespace(create_session=lambda *a, **kw: None),
    )
    monkeypatch.setattr(server, "_session_info", lambda _a, *a2: {"model": "x"})
    monkeypatch.setattr(server, "_probe_credentials", lambda _a: None)
    monkeypatch.setattr(server, "_wire_callbacks", lambda _sid: None)
    monkeypatch.setattr(server, "_emit", lambda *a, **kw: None)

    import tools.approval as _approval

    monkeypatch.setattr(_approval, "register_gateway_notify", lambda key, cb: None)
    monkeypatch.setattr(
        _approval,
        "unregister_gateway_notify",
        lambda key: unregistered_keys.append(key),
    )
    monkeypatch.setattr(_approval, "load_permanent_allowlist", lambda: None)

    # Isolate from sibling-test leakage: daemon build threads from prior
    # session.create tests in the same shard process mutate the shared
    # ``server._sessions`` dict under ``_sessions_lock`` and can replace/pop
    # entries mid-run, which would flip this build thread's ``replaced`` check
    # to True and trigger a spurious unregister. Snapshot, clear, and restore
    # so this test sees only its own session regardless of shard composition.
    _saved_sessions = dict(server._sessions)
    server._sessions.clear()

    try:
        resp = server.handle_request(
            {
                "id": "1",
                "method": "session.create",
                "params": {"cols": 80},
            }
        )
        sid = resp["result"]["session_id"]

        # Wait for the build to finish (ready event inside session dict).
        session = server._sessions[sid]
        built = session["agent_ready"].wait(timeout=10.0)
        assert built, "agent build did not complete within timeout"

        # Build finished without a close race — nothing should have been
        # cleaned up by the orphan check.  Scope the assertions to THIS
        # test's own session_key: a daemon build thread leaked from a prior
        # session.create test in the same shard process can fire close/
        # unregister against its own (foreign) key after we've patched the
        # global hooks, polluting these lists.  Filtering by this session's
        # key keeps the regression intent (this session's worker/notify must
        # survive) while making the test immune to shard composition.
        # (flaky under -j 8: foreign key e.g. 20260629_210208_d4f545)
        own_key = session["session_key"]
        own_closed = [k for k in closed_workers if k == own_key]
        own_unregistered = [k for k in unregistered_keys if k == own_key]
        assert (
            own_closed == []
        ), f"build thread closed its own worker despite no race: {own_closed}"
        assert (
            own_unregistered == []
        ), f"build thread unregistered its own notify despite no race: {own_unregistered}"

        # No pre-warmed worker: slash.exec spawns on demand, so a fresh
        # session that hasn't run a worker-routed command carries None.
        assert session.get("slash_worker") is None
    finally:
        # Cleanup + restore sibling sessions we snapshotted.
        server._sessions.clear()
        server._sessions.update(_saved_sessions)


def test_get_db_degrades_cleanly_when_sessiondb_init_fails(monkeypatch):
    fake_mod = types.ModuleType("hermes_state")

    class _BrokenSessionDB:
        def __init__(self):
            raise RuntimeError("locking protocol")

    fake_mod.SessionDB = _BrokenSessionDB

    def _broken_shared(_db_path=None):
        raise RuntimeError("locking protocol")

    monkeypatch.setitem(sys.modules, "hermes_state", fake_mod)
    fake_registry = types.ModuleType("hermes_state_registry")
    fake_registry.acquire = _broken_shared
    monkeypatch.setitem(sys.modules, "hermes_state_registry", fake_registry)
    monkeypatch.setattr(server, "_db", None)
    monkeypatch.setattr(server, "_db_error", None)

    assert server._get_db() is None
    assert server._db_error == "locking protocol"


def test_ensure_session_db_row_false_when_store_unavailable(monkeypatch):
    """Store unavailable → False, so prompt.submit can fail the send loudly
    instead of streaming into a store that will never save it (#98924)."""
    fake_mod = types.ModuleType("hermes_state")

    class _BrokenSessionDB:
        def __init__(self):
            raise RuntimeError("utf-8 boom")

    fake_mod.SessionDB = _BrokenSessionDB

    def _broken_shared(_db_path=None):
        raise RuntimeError("utf-8 boom")

    monkeypatch.setitem(sys.modules, "hermes_state", fake_mod)
    fake_registry = types.ModuleType("hermes_state_registry")
    fake_registry.acquire = _broken_shared
    monkeypatch.setitem(sys.modules, "hermes_state_registry", fake_registry)
    monkeypatch.setattr(server, "_db", None)
    monkeypatch.setattr(server, "_db_error", None)

    assert server._ensure_session_db_row({"session_key": "k1"}) is False


def test_ensure_session_db_row_true_when_row_persisted(monkeypatch):
    created = []

    class _FakeDB:
        def create_session(self, key, **_kwargs):
            created.append(key)

    monkeypatch.setattr(server, "_get_db", lambda: _FakeDB())
    monkeypatch.setattr(server, "_resolve_model", lambda: "test-model")

    assert server._ensure_session_db_row({"session_key": "k1", "cwd": "/tmp"}) is True
    assert created == ["k1"]


def test_prompt_submit_fails_loudly_when_store_unavailable(monkeypatch):
    """A send with no persistable store must fail the RPC with a real error
    (desktop maps it to a toast) instead of streaming the message into a
    store that will never save it (#98924)."""
    monkeypatch.setattr(server, "_load_cfg", lambda: {"dashboard": {}})
    monkeypatch.setattr(server, "_get_db", lambda: None)
    monkeypatch.setattr(server, "_db_error", "utf-8 decode failure")

    server._sessions["lost-sid"] = _session()
    try:
        resp = server.handle_request(
            {
                "id": "lost",
                "method": "prompt.submit",
                "params": {"session_id": "lost-sid", "text": "will vanish"},
            }
        )
    finally:
        server._sessions.pop("lost-sid", None)

    assert resp["error"]["code"] == 5072
    msg = resp["error"]["message"]
    assert "utf-8 decode failure" not in msg  # raw cause rides `data.details`, never the lead
    assert resp["error"]["data"]["code"] == "storage_unavailable"
    assert "utf-8 decode failure" in resp["error"]["data"]["details"]


@pytest.mark.real_agent_prewarm
def test_session_create_continues_when_state_db_is_unavailable(monkeypatch):
    class _FakeWorker:
        def __init__(self, key, model, profile_home=None):
            self.key = key

        def close(self):
            return None

    class _FakeAgent:
        def __init__(self):
            self.model = "x"
            self.provider = "openrouter"
            self.base_url = ""
            self.api_key = ""

    emits = []

    monkeypatch.setattr(server, "_make_agent", lambda sid, key, session_db=None, **_kwargs: _FakeAgent())
    monkeypatch.setattr(server, "_SlashWorker", _FakeWorker)
    monkeypatch.setattr(server, "_get_db", lambda: None)
    monkeypatch.setattr(server, "_session_info", lambda _a, *a2: {"model": "x"})
    monkeypatch.setattr(server, "_probe_credentials", lambda _a: None)
    monkeypatch.setattr(server, "_wire_callbacks", lambda _sid: None)
    monkeypatch.setattr(server, "_emit", lambda *a, **kw: emits.append(a))

    import tools.approval as _approval

    monkeypatch.setattr(_approval, "register_gateway_notify", lambda key, cb: None)
    monkeypatch.setattr(_approval, "load_permanent_allowlist", lambda: None)

    resp = server.handle_request(
        {"id": "1", "method": "session.create", "params": {"cols": 80}}
    )
    sid = resp["result"]["session_id"]
    session = server._sessions[sid]
    session["agent_ready"].wait(timeout=2.0)

    assert session["agent_error"] is None
    assert session["agent"] is not None
    assert not any(args and args[0] == "error" for args in emits)

    server._sessions.pop(sid, None)


def test_session_create_lazy_info_reports_desktop_contract(monkeypatch):
    """The lazy session.create info payload must carry desktop_contract, else
    the desktop GUI reads it as undefined and falsely warns "Backend out of
    date" on every launch even against a current backend."""

    class _FakeWorker:
        def __init__(self, key, model, profile_home=None):
            self.key = key

        def close(self):
            return None

    monkeypatch.setattr(server, "_SlashWorker", _FakeWorker)
    monkeypatch.setattr(server, "_get_db", lambda: None)
    monkeypatch.setattr(server, "_emit", lambda *a, **kw: None)
    monkeypatch.setattr(server, "_start_agent_build", lambda *a, **kw: None)

    resp = server.handle_request(
        {"id": "1", "method": "session.create", "params": {"cols": 80}}
    )
    info = resp["result"]["info"]

    assert info["desktop_contract"] == server.DESKTOP_BACKEND_CONTRACT

    server._sessions.pop(resp["result"]["session_id"], None)


def test_session_activate_lazy_info_reports_desktop_contract():
    """Activating an already-live *lazy* session (agent not built yet) must
    still advertise desktop_contract. _live_session_payload falls back to
    _fallback_session_info while session["agent"] is None; the desktop reads a
    missing field as contract 0 and falsely warns "Backend out of date" against
    a current backend (#68392). The sibling session.create path was fixed in
    #36112; this pins the session.activate path."""
    import threading

    sid = "lazy-activate-contract"
    server._sessions[sid] = {
        "agent": None,
        "created_at": 123.0,
        "history": [],
        "history_lock": threading.RLock(),
        "last_active": 123.0,
        "running": False,
        "session_key": sid,
        "transport": server._stdio_transport,
    }
    try:
        resp = server.handle_request(
            {
                "id": "activate-lazy",
                "method": "session.activate",
                "params": {"session_id": sid},
            }
        )
        info = resp["result"]["info"]
        assert info["lazy"] is True
        assert info["desktop_contract"] == server.DESKTOP_BACKEND_CONTRACT
    finally:
        server._sessions.pop(sid, None)


def test_session_list_returns_clean_error_when_state_db_is_unavailable(monkeypatch):
    monkeypatch.setattr(server, "_get_db", lambda: None)
    monkeypatch.setattr(server, "_db_error", "locking protocol")

    resp = server.handle_request({"id": "1", "method": "session.list", "params": {}})

    assert "error" in resp
    assert resp["error"]["data"]["code"] == "storage_unavailable"
    assert resp["error"]["data"]["details"] == "locking protocol"


def test_session_list_honors_params_profile_opens_profile_db(monkeypatch, tmp_path):
    """Issue #62503: session.list must read the profile's state.db, not launch."""
    profile_home = tmp_path / "profiles" / "mlperf"
    profile_home.mkdir(parents=True)
    (profile_home / "state.db").write_bytes(b"")
    seen: dict = {}

    class LaunchDB:
        def list_sessions_rich(self, **kwargs):
            seen["launch"] = True
            return [{"id": "launch-1", "source": "tui", "title": "L"}]

    class ProfileDB:
        def __init__(self, db_path=None):
            seen["db_path"] = db_path

        def list_sessions_rich(self, **kwargs):
            seen["profile"] = True
            return [
                {
                    "id": "ml-1",
                    "source": "tui",
                    "title": "M",
                    "preview": "",
                    "started_at": 1,
                    "message_count": 1,
                }
            ]

        def close(self):
            seen["closed"] = True

    monkeypatch.setattr(server, "_profile_home", lambda p: profile_home if p == "mlperf" else None)
    monkeypatch.setattr(server, "_get_db", lambda: LaunchDB())
    monkeypatch.setattr(
        "hermes_cli.web_server_sessions._open_session_db_at_path",
        lambda db_path, *, read_only: ProfileDB(db_path=db_path),
    )

    resp = server.handle_request(
        {
            "id": "1",
            "method": "session.list",
            "params": {"profile": "mlperf", "limit": 5},
        }
    )
    assert "result" in resp, resp
    assert resp["result"]["sessions"][0]["id"] == "ml-1"
    assert seen.get("profile") is True
    assert seen.get("launch") is None
    assert str(seen.get("db_path")).endswith("state.db")
    assert seen.get("closed") is True


def test_session_most_recent_honors_params_profile(monkeypatch, tmp_path):
    """Issue #62503: session.most_recent must not return the launch profile tip."""
    profile_home = tmp_path / "profiles" / "mlperf"
    profile_home.mkdir(parents=True)

    class LaunchDB:
        def list_sessions_rich(self, **kwargs):
            return [{"id": "launch-tip", "source": "tui", "title": "L", "started_at": 9}]

    class ProfileDB2:
        def __init__(self, db_path=None):
            self.db_path = db_path

        def list_sessions_rich(self, **kwargs):
            return [
                {"id": "tool-noise", "source": "tool", "title": "t", "started_at": 9},
                {"id": "ml-tip", "source": "desktop", "title": "M", "started_at": 3},
            ]

        def close(self):
            pass

    monkeypatch.setattr(server, "_profile_home", lambda p: profile_home if p == "mlperf" else None)
    monkeypatch.setattr(server, "_get_db", lambda: LaunchDB())
    monkeypatch.setattr(
        "hermes_cli.web_server_sessions._open_session_db_at_path",
        lambda db_path, *, read_only: ProfileDB2(db_path=db_path),
    )

    resp = server.handle_request(
        {
            "id": "1",
            "method": "session.most_recent",
            "params": {"profile": "mlperf"},
        }
    )
    assert resp["result"]["session_id"] == "ml-tip"


def test_handoff_request_uses_session_profile_home(monkeypatch, tmp_path):
    """Handoff validation must read the owning session's gateway config."""

    from gateway.config import GatewayConfig, HomeChannel, Platform, PlatformConfig
    from hermes_cli.config import get_hermes_home
    from hermes_runtime import methods_session

    methods_session.register(server)
    profile_home = tmp_path / "profiles" / "coder"
    profile_home.mkdir(parents=True)
    seen_homes = []

    def load_config():
        home = get_hermes_home()
        seen_homes.append(home)
        config = GatewayConfig()
        if home == profile_home:
            config.platforms[Platform.DISCORD] = PlatformConfig(
                enabled=True,
                home_channel=HomeChannel(
                    platform=Platform.DISCORD,
                    chat_id="discord-home",
                    name="Hermes / #chat-coding",
                ),
            )
        return config

    class ProfileDB:
        def get_session(self, _key):
            return {"id": _key}

        def request_handoff(self, _key, platform):
            return platform == "discord"

    @contextlib.contextmanager
    def profile_db(_session):
        yield ProfileDB()

    monkeypatch.setattr("gateway.config.load_gateway_config", load_config)
    monkeypatch.setattr(server, "_ensure_session_db_row", lambda _session: None)
    monkeypatch.setattr(server, "_session_db", profile_db)
    server._sessions["handoff-profile"] = {
        "running": False,
        "session_key": "desktop-coder-session",
        "profile_home": str(profile_home),
    }
    try:
        resp = server.handle_request(
            {
                "id": "1",
                "method": "handoff.request",
                "params": {
                    "session_id": "handoff-profile",
                    "platform": "discord",
                },
            }
        )
    finally:
        server._sessions.pop("handoff-profile", None)

    assert "result" in resp, resp
    assert resp["result"]["queued"] is True
    assert seen_homes == [profile_home]
    assert get_hermes_home() != profile_home


def test_session_create_reports_requested_profile_name(monkeypatch, tmp_path):
    """Issue #62503: session.create info.profile_name must not always be launch."""
    profile_home = tmp_path / "profiles" / "mlperf"
    profile_home.mkdir(parents=True)

    def _clear():
        for session in list(server._sessions.values()):
            server._teardown_session(session)
        server._sessions.clear()

    monkeypatch.setattr(server, "_start_agent_build", lambda *a, **k: None)
    monkeypatch.setattr(server, "_schedule_agent_build", lambda *a, **k: None)
    monkeypatch.setattr(server, "_schedule_session_cap_enforcement", lambda *a, **k: None)
    monkeypatch.setattr(server, "_completion_cwd", lambda params=None: str(tmp_path))
    monkeypatch.setattr(server, "_profile_home", lambda p: profile_home if p == "mlperf" else None)
    monkeypatch.setattr(server, "_current_profile_name", lambda: "default")
    monkeypatch.setattr(server, "_claim_active_session_slot", lambda *a, **k: (None, None))
    _clear()
    try:
        resp = server._methods["session.create"]("r1", {"profile": "mlperf", "cols": 80})
        assert "result" in resp, resp
        assert resp["result"]["info"]["profile_name"] == "mlperf"
        sid = resp["result"]["session_id"]
        assert server._sessions[sid]["profile_home"] == str(profile_home)
    finally:
        _clear()


def test_session_title_uses_session_profile_db_not_launch(monkeypatch, tmp_path):
    """session.title on a non-launch profile session must not touch launch DB."""
    profile_home = tmp_path / "profiles" / "mlperf"
    profile_home.mkdir(parents=True)
    seen: dict = {}

    class LaunchDB:
        def get_session_title(self, _key):
            seen["launch_read"] = True
            return "from-launch"

        def set_session_title(self, _key, _title):
            seen["launch_write"] = True
            return True

        def get_session(self, _key):
            return {"id": _key, "title": "from-launch"}

    class ProfileDB:
        def __init__(self, db_path=None):
            self.db_path = db_path
            seen["db_path"] = db_path

        def get_session_title(self, _key):
            return seen.get("title")

        def get_session(self, _key):
            if "title" in seen:
                return {"id": _key, "title": seen["title"]}
            return None

        def set_session_title(self, _key, title):
            seen["title"] = title
            seen["profile_write"] = True
            return True

        def close(self):
            seen["closed"] = True

    server._sessions["sid"] = {
        "session_key": "ml-sess",
        "history": [],
        "history_lock": __import__("threading").Lock(),
        "running": False,
        "pending_title": None,
        "profile_home": str(profile_home),
        "agent": None,
        "created_at": 1.0,
        "last_active": 1.0,
    }
    monkeypatch.setattr(server, "_get_db", lambda: LaunchDB())
    monkeypatch.setattr("hermes_state_registry.acquire", ProfileDB)
    try:
        set_resp = server.handle_request(
            {
                "id": "1",
                "method": "session.title",
                "params": {"session_id": "sid", "title": "profile-title"},
            }
        )
        assert "result" in set_resp, set_resp
        assert set_resp["result"]["title"] == "profile-title"
        assert seen.get("profile_write") is True
        assert seen.get("launch_write") is None
        assert str(seen.get("db_path")).endswith("state.db")

        get_resp = server.handle_request(
            {"id": "2", "method": "session.title", "params": {"session_id": "sid"}}
        )
        assert get_resp["result"]["title"] == "profile-title"
        assert seen.get("launch_read") is None
    finally:
        server._sessions.pop("sid", None)


def test_session_history_uses_session_profile_db(monkeypatch, tmp_path):
    """session.history must read durable messages from the profile state.db."""
    profile_home = tmp_path / "profiles" / "mlperf"
    profile_home.mkdir(parents=True)
    seen: dict = {}

    class LaunchDB:
        def get_messages_as_conversation(self, _key, include_ancestors=True, **_kwargs):
            seen["launch"] = True
            return [{"role": "user", "content": "launch"}]

    class ProfileDB:
        def __init__(self, db_path=None):
            seen["db_path"] = db_path

        def get_messages_as_conversation(self, _key, include_ancestors=True, **_kwargs):
            seen["profile"] = True
            return [{"role": "user", "content": "from-profile"}]

        def close(self):
            seen["closed"] = True

    server._sessions["sid"] = {
        "session_key": "ml-sess",
        "history": [{"role": "user", "content": "mem"}],
        "history_lock": __import__("threading").Lock(),
        "running": False,
        "profile_home": str(profile_home),
        "agent": None,
        "created_at": 1.0,
        "last_active": 1.0,
    }
    monkeypatch.setattr(server, "_get_db", lambda: LaunchDB())
    monkeypatch.setattr("hermes_state_registry.acquire", ProfileDB)
    try:
        resp = server.handle_request(
            {"id": "1", "method": "session.history", "params": {"session_id": "sid"}}
        )
        assert "result" in resp, resp
        assert seen.get("profile") is True
        assert seen.get("launch") is None
        # Count comes from profile-backed conversation (1 msg), not bare mem list alone.
        assert resp["result"]["count"] == 1
    finally:
        server._sessions.pop("sid", None)


def test_session_history_ships_durable_row_ids(monkeypatch):
    """session.history must request row-id stamps — clients resolve truncation
    targets by content against this projection (#87059 client half)."""
    seen: dict = {}

    class _Db:
        def get_messages_as_conversation(self, _key, include_ancestors=False, include_row_ids=False, **_kwargs):
            seen["include_row_ids"] = include_row_ids

            return [
                {"role": "user", "content": "hello", "_row_id": 41},
                {"role": "assistant", "content": "hi"},
            ]

    server._sessions["rowid-hist-sid"] = {
        "session_key": "rowid-sess",
        "history": [],
        "history_lock": __import__("threading").Lock(),
        "running": False,
        "agent": None,
        "created_at": 1.0,
        "last_active": 1.0,
    }
    monkeypatch.setattr(server, "_get_db", lambda: _Db())
    try:
        resp = server.handle_request(
            {"id": "1", "method": "session.history", "params": {"session_id": "rowid-hist-sid"}}
        )
        assert "result" in resp, resp
        assert seen.get("include_row_ids") is True
        user_rows = [m for m in resp["result"]["messages"] if m.get("role") == "user"]
        assert user_rows and user_rows[0].get("row_id") == 41
    finally:
        server._sessions.pop("rowid-hist-sid", None)


def test_session_status_uses_session_profile_db(monkeypatch, tmp_path):
    """session.status must load meta from the session profile state.db."""
    profile_home = tmp_path / "profiles" / "mlperf"
    profile_home.mkdir(parents=True)
    seen: dict = {}

    class LaunchDB:
        def get_session(self, _key):
            seen["launch"] = True
            return {"id": _key, "title": "launch-title", "started_at": 1}

    class ProfileDB:
        def __init__(self, db_path=None):
            seen["db_path"] = db_path

        def get_session(self, _key):
            seen["profile"] = True
            return {"id": _key, "title": "profile-title", "started_at": 42}

        def close(self):
            seen["closed"] = True

    server._sessions["sid"] = {
        "session_key": "ml-sess",
        "history": [],
        "history_lock": __import__("threading").Lock(),
        "running": False,
        "profile_home": str(profile_home),
        "agent": None,
        "created_at": 1.0,
        "last_active": 1.0,
    }
    monkeypatch.setattr(server, "_get_db", lambda: LaunchDB())
    monkeypatch.setattr("hermes_state_registry.acquire", ProfileDB)
    try:
        resp = server.handle_request(
            {"id": "1", "method": "session.status", "params": {"session_id": "sid"}}
        )
        assert "result" in resp, resp
        assert "profile-title" in resp["result"]["output"]
        assert seen.get("profile") is True
        assert seen.get("launch") is None
    finally:
        server._sessions.pop("sid", None)


def test_teardown_ends_session_in_profile_db(monkeypatch, tmp_path):
    """_teardown_session must end_session on the profile store, not launch."""
    profile_home = tmp_path / "profiles" / "mlperf"
    profile_home.mkdir(parents=True)
    seen: dict = {}

    class LaunchDB:
        def get_session(self, _key):
            seen["launch"] = True
            return {"id": _key, "source": "tui"}

        def end_session(self, _key, _reason):
            seen["launch_end"] = True

    class ProfileDB:
        def __init__(self, db_path=None):
            seen["db_path"] = db_path

        def get_session(self, _key):
            seen["profile"] = True
            return {"id": _key, "source": "tui"}

        def end_session(self, key, reason):
            seen["ended"] = (key, reason)

        def close(self):
            seen["closed"] = True

    monkeypatch.setattr(server, "_get_db", lambda: LaunchDB())
    monkeypatch.setattr("hermes_state_registry.acquire", ProfileDB)
    session = {
        "session_key": "ml-sess",
        "profile_home": str(profile_home),
        "agent": None,
        "history": [],
        "source": "tui",
    }
    server._teardown_session(session, end_reason="closed")
    assert seen.get("ended") == ("ml-sess", "closed")
    assert seen.get("launch_end") is None
    assert seen.get("launch") is None
    assert str(seen.get("db_path")).endswith("state.db")


def test_session_branch_writes_to_parent_profile_db(monkeypatch, tmp_path):
    """session.branch must copy history into the parent's profile state.db."""
    profile_home = tmp_path / "profiles" / "mlperf"
    profile_home.mkdir(parents=True)
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    seen: dict = {"msgs": []}

    class LaunchDB:
        def get_session_title(self, _key):
            seen["launch"] = True
            return "L"

        def create_session(self, *a, **k):
            seen["launch_create"] = True

        def append_message(self, **k):
            seen["launch_msg"] = True

        def set_session_title(self, *a, **k):
            return True

    class ProfileDB:
        def __init__(self, db_path=None):
            seen["db_path"] = db_path
            seen.setdefault("inits", 0)
            seen["inits"] += 1

        def get_session_title(self, _key):
            return "parent"

        def get_next_title_in_lineage(self, current):
            return f"{current} (branch)"

        def create_session(self, new_key, **kwargs):
            seen["created"] = new_key
            seen["parent"] = kwargs.get("parent_session_id")
            seen["profile_name"] = kwargs.get("profile_name")

        def append_message(self, **kwargs):
            seen["msgs"].append(kwargs)

        def append_messages_batch(self, session_id, messages, **kwargs):
            for m in messages:
                seen["msgs"].append(dict(m, session_id=session_id))
            return list(range(1, len(messages) + 1))

        def set_session_title(self, key, title):
            seen["title"] = (key, title)
            return True

        def get_session(self, key):
            return {"id": key, "cwd": str(tmp_path)}

        def update_session_cwd(self, *a, **k):
            return None

        def close(self):
            seen["closed"] = True

    class FakeAgent:
        def __init__(self):
            self.model = "test-model"
            self.session_id = None

    parent = {
        "session_key": "parent-key",
        "history": [{"role": "user", "content": "hi"}],
        "history_lock": __import__("threading").Lock(),
        "running": False,
        "cols": 80,
        "profile_home": str(profile_home),
        "source": "tui",
        "agent": FakeAgent(),
        "created_at": 1.0,
        "last_active": 1.0,
        "cwd": str(tmp_path),
    }
    server._sessions["parent"] = parent
    monkeypatch.setattr(server, "_get_db", lambda: LaunchDB())
    monkeypatch.setattr("hermes_state_registry.acquire", ProfileDB)
    monkeypatch.setattr(server, "_claim_active_session_slot", lambda *a, **k: (None, None))

    def _fake_make_agent(*a, **k):
        seen["agent_session_db"] = k.get("session_db")
        seen["agent_cwd"] = k.get("cwd_override")
        return FakeAgent()

    monkeypatch.setattr(server, "_make_agent", _fake_make_agent)
    monkeypatch.setattr(server, "_set_session_context", lambda *a, **k: {})
    monkeypatch.setattr(server, "_clear_session_context", lambda *a, **k: None)
    monkeypatch.setattr(server, "_resolve_model", lambda: "test-model")
    monkeypatch.setattr(server, "_session_cwd", lambda s: str(tmp_path))
    monkeypatch.setattr(server, "_register_session_cwd", lambda *a, **k: None)
    monkeypatch.setattr(server, "_attach_worker", lambda *a, **k: None)
    try:
        resp = server.handle_request(
            {
                "id": "1",
                "method": "session.branch_whole",
                "params": {"session_id": "parent", "name": "forked"},
            }
        )
        assert "result" in resp, resp
        assert seen.get("created")
        assert seen.get("parent") == "parent-key"
        # The branch row is self-describing: stamped with the parent's owning
        # profile, not left NULL for aggregators to mis-tag as "default".
        assert seen.get("profile_name") == "mlperf"
        assert resp["result"].get("messages_omitted") is True
        assert "messages" not in resp["result"]
        assert seen.get("title") == (seen["created"], "forked")
        assert len(seen["msgs"]) == 1
        assert seen.get("launch") is None
        assert seen.get("launch_create") is None
        child_sid = resp["result"]["session_id"]
        assert server._sessions[child_sid]["profile_home"] == str(profile_home)
        # The branched AGENT must be bound to the parent profile's state.db —
        # not just the row. Otherwise its own flushes (and a later compression
        # rotation) land on the launch db, splitting the lineage again.
        assert isinstance(seen.get("agent_session_db"), ProfileDB)
        assert seen.get("agent_cwd") == str(tmp_path)
    finally:
        for k in list(server._sessions):
            server._sessions.pop(k, None)


def test_session_create_persists_seeded_branch_child(monkeypatch):
    """A desktop branch (session.create with parent_session_id + seeded
    messages) must persist its row + transcript immediately (#93959).

    The renderer re-fetches the fresh child via REST and defer_history
    hydration right after create; both read the DB. An unpersisted child
    404s/hydrates empty, the client fail-latch refuses to bind it, and the
    user gets an infinite spinner whose optimistic row vanishes on restart.
    """

    class _FakeAgent:
        def __init__(self):
            self.model = "test-model"

    seen: dict = {}

    class _FakeDB:
        def get_session_title(self, key):
            seen["parent_title"] = key
            return "My Parent Session"

        def get_next_title_in_lineage(self, current):
            return f"{current} #2"

        def create_session(self, key, **kwargs):
            seen["created"] = key
            seen["parent"] = kwargs.get("parent_session_id")
            seen["branched_from"] = (kwargs.get("model_config") or {}).get("_branched_from")

        def append_messages_batch(self, session_id, messages, **kwargs):
            seen["messages"] = list(messages)

        def set_auto_title(self, key, title, *, source):
            seen["title"] = title
            seen["title_source"] = source
            return True

    monkeypatch.setattr(server, "_get_db", lambda: _FakeDB())
    monkeypatch.setattr(server, "_make_agent", lambda sid, key, session_db=None, **_kw: _FakeAgent())
    monkeypatch.setattr(server, "_SlashWorker", lambda *a, **k: None)
    monkeypatch.setattr(server, "_session_info", lambda _a, *a2: {"model": "x"})
    monkeypatch.setattr(server, "_probe_credentials", lambda _a: None)
    monkeypatch.setattr(server, "_wire_callbacks", lambda _sid: None)
    monkeypatch.setattr(server, "_emit", lambda *a, **kw: None)

    import tools.approval as _approval

    monkeypatch.setattr(_approval, "register_gateway_notify", lambda key, cb: None)
    monkeypatch.setattr(_approval, "load_permanent_allowlist", lambda: None)

    seeded = [
        {"role": "user", "content": "hello from parent"},
        {"role": "assistant", "content": "parent reply"},
    ]

    resp = server.handle_request(
        {
            "id": "1",
            "method": "session.create",
            "params": {
                "cols": 96,
                "source": "desktop",
                "parent_session_id": "20260823_084113_6de211",
                "messages": seeded,
            },
        }
    )

    assert "result" in resp, resp
    key = resp["result"]["stored_session_id"]

    # Row persisted up front with lineage linkage and a lineage title —
    # not deferred to the first prompt.
    assert seen.get("created") == key
    assert seen.get("parent") == "20260823_084113_6de211"
    assert seen.get("branched_from") == "20260823_084113_6de211"
    assert seen.get("title") == "My Parent Session #2"
    assert seen.get("title_source") == "derived"

    # Seeded transcript copied into the durable row so REST prefetch and
    # defer_history hydration both find it immediately.
    assert len(seen.get("messages") or []) == 2
    assert seen["messages"][0]["content"] == "hello from parent"

    # The live record no longer queues the title — the DB already holds it.
    runtime_sid = resp["result"]["session_id"]
    assert server._sessions[runtime_sid]["pending_title"] is None

    server._sessions.pop(runtime_sid, None)


def test_session_branch_stored_copies_parent_history_without_returning_transcript(monkeypatch):
    """Whole-session desktop branches read the parent in the gateway, not in the renderer."""

    class _Scope:
        def __init__(self, db):
            self.db = db

        def __enter__(self):
            return self.db

        def __exit__(self, *_args):
            return False

    class _FakeDB:
        def get_resume_conversations(self, key):
            assert key == "parent"
            return [], [
                {"role": "user", "content": "first question", "timestamp": 1},
                {"role": "assistant", "content": "first answer", "timestamp": 2},
            ]

    class _FakeAgent:
        model = "test-model"

    seen: dict = {}
    db = _FakeDB()

    monkeypatch.setattr(server, "_profile_db", lambda _params: _Scope(db))
    monkeypatch.setattr(
        server,
        "_seed_branch_row",
        lambda _record, _key, _parent, history, *_args: seen.update(history=list(history)),
    )
    monkeypatch.setattr(server, "_make_agent", lambda *_args, **_kwargs: _FakeAgent())
    monkeypatch.setattr(server, "_session_info", lambda *_args: {"model": "test-model"})

    resp = server.handle_request(
        {
            "id": "1",
            "method": "session.branch_stored",
            "params": {
                "cols": 96,
                "parent_session_id": "parent",
                "source": "desktop",
            },
        }
    )

    assert "result" in resp, resp
    assert [message["content"] for message in seen["history"]] == ["first question", "first answer"]
    assert resp["result"]["message_count"] == 2
    assert resp["result"].get("messages_omitted") is True
    assert "messages" not in resp["result"]


def test_session_create_branch_seed_failure_does_not_break_create(monkeypatch):
    """Best-effort persistence: a broken DB must not fail session.create."""

    class _FakeAgent:
        def __init__(self):
            self.model = "test-model"

    class _BrokenDB:
        def get_session_title(self, key):
            raise RuntimeError("db down")

    monkeypatch.setattr(server, "_get_db", lambda: _BrokenDB())
    monkeypatch.setattr(server, "_make_agent", lambda sid, key, session_db=None, **_kw: _FakeAgent())
    monkeypatch.setattr(server, "_SlashWorker", lambda *a, **k: None)
    monkeypatch.setattr(server, "_session_info", lambda _a, *a2: {"model": "x"})
    monkeypatch.setattr(server, "_probe_credentials", lambda _a: None)
    monkeypatch.setattr(server, "_wire_callbacks", lambda _sid: None)
    monkeypatch.setattr(server, "_emit", lambda *a, **kw: None)

    import tools.approval as _approval

    monkeypatch.setattr(_approval, "register_gateway_notify", lambda key, cb: None)
    monkeypatch.setattr(_approval, "load_permanent_allowlist", lambda: None)

    resp = server.handle_request(
        {
            "id": "1",
            "method": "session.create",
            "params": {
                "source": "desktop",
                "parent_session_id": "parent-1",
                "messages": [{"role": "user", "content": "seed"}],
            },
        }
    )

    # Create itself still succeeds — the lazy first-prompt path remains as
    # the fallback for the seed.
    assert "result" in resp

    server._sessions.pop(resp["result"]["stored_session_id"], None)


def test_session_create_seed_failure_after_row_compensates(monkeypatch):
    """Partial-failure compensation (#93959 review): if the row commits but
    the transcript copy fails, the just-created child is DELETED so the lazy
    first-prompt fallback can retry cleanly. Without this, a durable empty
    row defeats _ensure_session_db_row's INSERT OR IGNORE and the renderer
    fail-latches on a transcript-less session again."""

    class _FakeAgent:
        def __init__(self):
            self.model = "test-model"

    seen: dict = {}

    class _FakeDB:
        def get_session_title(self, key):
            return "Parent"

        def get_next_title_in_lineage(self, current):
            return f"{current} #2"

        def create_session(self, key, **kwargs):
            seen["created"] = key

        def append_messages_batch(self, session_id, messages, **kwargs):
            raise RuntimeError("transcript write failed")

        def discard_unadmitted_session(self, session_id):
            seen["deleted"] = session_id
            return True

    monkeypatch.setattr(server, "_get_db", lambda: _FakeDB())
    monkeypatch.setattr(server, "_make_agent", lambda sid, key, session_db=None, **_kw: _FakeAgent())
    monkeypatch.setattr(server, "_SlashWorker", lambda *a, **k: None)
    monkeypatch.setattr(server, "_session_info", lambda _a, *a2: {"model": "x"})
    monkeypatch.setattr(server, "_probe_credentials", lambda _a: None)
    monkeypatch.setattr(server, "_wire_callbacks", lambda _sid: None)
    monkeypatch.setattr(server, "_emit", lambda *a, **kw: None)

    import tools.approval as _approval

    monkeypatch.setattr(_approval, "register_gateway_notify", lambda key, cb: None)
    monkeypatch.setattr(_approval, "load_permanent_allowlist", lambda: None)

    resp = server.handle_request(
        {
            "id": "1",
            "method": "session.create",
            "params": {
                "source": "desktop",
                "parent_session_id": "parent-1",
                "title": "My Branch",
                "messages": [{"role": "user", "content": "seed"}],
            },
        }
    )

    assert "result" in resp
    key = resp["result"]["stored_session_id"]
    # The half-written child was rolled back — no durable empty row left to
    # shadow the lazy seed path.
    assert seen.get("deleted") == key
    # pending_title survived: it still lands via the lazy post-turn apply.
    runtime_sid = resp["result"]["session_id"]
    assert server._sessions[runtime_sid]["pending_title"] == "My Branch"

    server._sessions.pop(runtime_sid, None)



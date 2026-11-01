





import os




import sys


import threading


import time


import types




from pathlib import Path


from unittest.mock import patch


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


def test_session_title_clears_pending_after_persist(monkeypatch):
    class _FakeDB:
        def __init__(self):
            self.title = "old"

        def get_session_title(self, _key):
            return self.title

        def get_session(self, _key):
            return {"id": _key, "title": self.title}

        def set_session_title(self, _key, title):
            self.title = title
            return True

    db = _FakeDB()
    emitted = []
    server._sessions["sid"] = _session(pending_title="stale")
    monkeypatch.setattr(server, "_get_db", lambda: db)
    monkeypatch.setattr(server, "_emit", lambda *args: emitted.append(args))
    try:
        resp = server.handle_request(
            {
                "id": "1",
                "method": "session.title",
                "params": {"session_id": "sid", "title": "fresh"},
            }
        )

        assert resp["result"]["pending"] is False
        assert resp["result"]["title"] == "fresh"
        assert server._sessions["sid"]["pending_title"] is None
        assert emitted[-1][0:2] == ("session.info", "sid")
        assert emitted[-1][2]["title"] == "fresh"
    finally:
        server._sessions.pop("sid", None)


def test_session_title_does_not_queue_noop_when_row_exists(monkeypatch):
    class _FakeDB:
        def __init__(self):
            self.title = "same title"

        def get_session_title(self, _key):
            return self.title

        def get_session(self, _key):
            return {"id": _key, "title": self.title}

        def set_session_title(self, _key, _title):
            # Simulate sqlite UPDATE rowcount==0 for no-op update.
            return False

    server._sessions["sid"] = _session(pending_title="stale")
    monkeypatch.setattr(server, "_get_db", lambda: _FakeDB())
    try:
        resp = server.handle_request(
            {
                "id": "1",
                "method": "session.title",
                "params": {"session_id": "sid", "title": "same title"},
            }
        )

        assert resp["result"]["pending"] is False
        assert resp["result"]["title"] == "same title"
        assert server._sessions["sid"]["pending_title"] is None
    finally:
        server._sessions.pop("sid", None)


def test_session_title_get_falls_back_to_pending_when_db_read_throws(monkeypatch):
    class _FakeDB:
        def get_session_title(self, _key):
            raise RuntimeError("db temporarily locked")

    server._sessions["sid"] = _session(pending_title="queued title")
    monkeypatch.setattr(server, "_get_db", lambda: _FakeDB())
    try:
        resp = server.handle_request(
            {"id": "1", "method": "session.title", "params": {"session_id": "sid"}}
        )
        assert resp["result"]["title"] == "queued title"
    finally:
        server._sessions.pop("sid", None)


def test_session_title_get_retries_persist_for_pending_title(monkeypatch):
    class _FakeDB:
        def __init__(self):
            self.title = ""

        def get_session_title(self, _key):
            return self.title

        def set_session_title(self, _key, title):
            self.title = title
            return True

        def get_session(self, _key):
            return {"id": _key, "title": self.title}

    db = _FakeDB()
    server._sessions["sid"] = _session(pending_title="queued title")
    monkeypatch.setattr(server, "_get_db", lambda: db)
    try:
        resp = server.handle_request(
            {"id": "1", "method": "session.title", "params": {"session_id": "sid"}}
        )
        assert resp["result"]["title"] == "queued title"
        assert server._sessions["sid"]["pending_title"] is None
    finally:
        server._sessions.pop("sid", None)


def test_session_title_get_retries_pending_even_when_db_has_title(monkeypatch):
    class _FakeDB:
        def __init__(self):
            self.title = "auto title"

        def get_session_title(self, _key):
            return self.title

        def set_session_title(self, _key, title):
            self.title = title
            return True

        def get_session(self, _key):
            return {"id": _key, "title": self.title}

    db = _FakeDB()
    server._sessions["sid"] = _session(pending_title="queued title")
    monkeypatch.setattr(server, "_get_db", lambda: db)
    try:
        resp = server.handle_request(
            {"id": "1", "method": "session.title", "params": {"session_id": "sid"}}
        )
        assert resp["result"]["title"] == "queued title"
        assert server._sessions["sid"]["pending_title"] is None
    finally:
        server._sessions.pop("sid", None)


def test_session_title_rejects_empty_title_with_specific_error_code(monkeypatch):
    class _FakeDB:
        def get_session_title(self, _key):
            return ""

    server._sessions["sid"] = _session()
    monkeypatch.setattr(server, "_get_db", lambda: _FakeDB())
    try:
        resp = server.handle_request(
            {
                "id": "1",
                "method": "session.title",
                "params": {"session_id": "sid", "title": "   "},
            }
        )
        assert "error" in resp
        assert resp["error"]["code"] == 4021
    finally:
        server._sessions.pop("sid", None)


def test_session_title_set_maps_valueerror_to_user_error(monkeypatch):
    class _FakeDB:
        def get_session_title(self, _key):
            return ""

        def get_session(self, _key):
            return {"id": _key}

        def set_session_title(self, _key, _title):
            raise ValueError("Title already in use")

    server._sessions["sid"] = _session()
    monkeypatch.setattr(server, "_get_db", lambda: _FakeDB())
    try:
        resp = server.handle_request(
            {
                "id": "1",
                "method": "session.title",
                "params": {"session_id": "sid", "title": "dup"},
            }
        )
        assert "error" in resp
        assert resp["error"]["code"] == 4022
        assert "already in use" in resp["error"]["message"]
    finally:
        server._sessions.pop("sid", None)


def test_session_title_set_errors_when_row_lookup_fails_after_noop(monkeypatch):
    class _FakeDB:
        def get_session_title(self, _key):
            return ""

        def get_session(self, _key):
            raise RuntimeError("row lookup failed")

        def set_session_title(self, _key, _title):
            return False

    server._sessions["sid"] = _session()
    monkeypatch.setattr(server, "_get_db", lambda: _FakeDB())
    try:
        resp = server.handle_request(
            {
                "id": "1",
                "method": "session.title",
                "params": {"session_id": "sid", "title": "fresh"},
            }
        )
        assert "error" in resp
        assert resp["error"]["code"] == 5007
        assert "row lookup failed" in resp["error"]["message"]
    finally:
        server._sessions.pop("sid", None)


def test_session_create_drops_pending_title_on_valueerror(monkeypatch):
    """When set_session_title raises ValueError during post-message title flush,
    pending_title should be dropped (non-retryable). Updated for post-#18370
    lazy session creation where title is applied post-first-message.
    """

    class _Agent:
        session_id = "test-session"
        model = "x"
        provider = "openrouter"
        base_url = ""
        api_key = ""
        _cached_system_prompt = ""

        def run_conversation(self, prompt, **kw):
            return {
                "final_response": "ok",
                "messages": [{"role": "assistant", "content": "ok"}],
            }

    class _FakeDB:
        def set_session_title(self, _key, _title):
            raise ValueError("Title already in use")

    class _ImmediateThread:
        def __init__(self, target=None, daemon=None, **kw):
            self._target = target

        def start(self):
            self._target()

    agent = _Agent()
    session = {
        "agent": agent,
        "session_key": "test-session",
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
        "pending_title": "duplicate title",
    }

    server._sessions["sid"] = session
    monkeypatch.setattr(server, "_get_db", lambda: _FakeDB())
    monkeypatch.setattr(server, "_emit", lambda *a, **kw: None)
    monkeypatch.setattr(server, "make_stream_renderer", lambda cols: None)
    monkeypatch.setattr(server, "render_message", lambda raw, cols: None)
    monkeypatch.setattr(
        server, "_sync_session_key_after_compress", lambda *a, **kw: None
    )
    monkeypatch.setattr(server.threading, "Thread", _ImmediateThread)

    try:
        server.handle_request(
            {"id": "1", "method": "prompt.submit", "params": {"session_id": "sid", "text": "hello"}}
        )
        assert session["pending_title"] is None
    finally:
        server._sessions.pop("sid", None)


def test_config_set_yolo_toggles_session_scope():
    from tools.approval import clear_session, is_session_yolo_enabled

    server._sessions["sid"] = _session()
    try:
        resp_on = server.handle_request(
            {
                "id": "1",
                "method": "config.set",
                "params": {"session_id": "sid", "key": "yolo"},
            }
        )
        assert resp_on["result"]["value"] == "1"
        assert is_session_yolo_enabled("session-key") is True

        resp_off = server.handle_request(
            {
                "id": "2",
                "method": "config.set",
                "params": {"session_id": "sid", "key": "yolo"},
            }
        )
        assert resp_off["result"]["value"] == "0"
        assert is_session_yolo_enabled("session-key") is False
    finally:
        clear_session("session-key")
        server._sessions.clear()


def test_config_set_yolo_stale_session_id_is_refused_not_process_scoped(monkeypatch):
    """A runtime id the backend no longer holds must answer 4001 so the client resumes, not flip
    the process HERMES_YOLO_MODE that every child spawned afterwards inherits."""
    monkeypatch.setenv("HERMES_YOLO_MODE", "0")  # setenv, not delenv: undo must also drop a leaked "1"

    with patch.dict(server._sessions, {}, clear=True):
        resp = server.handle_request(
            {
                "id": "1",
                "method": "config.set",
                "params": {"session_id": "reaped-sid", "key": "yolo", "value": "1"},
            }
        )

    assert resp.get("error", {}).get("code") == 4001, resp
    assert os.environ["HERMES_YOLO_MODE"] == "0"


def test_config_set_yolo_global_scope_writes_approvals_mode(tmp_path, monkeypatch):
    """Shift+click the desktop zap -> scope="global" flips persistent approvals.mode."""
    import hermes_yaml as yaml

    cfg_path = tmp_path / "config.yaml"
    cfg_path.write_text(yaml.safe_dump({"approvals": {"mode": "manual"}}), encoding="utf-8")
    monkeypatch.setattr(server, "_hermes_home", tmp_path)

    resp_on = server.handle_request(
        {
            "id": "1",
            "method": "config.set",
            "params": {"key": "yolo", "scope": "global"},
        }
    )
    assert resp_on["result"]["value"] == "1"
    assert resp_on["result"]["scope"] == "global"
    assert yaml.safe_load(cfg_path.read_text(encoding="utf-8-sig"))["approvals"]["mode"] == "off"

    resp_off = server.handle_request(
        {
            "id": "2",
            "method": "config.set",
            "params": {"key": "yolo", "scope": "global"},
        }
    )
    assert resp_off["result"]["value"] == "0"
    assert yaml.safe_load(cfg_path.read_text(encoding="utf-8-sig"))["approvals"]["mode"] == "manual"


def test_config_get_approval_mode_uses_smart_default_when_key_is_missing(
    tmp_path, monkeypatch
):
    import hermes_yaml as yaml

    monkeypatch.setattr(server, "_hermes_home", tmp_path)
    # Point the canonical resolver (load_config → env HERMES_HOME) at the
    # temp home too, so the smart default is asserted against THIS config
    # rather than whatever the developer's real ~/.hermes happens to hold.
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    (tmp_path / "config.yaml").write_text(
        yaml.safe_dump({"approvals": {"timeout": 15}})
    )

    response = server.handle_request(
        {"id": "1", "method": "config.get", "params": {"key": "approvals.mode"}}
    )
    assert response["result"]["value"] == "smart"


def test_config_get_approval_mode_fails_safe_to_manual_for_invalid_explicit_value(
    tmp_path, monkeypatch
):
    import hermes_yaml as yaml

    monkeypatch.setattr(server, "_hermes_home", tmp_path)
    # _load_approval_mode delegates to the canonical resolver in
    # tools.approval, which reads via hermes_cli.config.load_config —
    # that path resolves HERMES_HOME from the environment, not
    # server._hermes_home.
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    (tmp_path / "config.yaml").write_text(
        yaml.safe_dump({"approvals": {"mode": "sometimes"}})
    )

    response = server.handle_request(
        {"id": "1", "method": "config.get", "params": {"key": "approvals.mode"}}
    )
    assert response["result"]["value"] == "manual"


def test_config_get_approval_mode_normalizes_yaml_off(tmp_path, monkeypatch):
    import hermes_yaml as yaml

    monkeypatch.setattr(server, "_hermes_home", tmp_path)
    # See fail-safe test above: the canonical resolver reads via
    # load_config, which resolves HERMES_HOME from the environment.
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    (tmp_path / "config.yaml").write_text(
        yaml.safe_dump({"approvals": {"mode": False}})
    )

    response = server.handle_request(
        {"id": "1", "method": "config.get", "params": {"key": "approvals.mode"}}
    )
    assert response["result"]["value"] == "off"


def test_config_set_approval_mode_persists_three_way_value_and_emits_live_status(
    tmp_path, monkeypatch
):
    import hermes_yaml as yaml

    monkeypatch.setattr(server, "_hermes_home", tmp_path)
    # config.set writes via server._hermes_home, but the post-write
    # session.info emit resolves the effective mode through the canonical
    # tools.approval resolver (load_config → env HERMES_HOME).
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    emitted = []
    monkeypatch.setattr(server, "_emit", lambda *args: emitted.append(args))
    server._sessions["sid"] = {"agent": object(), "session_key": "profile-session"}

    try:
        resp = server.handle_request(
            {
                "id": "1",
                "method": "config.set",
                "params": {"key": "approvals.mode", "value": "manual"},
            }
        )
    finally:
        server._sessions.clear()

    assert resp["result"] == {"key": "approvals.mode", "value": "manual"}
    assert yaml.safe_load((tmp_path / "config.yaml").read_text(encoding="utf-8-sig"))["approvals"]["mode"] == "manual"
    assert emitted and emitted[0][0:2] == ("session.info", "sid")
    assert emitted[0][2]["approval_mode"] == "manual"


def test_pet_gallery_quoted_false_enabled_reports_disabled(tmp_path, monkeypatch):
    """display.pet.enabled: "false" (quoted) must report enabled=False.

    The old check was bool(value) — bool('false') is True, so a hand-edited
    quoted YAML value kept the petdex mascot enabled against the operator's
    explicit intent.
    """
    import hermes_yaml as yaml

    monkeypatch.setattr(server, "_hermes_home", tmp_path)
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    (tmp_path / "config.yaml").write_text(
        yaml.safe_dump({"display": {"pet": {"enabled": "false"}}})
    )

    response = server.handle_request(
        {"id": "1", "method": "pet.gallery", "params": {}}
    )
    assert response["result"]["enabled"] is False


def test_pet_info_known_revision_elides_spritesheet(monkeypatch):
    """pet.info with a matching knownRevision must not resend the sheet bytes.

    The spritesheet payload is multi-MB; resending it on every backstop
    refresh stalls the WS write loop (#54730). A caller passing the revision
    it already holds gets metadata plus spritesheetUnchanged instead.
    """

    class _FakePet:
        slug = "codex"
        display_name = "Codex"
        exists = True
        spritesheet = None

    payload = {
        "slug": "codex",
        "displayName": "Codex",
        "mime": "image/png",
        "spritesheetBase64": "A" * 1024,
        "spritesheetRevision": "123:456",
        "frameW": 192,
        "frameH": 208,
        "scale": 0.33,
    }

    monkeypatch.setattr(server, "_pet_active_selection", lambda: (True, _FakePet(), 0.33))
    monkeypatch.setattr(server, "_pet_sprite_payload", lambda pet, *, scale: dict(payload))

    # Matching revision: bytes elided, unchanged marker set.
    resp = server.handle_request(
        {"id": "1", "method": "pet.info", "params": {"knownRevision": "123:456"}}
    )
    assert resp["result"]["enabled"] is True
    assert "spritesheetBase64" not in resp["result"]
    assert resp["result"]["spritesheetUnchanged"] is True
    assert resp["result"]["spritesheetRevision"] == "123:456"

    # Stale revision: full payload still flows.
    resp = server.handle_request(
        {"id": "2", "method": "pet.info", "params": {"knownRevision": "999:999"}}
    )
    assert resp["result"]["spritesheetBase64"] == "A" * 1024
    assert "spritesheetUnchanged" not in resp["result"]

    # No revision (legacy callers): full payload.
    resp = server.handle_request({"id": "3", "method": "pet.info", "params": {}})
    assert resp["result"]["spritesheetBase64"] == "A" * 1024


def test_config_set_approval_mode_rejects_unknown_value():
    resp = server.handle_request(
        {
            "id": "1",
            "method": "config.set",
            "params": {"key": "approvals.mode", "value": "sometimes"},
        }
    )

    assert resp["error"]["code"] == 4002


def test_config_set_yolo_global_scope_honors_explicit_value(tmp_path, monkeypatch):
    """An explicit value pins global approvals.mode regardless of prior state."""
    import hermes_yaml as yaml

    cfg_path = tmp_path / "config.yaml"
    cfg_path.write_text(yaml.safe_dump({"approvals": {"mode": "manual"}}), encoding="utf-8")
    monkeypatch.setattr(server, "_hermes_home", tmp_path)

    resp = server.handle_request(
        {
            "id": "1",
            "method": "config.set",
            "params": {"key": "yolo", "scope": "global", "value": "1"},
        }
    )
    assert resp["result"]["value"] == "1"
    assert yaml.safe_load(cfg_path.read_text(encoding="utf-8-sig"))["approvals"]["mode"] == "off"

    # Setting it on again is idempotent — stays off.
    resp_again = server.handle_request(
        {
            "id": "2",
            "method": "config.set",
            "params": {"key": "yolo", "scope": "global", "value": "1"},
        }
    )
    assert resp_again["result"]["value"] == "1"
    assert yaml.safe_load(cfg_path.read_text(encoding="utf-8-sig"))["approvals"]["mode"] == "off"


def test_config_set_fast_updates_live_agent_session_scoped(monkeypatch):
    """A session-targeted fast toggle updates the live agent + pins the
    per-session override, and NEVER writes global config — the desktop's
    per-model presets call this on every model pick, and a global write
    flipped the tier for every other session/profile (the "switch one
    session, switches everywhere" class)."""
    writes = []
    emits = []
    agent = types.SimpleNamespace(
        model="openai/gpt-5.4",
        request_overrides={"foo": "bar", "speed": "slow"},
        service_tier=None,
    )
    session = _session(agent=agent)
    server._sessions["sid"] = session

    monkeypatch.setattr(
        server, "_write_config_key", lambda path, value: writes.append((path, value))
    )
    monkeypatch.setattr(server, "_session_info", lambda _agent, *a: {"model": "x"})
    monkeypatch.setattr(server, "_emit", lambda *args: emits.append(args))
    monkeypatch.setattr(
        "hermes_cli.models.resolve_fast_mode_overrides",
        lambda _model_id, **_route: {"service_tier": "priority"},
    )

    try:
        resp = server.handle_request(
            {
                "id": "1",
                "method": "config.set",
                "params": {"session_id": "sid", "key": "fast", "value": "fast"},
            }
        )
        assert resp["result"]["value"] == "fast"
        assert agent.service_tier == "priority"
        assert agent.request_overrides == {
            "foo": "bar",
            "service_tier": "priority",
        }
        assert session["create_service_tier_override"] == "priority"
        assert writes == []
        assert ("session.info", "sid", {"model": "x"}) in emits

        resp_normal = server.handle_request(
            {
                "id": "2",
                "method": "config.set",
                "params": {"session_id": "sid", "key": "fast", "value": "normal"},
            }
        )
        assert resp_normal["result"]["value"] == "normal"
        assert agent.service_tier is None
        assert agent.request_overrides == {"foo": "bar"}
        # "" (not absent) so a rebuild pins normal instead of falling back to
        # the global default.
        assert session["create_service_tier_override"] == ""
        assert writes == []
    finally:
        server._sessions.pop("sid", None)


def test_config_set_fast_status_is_non_mutating(monkeypatch):
    writes = []
    emits = []
    agent = types.SimpleNamespace(service_tier="priority")
    server._sessions["sid"] = _session(agent=agent)

    monkeypatch.setattr(
        server, "_write_config_key", lambda path, value: writes.append((path, value))
    )
    monkeypatch.setattr(server, "_emit", lambda *args: emits.append(args))

    try:
        resp = server.handle_request(
            {
                "id": "1",
                "method": "config.set",
                "params": {"session_id": "sid", "key": "fast", "value": "status"},
            }
        )
        assert resp["result"]["value"] == "fast"
        assert writes == []
        assert emits == []
    finally:
        server._sessions.pop("sid", None)


def test_config_set_fast_rejects_unsupported_model(monkeypatch):
    writes = []
    agent = types.SimpleNamespace(
        model="unsupported-model",
        request_overrides={},
        service_tier=None,
    )
    server._sessions["sid"] = _session(agent=agent)

    monkeypatch.setattr(
        server, "_write_config_key", lambda path, value: writes.append((path, value))
    )
    monkeypatch.setattr(
        "hermes_cli.models.resolve_fast_mode_overrides",
        lambda _model_id, **_route: None,
    )

    try:
        resp = server.handle_request(
            {
                "id": "1",
                "method": "config.set",
                "params": {"session_id": "sid", "key": "fast", "value": "fast"},
            }
        )
        assert resp["error"]["code"] == 4002
        assert agent.service_tier is None
        assert agent.request_overrides == {}
        assert writes == []
    finally:
        server._sessions.pop("sid", None)


def test_config_set_fast_rejects_missing_model(monkeypatch):
    writes = []
    agent = types.SimpleNamespace(
        model="",
        request_overrides={},
        service_tier=None,
    )
    server._sessions["sid"] = _session(agent=agent)

    monkeypatch.setattr(
        server, "_write_config_key", lambda path, value: writes.append((path, value))
    )

    try:
        resp = server.handle_request(
            {
                "id": "1",
                "method": "config.set",
                "params": {"session_id": "sid", "key": "fast", "value": "fast"},
            }
        )
        assert resp["error"]["code"] == 4002
        assert agent.service_tier is None
        assert agent.request_overrides == {}
        assert writes == []
    finally:
        server._sessions.pop("sid", None)


def test_config_busy_get_and_set(monkeypatch):
    writes = []

    monkeypatch.setattr(
        server,
        "_load_cfg",
        lambda: {"display": {"busy_input_mode": "steer"}},
    )
    monkeypatch.setattr(
        server, "_write_config_key", lambda path, value: writes.append((path, value))
    )

    get_resp = server.handle_request(
        {"id": "1", "method": "config.get", "params": {"key": "busy"}}
    )
    assert get_resp["result"]["value"] == "steer"

    set_resp = server.handle_request(
        {
            "id": "2",
            "method": "config.set",
            "params": {"key": "busy", "value": "interrupt"},
        }
    )
    assert set_resp["result"]["value"] == "interrupt"
    assert ("display.busy_input_mode", "interrupt") in writes


def test_config_set_yolo_process_scope_treats_false_like_env_as_disabled(monkeypatch):
    monkeypatch.setenv("HERMES_YOLO_MODE", "false")

    resp = server.handle_request(
        {
            "id": "1",
            "method": "config.set",
            "params": {"key": "yolo"},
        }
    )

    assert resp["result"]["value"] == "1"
    assert os.environ.get("HERMES_YOLO_MODE") == "1"


def test_config_get_statusbar_survives_non_dict_display(monkeypatch):
    monkeypatch.setattr(server, "_load_cfg", lambda: {"display": "broken"})

    resp = server.handle_request(
        {"id": "1", "method": "config.get", "params": {"key": "statusbar"}}
    )

    assert resp["result"]["value"] == "top"


def test_config_get_busy_survives_non_dict_display(monkeypatch):
    monkeypatch.setattr(server, "_load_cfg", lambda: {"display": "broken"})

    resp = server.handle_request(
        {"id": "1", "method": "config.get", "params": {"key": "busy"}}
    )

    assert resp["result"]["value"] == "interrupt"


def test_config_set_statusbar_survives_non_dict_display(tmp_path, monkeypatch):
    import hermes_yaml as yaml

    cfg_path = tmp_path / "config.yaml"
    cfg_path.write_text(yaml.safe_dump({"display": "broken"}), encoding="utf-8")
    monkeypatch.setattr(server, "_hermes_home", tmp_path)

    resp = server.handle_request(
        {
            "id": "1",
            "method": "config.set",
            "params": {"key": "statusbar", "value": "bottom"},
        }
    )

    assert resp["result"]["value"] == "bottom"
    saved = yaml.safe_load(cfg_path.read_text(encoding="utf-8-sig"))
    assert saved["display"]["tui_statusbar"] == "bottom"


def test_config_set_details_mode_pins_all_sections(tmp_path, monkeypatch):
    import hermes_yaml as yaml

    cfg_path = tmp_path / "config.yaml"
    cfg_path.write_text(
        yaml.safe_dump(
            {"display": {"sections": {"tools": "expanded", "activity": "hidden"}}}
        )
    )
    monkeypatch.setattr(server, "_hermes_home", tmp_path)

    resp = server.handle_request(
        {
            "id": "1",
            "method": "config.set",
            "params": {"key": "details_mode", "value": "collapsed"},
        }
    )

    assert resp["result"] == {"key": "details_mode", "value": "collapsed"}
    saved = yaml.safe_load(cfg_path.read_text(encoding="utf-8-sig"))
    assert saved["display"]["details_mode"] == "collapsed"
    assert saved["display"]["sections"] == {
        "thinking": "collapsed",
        "tools": "collapsed",
        "subagents": "collapsed",
        "activity": "collapsed",
    }


def test_config_set_section_writes_per_section_override(tmp_path, monkeypatch):
    import hermes_yaml as yaml

    cfg_path = tmp_path / "config.yaml"
    monkeypatch.setattr(server, "_hermes_home", tmp_path)

    resp = server.handle_request(
        {
            "id": "1",
            "method": "config.set",
            "params": {"key": "details_mode.activity", "value": "hidden"},
        }
    )

    assert resp["result"] == {"key": "details_mode.activity", "value": "hidden"}
    saved = yaml.safe_load(cfg_path.read_text(encoding="utf-8-sig"))
    assert saved["display"]["sections"] == {"activity": "hidden"}


def test_config_set_section_clears_override_on_empty_value(tmp_path, monkeypatch):
    import hermes_yaml as yaml

    cfg_path = tmp_path / "config.yaml"
    cfg_path.write_text(
        yaml.safe_dump(
            {"display": {"sections": {"activity": "hidden", "tools": "expanded"}}}
        )
    )
    monkeypatch.setattr(server, "_hermes_home", tmp_path)

    resp = server.handle_request(
        {
            "id": "1",
            "method": "config.set",
            "params": {"key": "details_mode.activity", "value": ""},
        }
    )

    assert resp["result"] == {"key": "details_mode.activity", "value": ""}
    saved = yaml.safe_load(cfg_path.read_text(encoding="utf-8-sig"))
    assert saved["display"]["sections"] == {"tools": "expanded"}


def test_config_set_section_rejects_unknown_section_or_mode(tmp_path, monkeypatch):
    monkeypatch.setattr(server, "_hermes_home", tmp_path)

    bad_section = server.handle_request(
        {
            "id": "1",
            "method": "config.set",
            "params": {"key": "details_mode.bogus", "value": "hidden"},
        }
    )
    assert bad_section["error"]["code"] == 4002

    bad_mode = server.handle_request(
        {
            "id": "2",
            "method": "config.set",
            "params": {"key": "details_mode.tools", "value": "maximised"},
        }
    )
    assert bad_mode["error"]["code"] == 4002


def test_config_mouse_uses_documented_key_with_legacy_fallback(monkeypatch):
    cfg = {"display": {"tui_mouse": False}}
    writes = []

    monkeypatch.setattr(server, "_load_cfg", lambda: cfg)
    monkeypatch.setattr(
        server, "_write_config_key", lambda path, value: writes.append((path, value))
    )

    get_legacy = server.handle_request(
        {"id": "1", "method": "config.get", "params": {"key": "mouse"}}
    )
    assert get_legacy["result"]["value"] == "off"

    set_toggle = server.handle_request(
        {"id": "2", "method": "config.set", "params": {"key": "mouse"}}
    )
    # /mouse (no arg) toggles between 'all' and 'off'. Starting from
    # tui_mouse: False (→ 'off'), the toggle flips to 'all'.
    assert set_toggle["result"] == {"key": "mouse", "value": "all"}
    assert writes == [("display.mouse_tracking", "all")]

    cfg["display"] = {"mouse_tracking": 0, "tui_mouse": True}
    get_canonical = server.handle_request(
        {"id": "3", "method": "config.get", "params": {"key": "mouse"}}
    )
    assert get_canonical["result"]["value"] == "off"

    cfg["display"] = {"mouse_tracking": None, "tui_mouse": False}
    get_null = server.handle_request(
        {"id": "4", "method": "config.get", "params": {"key": "mouse"}}
    )
    # mouse_tracking present-but-None defers neither to tui_mouse nor to
    # the legacy off bucket: it falls through to the 'all' default.
    assert get_null["result"]["value"] == "all"


def test_config_mouse_accepts_preset_strings_and_aliases(monkeypatch):
    cfg = {"display": {"mouse_tracking": "all"}}
    writes = []

    monkeypatch.setattr(server, "_load_cfg", lambda: cfg)
    monkeypatch.setattr(
        server, "_write_config_key", lambda path, value: writes.append((path, value))
    )

    # Direct preset.
    set_wheel = server.handle_request(
        {
            "id": "1",
            "method": "config.set",
            "params": {"key": "mouse", "value": "wheel"},
        }
    )
    assert set_wheel["result"] == {"key": "mouse", "value": "wheel"}
    assert writes[-1] == ("display.mouse_tracking", "wheel")

    # Alias for buttons.
    set_click = server.handle_request(
        {
            "id": "2",
            "method": "config.set",
            "params": {"key": "mouse", "value": "click"},
        }
    )
    assert set_click["result"] == {"key": "mouse", "value": "buttons"}
    assert writes[-1] == ("display.mouse_tracking", "buttons")

    # Unknown value → 4002.
    bad = server.handle_request(
        {
            "id": "3",
            "method": "config.set",
            "params": {"key": "mouse", "value": "rainbows"},
        }
    )
    assert bad["error"]["code"] == 4002


def test_enable_gateway_prompts_sets_gateway_env(monkeypatch):
    monkeypatch.delenv("HERMES_EXEC_ASK", raising=False)
    monkeypatch.delenv("HERMES_GATEWAY_SESSION", raising=False)
    monkeypatch.delenv("HERMES_INTERACTIVE", raising=False)

    server._enable_gateway_prompts()

    assert server.os.environ["HERMES_GATEWAY_SESSION"] == "1"
    assert server.os.environ["HERMES_EXEC_ASK"] == "1"
    assert server.os.environ["HERMES_INTERACTIVE"] == "1"


def test_setup_status_answers_from_the_bootstrap_record_once_it_exists(monkeypatch):
    """Under ``hermes serve`` the boot bootstrap records who carries inference; ``setup.status``
    reports its record (blocking for it while it is in flight) instead of re-probing, so a client's
    first poll sees the boot's answer rather than racing it."""
    import threading
    from hermes_cli import free_tier_bootstrap as fb
    fb.reset_for_tests()
    monkeypatch.setattr("hermes_cli.main._has_any_provider_configured",
                        lambda **_kw: pytest.fail("setup.status must read the record, not re-probe"))
    release = threading.Event()

    def slow_bootstrap():
        release.wait(5)
        with fb._lock:
            fb._record = fb.SetupRecord(provider_configured=True, inference_provider="nous", free_tier_account=True,
                                        has_identity=True, other_providers=False)
            fb._done.set()
    with fb._lock:
        fb._started = True
    threading.Thread(target=slow_bootstrap, daemon=True).start()
    try:
        release.set()
        resp = server.handle_request({"id": "1", "method": "setup.status", "params": {}})
        assert resp["result"]["provider_configured"] is True
        assert resp["result"]["ready"] is True and resp["result"]["free_tier_account"] is True
        assert resp["result"]["inference_provider"] == "nous"
    finally:
        fb.reset_for_tests()


def test_invalid_params_and_unknown_method_name_the_version_skew_fix():
    """The only signal of a TUI/backend version mismatch; the lead phrases stay for clients."""
    resp = server.handle_request({"id": "1", "method": "no.such.method", "params": {}})
    assert resp["error"]["code"] == -32601
    assert resp["error"]["message"].startswith("unknown method: no.such.method")

    resp = server.handle_request(
        {"id": "2", "method": "session.status", "params": {"session_id": "x", "turn_author": "y"}})
    assert resp["error"]["code"] == 4000
    assert resp["error"]["message"].startswith("invalid params for session.status: turn_author")


def test_probe_credentials_emits_exact_empty_key_warning():
    agent = types.SimpleNamespace(api_key="", provider="openrouter")

    warning = server._probe_credentials(agent)
    assert warning and "openrouter" in warning


def test_probe_credentials_allows_keyless_custom_runtime():
    agent = types.SimpleNamespace(api_key="no-key-required", provider="custom")

    assert server._probe_credentials(agent) == ""


def test_setup_runtime_check_rejects_empty_runtime_key(monkeypatch):
    monkeypatch.setattr("hermes_cli.main._has_any_provider_configured", lambda **_kw: True)
    monkeypatch.setattr(server, "_resolve_startup_runtime", lambda: ("openrouter/test-model", None))
    # No pin anywhere (startup pin None, config model pin absent): the failure keeps naming the
    # resolved route's provider.
    monkeypatch.setattr(server, "_load_cfg", lambda: {"model": {}})
    monkeypatch.setattr(
        "hermes_cli.runtime_provider.resolve_runtime_provider",
        lambda requested=None, **_kw: {
            "provider": "openrouter",
            "api_key": "",
            "source": "env/config",
        },
    )

    resp = server.handle_request({"id": "1", "method": "setup.runtime_check", "params": {}})

    result = resp["result"]
    assert result["ok"] is False
    assert (result["provider"], result["model"], result["source"]) == (
        "openrouter", "openrouter/test-model", "env/config"
    )
    assert result["error"]


def test_setup_runtime_check_failure_names_the_configured_pin_not_the_chain_tail(monkeypatch):
    """#124939: without an explicit ``provider`` the probe runs the startup pin and then the
    fallback chain. When the chain only resolves at its tail (BWS secrets not yet hydrated at
    boot), the runtime stops on the tail provider and the failure used to blame it — pointing
    the user at a provider they never pinned. The failure must name the pin."""
    monkeypatch.setattr("hermes_cli.main._has_any_provider_configured", lambda **_kw: True)
    monkeypatch.setattr(server, "_resolve_startup_runtime", lambda: ("glm-5.3", None))
    monkeypatch.setattr(server, "_load_cfg", lambda: {"model": {"provider": "zai"}})
    monkeypatch.setattr(
        "hermes_cli.runtime_provider.resolve_runtime_provider",
        lambda requested=None, **_kw: {
            "provider": "openrouter",  # the chain tail that "resolved"
            "api_key": "",
            "source": "pool",
        },
    )

    resp = server.handle_request({"id": "1", "method": "setup.runtime_check", "params": {}})

    result = resp["result"]
    assert result["ok"] is False
    assert result["provider"] == "zai"
    assert result["error"] == "No usable credentials found for zai."
    assert "openrouter" not in result["error"]


def test_setup_runtime_check_failure_names_startup_env_pin(monkeypatch):
    """The startup env pin (HERMES_TUI_PROVIDER) outranks the config model pin."""
    monkeypatch.setattr("hermes_cli.main._has_any_provider_configured", lambda **_kw: True)
    monkeypatch.setattr(server, "_resolve_startup_runtime", lambda: ("glm-5.3", "zai"))
    monkeypatch.setattr(server, "_load_cfg", lambda: {"model": {"provider": "openai-api"}})
    monkeypatch.setattr(
        "hermes_cli.runtime_provider.resolve_runtime_provider",
        lambda requested=None, **_kw: {
            "provider": "openrouter",
            "api_key": "",
            "source": "pool",
        },
    )

    resp = server.handle_request({"id": "1", "method": "setup.runtime_check", "params": {}})

    result = resp["result"]
    assert result["ok"] is False
    assert result["error"] == "No usable credentials found for zai."


def test_setup_runtime_check_allows_no_key_custom_runtime(monkeypatch):
    monkeypatch.setattr("hermes_cli.main._has_any_provider_configured", lambda **_kw: True)
    monkeypatch.setattr(
        "hermes_cli.runtime_provider.resolve_runtime_provider",
        lambda requested=None, **_kw: {
            "provider": "custom",
            "api_key": "no-key-required",
            "source": "env/config",
        },
    )

    resp = server.handle_request({"id": "1", "method": "setup.runtime_check", "params": {}})

    assert resp["result"]["ok"] is True
    assert resp["result"]["provider"] == "custom"


def test_setup_runtime_check_rejects_implicit_bedrock_when_unconfigured(monkeypatch):
    monkeypatch.setattr("hermes_cli.main._has_any_provider_configured", lambda **_kw: False)
    monkeypatch.setattr(server, "_load_cfg", dict)
    monkeypatch.setattr(
        "hermes_cli.runtime_provider.resolve_runtime_provider",
        lambda requested=None, **_kw: {
            "provider": "bedrock",
            "api_key": "aws-sdk",
            "source": "iam-role",
        },
    )

    resp = server.handle_request({"id": "1", "method": "setup.runtime_check", "params": {}})

    assert resp["result"]["ok"] is False
    assert resp["result"]["provider"] == "bedrock"


def test_setup_runtime_check_honors_requested_provider(monkeypatch):
    """Onboarding must be able to validate the provider the user just connected."""
    monkeypatch.setattr("hermes_cli.main._has_any_provider_configured", lambda **_kw: True)
    monkeypatch.setattr(server, "_load_cfg", dict)

    def fake_resolve(requested=None, **kwargs):
        if requested == "nous":
            return {
                "provider": "nous",
                "api_key": "invoke-jwt",
                "source": "portal",
            }
        return {
            "provider": "anthropic",
            "api_key": "",
            "source": "config",
        }

    monkeypatch.setattr(
        "hermes_cli.runtime_provider.resolve_runtime_provider",
        fake_resolve,
    )

    scoped = server.handle_request(
        {"id": "1", "method": "setup.runtime_check", "params": {"provider": "nous"}}
    )
    assert scoped["result"]["ok"] is True
    assert scoped["result"]["provider"] == "nous"

    default = server.handle_request({"id": "1", "method": "setup.runtime_check", "params": {}})
    assert default["result"]["ok"] is False
    assert default["result"]["provider"] == "anthropic"



import contextlib










import sys


import threading


import time


import types




from pathlib import Path


from unittest.mock import Mock, patch


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


def test_session_create_seed_disk_full_keeps_row_for_retry(monkeypatch):
    """Disk-full is NOT compensated: the row stays (deleting data on a full
    disk can make things worse), create still succeeds, and the failure is
    observable at warning level (#93959 review)."""

    import logging as _logging

    class _FakeAgent:
        def __init__(self):
            self.model = "test-model"

    class _FakeDB:
        def get_session_title(self, key):
            return "Parent"

        def get_next_title_in_lineage(self, current):
            return f"{current} #2"

        def create_session(self, key, **kwargs):
            pass

        def append_messages_batch(self, session_id, messages, **kwargs):
            raise OSError(28, "No space left on device")

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

    records: list = []

    class _Capture(_logging.Handler):
        def emit(self, record):
            records.append(record)

    handler = _Capture(level=_logging.WARNING)
    root = _logging.getLogger()
    root.addHandler(handler)
    try:
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
    finally:
        root.removeHandler(handler)

    assert "result" in resp
    # The failure surfaced at WARNING (observable), not buried at debug.
    warnings = [r for r in records if r.levelno >= _logging.WARNING]
    assert warnings

    server._sessions.pop(resp["result"]["stored_session_id"], None)


def test_session_create_without_parent_still_defers_row(monkeypatch):
    """Plain drafts keep the lazy-row contract: no parent + no explicit branch
    intent means no eager persistence (the original draft-hygiene invariant)."""

    class _FakeAgent:
        def __init__(self):
            self.model = "test-model"

    calls: dict = {"create": 0}

    class _FakeDB:
        def create_session(self, *a, **k):
            calls["create"] += 1

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
        {"id": "1", "method": "session.create", "params": {"cols": 80}}
    )
    sid = resp["result"]["session_id"]
    server._sessions[sid]["agent_ready"].wait(timeout=2.0)

    assert calls["create"] == 0, "plain drafts must not persist eagerly"

    server._sessions.pop(sid, None)


def test_session_branch_installs_parent_profile_secret_scope(monkeypatch, tmp_path):
    """The branched agent must be built under the parent profile's secrets.

    session.branch already binds the parent's HERMES_HOME and state.db, but the
    secret scope is what makes get_secret() resolve that profile's .env. Without
    it the build falls through to process os.environ — the LAUNCH profile's
    credentials — which is exactly the cross-profile resolution #67605 fixed for
    session.create / session.resume.
    """
    import threading

    from agent.secret_scope import current_secret_scope

    profile_home = tmp_path / "profiles" / "mlperf"
    profile_home.mkdir(parents=True)
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    (profile_home / ".env").write_text(
        "PROXMOX_TOKEN=mlperf-secret\n", encoding="utf-8"
    )
    seen: dict = {"msgs": []}

    class ProfileDB:
        def __init__(self, db_path=None):
            pass

        def get_session_title(self, _key):
            return "parent"

        def get_next_title_in_lineage(self, current):
            return f"{current} (branch)"

        def create_session(self, new_key, **kwargs):
            seen["created"] = new_key

        def append_message(self, **kwargs):
            seen["msgs"].append(kwargs)

        def append_messages_batch(self, session_id, messages, **kwargs):
            for m in messages:
                seen["msgs"].append(dict(m, session_id=session_id))
            return list(range(1, len(messages) + 1))

        def set_session_title(self, key, title):
            return True

        def get_session(self, key):
            return {"id": key, "cwd": str(tmp_path)}

        def update_session_cwd(self, *a, **k):
            return None

        def close(self):
            return None

    class FakeAgent:
        def __init__(self):
            self.model = "test-model"
            self.session_id = None

    parent = {
        "session_key": "parent-key",
        "history": [{"role": "user", "content": "hi"}],
        "history_lock": threading.Lock(),
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
    monkeypatch.setattr(server, "_get_db", lambda: ProfileDB())
    monkeypatch.setattr("hermes_state_registry.acquire", ProfileDB)
    monkeypatch.setattr(server, "_claim_active_session_slot", lambda *a, **k: (None, None))

    def _fake_make_agent(*a, **k):
        scope = current_secret_scope()
        seen["scope"] = dict(scope) if scope else None
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
                "method": "session.branch",
                "params": {"session_id": "parent", "name": "forked"},
            }
        )
        assert "result" in resp, resp
        assert seen.get("scope") == {"PROXMOX_TOKEN": "mlperf-secret"}
    finally:
        for k in list(server._sessions):
            server._sessions.pop(k, None)


def test_session_branch_uses_persisted_display_history_after_compaction(monkeypatch, tmp_path):
    """A live branch must copy the complete visible transcript, not the compacted model tail."""
    profile_home = tmp_path / "profiles" / "mlperf"
    profile_home.mkdir(parents=True)
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    seen: dict = {"msgs": []}

    display_history = [
        {"role": "user", "content": "first question", "timestamp": 1.0},
        {"role": "assistant", "content": "first answer", "timestamp": 2.0},
        {"role": "assistant", "content": "", "tool_calls": [{"id": "call-1"}]},
        {"role": "tool", "content": "tool output", "tool_call_id": "call-1"},
        {"role": "user", "content": "second question", "timestamp": 3.0},
        {"role": "assistant", "content": "second answer", "timestamp": 4.0},
    ]

    class LaunchDB:
        def get_session_title(self, _key):
            return "launch"

    class ProfileDB:
        def __init__(self, db_path=None):
            seen.setdefault("inits", 0)
            seen["inits"] += 1

        def get_session_title(self, _key):
            return "parent"

        def get_next_title_in_lineage(self, current):
            return f"{current} (branch)"

        def get_resume_conversations(self, key):
            assert key == "parent-key"
            # The model projection has already been compacted to a summary + tail;
            # the display projection still contains every visible turn.
            return (
                [{"role": "assistant", "content": "compact summary"}],
                display_history,
            )

        def create_session(self, _new_key, **_kwargs):
            return None

        def append_message(self, **kwargs):
            seen["msgs"].append(kwargs)

        def append_messages_batch(self, session_id, messages, **kwargs):
            for message in messages:
                seen["msgs"].append(dict(message, session_id=session_id))
            return list(range(1, len(messages) + 1))

        def set_session_title(self, _key, _title):
            return True

        def set_auto_title(self, _key, _title, *, source="llm"):
            seen["title_source"] = source
            return True

        def get_session(self, key):
            return {"id": key, "cwd": str(tmp_path)}

        def update_session_cwd(self, *args, **kwargs):
            return None

        def close(self):
            return None

    class FakeAgent:
        model = "test-model"
        session_id = None

    parent = {
        "session_key": "parent-key",
        # This is the model-fed projection after compaction: the old turns are
        # absent here even though the display projection above retains them.
        "history": [
            {"role": "assistant", "content": "compact summary"},
            {"role": "user", "content": "second question"},
            {"role": "assistant", "content": "second answer"},
        ],
        "history_lock": threading.Lock(),
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
    monkeypatch.setattr(server, "_claim_active_session_slot", lambda *args, **kwargs: (None, None))
    monkeypatch.setattr(server, "_make_agent", lambda *args, **kwargs: FakeAgent())
    monkeypatch.setattr(server, "_set_session_context", lambda *args, **kwargs: {})
    monkeypatch.setattr(server, "_clear_session_context", lambda *args, **kwargs: None)
    monkeypatch.setattr(server, "_resolve_model", lambda: "test-model")
    monkeypatch.setattr(server, "_session_cwd", lambda _session: str(tmp_path))
    monkeypatch.setattr(server, "_register_session_cwd", lambda *args, **kwargs: None)
    monkeypatch.setattr(server, "_attach_worker", lambda *args, **kwargs: None)

    try:
        response = server.handle_request(
            {
                "id": "1",
                "method": "session.branch",
                "params": {"session_id": "parent", "count": 4},
            }
        )

        assert "result" in response, response
        assert seen.get("title_source") == "derived"
        assert [message["content"] for message in seen["msgs"]] == [
            "first question",
            "first answer",
            "second question",
            "second answer",
        ]
        assert [message["text"] for message in response["result"]["messages"]] == [
            "first question",
            "first answer",
            "second question",
            "second answer",
        ]
    finally:
        for key in list(server._sessions):
            server._sessions.pop(key, None)


def test_model_options_does_not_overwrite_curated_models(monkeypatch):
    """The TUI model.options handler must surface the same curated model
    list as `hermes model` and the classic CLI /model picker.

    Regression: earlier versions of this handler unconditionally replaced
    each provider's curated ``models`` field with ``provider_model_ids()``
    (live /models catalog).  That pulled in hundreds of non-agentic models
    for providers like Nous whose /models endpoint returns image/video
    generators, rerankers, embeddings, and TTS models alongside chat models.
    """
    curated_providers = [
        {
            "slug": "nous",
            "name": "Nous",
            "models": ["moonshotai/kimi-k2.5", "anthropic/claude-opus-4.7"],
            "total_models": 30,
            "source": "built-in",
            "is_current": False,
            "is_user_defined": False,
        },
    ]

    monkeypatch.setattr(
        server,
        "_load_cfg",
        lambda: {"providers": {}, "custom_providers": []},
    )

    with patch(
        "hermes_cli.model_switch.list_authenticated_providers",
        return_value=curated_providers,
    ):
        # If provider_model_ids gets called at all, the handler is still
        # overwriting curated with live — that's the regression we're
        # guarding against.
        with patch("hermes_cli.models.provider_model_ids") as live_fetch:
            resp = server._methods["model.options"](99, {"session_id": ""})

    assert "result" in resp, resp
    providers = resp["result"]["providers"]
    nous = next((p for p in providers if p.get("slug") == "nous"), None)
    assert nous is not None
    assert nous["models"] == [
        "moonshotai/kimi-k2.5",
        "anthropic/claude-opus-4.7",
    ]
    assert nous["total_models"] == 30
    # Handler must not consult the live catalog — curated is the truth.
    live_fetch.assert_not_called()


def test_model_options_propagates_list_exception(monkeypatch):
    """If list_authenticated_providers itself raises, surface as an RPC
    error rather than swallowing to a blank picker."""
    monkeypatch.setattr(
        server,
        "_load_cfg",
        lambda: {"providers": {}, "custom_providers": []},
    )
    with patch(
        "hermes_cli.model_switch.list_authenticated_providers",
        side_effect=RuntimeError("catalog blew up"),
    ):
        resp = server._methods["model.options"](77, {"session_id": ""})
    assert "error" in resp
    assert resp["error"]["code"] == 5033
    assert "catalog blew up" in resp["error"]["message"]


def test_model_options_preserves_canonical_custom_row_after_agent_init(monkeypatch):
    from hermes_cli.inventory import ConfigContext

    class _Agent:
        provider = "custom"
        model = "qwen3.6:35b-65k"
        base_url = "http://127.0.0.1:11434/v1"

    server._sessions["custom-session"] = _session(agent=_Agent())
    monkeypatch.setattr(server, "_resolve_model", lambda: "")
    monkeypatch.setattr(
        "hermes_cli.inventory.load_picker_context",
        lambda: ConfigContext(
            current_provider="custom:local-ollama",
            current_model="qwen3.6:35b-65k",
            current_base_url="http://127.0.0.1:11434/v1",
            user_providers={},
            custom_providers=[],
        ),
    )
    canonical = Mock(return_value="custom:local-ollama")
    monkeypatch.setattr(
        "hermes_cli.runtime_provider.canonical_custom_identity",
        canonical,
    )
    monkeypatch.setattr(
        "hermes_cli.model_switch.list_authenticated_providers",
        lambda **_kwargs: [
            {
                "slug": "custom:local-ollama",
                "name": "Local Ollama",
                "is_current": True,
                "is_user_defined": True,
                "models": ["qwen3.6:35b-65k"],
                "total_models": 1,
            },
            {
                "slug": "anthropic",
                "name": "Anthropic",
                "is_current": False,
                "is_user_defined": False,
                "models": ["claude-sonnet-4.6"],
                "total_models": 1,
            },
        ],
    )
    monkeypatch.setattr(
        "hermes_cli.auth.is_provider_explicitly_configured",
        lambda _slug: False,
    )
    # A host signed in to Claude Code / Anthropic OAuth would otherwise keep the anthropic row.
    monkeypatch.setattr("hermes_cli.inventory._anthropic_oauth_credentials_present", lambda: False)
    monkeypatch.setattr("hermes_cli.inventory._apply_pricing", lambda *_args, **_kwargs: None)
    monkeypatch.setattr("hermes_cli.inventory._apply_capabilities", lambda *_args, **_kwargs: None)

    resp = server._methods["model.options"](
        102,
        {"session_id": "custom-session", "explicit_only": True},
    )

    assert "result" in resp, resp
    assert resp["result"]["provider"] == "custom:local-ollama"
    assert [row["slug"] for row in resp["result"]["providers"]] == [
        "custom:local-ollama"
    ]


def test_model_save_key_uses_credential_lifecycle_and_picker_context(monkeypatch):
    env_var = "TEST_PROVIDER_API_KEY"
    agent = object()
    picker_ctx = object()
    provider = {
        "slug": "test-provider",
        "name": "Test Provider",
        "models": ["test-model"],
        "total_models": 1,
    }
    server._sessions["save-key-session"] = _session(agent=agent)
    monkeypatch.setattr(
        "hermes_cli.auth.PROVIDER_REGISTRY",
        {
            "test-provider": types.SimpleNamespace(
                name="Test Provider",
                auth_type="api_key",
                api_key_env_vars=(env_var,),
            )
        },
    )
    monkeypatch.setattr("hermes_cli.config.is_managed", lambda: False)
    save_credential = Mock()
    monkeypatch.setattr(
        "hermes_cli.credential_lifecycle.save_provider_env_credential",
        save_credential,
    )
    picker_context = Mock(return_value=picker_ctx)
    monkeypatch.setattr(server, "_model_picker_context", picker_context)
    build_payload = Mock(return_value={"providers": [provider]})
    monkeypatch.setattr(
        "hermes_cli.inventory.build_models_payload",
        build_payload,
    )
    monkeypatch.setenv(env_var, "previous-value")
    fake_key = "replacement-" + "value"

    resp = server._methods["model.save_key"](
        103,
        {
            "slug": "test-provider",
            "api_key": fake_key,
            "session_id": "save-key-session",
        },
    )

    assert "result" in resp, resp
    assert resp["result"]["provider"] == {**provider, "authenticated": True}
    save_credential.assert_called_once_with(env_var, fake_key)


def test_model_save_key_reconciles_the_launch_profiles_stale_setup_record(monkeypatch, tmp_path):
    """The gated picker's own chat waits on ``setup.status``, which answers from the boot record:
    a key saved for the launch profile must flip a ``False`` record (+ ``setup.ready``) at once;
    a key saved for another profile (``profile`` param) must leave the launch record alone."""
    from hermes_cli import free_tier_bootstrap as fb

    monkeypatch.setattr("hermes_cli.auth.PROVIDER_REGISTRY", {"test-provider": types.SimpleNamespace(
        name="Test Provider", auth_type="api_key", api_key_env_vars=("TEST_PROVIDER_API_KEY",))})
    monkeypatch.setattr("hermes_cli.config.is_managed", lambda: False)
    monkeypatch.setattr("hermes_cli.credential_lifecycle.save_provider_env_credential", Mock())
    monkeypatch.setattr("hermes_cli.inventory.build_models_payload", Mock(return_value={"providers": []}))
    monkeypatch.setenv("TEST_PROVIDER_API_KEY", "previous-value")  # save_key exports the new key
    monkeypatch.setattr(fb, "_inventory_other_providers", lambda: True)
    monkeypatch.setattr(fb, "_resolve_inference", lambda: "test-provider")
    broadcasts = []
    monkeypatch.setattr(fb, "_broadcast", broadcasts.append)
    other_home = tmp_path / "profiles" / "other"  # model.save_key is profile scoped: "other" must exist
    other_home.mkdir(parents=True)
    monkeypatch.setattr(server, "_profile_home", lambda name: other_home if name == "other" else None)
    fb.reset_for_tests()
    stale = fb.SetupRecord(provider_configured=False, inference_provider="", free_tier_account=False,
                           has_identity=False, other_providers=False)
    with fb._lock:
        fb._record, fb._started = stale, True
        fb._done.set()
    try:
        params = {"slug": "test-provider", "api_key": "k-" + "1"}
        assert "result" in server._methods["model.save_key"](104, {**params, "profile": "other"})
        assert fb.current_record() is stale and broadcasts == [], "another profile's key is not ours"
        assert "result" in server._methods["model.save_key"](105, params)
        record = fb.current_record()
        assert record.provider_configured is True and record.inference_provider == "test-provider"
        assert broadcasts == [record]
    finally:
        fb.reset_for_tests()


def test_prompt_submit_releases_old_history_before_heap_trim(monkeypatch, tmp_path):
    """The post-turn heap trim must run after the turn's pre-turn history snapshots are dropped:
    malloc_trim cannot return pages still referenced, so a retained snapshot of a large tool
    result pins them for the life of the process. Observed through a weak reference to the old
    message, not by reading the finisher's local variable names."""
    import gc
    import weakref

    class _Msg(dict):
        """dict itself is not weakref-able."""

    observed = {}
    order = []

    class _Agent:
        def run_conversation(self, prompt, conversation_history=None, stream_callback=None, **_kw):
            return {"final_response": "reply", "messages": [{"role": "assistant", "content": "reply"}]}

    def _trim(**_kwargs):
        order.append("trim")
        gc.collect()
        observed["alive_at_trim"] = old_ref() is not None

    old = _Msg(role="tool", tool_call_id="old", content="x" * 20_000)
    old_ref = weakref.ref(old)
    session = _session(agent=_Agent())
    profile_home = tmp_path / "profiles" / "worker"
    profile_home.mkdir(parents=True)
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    session["profile_home"] = str(profile_home)
    session["history"] = [old]
    del old
    server._sessions["sid_trim"] = session
    try:
        monkeypatch.setattr(server.threading, "Thread", _ImmediateThread)
        monkeypatch.setattr(server, "_get_usage", lambda _a: {})
        monkeypatch.setattr(server, "render_message", lambda _t, _c: "")
        monkeypatch.setattr(server, "_emit", lambda *a: None)
        monkeypatch.setattr(server, "set_hermes_home_override", lambda _home: object())
        monkeypatch.setattr(server, "reset_hermes_home_override", lambda _token: order.append("reset_home"))
        monkeypatch.setattr(server, "_session_profile_runtime_scope", lambda _session, **_kw: contextlib.nullcontext())
        monkeypatch.setattr("hermes_cli.mem_trim.trim_memory", _trim)

        resp = server.handle_request(
            {"id": "1", "method": "prompt.submit", "params": {"session_id": "sid_trim", "text": "hi"}})

        assert resp is not None and resp.get("result")
        assert order == ["trim", "reset_home"]
        assert observed["alive_at_trim"] is False, "a pre-turn history snapshot survived to the heap trim"
    finally:
        server._sessions.pop("sid_trim", None)


def test_prompt_submit_wires_live_title_rename_callback(monkeypatch):
    """The gateway hands the agent a hook so a new title repaints the sidebar.

    Titling itself moved into the shared turn prologue (agent/turn_context.py),
    so the gateway's only remaining job is delivering the rename event. Asserted
    by calling the hook the gateway installed and checking what it emits.
    """

    class _Agent:
        model = "gpt-5.6-sol"
        provider = "openai-codex"
        base_url = "https://chatgpt.example.test/backend-api/codex"
        api_key = object()
        api_mode = "codex_responses"

        def run_conversation(self, prompt, conversation_history=None, stream_callback=None, **_kwargs):
            return {
                "final_response": "Rome was founded in 753 BC.",
                "messages": [
                    {"role": "user", "content": "Tell me about Rome"},
                    {"role": "assistant", "content": "Rome was founded in 753 BC."},
                ],
            }

    agent = _Agent()
    server._sessions["sid"] = _session(agent=agent)
    emitted = []
    monkeypatch.setattr(server.threading, "Thread", _ImmediateThread)
    monkeypatch.setattr(
        server, "_emit", lambda kind, sid, payload=None, **kw: emitted.append((kind, payload))
    )
    monkeypatch.setattr(server, "make_stream_renderer", lambda cols: None)
    monkeypatch.setattr(server, "render_message", lambda raw, cols: None)
    monkeypatch.setattr(server, "_get_db", lambda: None)

    server.handle_request(
        {
            "id": "1",
            "method": "prompt.submit",
            "params": {"session_id": "sid", "text": "Tell me about Rome"},
        }
    )

    hook = getattr(agent, "_on_session_title", None)
    assert callable(hook), "gateway did not install a live title-rename hook"
    # Titling is two-stage, and a local surface wants both: the sidebar renames
    # off the derived slice instantly and sharpens when the model's lands. Only
    # the lanes that spend a rate-limited remote rename filter by stage.
    hook("tell me about rome", "derived")
    hook("Founding of Rome", "llm")
    assert [payload["title"] for kind, payload in emitted if kind == "session.title"] == [
        "tell me about rome",
        "Founding of Rome",
    ]
    assert (
        "session.title",
        {"session_id": "session-key", "title": "Founding of Rome"},
    ) in emitted


def test_prompt_submit_surfaces_backend_error_as_visible_text(monkeypatch):
    """When the backend fails with no visible response (e.g. invalid model slug
    → provider 4xx), the TUI must surface result['error'] as visible text
    instead of emitting a blank message.complete turn."""

    class _Agent:
        def run_conversation(self, prompt, conversation_history=None, stream_callback=None, **_kwargs):
            return {
                "final_response": None,
                "messages": [],
                "api_calls": 0,
                "completed": False,
                "failed": True,
                "error": "HTTP 400: invalid model id 'kimi-k2.6'",
            }

    server._sessions["sid"] = _session(agent=_Agent())
    monkeypatch.setattr(server.threading, "Thread", _ImmediateThread)

    emitted: list[tuple[str, str, dict]] = []
    monkeypatch.setattr(
        server,
        "_emit",
        lambda event, sid, payload=None: emitted.append((event, sid, payload or {})),
    )
    monkeypatch.setattr(server, "make_stream_renderer", lambda cols: None)
    monkeypatch.setattr(server, "render_message", lambda raw, cols: None)
    monkeypatch.setattr(server, "_get_db", lambda: None)

    server.handle_request(
        {
            "id": "1",
            "method": "prompt.submit",
            "params": {"session_id": "sid", "text": "hello"},
        }
    )

    complete_events = [e for e in emitted if e[0] == "message.complete"]
    assert complete_events, "expected message.complete to be emitted"
    payload = complete_events[-1][2]
    assert payload.get("status") == "error"
    text = payload.get("text", "")
    assert "invalid model id 'kimi-k2.6'" in text
    assert payload.get("error") == "HTTP 400: invalid model id 'kimi-k2.6'"


def test_prompt_submit_preserves_empty_response_without_error(monkeypatch):
    """An empty final_response with NO backend error must stay empty — do not
    synthesize an error string. Preserves the existing None/empty-sentinel
    semantics owned by downstream handlers."""

    class _Agent:
        def run_conversation(self, prompt, conversation_history=None, stream_callback=None, **_kwargs):
            return {
                "final_response": None,
                "messages": [],
                "api_calls": 1,
                "completed": True,
            }

    server._sessions["sid"] = _session(agent=_Agent())
    monkeypatch.setattr(server.threading, "Thread", _ImmediateThread)

    emitted: list[tuple[str, str, dict]] = []
    monkeypatch.setattr(
        server,
        "_emit",
        lambda event, sid, payload=None: emitted.append((event, sid, payload or {})),
    )
    monkeypatch.setattr(server, "make_stream_renderer", lambda cols: None)
    monkeypatch.setattr(server, "render_message", lambda raw, cols: None)
    monkeypatch.setattr(server, "_get_db", lambda: None)

    server.handle_request(
        {
            "id": "1",
            "method": "prompt.submit",
            "params": {"session_id": "sid", "text": "hello"},
        }
    )

    complete_events = [e for e in emitted if e[0] == "message.complete"]
    assert complete_events, "expected message.complete to be emitted"
    payload = complete_events[-1][2]
    # Status stays "complete" because no error flag was set
    assert payload.get("status") == "complete"
    # Text stays empty — we did NOT fabricate an "Error:" string
    text = payload.get("text", "")
    assert text in {"", None}, f"expected empty text, got {text!r}"


def test_session_active_list_reports_live_sessions(monkeypatch):
    class _DB:
        def get_session_title(self, key):
            return {"key-a": "Research", "key-b": "Implement"}.get(key, "")

    previous_sessions = dict(server._sessions)
    server._sessions.clear()
    monkeypatch.setattr(server, "_get_db", lambda: _DB())
    server._sessions["sid-a"] = _session(
        agent=types.SimpleNamespace(model="model-a"),
        history=[{"role": "user", "content": "find docs"}],
        session_key="key-a",
        created_at=10.0,
        last_active=20.0,
    )
    server._sessions["sid-b"] = _session(
        agent=types.SimpleNamespace(model="model-b"),
        history=[{"role": "assistant", "content": "writing code"}],
        running=True,
        session_key="key-b",
        created_at=11.0,
        last_active=30.0,
    )
    try:
        resp = server.handle_request(
            {
                "id": "1",
                "method": "session.active_list",
                "params": {"current_session_id": "sid-b"},
            }
        )
    finally:
        server._sessions.clear()
        server._sessions.update(previous_sessions)

    session_rows = resp["result"]["sessions"]
    assert [row["id"] for row in session_rows] == ["sid-a", "sid-b"]

    rows = {row["id"]: row for row in session_rows}
    assert rows["sid-a"] == {
        "current": False,
        "id": "sid-a",
        "last_active": 20.0,
        "message_count": 1,
        "model": "model-a",
        "preview": "find docs",
        "session_key": "key-a",
        "started_at": 10.0,
        "status": "idle",
        "title": "Research",
    }
    assert rows["sid-b"]["current"] is True
    assert rows["sid-b"]["status"] == "working"
    assert rows["sid-b"]["title"] == "Implement"
    assert rows["sid-b"]["preview"] == "writing code"


def test_session_active_list_excludes_finalized_sessions(monkeypatch):
    """#38950: a finalized-but-not-yet-popped session must not inflate the count.

    The WS grace-reap and idle reaper set ``_finalized`` inside
    ``_teardown_session`` before popping the entry from ``_sessions``. During
    that window ``session.active_list`` would otherwise still report the dead
    session, which is exactly the footer "N sessions" count that only ever grew
    until a gateway restart. A live session on the real stdio transport (the
    standalone ``hermes --tui`` case) must still be reported.
    """
    class _DB:
        def get_session_title(self, key):
            return {"key-live": "Live", "key-dead": "Dead"}.get(key, "")

    previous_sessions = dict(server._sessions)
    server._sessions.clear()
    monkeypatch.setattr(server, "_get_db", lambda: _DB())
    server._sessions["sid-live"] = _session(
        agent=types.SimpleNamespace(model="model-live"),
        history=[{"role": "user", "content": "still here"}],
        session_key="key-live",
        created_at=10.0,
        last_active=20.0,
    )
    dead = _session(
        agent=types.SimpleNamespace(model="model-dead"),
        history=[{"role": "user", "content": "gone"}],
        session_key="key-dead",
        created_at=11.0,
        last_active=21.0,
    )
    dead["_finalized"] = True
    server._sessions["sid-dead"] = dead
    try:
        resp = server.handle_request(
            {
                "id": "1",
                "method": "session.active_list",
                "params": {},
            }
        )
    finally:
        server._sessions.clear()
        server._sessions.update(previous_sessions)

    session_rows = resp["result"]["sessions"]
    assert [row["id"] for row in session_rows] == ["sid-live"]


def test_session_activate_returns_inflight_stream_before_completion(monkeypatch):
    """Switching into a still-running live session must hydrate partial output.

    The committed session history is only updated after run_conversation returns,
    so session.activate needs an explicit in-flight payload sourced from the
    backend stream callback.
    """
    started = threading.Event()
    release = threading.Event()
    done = threading.Event()

    class _Agent:
        model = "model-live"

        def run_conversation(self, prompt, conversation_history=None, stream_callback=None, **_kwargs):
            assert prompt == "write a long answer"
            assert conversation_history == []
            stream_callback("partial ")
            stream_callback("answer")
            started.set()
            assert release.wait(2), "test timed out waiting to finish fake model turn"
            return {
                "final_response": "partial answer complete",
                "messages": [
                    {"role": "user", "content": "write a long answer"},
                    {"role": "assistant", "content": "partial answer complete"},
                ],
            }

    server._sessions["sid-live"] = _session(agent=_Agent())
    monkeypatch.setattr(server, "make_stream_renderer", lambda cols: None)
    monkeypatch.setattr(server, "render_message", lambda raw, cols: None)
    monkeypatch.setattr(server, "_get_db", lambda: None)
    monkeypatch.setattr(server, "_session_info", lambda agent: {"model": agent.model})

    def _emit(event, sid, payload=None):
        if event == "message.complete":
            done.set()

    monkeypatch.setattr(server, "_emit", _emit)

    try:
        submit = server.handle_request(
            {
                "id": "submit",
                "method": "prompt.submit",
                "params": {"session_id": "sid-live", "text": "write a long answer"},
            }
        )
        assert submit["result"]["status"] == "streaming"
        assert started.wait(2), "fake model did not stream before activation"

        resp = server.handle_request(
            {
                "id": "activate",
                "method": "session.activate",
                "params": {"session_id": "sid-live"},
            }
        )

        inflight = resp["result"].get("inflight")
        assert inflight == {
            "assistant": "partial answer",
            "streaming": True,
            "user": "write a long answer",
        }
        turn_started_at = resp["result"]["turn_started_at"]
        assert turn_started_at == server._sessions["sid-live"]["inflight_turn"]["started_at"]
        assert turn_started_at > 0
        assert resp["result"]["messages"] == []

        release.set()
        assert done.wait(2), "fake model turn did not complete"
        completed = server.handle_request(
            {
                "id": "activate-done",
                "method": "session.activate",
                "params": {"session_id": "sid-live"},
            }
        )
        assert completed["result"].get("inflight") is None
        assert completed["result"]["turn_started_at"] is None
        assert completed["result"]["messages"] == [
            {"role": "user", "text": "write a long answer"},
            {"role": "assistant", "text": "partial answer complete"},
        ]
    finally:
        release.set()
        done.wait(2)
        server._sessions.pop("sid-live", None)


def test_session_activate_returns_prompt_queued_during_busy_turn(monkeypatch):
    """A full client restart must recover an accepted next-turn prompt.

    Busy prompts are intentionally not durable until they drain. Their only
    authoritative copy is ``queued_prompt``, so the live projection must expose
    that copy without leaking the transport object.
    """
    monkeypatch.setattr(server, "_load_busy_input_mode", lambda: "queue")
    monkeypatch.setattr(server, "_session_info", lambda agent: {"model": agent.model})
    agent = types.SimpleNamespace(model="model-live")
    session = _session(
        agent=agent,
        running=True,
        inflight_turn={
            "assistant": "partial answer",
            "streaming": True,
            "user": "current prompt",
        },
    )
    server._sessions["sid-live"] = session
    try:
        queued = server._handle_busy_submit(
            "submit", "sid-live", session, "newest prompt", object()
        )
        assert queued["result"]["status"] == "queued"

        activated = server.handle_request(
            {
                "id": "activate",
                "method": "session.activate",
                "params": {"session_id": "sid-live"},
            }
        )

        assert activated["result"]["queued"] == {"user": "newest prompt"}
        assert "transport" not in activated["result"]["queued"]
    finally:
        server._sessions.pop("sid-live", None)


def test_session_activate_switches_live_session_without_closing_siblings(monkeypatch):
    monkeypatch.setattr(server, "_session_info", lambda agent: {"model": agent.model})
    server._sessions["sid-a"] = _session(
        agent=types.SimpleNamespace(model="model-a"),
        history=[{"role": "user", "content": "old"}],
        session_key="key-a",
    )
    server._sessions["sid-b"] = _session(
        agent=types.SimpleNamespace(model="model-b"),
        history=[
            {"role": "user", "content": "new prompt"},
            {"role": "assistant", "content": "new answer"},
        ],
        running=True,
        session_key="key-b",
    )
    try:
        resp = server.handle_request(
            {"id": "1", "method": "session.activate", "params": {"session_id": "sid-b"}}
        )

        assert "sid-a" in server._sessions
        assert "sid-b" in server._sessions
        assert resp["result"]["session_id"] == "sid-b"
        assert resp["result"]["session_key"] == "key-b"
        assert resp["result"]["running"] is True
        assert resp["result"]["status"] == "working"
        assert resp["result"]["info"] == {"model": "model-b"}
        assert resp["result"]["messages"] == [
            {"role": "user", "text": "new prompt"},
            {"role": "assistant", "text": "new answer"},
        ]
    finally:
        server._sessions.pop("sid-a", None)
        server._sessions.pop("sid-b", None)


def test_session_activate_can_omit_duplicate_desktop_transcript(monkeypatch):
    monkeypatch.setattr(server, "_session_info", lambda agent: {"model": agent.model})
    server._sessions["sid-large"] = _session(
        agent=types.SimpleNamespace(model="model-large"),
        history=[
            {"role": "user", "content": "large prompt"},
            {"role": "assistant", "content": "large answer"},
        ],
        session_key="key-large",
    )
    try:
        resp = server.handle_request(
            {
                "id": "1",
                "method": "session.activate",
                "params": {"session_id": "sid-large", "omit_messages": True},
            }
        )

        assert resp["result"]["messages"] == []
        assert resp["result"]["message_count"] == 2
        assert resp["result"]["messages_omitted"] is True
        assert resp["result"]["session_key"] == "key-large"
    finally:
        server._sessions.pop("sid-large", None)


def test_session_most_recent_returns_first_non_denied(monkeypatch):
    """Drops `tool` rows like session.list does, returns the first hit."""

    class _DB:
        def list_sessions_rich(self, *, source=None, limit=200, order_by_last_active=False, compact_rows=False):
            return [
                {"id": "tool-1", "source": "tool", "title": "noise", "started_at": 100},
                {"id": "tui-1", "source": "tui", "title": "real", "started_at": 99},
            ]

    monkeypatch.setattr(server, "_get_db", lambda: _DB())

    resp = server.handle_request(
        {"id": "1", "method": "session.most_recent", "params": {}}
    )

    assert resp["result"]["session_id"] == "tui-1"
    assert resp["result"]["title"] == "real"
    assert resp["result"]["source"] == "tui"


def test_session_most_recent_returns_null_when_only_tool_rows(monkeypatch):
    class _DB:
        def list_sessions_rich(self, *, source=None, limit=200, order_by_last_active=False, compact_rows=False):
            return [{"id": "tool-1", "source": "tool", "started_at": 1}]

    monkeypatch.setattr(server, "_get_db", lambda: _DB())

    resp = server.handle_request(
        {"id": "1", "method": "session.most_recent", "params": {}}
    )

    assert resp["result"]["session_id"] is None


def test_session_most_recent_skips_unknown_source_rows(monkeypatch):
    """#54320: a token-accounting guard placeholder (source='unknown') must never be
    picked for auto-resume — the row can outrank the session the user actually opened."""

    class _DB:
        def list_sessions_rich(self, *, source=None, limit=200, order_by_last_active=False, compact_rows=False):
            return [
                {"id": "guard-1", "source": "unknown", "title": "", "started_at": 101},
                {"id": "tui-1", "source": "tui", "title": "real", "started_at": 100},
            ]

    monkeypatch.setattr(server, "_get_db", lambda: _DB())

    resp = server.handle_request(
        {"id": "1", "method": "session.most_recent", "params": {}}
    )

    assert resp["result"]["session_id"] == "tui-1"
    assert resp["result"]["source"] == "tui"


def test_session_most_recent_returns_null_when_only_unknown_rows(monkeypatch):
    class _DB:
        def list_sessions_rich(self, *, source=None, limit=200, order_by_last_active=False, compact_rows=False):
            return [{"id": "guard-1", "source": "unknown", "started_at": 1}]

    monkeypatch.setattr(server, "_get_db", lambda: _DB())

    resp = server.handle_request(
        {"id": "1", "method": "session.most_recent", "params": {}}
    )

    assert resp["result"]["session_id"] is None


def test_session_most_recent_folds_db_exception_into_null_result(monkeypatch):
    """Per contract, errors are folded into the null-result shape so
    callers don't have to special-case JSON-RPC error envelopes for
    'no answer' (Copilot review on #17130)."""

    class _BrokenDB:
        def list_sessions_rich(self, *, source=None, limit=200, order_by_last_active=False, compact_rows=False):
            raise RuntimeError("db locked")

    monkeypatch.setattr(server, "_get_db", lambda: _BrokenDB())

    resp = server.handle_request(
        {"id": "1", "method": "session.most_recent", "params": {}}
    )

    assert "error" not in resp
    assert resp["result"]["session_id"] is None


def test_session_most_recent_handles_db_unavailable(monkeypatch):
    monkeypatch.setattr(server, "_get_db", lambda: None)

    resp = server.handle_request(
        {"id": "1", "method": "session.most_recent", "params": {}}
    )

    assert resp["result"]["session_id"] is None



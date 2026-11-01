









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


def test_run_prompt_submit_registers_turn_thread_for_interrupt(monkeypatch):
    """_run_prompt_submit must expose the actual turn thread to session.interrupt.

    prompt.submit's outer wrapper only waits for agent initialization, then
    _run_prompt_submit starts the real conversation thread. If the session keeps
    the wrapper thread handle, stop/esc sees a dead thread and never calls
    agent.interrupt() on the live turn.
    """
    calls = {"interrupted": False, "started": False}

    class _FakeThread:
        def __init__(self, target=None, daemon=None, **_thread_options):
            self.target = target

        def start(self):
            calls["started"] = True

        def is_alive(self):
            return True

    agent = types.SimpleNamespace(
        interrupt=lambda: calls.__setitem__("interrupted", True),
        run_conversation=lambda *args, **kwargs: {},
    )
    session = _session(agent=agent, running=True)
    server._sessions["sid"] = session

    try:
        monkeypatch.setattr(server.threading, "Thread", _FakeThread)
        monkeypatch.setattr(server, "_emit", lambda *args, **kwargs: None)

        server._run_prompt_submit("1", "sid", session, "hello")

        assert session.get("_run_thread") is not None
        resp = server.handle_request(
            {"id": "2", "method": "session.interrupt", "params": {"session_id": "sid"}}
        )

        assert resp.get("result"), f"got error: {resp.get('error')}"
        assert calls["interrupted"] is True
    finally:
        server._sessions.pop("sid", None)


def test_interrupt_drops_queued_prompt_for_session():
    """Explicit stop cancels a queued next turn instead of auto-draining it."""
    calls = {"interrupted": False}

    class _LiveThread:
        def is_alive(self):
            return True

    session = _session(
        agent=types.SimpleNamespace(
            interrupt=lambda: calls.__setitem__("interrupted", True)
        ),
        running=True,
        queued_prompt={"text": "next prompt", "transport": None},
        queued_prompts=[{"text": "later prompt", "image_paths": ["/tmp/later.png"], "transport": None}],
        _run_thread=_LiveThread(),
    )
    server._sessions["sid"] = session

    try:
        resp = server.handle_request(
            {"id": "1", "method": "session.interrupt", "params": {"session_id": "sid"}}
        )

        assert resp.get("result"), f"got error: {resp.get('error')}"
        assert calls["interrupted"] is True
        assert session.get("queued_prompt") is None
        assert session.get("queued_prompts") is None
    finally:
        server._sessions.pop("sid", None)


def test_interrupt_before_agent_ready_prevents_late_turn_start(monkeypatch):
    """Stop during lazy agent startup must not start the turn after init finishes."""
    threads = []
    calls = {"run_prompt": 0}

    class _FakeThread:
        def __init__(self, target=None, daemon=None, **_thread_options):
            self.target = target
            threads.append(self)

        def start(self):
            return None

        def is_alive(self):
            return True

    session = _session()
    session["agent"] = None
    server._sessions["sid"] = session

    try:
        monkeypatch.setattr(server.threading, "Thread", _FakeThread)
        monkeypatch.setattr(server, "_emit", lambda *args, **kwargs: None)
        monkeypatch.setattr(server, "_ensure_session_db_row", lambda session: None)
        monkeypatch.setattr(server, "_persist_branch_seed", lambda session: None)
        monkeypatch.setattr(server, "_start_agent_build", lambda sid, session: None)
        monkeypatch.setattr(server, "_wait_agent", lambda session, rid: None)
        monkeypatch.setattr(
            server,
            "_run_prompt_submit",
            lambda *args, **kwargs: calls.__setitem__(
                "run_prompt", calls["run_prompt"] + 1
            ),
        )

        submit = server.handle_request(
            {
                "id": "1",
                "method": "prompt.submit",
                "params": {"session_id": "sid", "text": "hello"},
            }
        )
        assert submit.get("result"), f"got error: {submit.get('error')}"
        assert session["running"] is True
        assert len(threads) == 1

        stop = server.handle_request(
            {"id": "2", "method": "session.interrupt", "params": {"session_id": "sid"}}
        )
        assert stop.get("result"), f"got error: {stop.get('error')}"

        threads[0].target()

        assert calls["run_prompt"] == 0
        assert session["running"] is False
        assert session.get("inflight_turn") is None
    finally:
        server._sessions.pop("sid", None)


def test_cancelled_turn_before_agent_ready_emits_error_event(monkeypatch):
    """A turn cancelled during lazy agent startup must surface an error event.

    Sibling of test_interrupt_before_agent_ready_prevents_late_turn_start: that
    test only asserts `_run_prompt_submit` is skipped, mocking `_emit` to a
    no-op so it cannot catch a silent drop. This test captures `_emit` and
    asserts the client receives an `error` event with a human-readable message,
    so the Desktop composer can show feedback instead of hanging on a
    `{"status":"streaming"}` reply that never produces a turn (issue #63078
    server-side half).
    """
    threads = []
    emitted = []
    calls = {"run_prompt": 0}

    class _FakeThread:
        def __init__(self, target=None, daemon=None, **_thread_options):
            self.target = target
            threads.append(self)

        def start(self):
            return None

        def is_alive(self):
            return True

    session = _session()
    session["agent"] = None
    server._sessions["sid"] = session

    try:
        monkeypatch.setattr(server.threading, "Thread", _FakeThread)
        monkeypatch.setattr(server, "_emit", lambda *args, **kwargs: emitted.append(args))
        monkeypatch.setattr(server, "_ensure_session_db_row", lambda session: None)
        monkeypatch.setattr(server, "_persist_branch_seed", lambda session: None)
        monkeypatch.setattr(server, "_start_agent_build", lambda sid, session: None)
        monkeypatch.setattr(server, "_wait_agent", lambda session, rid: None)
        monkeypatch.setattr(
            server,
            "_run_prompt_submit",
            lambda *args, **kwargs: calls.__setitem__(
                "run_prompt", calls["run_prompt"] + 1
            ),
        )

        submit = server.handle_request(
            {
                "id": "1",
                "method": "prompt.submit",
                "params": {"session_id": "sid", "text": "hello"},
            }
        )
        assert submit.get("result"), f"got error: {submit.get('error')}"
        assert session["running"] is True

        # User hits Stop while the agent is still building.
        stop = server.handle_request(
            {"id": "2", "method": "session.interrupt", "params": {"session_id": "sid"}}
        )
        assert stop.get("result"), f"got error: {stop.get('error')}"
        assert session.get("_turn_cancel_requested") is True

        # The deferred run thread now wakes up; without the emit it would bail
        # silently and the Desktop would never learn the turn was dropped.
        threads[0].target()

        assert calls["run_prompt"] == 0
        assert session["running"] is False
        assert session.get("inflight_turn") is None
        # Exactly one error event addressed to this session.
        error_events = [e for e in emitted if e and len(e) >= 2 and e[0] == "error" and e[1] == "sid"]
        assert len(error_events) == 1, f"expected one error event, got: {emitted}"
        msg = error_events[0][2].get("message", "")
        assert "cancelled" in msg.lower(), f"unexpected message: {msg}"
    finally:
        server._sessions.pop("sid", None)


def test_session_not_running_before_agent_ready_emits_error_event(monkeypatch):
    """When `running` is cleared by something other than an explicit interrupt
    (e.g. a concurrent session.create race that resets the flag), the deferred
    run thread must still emit an error event rather than disappearing silently.
    """
    threads = []
    emitted = []
    calls = {"run_prompt": 0}

    class _FakeThread:
        def __init__(self, target=None, daemon=None, **_thread_options):
            self.target = target
            threads.append(self)

        def start(self):
            return None

        def is_alive(self):
            return True

    session = _session()
    session["agent"] = None
    server._sessions["sid"] = session

    try:
        monkeypatch.setattr(server.threading, "Thread", _FakeThread)
        monkeypatch.setattr(server, "_emit", lambda *args, **kwargs: emitted.append(args))
        monkeypatch.setattr(server, "_ensure_session_db_row", lambda session: None)
        monkeypatch.setattr(server, "_persist_branch_seed", lambda session: None)
        monkeypatch.setattr(server, "_start_agent_build", lambda sid, session: None)
        monkeypatch.setattr(server, "_wait_agent", lambda session, rid: None)
        monkeypatch.setattr(
            server,
            "_run_prompt_submit",
            lambda *args, **kwargs: calls.__setitem__(
                "run_prompt", calls["run_prompt"] + 1
            ),
        )

        submit = server.handle_request(
            {
                "id": "1",
                "method": "prompt.submit",
                "params": {"session_id": "sid", "text": "hello"},
            }
        )
        assert submit.get("result"), f"got error: {submit.get('error')}"
        assert session["running"] is True

        # Simulate a concurrent path clearing `running` without setting the
        # cancel flag (the other branch of the guard).
        with session["history_lock"]:
            session["running"] = False

        threads[0].target()

        assert calls["run_prompt"] == 0
        assert session.get("inflight_turn") is None
        error_events = [e for e in emitted if e and len(e) >= 2 and e[0] == "error" and e[1] == "sid"]
        assert len(error_events) == 1, f"expected one error event, got: {emitted}"
        msg = error_events[0][2].get("message", "")
        assert msg.strip(), f"empty error message: {emitted}"
    finally:
        server._sessions.pop("sid", None)


def test_slow_agent_build_delivers_prompt_instead_of_timing_out(monkeypatch):
    """#63078 server-side half: a deferred build slower than the old 30s
    ``_wait_agent`` cliff must NOT eat the first message. The patient wait
    keeps the pending prompt attached and delivers it as soon as the
    still-running build completes."""
    threads = []
    emitted = []
    calls = {"run_prompt": 0}

    class _FakeThread:
        def __init__(self, target=None, daemon=None, **_thread_options):
            self.target = target
            threads.append(self)

        def start(self):
            return None

        def is_alive(self):
            return True

    ready = threading.Event()
    session = _session(agent_ready=ready)
    session["agent"] = None
    server._sessions["sid"] = session

    # The build "completes" only after the wait loop has already gone through
    # several empty slices — i.e. well past what a single fixed-timeout wait
    # slice would tolerate.
    slices = {"n": 0}

    class _SlowReady:
        def wait(self, timeout=None):
            slices["n"] += 1
            if slices["n"] >= 3:
                ready.set()
                session["agent"] = types.SimpleNamespace()
                return True
            return False

        def is_set(self):
            return ready.is_set()

    session["agent_ready"] = _SlowReady()

    try:
        monkeypatch.setattr(server.threading, "Thread", _FakeThread)
        monkeypatch.setattr(server, "_emit", lambda *args, **kwargs: emitted.append(args))
        monkeypatch.setattr(server, "_ensure_session_db_row", lambda session: None)
        monkeypatch.setattr(server, "_persist_branch_seed", lambda session: None)
        monkeypatch.setattr(server, "_start_agent_build", lambda sid, session: None)
        monkeypatch.setattr(
            server,
            "_run_prompt_submit",
            lambda *args, **kwargs: calls.__setitem__(
                "run_prompt", calls["run_prompt"] + 1
            ),
        )

        submit = server.handle_request(
            {
                "id": "1",
                "method": "prompt.submit",
                "params": {"session_id": "sid", "text": "first message"},
            }
        )
        assert submit.get("result"), f"got error: {submit.get('error')}"

        threads[0].target()

        # The message was DELIVERED, not dropped, and no error event fired.
        assert calls["run_prompt"] == 1
        error_events = [e for e in emitted if e and e[0] == "error"]
        assert not error_events, f"unexpected error events: {error_events}"
    finally:
        server._sessions.pop("sid", None)


def test_slow_agent_build_emits_keyed_progress_notice(monkeypatch):
    """Past the slow threshold the patient wait must tell the user once
    (keyed notification.show) and clear the notice when the build lands —
    a long wait is acceptable, a silent one is not."""
    threads = []
    emitted = []
    calls = {"run_prompt": 0}

    class _FakeThread:
        def __init__(self, target=None, daemon=None, **_thread_options):
            self.target = target
            threads.append(self)

        def start(self):
            return None

        def is_alive(self):
            return True

    ready = threading.Event()
    session = _session(agent_ready=ready)
    session["agent"] = None
    server._sessions["sid"] = session

    slices = {"n": 0}

    class _SlowReady:
        def wait(self, timeout=None):
            slices["n"] += 1
            if slices["n"] >= 3:
                ready.set()
                session["agent"] = types.SimpleNamespace()
                return True
            return False

        def is_set(self):
            return ready.is_set()

    session["agent_ready"] = _SlowReady()

    try:
        monkeypatch.setattr(server.threading, "Thread", _FakeThread)
        # Every wait slice lands past the slow threshold.
        monkeypatch.setattr(server, "_AGENT_BUILD_SLOW_NOTICE_AFTER", 0.0)
        monkeypatch.setattr(server, "_emit", lambda *args, **kwargs: emitted.append(args))
        monkeypatch.setattr(server, "_ensure_session_db_row", lambda session: None)
        monkeypatch.setattr(server, "_persist_branch_seed", lambda session: None)
        monkeypatch.setattr(server, "_start_agent_build", lambda sid, session: None)
        monkeypatch.setattr(
            server,
            "_run_prompt_submit",
            lambda *args, **kwargs: calls.__setitem__(
                "run_prompt", calls["run_prompt"] + 1
            ),
        )

        submit = server.handle_request(
            {
                "id": "1",
                "method": "prompt.submit",
                "params": {"session_id": "sid", "text": "first message"},
            }
        )
        assert submit.get("result"), f"got error: {submit.get('error')}"

        threads[0].target()

        assert calls["run_prompt"] == 1
        shows = [e for e in emitted if e and e[0] == "notification.show" and e[1] == "sid"]
        clears = [e for e in emitted if e and e[0] == "notification.clear" and e[1] == "sid"]
        # Exactly one keyed notice, replaced-in-place semantics, then cleared.
        assert len(shows) == 1, f"expected one slow-build notice, got: {shows}"
        assert shows[0][2].get("key") == server._AGENT_BUILD_SLOW_NOTICE_KEY
        assert len(clears) == 1 and clears[0][2].get("key") == server._AGENT_BUILD_SLOW_NOTICE_KEY
    finally:
        server._sessions.pop("sid", None)


def test_agent_build_failure_surfaces_error_and_drops_turn(monkeypatch):
    """When the build itself FAILS (agent_error set when ready fires), the
    prompt must not run and the failure must reach the client as a visible
    error event — never a silent drop.

    prompt.submit retries a completed failed build once (fresh provider
    resolution un-wedges sessions whose failure cause was fixed), so the
    build stub here is a faithful failing build: it sets agent_error and
    fires the session's CURRENT ready event (the retry installs a new one).
    A no-op stub would leave that event unset and hang the patient wait."""
    threads = []
    emitted = []
    calls = {"run_prompt": 0}

    class _FakeThread:
        def __init__(self, target=None, daemon=None, **_thread_options):
            self.target = target
            threads.append(self)

        def start(self):
            return None

        def is_alive(self):
            return True

    ready = threading.Event()
    ready.set()  # build finished...
    session = _session(agent_ready=ready)
    session["agent"] = None
    session["agent_error"] = "No LLM provider configured"  # ...but failed
    server._sessions["sid"] = session

    def _failing_build(sid, session):
        session["agent_error"] = "No LLM provider configured"
        session["agent_ready"].set()

    try:
        monkeypatch.setattr(server.threading, "Thread", _FakeThread)
        monkeypatch.setattr(server, "_emit", lambda *args, **kwargs: emitted.append(args))
        monkeypatch.setattr(server, "_ensure_session_db_row", lambda session: None)
        monkeypatch.setattr(server, "_persist_branch_seed", lambda session: None)
        monkeypatch.setattr(server, "_start_agent_build", _failing_build)
        monkeypatch.setattr(
            server,
            "_run_prompt_submit",
            lambda *args, **kwargs: calls.__setitem__(
                "run_prompt", calls["run_prompt"] + 1
            ),
        )

        submit = server.handle_request(
            {
                "id": "1",
                "method": "prompt.submit",
                "params": {"session_id": "sid", "text": "first message"},
            }
        )
        assert submit.get("result"), f"got error: {submit.get('error')}"

        threads[0].target()

        assert calls["run_prompt"] == 0
        assert session["running"] is False
        # #71184 upgraded failure delivery from a bare "error" event to a
        # terminal message.complete frame (status=error, recoverable) so
        # failed turns are retained as replayable inflight snapshots. The
        # contract this test pins is unchanged: the build failure must reach
        # the client VISIBLY — never a silent drop.
        failure_frames = [
            e
            for e in emitted
            if e
            and e[0] in ("error", "message.complete")
            and e[1] == "sid"
            and (
                "No LLM provider configured" in str(e[2].get("message", ""))
                or "No LLM provider configured" in str(e[2].get("error", ""))
                or "No LLM provider configured" in str(e[2].get("text", ""))
            )
        ]
        assert len(failure_frames) == 1, f"expected one visible failure frame, got: {emitted}"
        frame = failure_frames[0]
        if frame[0] == "message.complete":
            assert frame[2].get("status") == "error"
    finally:
        server._sessions.pop("sid", None)


def test_dead_build_thread_fails_fast_not_full_cap(monkeypatch):
    """A build thread that died without setting agent_ready means the build
    died hard — the waiter must fail promptly with a visible error instead of
    sitting out the full wait cap on a corpse."""
    emitted = []

    class _DeadThread:
        def is_alive(self):
            return False

    ready = threading.Event()  # never set
    session = _session(agent_ready=ready)
    session["agent"] = None
    session["running"] = True
    session["_agent_build_thread"] = _DeadThread()
    session["agent_error"] = "agent init failed: boom"
    server._sessions["sid"] = session

    try:
        monkeypatch.setattr(server, "_emit", lambda *args, **kwargs: emitted.append(args))
        # Short slices so the test is fast; the dead-thread check fires on the
        # first empty slice, far below the cap.
        monkeypatch.setattr(server, "_AGENT_BUILD_WAIT_SLICE", 0.01)

        start = time.monotonic()
        err = server._wait_agent_for_prompt(session, "rid-1", "sid")
        elapsed = time.monotonic() - start

        assert err is not None
        assert "boom" in (err.get("error") or {}).get("message", "")
        assert elapsed < 5.0, f"dead-thread detection took {elapsed:.1f}s"
    finally:
        server._sessions.pop("sid", None)


def test_wait_agent_for_prompt_honors_cancel_mid_wait(monkeypatch):
    """A cancel arriving during the patient wait must end it promptly and
    return None (the caller's cancel branch owns the user-visible event)."""
    ready = threading.Event()  # never set
    session = _session(agent_ready=ready)
    session["agent"] = None
    session["running"] = True
    server._sessions["sid"] = session

    try:
        monkeypatch.setattr(server, "_AGENT_BUILD_WAIT_SLICE", 0.01)

        def cancel_soon():
            time.sleep(0.05)
            with session["history_lock"]:
                session["_turn_cancel_requested"] = True

        canceller = threading.Thread(target=cancel_soon)
        canceller.start()
        start = time.monotonic()
        err = server._wait_agent_for_prompt(session, "rid-1", "sid")
        elapsed = time.monotonic() - start
        canceller.join()

        assert err is None
        assert elapsed < 5.0, f"cancel honored only after {elapsed:.1f}s"
    finally:
        server._sessions.pop("sid", None)


def test_agent_build_wait_cap_config_override(monkeypatch):
    """agent.build_wait_timeout in config.yaml overrides the default cap;
    invalid/absent values fall back to 600s."""
    monkeypatch.setattr(server, "_load_cfg", lambda: {"agent": {"build_wait_timeout": 90}})
    assert server._agent_build_wait_cap() == 90.0

    monkeypatch.setattr(server, "_load_cfg", lambda: {"agent": {}})
    default_cap = server._agent_build_wait_cap()
    assert default_cap > 0 and default_cap != 90.0

    monkeypatch.setattr(server, "_load_cfg", lambda: {"agent": {"build_wait_timeout": 0}})
    assert server._agent_build_wait_cap() == default_cap

    monkeypatch.setattr(server, "_load_cfg", lambda: {"agent": {"build_wait_timeout": "nonsense"}})
    assert server._agent_build_wait_cap() == default_cap


def test_wait_agent_for_prompt_expires_at_cap(monkeypatch):
    """A genuinely hung build (thread alive, never ready) still fails at the
    bounded cap with a message that tells the user their text was not sent."""
    class _AliveThread:
        def is_alive(self):
            return True

    ready = threading.Event()  # never set
    session = _session(agent_ready=ready)
    session["agent"] = None
    session["running"] = True
    session["_agent_build_thread"] = _AliveThread()
    server._sessions["sid"] = session

    try:
        monkeypatch.setattr(server, "_AGENT_BUILD_WAIT_SLICE", 0.01)
        monkeypatch.setattr(server, "_agent_build_wait_cap", lambda: 0.05)

        err = server._wait_agent_for_prompt(session, "rid-1", "sid")

        assert err is not None
        message = (err.get("error") or {}).get("message", "")
        assert "timed out" in message and "was not sent" in message
    finally:
        server._sessions.pop("sid", None)


def test_clear_pending_without_sid_clears_all():
    """_clear_pending(None) is the process-exit path — every open request is withdrawn, and a response for
    a withdrawn id is dropped quietly (no error frame back to the client)."""
    from hermes_runtime import server_requests
    reqs = [_open_request("sid-x"), _open_request("sid-y", "sudo")]
    try:
        server._clear_pending(None)
        assert all(r.event.is_set() and not r.answered for r in reqs)
        assert server.dispatch({"jsonrpc": "2.0", "id": reqs[0].id, "result": {"answer": "late"}}) is None
    finally:
        server_requests.reset_for_tests()


def test_config_set_model_defers_while_running(monkeypatch):
    """/model via config.set queues the pick during an in-flight turn instead
    of rejecting or racing the worker thread."""
    seen = {"called": False}

    def _fake_apply(sid, session, raw, **_kwargs):
        seen["called"] = True
        return {"value": raw, "warning": ""}

    monkeypatch.setattr(server, "_apply_model_switch", _fake_apply)

    server._sessions["sid"] = _session(running=True)
    try:
        resp = server.handle_request(
            {
                "id": "1",
                "method": "config.set",
                "params": {
                    "session_id": "sid",
                    "key": "model",
                    "value": "anthropic/claude-sonnet-4.6",
                },
            }
        )
        assert not resp.get("error")
        result = resp["result"]
        assert result["deferred"] is True
        assert result["value"] == "anthropic/claude-sonnet-4.6"
        assert not seen["called"], (
            "_apply_model_switch ran mid-turn — would race the worker thread "
            "reading agent.model / agent.client; it must defer to turn start"
        )
        pending = server._sessions["sid"].get("pending_model_switch")
        assert pending and pending["raw"] == "anthropic/claude-sonnet-4.6"
    finally:
        server._sessions.pop("sid", None)


def test_apply_pending_model_switch_runs_queued_pick(monkeypatch):
    """The queued pick is consumed once, on the turn thread, via
    _apply_model_switch — and cleared so it can't re-fire next turn."""
    calls = []

    def _fake_apply(sid, session, raw, **kwargs):
        calls.append(raw)
        return {"value": raw, "warning": "", "confirm_required": False}

    monkeypatch.setattr(server, "_apply_model_switch", _fake_apply)

    session = _session(running=False)
    session["agent"] = object()
    session["pending_model_switch"] = {
        "raw": "anthropic/claude-sonnet-4.6",
        "confirm_expensive_model": False,
    }

    server._apply_pending_model_switch("sid", session)
    assert calls == ["anthropic/claude-sonnet-4.6"]
    assert "pending_model_switch" not in session

    # Idempotent: a second turn start with nothing queued is a no-op.
    server._apply_pending_model_switch("sid", session)
    assert calls == ["anthropic/claude-sonnet-4.6"]


def test_mirror_slash_side_effects_rejects_mutating_commands_while_running(monkeypatch):
    """Slash worker passthrough (e.g. /model, /personality, /prompt,
    /compress) must reject during an in-flight turn.  Same race as
    config.set — mutates live agent state while run_conversation is
    reading it."""
    import types

    applied = {"model": False, "compress": False}

    def _fake_apply_model(sid, session, arg):
        applied["model"] = True
        return {"value": arg, "warning": ""}

    def _fake_compress(session, focus):
        applied["compress"] = True
        return (0, {})

    monkeypatch.setattr(server, "_apply_model_switch", _fake_apply_model)
    monkeypatch.setattr(server, "_compress_session_history", _fake_compress)

    session = _session(running=True)
    session["agent"] = types.SimpleNamespace(model="x")

    for cmd, expected_name in [
        ("/model new/model", "model"),
        ("/personality default", "personality"),
        ("/compress", "compress"),
    ]:
        warning = server._mirror_slash_side_effects("sid", session, cmd)
        assert (
            "session busy" in warning
        ), f"{cmd} should have returned busy warning, got: {warning!r}"
        assert f"/{expected_name}" in warning

    # None of the mutating side-effect helpers should have fired.
    assert not applied["model"], "model switch fired despite running session"
    assert not applied["compress"], "compress fired despite running session"


def test_mirror_slash_side_effects_allowed_when_idle(monkeypatch):
    """Regression guard: idle session still runs the side effects."""
    import types

    applied = {"model": False}

    def _fake_apply_model(sid, session, arg):
        applied["model"] = True
        return {"value": arg, "warning": ""}

    monkeypatch.setattr(server, "_apply_model_switch", _fake_apply_model)

    session = _session(running=False)
    session["agent"] = types.SimpleNamespace(model="x")

    warning = server._mirror_slash_side_effects("sid", session, "/model foo")
    # Should NOT contain "session busy" — the switch went through.
    assert "session busy" not in warning
    assert applied["model"]


def test_mirror_slash_compress_does_not_prelock_history(monkeypatch):
    """Regression guard: /compress side effect must not hold history_lock
    when calling _compress_session_history (the helper snapshots under
    the same non-reentrant lock internally). It also returns a before/after
    summary string (#46686)."""
    import types

    seen = {"compress": False, "sync": False}
    emitted = []

    def _fake_compress(session, focus_topic=None, **_kw):
        seen["compress"] = True
        assert not session["history_lock"].locked()
        # Simulate a real compaction shrinking the transcript.
        session["history"] = [{"role": "user", "content": "summary"}]
        return (1, {"total": 0})

    def _fake_sync(_sid, _session):
        seen["sync"] = True

    monkeypatch.setattr(server, "_compress_session_history", _fake_compress)
    monkeypatch.setattr(server, "_sync_session_key_after_compress", _fake_sync)
    monkeypatch.setattr(server, "_session_info", lambda _agent, *a: {"model": "x"})
    monkeypatch.setattr(server, "_emit", lambda *args: emitted.append(args))

    session = _session(running=False)
    session["history"] = [
        {"role": "user", "content": f"m{i}"} for i in range(6)
    ]
    session["agent"] = types.SimpleNamespace(model="x", _cached_system_prompt="", tools=None)

    warning = server._mirror_slash_side_effects("sid", session, "/compress")

    # Now returns a before/after summary (was "" before #46686).
    assert seen["compress"]
    assert seen["sync"]
    assert ("session.info", "sid", {"model": "x"}) in emitted
    assert warning  # before/after summary returned (#46686)


def test_compress_session_history_here_triggers_partial_compress():
    """/compress here [N] must split history into head/tail and rejoin after
    compression — the partial_compress module is used, not full compress.

    Before this fix, /compress here 3 passed "here 3" as focus_topic to the
    full compress, silently ignoring the boundary intent. The parsing lives
    in _compress_session_history — the choke point every manual-compress
    route (session.compress RPC, command.dispatch, slash-exec mirror)
    converges on — so 'here [N]' works everywhere (#35533).
    """
    compress_context_calls = []
    agent = _partial_compress_agent(compress_context_calls)

    session = _session(agent=agent)
    session["history"] = list(_PARTIAL_FAKE_HISTORY)
    session["history_version"] = 7

    removed, _usage = server._compress_session_history(session, "here 1")

    # agent._compress_context must have been called with the HEAD only
    assert len(compress_context_calls) == 1
    head_passed, focus_passed = compress_context_calls[0]
    assert head_passed == _PARTIAL_FAKE_HISTORY[:-2]
    assert focus_passed is None  # partial compress has no focus topic
    # Session history must now contain the rejoined transcript: compressed
    # head + the last exchange verbatim.
    assert session["history"] == _PARTIAL_COMPRESSED_HEAD + _PARTIAL_FAKE_HISTORY[-2:]
    assert session["history_version"] == 8
    assert removed == len(_PARTIAL_FAKE_HISTORY) - len(session["history"])


def test_compress_session_history_here_falls_back_on_degenerate_split():
    """/compress here with keep_last >= exchanges produces an empty tail —
    must fall back to full compression (whole history, no rejoined tail)."""
    compress_context_calls = []
    agent = _partial_compress_agent(compress_context_calls)

    # 4 messages = 2 exchanges; keep_last=5 leaves nothing to compress.
    short_history = _PARTIAL_FAKE_HISTORY[:4]
    session = _session(agent=agent)
    session["history"] = list(short_history)

    server._compress_session_history(session, "here 5")

    # Degenerate split → full compress of the whole history, focus_topic=None
    assert len(compress_context_calls) == 1
    head_passed, focus_passed = compress_context_calls[0]
    assert head_passed == short_history
    assert focus_passed is None
    assert session["history"] == _PARTIAL_COMPRESSED_HEAD


def test_compress_session_history_plain_focus_topic_not_parsed_as_partial():
    """/compress my topic must still do full compress with focus_topic set."""
    compress_context_calls = []
    agent = _partial_compress_agent(compress_context_calls)

    session = _session(agent=agent)
    session["history"] = list(_PARTIAL_FAKE_HISTORY)

    server._compress_session_history(session, "my topic")

    assert len(compress_context_calls) == 1
    head_passed, focus_passed = compress_context_calls[0]
    assert head_passed == _PARTIAL_FAKE_HISTORY  # full history, no split
    assert focus_passed == "my topic"
    assert session["history"] == _PARTIAL_COMPRESSED_HEAD


def test_session_compress_rpc_honors_here_argument(monkeypatch):
    """Route 1/3: the session.compress RPC must honor 'here [N]'."""
    compress_context_calls = []
    agent = _partial_compress_agent(compress_context_calls)
    session = _session(agent=agent)
    session["history"] = list(_PARTIAL_FAKE_HISTORY)
    server._sessions["sid"] = session

    monkeypatch.setattr(server, "_session_info", lambda *_a, **_kw: {"model": "x"})
    monkeypatch.setattr(server, "_sync_session_key_after_compress", lambda *a, **kw: None)
    monkeypatch.setattr(server, "_emit", lambda *args: None)

    try:
        resp = server.handle_request(
            {
                "id": "1",
                "method": "session.compress",
                "params": {"session_id": "sid", "focus_topic": "here 1"},
            }
        )
    finally:
        server._sessions.pop("sid", None)

    assert resp["result"]["status"] == "compressed"
    assert len(compress_context_calls) == 1
    head_passed, focus_passed = compress_context_calls[0]
    assert head_passed == _PARTIAL_FAKE_HISTORY[:-2]
    assert focus_passed is None
    assert session["history"] == _PARTIAL_COMPRESSED_HEAD + _PARTIAL_FAKE_HISTORY[-2:]


def test_command_dispatch_compress_honors_here_argument(monkeypatch):
    """Route 2/3: command.dispatch /compress must honor 'here [N]'."""
    compress_context_calls = []
    agent = _partial_compress_agent(compress_context_calls)
    session = _session(agent=agent)
    session["history"] = list(_PARTIAL_FAKE_HISTORY)
    server._sessions["sid"] = session

    monkeypatch.setattr(server, "_session_uses_compute_host", lambda *_a, **_kw: False)
    monkeypatch.setattr(server, "_session_info", lambda *_a, **_kw: {"model": "x"})
    monkeypatch.setattr(server, "_sync_session_key_after_compress", lambda *a, **kw: None)
    monkeypatch.setattr(server, "_emit", lambda *args: None)

    try:
        resp = server.handle_request(
            {
                "id": "1",
                "method": "command.dispatch",
                "params": {"session_id": "sid", "name": "compress", "arg": "here 1"},
            }
        )
    finally:
        server._sessions.pop("sid", None)

    assert resp["result"]["type"] == "exec"
    assert len(compress_context_calls) == 1
    head_passed, focus_passed = compress_context_calls[0]
    assert head_passed == _PARTIAL_FAKE_HISTORY[:-2]
    assert focus_passed is None
    assert session["history"] == _PARTIAL_COMPRESSED_HEAD + _PARTIAL_FAKE_HISTORY[-2:]


def test_mirror_slash_compress_honors_here_argument(monkeypatch):
    """Route 3/3: the slash-exec mirror must honor 'here [N]'."""
    compress_context_calls = []
    agent = _partial_compress_agent(compress_context_calls)
    session = _session(agent=agent)
    session["history"] = list(_PARTIAL_FAKE_HISTORY)

    monkeypatch.setattr(server, "_session_info", lambda *_a, **_kw: {"model": "x"})
    monkeypatch.setattr(server, "_sync_session_key_after_compress", lambda *a, **kw: None)
    monkeypatch.setattr(server, "_emit", lambda *args: None)

    warning = server._mirror_slash_side_effects("sid", session, "/compress here 1")

    assert "Compressed:" in warning
    assert len(compress_context_calls) == 1
    head_passed, focus_passed = compress_context_calls[0]
    assert head_passed == _PARTIAL_FAKE_HISTORY[:-2]
    assert focus_passed is None
    assert session["history"] == _PARTIAL_COMPRESSED_HEAD + _PARTIAL_FAKE_HISTORY[-2:]


@pytest.mark.real_agent_prewarm
def test_session_create_close_race_does_not_orphan_resources(monkeypatch):
    """Regression guard: if session.close runs while session.create's
    _build thread is still constructing the agent, the build thread
    must detect the orphan, close the just-built agent, and unregister
    the notify registration it's about to install - avoiding installing
    a slash_worker or notify callback for the dead session.  It must also
    never pre-warm a slash worker (each worker forks the full stdio MCP
    fleet; spawn is on-demand in slash.exec) - a worker constructed here
    would be a regression.  Without the early abort the agent outlives
    the session until gateway shutdown."""
    import threading

    created_workers: list[str] = []
    closed_workers: list[str] = []
    closed_agents: list[str] = []
    registered_keys: list[str] = []
    unregistered_keys: list[str] = []

    class _FakeWorker:
        def __init__(self, key, model, profile_home=None):
            self.key = key
            self._closed = False
            created_workers.append(key)

        def close(self):
            self._closed = True
            closed_workers.append(self.key)

    class _FakeAgent:
        def __init__(self):
            self.model = "x"
            self.provider = "openrouter"
            self.base_url = ""
            self.api_key = ""

        def close(self):
            closed_agents.append("closed")

    # Make _build block until we release it — simulates slow agent init.
    # Also signal when _build actually reaches _make_agent so the test
    # can close the session at the right moment: session.create now
    # defers _start_agent_build behind a 50ms timer (see the
    # `_deferred_build` path in @method("session.create")), so closing
    # before the build thread has even started would skip the orphan
    # detection entirely and the test would race a non-event.
    build_started = threading.Event()
    release_build = threading.Event()
    build_entered = threading.Event()

    def _slow_make_agent(sid, key, session_id=None, session_db=None, **_kwargs):
        build_started.set()
        build_entered.set()
        release_build.wait(timeout=3.0)
        return _FakeAgent()

    # Stub everything _build touches
    monkeypatch.setattr(server, "_make_agent", _slow_make_agent)
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

    # Shim register/unregister to observe leaks
    import tools.approval as _approval

    monkeypatch.setattr(
        _approval,
        "register_gateway_notify",
        lambda key, cb: registered_keys.append(key),
    )
    monkeypatch.setattr(
        _approval,
        "unregister_gateway_notify",
        lambda key: unregistered_keys.append(key),
    )
    monkeypatch.setattr(_approval, "load_permanent_allowlist", lambda: None)

    # Start: session.create spawns _build thread, returns synchronously
    resp = server.handle_request(
        {
            "id": "1",
            "method": "session.create",
            "params": {"cols": 80},
        }
    )
    assert resp.get("result"), f"got error: {resp.get('error')}"
    sid = resp["result"]["session_id"]
    own_key = resp["result"]["stored_session_id"]
    assert build_entered.wait(timeout=1.0), "deferred build did not start"

    # Wait until the (deferred) build thread has actually entered
    # _make_agent — otherwise session.close pops _sessions[sid] before
    # _build ever runs, _start_agent_build never calls _build, and we
    # never exercise the orphan-cleanup path.
    assert build_started.wait(timeout=2.0), "build thread never entered _make_agent"

    # Build thread is blocked in _slow_make_agent.  Close the session
    # NOW — this pops _sessions[sid] before _build can install the
    # worker/notify.
    close_resp = server.handle_request(
        {
            "id": "2",
            "method": "session.close",
            "params": {"session_id": sid},
        }
    )
    assert close_resp.get("result", {}).get("closed") is True

    # At this point session.close saw agent=None and slash_worker=None (never
    # eagerly installed) so it had nothing to close.  Release the build thread:
    # it must close the agent that finishes late and abort before registering
    # any more session resources - without ever having constructed a worker.
    release_build.set()

    # Give the build thread a moment to run through its finally.
    for _ in range(100):
        if closed_agents:
            break
        import time

        time.sleep(0.02)

    assert closed_agents == ["closed"], (
        "agent built after session.close was never closed - "
        f"closed_agents={closed_agents}"
    )
    assert created_workers == [], (
        f"build thread pre-warmed a slash worker (spawn must stay on-demand "
        f"in slash.exec) — created_workers={created_workers}"
    )
    # Notify may be unregistered by both session.close (unconditional)
    # and the orphan-cleanup path; the key guarantee is that THIS session's
    # key gets unregistered (any prior close already popped the callback; the
    # duplicate is a no-op). Match on our own key, not the global count: the
    # registry is process-wide and a leaked _build thread from another
    # session.create test can append a foreign key here and falsely satisfy
    # a bare `>= 1`.
    assert own_key in unregistered_keys, (
        f"orphan notify registration was not unregistered — "
        f"{own_key} not in unregistered_keys={unregistered_keys}"
    )
    assert own_key not in registered_keys, (
        f"the abandoned build registered {own_key} for a closed session"
    )



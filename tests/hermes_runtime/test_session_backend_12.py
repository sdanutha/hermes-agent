









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


def test_rollback_restore_rejects_full_history_while_running(monkeypatch):
    """Full-history rollback must reject; file-scoped rollback still allowed."""
    server._sessions["sid"] = _session(running=True)
    try:
        resp = server.handle_request(
            {
                "id": "1",
                "method": "rollback.restore",
                "params": {"session_id": "sid", "hash": "abc"},
            }
        )
        assert resp.get("error"), "full-history rollback should reject while running"
        assert resp["error"]["code"] == 4009
    finally:
        server._sessions.pop("sid", None)


def test_prompt_submit_history_version_mismatch_surfaces_warning(monkeypatch):
    """Fix for TUI silent-drop #2: the defensive backstop at prompt.submit
    must attach a 'warning' to message.complete when history was
    mutated externally during the turn (instead of silently dropping
    the agent's output)."""
    # Agent bumps history_version itself mid-run to simulate an external
    # mutation slipping past the guards.
    session_ref = {"s": None}

    class _RacyAgent:
        def run_conversation(self, prompt, conversation_history=None, stream_callback=None, **_kwargs):
            # Simulate: something external bumped history_version
            # while we were running.
            with session_ref["s"]["history_lock"]:
                session_ref["s"]["history_version"] += 1
            return {
                "final_response": "agent reply",
                "messages": [{"role": "assistant", "content": "agent reply"}],
            }

    class _ImmediateThread:
        def __init__(self, target=None, daemon=None, **_thread_options):
            self._target = target

        def start(self):
            self._target()

    server._sessions["sid"] = _session(agent=_RacyAgent())
    session_ref["s"] = server._sessions["sid"]
    emits: list[tuple] = []
    try:
        monkeypatch.setattr(server.threading, "Thread", _ImmediateThread)
        monkeypatch.setattr(server, "_get_usage", lambda _a: {})
        monkeypatch.setattr(server, "render_message", lambda _t, _c: "")
        monkeypatch.setattr(server, "_emit", lambda *a: emits.append(a))

        resp = server.handle_request(
            {
                "id": "1",
                "method": "prompt.submit",
                "params": {"session_id": "sid", "text": "hi"},
            }
        )
        assert resp.get("result"), f"got error: {resp.get('error')}"

        # History should NOT contain the agent's output (version mismatch)
        assert server._sessions["sid"]["history"] == []

        # message.complete must carry a 'warning' so the UI / operator
        # knows the output was not persisted.
        complete_calls = [a for a in emits if a[0] == "message.complete"]
        assert len(complete_calls) == 1
        _, _, payload = complete_calls[0]
        assert "warning" in payload, (
            "message.complete must include a 'warning' field on "
            "history_version mismatch — otherwise the UI silently "
            "shows output that was never persisted"
        )
        assert (
            "not saved" in payload["warning"].lower()
            or "changed" in payload["warning"].lower()
        )
    finally:
        server._sessions.pop("sid", None)


def test_prompt_submit_merges_on_model_switch_marker(monkeypatch):
    """#76870: when a model-switch marker is the only history mutation during
    a turn, the agent's output must be merged into the current history (which
    now contains the marker) instead of being discarded.

    This test covers BOTH cases:
    - No prior marker in turn-start history (first switch in a session)
    - Prior marker existed (every subsequent switch — the original PR #77274
      fix was dead code here because _append_model_switch_marker strips the
      old marker before appending the new one, producing a net-zero length
      delta that the positional slice missed).
    """
    from hermes_runtime.server import _MODEL_SWITCH_MARKER_PREFIX

    session_ref = {"s": None}

    def _make_marker(model: str) -> dict:
        return {
            "role": "user",
            "content": f"{_MODEL_SWITCH_MARKER_PREFIX}{model}.]",
            "display_kind": "model_switch",
        }

    class _MarkerAgent:
        def __init__(self, new_history_state: list):
            self._new_history_state = new_history_state

        def run_conversation(self, prompt, conversation_history=None, stream_callback=None, **_kwargs):
            # Simulate _append_model_switch_marker: strip prior markers, append new one.
            with session_ref["s"]["history_lock"]:
                hist = session_ref["s"]["history"]
                hist[:] = [h for h in hist if not _is_marker(h)]
                hist.append(_make_marker("new-model"))
                session_ref["s"]["history_version"] += 1
            # result["messages"] = conversation_history + user msg + assistant reply
            return {
                "final_response": "agent reply",
                "messages": list(conversation_history) + [
                    {"role": "user", "content": "hi"},
                    {"role": "assistant", "content": "agent reply"},
                ],
            }

    def _is_marker(entry) -> bool:
        from hermes_runtime.server import _is_model_switch_marker
        return _is_model_switch_marker(entry)

    class _ImmediateThread:
        def __init__(self, target=None, daemon=None, **_thread_options):
            self._target = target

        def start(self):
            self._target()

    # Test both: no prior marker, and prior marker present
    for label, prior_history in [
        ("no prior marker", [{"role": "user", "content": "hello"}]),
        ("with prior marker", [
            {"role": "user", "content": "hello"},
            _make_marker("old-model"),
            {"role": "assistant", "content": "hi there"},
        ]),
    ]:
        server._sessions["sid"] = _session(
            agent=_MarkerAgent([]),
            history=list(prior_history),
        )
        session_ref["s"] = server._sessions["sid"]
        emits: list[tuple] = []
        try:
            monkeypatch.setattr(server.threading, "Thread", _ImmediateThread)
            monkeypatch.setattr(server, "_get_usage", lambda _a: {})
            monkeypatch.setattr(server, "render_message", lambda _t, _c: "")
            monkeypatch.setattr(server, "_emit", lambda *a: emits.append(a))

            resp = server.handle_request(
                {
                    "id": "1",
                    "method": "prompt.submit",
                    "params": {"session_id": "sid", "text": "hi"},
                }
            )
            assert resp.get("result"), f"[{label}] got error: {resp.get('error')}"

            final_history = server._sessions["sid"]["history"]

            # The agent's new messages must be present in the persisted history.
            assistant_msgs = [
                e for e in final_history
                if isinstance(e, dict) and e.get("role") == "assistant"
                and e.get("content") == "agent reply"
            ]
            assert len(assistant_msgs) == 1, (
                f"[{label}] agent output was not merged into history "
                f"(got {len(assistant_msgs)} assistant 'agent reply' messages)"
            )

            # The model-switch marker must be present.
            markers = [e for e in final_history if _is_marker(e)]
            assert len(markers) == 1, (
                f"[{label}] expected exactly 1 model-switch marker, got {len(markers)}"
            )
            assert "new-model" in markers[0]["content"]

            # No warning should be surfaced — the merge succeeded.
            complete_calls = [a for a in emits if a[0] == "message.complete"]
            assert len(complete_calls) == 1
            _, _, payload = complete_calls[0]
            assert "warning" not in payload, (
                f"[{label}] merge path should not surface a warning"
            )
        finally:
            server._sessions.pop("sid", None)


def test_prompt_submit_merges_on_personality_pivot_marker(monkeypatch):
    """A personality pivot injected mid-turn must merge like a model switch.

    `/personality` applies immediately — there is no deferred queue for it the
    way `pending_model_switch` defers a mid-turn model change — so choosing a
    personality while a turn is running bumps `history_version` from the RPC
    thread. The mid-turn reconciliation only recognized the model-switch
    marker, so the pivot read as a genuine desync and the finished turn was
    dropped from session history: the user saw the reply and it was never
    stored (#82756).
    """
    session_ref: dict[str, dict | None] = {"s": None}

    class _PivotAgent:
        def run_conversation(
            self, prompt, conversation_history=None, stream_callback=None, **_kwargs
        ):
            # Real injection point, mid-turn, exactly as the personality RPC
            # would reach it from the other thread.
            server._apply_personality_to_session(
                "sid", session_ref["s"], "Answer tersely.", "terse"
            )
            return {
                "final_response": "agent reply",
                "messages": list(conversation_history)
                + [
                    {"role": "user", "content": "hi"},
                    {"role": "assistant", "content": "agent reply"},
                ],
            }

    class _ImmediateThread:
        def __init__(self, target=None, daemon=None, **_thread_options):
            self._target = target

        def start(self):
            self._target()

    server._sessions["sid"] = _session(
        agent=_PivotAgent(),
        history=[{"role": "user", "content": "hello"}],
    )
    session_ref["s"] = server._sessions["sid"]
    emits: list[tuple] = []
    try:
        monkeypatch.setattr(server.threading, "Thread", _ImmediateThread)
        monkeypatch.setattr(server, "_get_usage", lambda _a: {})
        monkeypatch.setattr(server, "render_message", lambda _t, _c: "")
        monkeypatch.setattr(server, "_session_info", lambda *a, **k: {})
        monkeypatch.setattr(server, "_emit", lambda *a: emits.append(a))

        resp = server.handle_request(
            {
                "id": "1",
                "method": "prompt.submit",
                "params": {"session_id": "sid", "text": "hi"},
            }
        )
        assert resp.get("result"), f"got error: {resp.get('error')}"

        final_history = server._sessions["sid"]["history"]

        assistant_msgs = [
            e
            for e in final_history
            if isinstance(e, dict)
            and e.get("role") == "assistant"
            and e.get("content") == "agent reply"
        ]
        assert len(assistant_msgs) == 1, (
            "the personality pivot discarded the finished turn instead of "
            f"merging it (got {len(assistant_msgs)} assistant replies)"
        )

        pivots = [
            e
            for e in final_history
            if isinstance(e, dict) and e.get("display_kind") == "personality_switch"
        ]
        assert len(pivots) == 1, f"expected exactly 1 pivot, got {len(pivots)}"

        complete_calls = [a for a in emits if a[0] == "message.complete"]
        assert len(complete_calls) == 1
        _, _, payload = complete_calls[0]
        assert "warning" not in payload, "merge path should not surface a warning"
    finally:
        server._sessions.pop("sid", None)


def test_prompt_submit_sanitizes_bracketed_paste_before_agent(monkeypatch):
    """prompt.submit must sanitize corrupted user text before run_conversation."""
    captured: dict[str, str] = {}

    class _Agent:
        def run_conversation(self, prompt, conversation_history=None, stream_callback=None, **_kwargs):
            captured["prompt"] = prompt
            return {
                "final_response": "ok",
                "messages": [{"role": "assistant", "content": "ok"}],
            }

    class _ImmediateThread:
        def __init__(self, target=None, daemon=None, **kw):
            self._target = target

        def start(self):
            self._target()

    corrupted = "hello[" + "~[[e" * 8
    server._sessions["sid"] = _session(agent=_Agent())
    try:
        monkeypatch.setattr(server.threading, "Thread", _ImmediateThread)
        monkeypatch.setattr(server, "_get_usage", lambda _a: {})
        monkeypatch.setattr(server, "render_message", lambda _t, _c: "")
        monkeypatch.setattr(server, "_emit", lambda *a: None)
        monkeypatch.setattr(server, "_start_agent_build", lambda *a, **k: None)
        monkeypatch.setattr(server, "_ensure_session_db_row", lambda *a, **k: None)
        monkeypatch.setattr(server, "_persist_branch_seed", lambda *a, **k: None)

        resp = server.handle_request(
            {
                "id": "1",
                "method": "prompt.submit",
                "params": {"session_id": "sid", "text": corrupted},
            }
        )
        assert resp.get("result"), f"got error: {resp.get('error')}"
        assert captured["prompt"] == "hello"
    finally:
        server._sessions.pop("sid", None)


def test_prompt_submit_history_version_match_persists_normally(monkeypatch):
    """Regression guard: the backstop does not affect the happy path."""

    class _Agent:
        def run_conversation(self, prompt, conversation_history=None, stream_callback=None, **_kwargs):
            return {
                "final_response": "reply",
                "messages": [{"role": "assistant", "content": "reply"}],
            }

    class _ImmediateThread:
        def __init__(self, target=None, daemon=None, **_thread_options):
            self._target = target

        def start(self):
            self._target()

    server._sessions["sid"] = _session(agent=_Agent())
    emits: list[tuple] = []
    try:
        monkeypatch.setattr(server.threading, "Thread", _ImmediateThread)
        monkeypatch.setattr(server, "_get_usage", lambda _a: {})
        monkeypatch.setattr(server, "render_message", lambda _t, _c: "")
        monkeypatch.setattr(server, "_emit", lambda *a: emits.append(a))

        resp = server.handle_request(
            {
                "id": "1",
                "method": "prompt.submit",
                "params": {"session_id": "sid", "text": "hi"},
            }
        )
        assert resp.get("result")

        # History was written
        assert server._sessions["sid"]["history"] == [
            {"role": "assistant", "content": "reply"}
        ]
        assert server._sessions["sid"]["history_version"] == 1

        # No warning should be attached
        complete_calls = [a for a in emits if a[0] == "message.complete"]
        assert len(complete_calls) == 1
        _, _, payload = complete_calls[0]
        assert "warning" not in payload
    finally:
        server._sessions.pop("sid", None)


def test_prompt_submit_snapshots_history_after_pending_model_switch(monkeypatch):
    marker = {"role": "user", "content": "[model switched]"}
    seen = {}

    class _Agent:
        def run_conversation(self, prompt, conversation_history=None, **_kwargs):
            seen["history"] = conversation_history
            return {
                "final_response": "reply",
                "messages": [
                    *(conversation_history or []),
                    {"role": "user", "content": prompt},
                    {"role": "assistant", "content": "reply"},
                ],
            }

    class _ImmediateThread:
        def __init__(self, target=None, **_kwargs):
            self._target = target

        def start(self):
            self._target()

    def _apply_pending(_sid, session):
        with session["history_lock"]:
            session["history"].append(marker)
            session["history_version"] += 1

    server._sessions["sid"] = _session(agent=_Agent())
    server._sessions["sid"]["pending_model_switch"] = {"raw": "new-model"}
    emits = []
    try:
        monkeypatch.setattr(server.threading, "Thread", _ImmediateThread)
        monkeypatch.setattr(server, "_apply_pending_model_switch", _apply_pending)
        monkeypatch.setattr(server, "_sync_agent_model_with_config", lambda *_a: None)
        monkeypatch.setattr(server, "_get_usage", lambda _a: {})
        monkeypatch.setattr(server, "render_message", lambda *_a: "")
        monkeypatch.setattr(server, "_emit", lambda *a: emits.append(a))

        server.handle_request(
            {"id": "1", "method": "prompt.submit", "params": {"session_id": "sid", "text": "hi"}}
        )

        assert seen["history"] == [marker]
        assert server._sessions["sid"]["history"][-1] == {
            "role": "assistant", "content": "reply"
        }
        complete = [a for a in emits if a[0] == "message.complete"]
        assert "warning" not in complete[0][2]
    finally:
        server._sessions.pop("sid", None)


def test_prompt_submit_can_truncate_before_user_ordinal(monkeypatch):
    """Desktop user-message edits should restart the turn from the edited user."""

    seen = {}

    class _Agent:
        def run_conversation(self, prompt, conversation_history=None, stream_callback=None, **_kwargs):
            seen["prompt"] = prompt
            seen["history"] = conversation_history
            return {
                "final_response": "edited reply",
                "messages": [
                    *(conversation_history or []),
                    {"role": "user", "content": prompt},
                    {"role": "assistant", "content": "edited reply"},
                ],
            }

    class _ImmediateThread:
        def __init__(self, target=None, daemon=None, **_thread_options):
            self._target = target

        def start(self):
            self._target()

    original_history = [
        {"role": "user", "content": "first"},
        {"role": "assistant", "content": "first reply"},
        {"role": "user", "content": "second"},
        {"role": "assistant", "content": "second reply"},
    ]
    server._sessions["sid"] = _session(agent=_Agent(), history=original_history)

    class _StubDb:
        def __init__(self):
            self.replaced = []

        def get_messages_as_conversation(self, *_args, **_kwargs):
            return []

        def replace_messages(
            self,
            session_id,
            messages,
            active_only=False,
            archive_dropped=False,
            reject_active_turn_lease=False,
        ):
            self.replaced.append((session_id, list(messages)))

    stub_db = _StubDb()

    try:
        monkeypatch.setattr(server.threading, "Thread", _ImmediateThread)
        monkeypatch.setattr(server, "_get_usage", lambda _a: {})
        monkeypatch.setattr(server, "render_message", lambda _t, _c: "")
        monkeypatch.setattr(server, "_emit", lambda *a: None)
        monkeypatch.setattr(server, "_get_db", lambda: stub_db)

        resp = server.handle_request(
            {
                "id": "1",
                "method": "prompt.submit",
                "params": {
                    "session_id": "sid",
                    "text": "edited second",
                    "truncate_before_user_ordinal": 1,
                    "confirm_truncate": True,
                },
            }
        )
        assert resp.get("result"), f"got error: {resp.get('error')}"

        assert seen["prompt"] == "edited second"
        assert seen["history"] == original_history[:2]
        assert server._sessions["sid"]["history"] == [
            *original_history[:2],
            {"role": "user", "content": "edited second"},
            {"role": "assistant", "content": "edited reply"},
        ]
        assert server._sessions["sid"]["history_version"] == 2
        assert stub_db.replaced == [("session-key", original_history[:2])]
    finally:
        server._sessions.pop("sid", None)


def test_prompt_submit_truncation_signals_busy_instead_of_queueing(monkeypatch):
    """#113942: editing a message while the turn it belongs to is still running

    must not be silently absorbed as a steered correction or a plain follow-up
    queued to run AFTER the live turn — both would drop the truncation and
    leave the original (un-edited) turn's reply intact, which reads to the
    user as "my edit was rejected". It must surface as the same "session busy"
    (4009) the desktop client's edit path already retries through (interrupt,
    then poll) until the turn actually clears.
    """
    agent = types.SimpleNamespace()  # no redirect/steer support -> would fall to queue+interrupt
    server._sessions["sid"] = _session(agent=agent, running=True)

    try:
        resp = server.handle_request(
            {
                "id": "1",
                "method": "prompt.submit",
                "params": {
                    "session_id": "sid",
                    "text": "edited while thinking",
                    "truncate_before_user_ordinal": 0,
                    "confirm_truncate": True,
                    "confirm_empty_truncate": True,
                },
            }
        )

        assert resp.get("error", {}).get("code") == 4009
        # The edit must not have been silently accepted as a queued follow-up.
        assert server._sessions["sid"].get("queued_prompt") is None
        assert server._sessions["sid"]["running"] is True
    finally:
        server._sessions.pop("sid", None)


def test_prompt_submit_truncation_refuses_redirect_of_live_turn():
    """The redirect-capable agent (the Desktop's normal case) is the branch that silently
    absorbed the edit: `busy_input_mode=interrupt` turned the rewind into a mid-turn
    redirect and left the un-edited transcript in place."""
    redirected = []
    agent = types.SimpleNamespace(
        _supports_active_turn_redirect=True, redirect=lambda text: redirected.append(text) or True)
    server._sessions["sid"] = _session(agent=agent, running=True, history=[{"role": "user", "content": "original"}])

    try:
        resp = server.handle_request({
            "id": "1", "method": "prompt.submit",
            "params": {"session_id": "sid", "text": "edited while thinking", "truncate_before_user_ordinal": 0,
                       "confirm_truncate": True, "confirm_empty_truncate": True}})

        assert resp.get("error", {}).get("code") == 4009
        assert redirected == []
        assert server._sessions["sid"]["history"] == [{"role": "user", "content": "original"}]
    finally:
        server._sessions.pop("sid", None)


def test_prompt_submit_refuses_turn_when_truncate_persist_fails(monkeypatch):
    """If replace_messages fails during edit/regenerate truncate, do not run the turn.

    Memory-first + fail-open left session['history'] short while state.db kept
    the old tail. The agent flush then appends the new exchange on top of the
    'undone' turns — durable zombie history. Write first; on failure leave
    memory and DB unchanged and return 5008.
    """
    original_history = [
        {"role": "user", "content": "first"},
        {"role": "assistant", "content": "first reply"},
        {"role": "user", "content": "second"},
        {"role": "assistant", "content": "second reply"},
    ]
    sess = _session(history=list(original_history))
    server._sessions["trunc-fail-sid"] = sess

    class _FailDb:
        def get_messages_as_conversation(self, *_args, **_kwargs):
            return []

        def replace_messages(
            self,
            session_id,
            messages,
            active_only=False,
            archive_dropped=False,
            reject_active_turn_lease=False,
        ):
            raise OSError("disk full")

    monkeypatch.setattr(server, "_get_db", lambda: _FailDb())
    monkeypatch.setattr(
        server, "_start_inflight_turn", lambda *a, **k: pytest.fail("must not start a turn")
    )
    monkeypatch.setattr(
        server, "_start_agent_build", lambda *a, **k: pytest.fail("must not start a turn")
    )

    try:
        resp = server.handle_request(
            {
                "id": "1",
                "method": "prompt.submit",
                "params": {
                    "session_id": "trunc-fail-sid",
                    "text": "edited second",
                    "truncate_before_user_ordinal": 1,
                    "confirm_truncate": True,
                },
            }
        )
        assert "error" in resp
        assert resp["error"]["code"] == 5008
        # Memory left intact — same list contents as before the refused cut.
        assert sess["history"] == original_history
        assert sess["history_version"] == 0
        assert sess.get("running") is not True
    finally:
        server._sessions.pop("trunc-fail-sid", None)


def test_prompt_submit_truncate_ordinal_skips_display_kind_rows(monkeypatch):
    """truncate_before_user_ordinal must count only real user turns.

    display_kind timeline rows (model_switch, async_delegation_complete, …)
    are role=user but no client counts them as user turns. Without the
    filter, a trailing marker shifts the ordinal so the wrong message is
    targeted for truncation.
    """

    seen = {}

    class _Agent:
        def run_conversation(self, prompt, conversation_history=None, stream_callback=None, **_kwargs):
            seen["prompt"] = prompt
            seen["history"] = conversation_history
            return {
                "final_response": "reply",
                "messages": [
                    *(conversation_history or []),
                    {"role": "user", "content": prompt},
                    {"role": "assistant", "content": "reply"},
                ],
            }

    class _ImmediateThread:
        def __init__(self, target=None, daemon=None, **_thread_options):
            self._target = target

        def start(self):
            self._target()

    original_history = [
        {"role": "user", "content": "first"},
        {"role": "assistant", "content": "first reply"},
        {"role": "user", "content": "second"},
        {"role": "assistant", "content": "second reply"},
        {
            "role": "user",
            "content": "background agent finished",
            "display_kind": "async_delegation_complete",
        },
    ]
    server._sessions["sid"] = _session(agent=_Agent(), history=original_history)

    class _StubDb:
        def __init__(self):
            self.replaced = []

        def get_messages_as_conversation(self, *_args, **_kwargs):
            return []

        def replace_messages(
            self,
            session_id,
            messages,
            active_only=False,
            archive_dropped=False,
            reject_active_turn_lease=False,
        ):
            self.replaced.append((session_id, list(messages)))

    stub_db = _StubDb()

    try:
        monkeypatch.setattr(server.threading, "Thread", _ImmediateThread)
        monkeypatch.setattr(server, "_get_usage", lambda _a: {})
        monkeypatch.setattr(server, "render_message", lambda _t, _c: "")
        monkeypatch.setattr(server, "_emit", lambda *a: None)
        monkeypatch.setattr(server, "_get_db", lambda: stub_db)

        # ordinal=1 means "truncate before the 2nd-from-last real user turn"
        # which is "first". The display_kind marker must NOT shift the ordinal.
        resp = server.handle_request(
            {
                "id": "1",
                "method": "prompt.submit",
                "params": {
                    "session_id": "sid",
                    "text": "edited first",
                    "truncate_before_user_ordinal": 1,
                    "confirm_truncate": True,
                },
            }
        )
        assert resp.get("result"), f"got error: {resp.get('error')}"

        # With display_kind filter: user_indices = [0, 2] (indices of "first" and "second").
        # ordinal=1 → user_indices[1] = 2, truncated = history[:2] = [first, first reply].
        # Without the filter: user_indices = [0, 2, 4] (includes the marker),
        # ordinal=1 → user_indices[1] = 2, same result by luck — but ordinal=0
        # would truncate to history[:0] vs history[:0], and higher ordinals shift.
        assert seen["history"] == original_history[:2], (
            f"Expected truncation to first 2 messages, got {seen['history']}"
        )
        assert stub_db.replaced == [("session-key", original_history[:2])], (
            f"Expected DB replace with first 2 messages, got {stub_db.replaced}"
        )
    finally:
        server._sessions.pop("sid", None)


def test_prompt_submit_truncate_translates_display_prefix_ordinal(monkeypatch):
    """Full-lineage Desktop ordinals must truncate the tip segment (#82462).

    After compression, session["history"] is the tip while display_history_prefix
    still holds ancestor user turns the UI shows. A client ordinal that includes
    those ancestors must map into the tip, not 4018.
    """

    seen = {}

    class _Agent:
        def run_conversation(self, prompt, conversation_history=None, stream_callback=None, **_kwargs):
            seen["prompt"] = prompt
            seen["history"] = conversation_history
            return {
                "final_response": "edited reply",
                "messages": [
                    *(conversation_history or []),
                    {"role": "user", "content": prompt},
                    {"role": "assistant", "content": "edited reply"},
                ],
            }

    class _ImmediateThread:
        def __init__(self, target=None, daemon=None, **_thread_options):
            self._target = target

        def start(self):
            self._target()

    tip_history = [
        {"role": "user", "content": "post-compress A"},
        {"role": "assistant", "content": "reply A"},
        {"role": "user", "content": "post-compress B"},
        {"role": "assistant", "content": "reply B"},
    ]
    display_prefix = [
        {"role": "user", "content": "pre-compress 1"},
        {"role": "assistant", "content": "pre reply 1"},
        {"role": "user", "content": "pre-compress 2"},
        {"role": "assistant", "content": "pre reply 2"},
    ]
    # Desktop lineage: pre1=0, pre2=1, postA=2, postB=3
    desktop_ordinal_for_post_b = 3

    server._sessions["sid"] = _session(
        agent=_Agent(),
        history=tip_history,
        display_history_prefix=display_prefix,
    )

    class _StubDb:
        def __init__(self):
            self.replaced = []

        def get_messages_as_conversation(self, *_args, **_kwargs):
            # Empty durable transcript: proves this ephemeral-style session
            # may take the ordinal-only path past the durability gate.
            return []

        def replace_messages(
            self,
            session_id,
            messages,
            active_only=False,
            archive_dropped=False,
            reject_active_turn_lease=False,
        ):
            assert reject_active_turn_lease is True
            self.replaced.append((session_id, list(messages)))

    stub_db = _StubDb()

    try:
        monkeypatch.setattr(server.threading, "Thread", _ImmediateThread)
        monkeypatch.setattr(server, "_get_usage", lambda _a: {})
        monkeypatch.setattr(server, "render_message", lambda _t, _c: "")
        monkeypatch.setattr(server, "_emit", lambda *a: None)
        monkeypatch.setattr(server, "_get_db", lambda: stub_db)

        resp = server.handle_request(
            {
                "id": "1",
                "method": "prompt.submit",
                "params": {
                    "session_id": "sid",
                    "text": "edited post B",
                    "truncate_before_user_ordinal": desktop_ordinal_for_post_b,
                    "confirm_truncate": True,
                },
            }
        )
        assert resp.get("result"), f"got error: {resp.get('error')}"
        assert seen["prompt"] == "edited post B"
        assert seen["history"] == tip_history[:2]
        assert stub_db.replaced == [("session-key", tip_history[:2])]
    finally:
        server._sessions.pop("sid", None)


def test_prompt_submit_truncate_oor_includes_structured_user_turn_count(monkeypatch):
    """4018 after compaction must carry recovery fields for Desktop (#82462)."""

    tip_history = [
        {"role": "user", "content": "post-compress A"},
        {"role": "assistant", "content": "reply A"},
    ]
    display_prefix = [
        {"role": "user", "content": "pre-compress 1"},
        {"role": "assistant", "content": "pre reply 1"},
    ]
    # Ordinal 0 points at the ancestor prefix — not editable from the tip.
    server._sessions["sid"] = _session(
        history=tip_history,
        display_history_prefix=display_prefix,
    )

    try:
        monkeypatch.setattr(
            server, "_start_inflight_turn", lambda *a, **k: pytest.fail("must not start a turn")
        )
        resp = server.handle_request(
            {
                "id": "1",
                "method": "prompt.submit",
                "params": {
                    "session_id": "sid",
                    "text": "edit ancestor",
                    "truncate_before_user_ordinal": 0,
                    "confirm_truncate": True,
                },
            }
        )
        err = resp.get("error") or {}
        assert err.get("code") == 4018
        data = err.get("data") or {}
        assert data.get("user_turn_count") == 1
        assert data.get("ordinal") == 0
        assert data.get("segment_ordinal") == -1
        assert data.get("prefix_user_count") == 1
    finally:
        server._sessions.pop("sid", None)


def test_prompt_submit_row_id_accepts_full_lineage_ordinal(monkeypatch):
    """Desktop sends rowId + a full-lineage ordinal; both must agree (#82462).

    After compression the client's visible-user ordinal counts the ancestor
    prefix turns while the gateway resolves the row id tip-relative. The
    reconcile cross-check must treat `tip_ordinal + prefix_user_count` as
    agreement, not #82756 drift — and the cut stays aimed by the row id.
    """
    from agent.context_compressor import (
        HISTORICAL_TASK_HEADING,
        SUMMARY_PREFIX,
        _SUMMARY_END_MARKER,
    )

    tip_history = [
        {"_row_id": 501, "role": "user", "content": "post-compress A"},
        {"_row_id": 502, "role": "assistant", "content": "reply A"},
        {"_row_id": 503, "role": "user", "content": "post-compress B"},
        {"_row_id": 504, "role": "assistant", "content": "reply B"},
    ]
    display_prefix = [
        {"role": "user", "content": "pre-compress 1"},
        {"role": "assistant", "content": "pre reply 1"},
        {
            # Legacy pure handoffs did not always carry display_kind=hidden.
            # They are physically user rows but not visible/user-originated
            # turns, so they must not shift the Desktop lineage ordinal.
            "role": "user",
            "content": (
                f"{SUMMARY_PREFIX}\n{HISTORICAL_TASK_HEADING}\nold task\n\n"
                f"{_SUMMARY_END_MARKER}"
            ),
        },
        {"role": "user", "content": "pre-compress 2"},
        {"role": "assistant", "content": "pre reply 2"},
    ]

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
            assert reject_active_turn_lease is True
            replaced.append((key, list(messages)))

    sess = _session(history=list(tip_history), display_history_prefix=display_prefix)
    server._sessions["lineage-row-sid"] = sess
    monkeypatch.setattr(server, "_get_db", lambda: _FakeDB())
    monkeypatch.setattr(server, "_start_agent_build", lambda *a, **k: None)

    try:
        resp = server.handle_request(
            {
                "id": "1",
                "method": "prompt.submit",
                "params": {
                    "session_id": "lineage-row-sid",
                    "text": "edited post B",
                    # Row id resolves to tip ordinal 1; the client counted the
                    # 2 ancestor user turns, so its lineage ordinal is 3.
                    "truncate_before_row_id": 503,
                    "truncate_before_user_ordinal": 3,
                    "confirm_truncate": True,
                },
            }
        )
        assert resp.get("error") is None, f"got error: {resp.get('error')}"
        assert sess["history"] == tip_history[:2]
    finally:
        # Release the slot the first turn claimed, as _finalize_session does in
        # production. Popping alone leaks the lease, and the second session below
        # uses the same session_key -- so without this the test fences itself out
        # of its own key and never reaches the mismatch it is checking.
        server._release_active_session_slot(sess)
        server._sessions.pop("lineage-row-sid", None)

    # A genuinely stale ordinal (matches neither the tip space nor the
    # lineage space) must still refuse with the #82756 mismatch.
    sess2 = _session(history=list(tip_history), display_history_prefix=display_prefix)
    server._sessions["lineage-row-sid-2"] = sess2
    monkeypatch.setattr(
        server, "_start_agent_build", lambda *a, **k: pytest.fail("must not start a turn")
    )
    try:
        resp = server.handle_request(
            {
                "id": "2",
                "method": "prompt.submit",
                "params": {
                    "session_id": "lineage-row-sid-2",
                    "text": "edited post B",
                    "truncate_before_row_id": 503,
                    "truncate_before_user_ordinal": 2,
                    "confirm_truncate": True,
                },
            }
        )
        assert resp.get("error") is not None
        assert resp["error"]["code"] == 4030
    finally:
        server._release_active_session_slot(sess2)
        server._sessions.pop("lineage-row-sid-2", None)


def test_interrupt_only_clears_own_session_pending():
    """session.interrupt on session A withdraws A's open server→client requests (a request.cancel each)
    and must NOT touch session B's — otherwise B's clarify/sudo/secret prompt silently resolves as if
    the user cancelled it."""
    import types
    from hermes_runtime import server_requests

    session_a = _session()
    session_a["agent"] = types.SimpleNamespace(interrupt=lambda: None)
    session_b = _session()
    session_b["agent"] = types.SimpleNamespace(interrupt=lambda: None)
    server._sessions["sid_a"] = session_a
    server._sessions["sid_b"] = session_b
    req_a1, req_a2, req_b = _open_request("sid_a"), _open_request("sid_a", "sudo"), _open_request("sid_b")

    try:
        resp = server.handle_request({"id": "1", "method": "session.interrupt", "params": {"session_id": "sid_a"}})
        assert resp.get("result"), f"got error: {resp.get('error')}"

        assert req_a1.event.is_set() and req_a2.event.is_set() and not req_a1.answered
        assert not req_b.event.is_set(), (
            "CRITICAL: session.interrupt on sid_a released a prompt belonging to sid_b")
        assert server_requests.open_requests("sid_a") == []
        assert [r["id"] for r in server_requests.open_requests("sid_b")] == [req_b.id]
    finally:
        server._sessions.pop("sid_a", None)
        server._sessions.pop("sid_b", None)
        server_requests.reset_for_tests()


def test_accepted_interrupt_resumes_wake_mismatch_does_not(monkeypatch):
    """An accepted interrupt re-arms the caller's paused detector; a hosted-task mismatch does not."""
    wake_word, state, pause_listening, resume_listening = _wake_lease_double()
    owner = types.SimpleNamespace(_closed=False)
    state["lease"] = owner
    session = _session(
        agent=types.SimpleNamespace(interrupt=lambda: None),
        running=True,
        _hosted_room_task={"task_id": "active"},
    )
    server._sessions["sid"] = session
    monkeypatch.setattr(wake_word, "pause_listening", pause_listening)
    monkeypatch.setattr(wake_word, "resume_listening", resume_listening)
    monkeypatch.setattr(server, "_voice_wake_owner", None)
    try:
        paused = _dispatch_sync(
            {"id": "pause", "method": "wake.pause", "params": {}}, transport=owner
        )
        mismatch = _dispatch_sync(
            {
                "id": "mismatch",
                "method": "session.interrupt",
                "params": {"session_id": "sid", "expected_hosted_task_id": "stale"},
            },
            transport=owner,
        )
        assert paused["result"]["paused"] is True
        assert mismatch["result"] == {"status": "not_interrupted", "interrupted": False}
        assert state["paused"] is True
        assert state["resumed"] == []

        accepted = _dispatch_sync(
            {"id": "ok", "method": "session.interrupt", "params": {"session_id": "sid"}},
            transport=owner,
        )
        assert accepted.get("result", {}).get("status") == "interrupted"
        assert state == {"lease": owner, "paused": False, "resumed": [owner]}
    finally:
        server._sessions.pop("sid", None)


def test_interrupt_error_after_tts_stop_still_resumes_wake(monkeypatch):
    """TTS is cut before session lookup and the compute-host call; a later error must still re-arm."""
    wake_word, state, pause_listening, resume_listening = _wake_lease_double()
    owner = types.SimpleNamespace(_closed=False)
    state["lease"] = owner
    session = _session(agent=types.SimpleNamespace(interrupt=lambda: None), running=True)
    server._sessions["sid"] = session
    monkeypatch.setattr(wake_word, "pause_listening", pause_listening)
    monkeypatch.setattr(wake_word, "resume_listening", resume_listening)
    monkeypatch.setattr(server, "_voice_wake_owner", None)
    monkeypatch.setattr(server, "_session_uses_compute_host", lambda _session: True)

    def _host_failed(*_args, **_kwargs):
        raise RuntimeError("host failed")

    monkeypatch.setattr(server, "_interrupt_session_turn", _host_failed)
    try:
        paused = _dispatch_sync(
            {"id": "pause", "method": "wake.pause", "params": {}}, transport=owner
        )
        missing = _dispatch_sync(
            {"id": "missing", "method": "session.interrupt", "params": {"session_id": "gone"}},
            transport=owner,
        )
        assert paused["result"]["paused"] is True
        assert missing["error"]["code"] == 4001
        assert state["paused"] is False
        assert state["resumed"] == [owner]

        _dispatch_sync({"id": "pause-2", "method": "wake.pause", "params": {}}, transport=owner)
        failed = _dispatch_sync(
            {"id": "host", "method": "session.interrupt", "params": {"session_id": "sid"}},
            transport=owner,
        )
        assert failed["result"] == {"status": "interrupted", "turn_isolation": True}
        assert state["paused"] is False
        assert state["resumed"] == [owner, owner]
    finally:
        server._sessions.pop("sid", None)


def test_interrupt_resumes_voice_wake_owner_not_a_foreign_lease(monkeypatch):
    """Interrupt resumes a voice-owned pause even when the caller differs, and leaves a foreign lease paused."""
    wake_word, state, pause_listening, resume_listening = _wake_lease_double()
    voice_owner = types.SimpleNamespace(_closed=False, role="voice")
    caller = types.SimpleNamespace(_closed=False, role="caller")
    foreign = types.SimpleNamespace(_closed=False, role="foreign")
    state["lease"] = voice_owner
    session = _session(agent=types.SimpleNamespace(interrupt=lambda: None), running=True)
    server._sessions["sid"] = session
    monkeypatch.setattr(wake_word, "pause_listening", pause_listening)
    monkeypatch.setattr(wake_word, "resume_listening", resume_listening)
    monkeypatch.setattr(server, "_voice_wake_owner", voice_owner)
    try:
        paused = _dispatch_sync(
            {"id": "pause", "method": "wake.pause", "params": {}}, transport=voice_owner
        )
        accepted = _dispatch_sync(
            {"id": "ok", "method": "session.interrupt", "params": {"session_id": "sid"}},
            transport=caller,
        )
        assert paused["result"]["paused"] is True
        assert accepted.get("result", {}).get("status") == "interrupted"
        assert state["paused"] is False
        assert any(item is voice_owner for item in state["resumed"])
        assert all(item is not caller for item in state["resumed"])

        state["lease"] = foreign
        state["paused"] = True
        state["resumed"] = []
        monkeypatch.setattr(server, "_voice_wake_owner", None)
        foreign_held = _dispatch_sync(
            {"id": "foreign", "method": "session.interrupt", "params": {"session_id": "sid"}},
            transport=caller,
        )
        assert foreign_held.get("result", {}).get("status") == "interrupted"
        assert state["paused"] is True
        assert state["resumed"] == []
    finally:
        server._sessions.pop("sid", None)









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


def test_config_set_model_once_keeps_env_and_records_restore(monkeypatch):
    class Agent:
        model = "old/model"
        provider = "openrouter"
        base_url = "https://openrouter.ai/api/v1"
        api_key = "sk-old"
        api_mode = "chat_completions"

        def switch_model(self, **kwargs):
            self.model = kwargs["new_model"]
            self.provider = kwargs["new_provider"]
            self.api_key = kwargs["api_key"]
            self.base_url = kwargs["base_url"]
            self.api_mode = kwargs["api_mode"]

    result = types.SimpleNamespace(
        success=True,
        new_model="claude-sonnet-4.6",
        target_provider="anthropic",
        api_key="sk-ant",
        base_url="https://api.anthropic.com",
        api_mode="anthropic_messages",
        warning_message="",
    )
    seen = {}
    agent = Agent()
    session = _session(agent=agent)
    server._sessions["sid"] = session
    monkeypatch.setenv("HERMES_INFERENCE_PROVIDER", "openrouter")
    monkeypatch.setenv("HERMES_MODEL", "old/model")
    monkeypatch.setattr(
        "hermes_cli.model_switch.switch_model",
        lambda **kwargs: seen.update(kwargs) or result,
    )
    monkeypatch.setattr(server, "_restart_slash_worker", lambda *args, **kwargs: None)
    monkeypatch.setattr(server, "_emit", lambda *args, **kwargs: None)

    try:
        resp = server.handle_request(
            {
                "id": "1",
                "method": "config.set",
                "params": {
                    "session_id": "sid",
                    "key": "model",
                    "value": "claude-sonnet-4.6 --provider anthropic --once",
                },
            }
        )

        assert resp["result"]["scope"] == "once"
        assert seen["is_global"] is False
        assert agent.model == "claude-sonnet-4.6"
        assert session["one_turn_model_restore"]["model"] == "old/model"
        assert os.environ["HERMES_INFERENCE_PROVIDER"] == "openrouter"
        assert os.environ["HERMES_MODEL"] == "old/model"
    finally:
        server._sessions.clear()


def test_config_set_model_once_requires_live_session(monkeypatch):
    monkeypatch.setattr(
        "hermes_cli.model_switch.switch_model",
        lambda **_: (_ for _ in ()).throw(AssertionError("switch should not run")),
    )

    resp = server.handle_request(
        {
            "id": "1",
            "method": "config.set",
            "params": {
                "key": "model",
                "value": "claude-sonnet-4.6 --provider anthropic --once",
            },
        }
    )

    assert resp["error"]["code"] == 5001


def test_config_set_model_sessionless_rejected(monkeypatch):
    """Sessionless config.set model must 4001 before _apply_model_switch.

    Missing session_id and a stale session_id miss both take the sessionless
    branch; unscoped values and legacy --global must be rejected the same way
    so a Desktop client cannot persist model.default before session.create.
    """
    called = {"n": 0}

    def boom(*a, **k):
        called["n"] += 1
        raise AssertionError("_apply_model_switch must not run")

    monkeypatch.setattr(server, "_apply_model_switch", boom)
    for value in ["some-model", "some-model --provider openai-codex --global"]:
        resp = server.handle_request({
            "id": "1", "method": "config.set",
            "params": {"key": "model", "value": value},
        })
        assert resp["error"]["code"] == 4001
        assert called["n"] == 0

    resp = server.handle_request({
        "id": "1", "method": "config.set",
        "params": {"session_id": "missing-sid", "key": "model", "value": "some-model --global"},
    })
    assert resp["error"]["code"] == 4001
    assert called["n"] == 0


def test_config_set_model_live_session_still_applies_switch(monkeypatch):
    """CONTROL: a live session still reaches _apply_model_switch, including --global."""
    called = {"raw": []}

    def fake_apply(sid, session, raw, **_kwargs):
        called["raw"].append(raw)
        return {"value": "some-model", "warning": "", "scope": "global"}

    server._sessions["sid"] = _session()
    monkeypatch.setattr(server, "_apply_model_switch", fake_apply)
    try:
        resp = server.handle_request({
            "id": "1", "method": "config.set",
            "params": {
                "session_id": "sid",
                "key": "model",
                "value": "some-model --provider openai-codex --global",
            },
        })
        assert "error" not in resp
        assert called["raw"] == ["some-model --provider openai-codex --global"]
    finally:
        server._sessions.pop("sid", None)


def test_config_set_model_session_switch_clears_pending_once_restore(monkeypatch):
    class Agent:
        model = "temp/model"
        provider = "anthropic"
        base_url = "https://api.anthropic.com"
        api_key = "sk-temp"
        api_mode = "anthropic_messages"

        def switch_model(self, **kwargs):
            self.model = kwargs["new_model"]
            self.provider = kwargs["new_provider"]
            self.api_key = kwargs["api_key"]
            self.base_url = kwargs["base_url"]
            self.api_mode = kwargs["api_mode"]

    result = types.SimpleNamespace(
        success=True,
        new_model="new/model",
        target_provider="openrouter",
        api_key="sk-new",
        base_url="https://openrouter.ai/api/v1",
        api_mode="chat_completions",
        warning_message="",
    )
    session = _session(agent=Agent())
    session["one_turn_model_restore"] = {"model": "old/model"}
    server._sessions["sid"] = session
    monkeypatch.setattr("hermes_cli.model_switch.switch_model", lambda **_kwargs: result)
    monkeypatch.setattr(server, "_restart_slash_worker", lambda *args, **kwargs: None)
    monkeypatch.setattr(server, "_emit", lambda *args, **kwargs: None)

    try:
        resp = server.handle_request(
            {
                "id": "1",
                "method": "config.set",
                "params": {
                    "session_id": "sid",
                    "key": "model",
                    "value": "new/model --provider openrouter --session",
                },
            }
        )

        assert resp["result"]["scope"] == "session"
        assert "one_turn_model_restore" not in session
    finally:
        server._sessions.clear()


def test_restore_agent_model_runtime_falls_back_to_switch_model():
    class Agent:
        model = "temp/model"
        provider = "anthropic"
        base_url = "https://api.anthropic.com"
        api_key = "sk-temp"
        api_mode = "anthropic_messages"

        def switch_model(self, **kwargs):
            self.model = kwargs["new_model"]
            self.provider = kwargs["new_provider"]
            self.api_key = kwargs["api_key"]
            self.base_url = kwargs["base_url"]
            self.api_mode = kwargs["api_mode"]

    agent = Agent()

    server._restore_agent_model_runtime(
        agent,
        {
            "model": "old/model",
            "provider": "openrouter",
            "api_key": "sk-old",
            "base_url": "https://openrouter.ai/api/v1",
            "api_mode": "chat_completions",
        },
    )

    assert agent.model == "old/model"
    assert agent.provider == "openrouter"
    assert agent.base_url == "https://openrouter.ai/api/v1"


def test_config_set_personality_rejects_unknown_name(monkeypatch):
    monkeypatch.setattr(
        server,
        "_available_personalities",
        lambda cfg=None: {"helpful": "You are helpful."},
    )
    resp = server.handle_request(
        {
            "id": "1",
            "method": "config.set",
            "params": {"key": "personality", "value": "bogus"},
        }
    )

    assert "error" in resp


def test_config_set_personality_preserves_history_and_returns_info(monkeypatch):
    agent = types.SimpleNamespace(
        ephemeral_system_prompt=None, _cached_system_prompt="old"
    )
    session = _session(
        agent=agent,
        history=[{"role": "user", "text": "hi"}],
        history_version=4,
    )
    emits = []
    writes = []

    server._sessions["sid"] = session
    monkeypatch.setattr(
        server,
        "_available_personalities",
        lambda cfg=None: {"helpful": "You are helpful."},
    )
    monkeypatch.setattr(
        server, "_session_info", lambda agent, *a: {"model": getattr(agent, "model", "?")}
    )
    monkeypatch.setattr(server, "_emit", lambda *args: emits.append(args))
    # Persistence now flows through the single owner (hermes_cli.personality),
    # never _write_config_key / agent.system_prompt.
    import hermes_cli.personality as personality_mod

    monkeypatch.setattr(
        personality_mod,
        "persist_personality",
        lambda name: writes.append(("display.personality", name)) or True,
    )
    monkeypatch.setattr(
        server,
        "_write_config_key",
        lambda path, value: writes.append((path, value)),
    )

    resp = server.handle_request(
        {
            "id": "1",
            "method": "config.set",
            "params": {"session_id": "sid", "key": "personality", "value": "helpful"},
        }
    )

    assert resp["result"]["history_reset"] is False
    assert resp["result"]["info"] == {"model": "?"}
    # History is preserved with a pivot marker appended
    assert len(session["history"]) == 2
    assert session["history"][0] == {"role": "user", "text": "hi"}
    assert session["history"][1]["role"] == "user"
    assert "personality" in session["history"][1]["content"].lower()
    assert "You are helpful." in session["history"][1]["content"]
    assert session["history_version"] == 5
    # Agent's system prompt was updated in-place; cached prompt untouched
    assert agent.ephemeral_system_prompt == "You are helpful."
    assert agent._cached_system_prompt == "old"
    assert ("session.info", "sid", {"model": "?"}) in emits
    assert ("display.personality", "helpful") in writes
    assert not any(path == "agent.system_prompt" for path, _ in writes)


def test_compress_session_history_passes_force():
    """_compress_session_history is manual-only (session.compress RPC, slash
    compress/compact, slash-worker mirror) — it must bypass the
    summary-failure cooldown via force=True, matching the CLI and gateway
    manual-compress handlers."""
    from unittest.mock import MagicMock

    agent = MagicMock()
    agent.context_compressor = None  # keep _get_usage on the simple path
    compressed = [{"role": "user", "content": "summary"}]
    agent._compress_context.return_value = (compressed, "")
    # Explicit non-lock-skip: MagicMock getattr would return a truthy mock.
    agent._compression_skipped_due_to_lock = False
    session = _session(
        agent=agent,
        history=[
            {"role": "user", "content": "one"},
            {"role": "assistant", "content": "two"},
            {"role": "user", "content": "three"},
            {"role": "assistant", "content": "four"},
        ],
    )

    removed, _usage = server._compress_session_history(session)

    assert removed == 3
    assert session["history"] == compressed
    assert agent._compress_context.call_args.kwargs.get("force") is True


def test_compress_session_history_works_when_auto_compaction_disabled():
    """compression.enabled: false disables *automatic* compaction only —
    manual /compress must still work on every TUI route (session.compress
    RPC, slash compress/compact, slash-worker mirror), all of which converge
    on _compress_session_history. Pin that the helper never gates on
    agent.compression_enabled (#64438)."""
    from unittest.mock import MagicMock

    agent = MagicMock()
    agent.compression_enabled = False
    agent.context_compressor = None  # keep _get_usage on the simple path
    compressed = [{"role": "user", "content": "summary"}]
    agent._compress_context.return_value = (compressed, "")
    # Explicit non-lock-skip: MagicMock getattr would return a truthy mock.
    agent._compression_skipped_due_to_lock = False
    session = _session(
        agent=agent,
        history=[
            {"role": "user", "content": "one"},
            {"role": "assistant", "content": "two"},
            {"role": "user", "content": "three"},
            {"role": "assistant", "content": "four"},
        ],
    )

    removed, _usage = server._compress_session_history(session)

    assert removed == 3
    assert session["history"] == compressed
    agent._compress_context.assert_called_once()
    assert agent._compress_context.call_args.kwargs.get("force") is True


def test_session_compress_uses_compress_helper(monkeypatch):
    agent = types.SimpleNamespace()
    server._sessions["sid"] = _session(agent=agent)

    monkeypatch.setattr(
        server,
        "_compress_session_history",
        lambda session, focus_topic=None, **_kw: (2, {"total": 42, "compressions": 2}),
    )
    monkeypatch.setattr(server, "_session_info", lambda _agent, *a: {"model": "x"})

    with patch("hermes_runtime.server._emit") as emit:
        resp = server.handle_request(
            {"id": "1", "method": "session.compress", "params": {"session_id": "sid"}}
        )

    assert resp["result"]["removed"] == 2
    assert resp["result"]["usage"] == {"total": 42, "compressions": 2}
    emit.assert_any_call("session.info", "sid", {"model": "x"})
    # Final status.update clears the pinned "compressing" indicator so the
    # status bar can revert to the neutral state when compaction finishes.
    emit.assert_any_call("status.update", "sid", {"kind": "status", "text": "ready"})


def test_session_compress_normalizes_messages_for_desktop_transcript(monkeypatch):
    history = [
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "id": "call-1",
                    "function": {"name": "read_file", "arguments": '{"path":"secret.txt"}'},
                }
            ],
        },
        {"role": "tool", "tool_call_id": "call-1", "content": "very sensitive tool output"},
    ]
    agent = types.SimpleNamespace()
    server._sessions["sid"] = _session(agent=agent, history=history)
    monkeypatch.setattr(server, "_compress_session_history", lambda *_args, **_kwargs: (0, {}))
    monkeypatch.setattr(server, "_session_info", lambda *_args: {})

    try:
        response = server.handle_request(
            {"id": "1", "method": "session.compress", "params": {"session_id": "sid"}}
        )
    finally:
        server._sessions.pop("sid", None)

    assert response["result"]["messages"] == server._history_to_messages(history)
    assert "very sensitive tool output" not in str(response["result"]["messages"])


def test_session_compress_returns_compute_host_history(monkeypatch):
    session = _session(agent=None, _compute_host_active=True)
    server._sessions["sid"] = session
    ack = {
        "type": "control.ack",
        "output": "Compressed 4 → 2 messages",
        "messages": [{"role": "user", "content": "compressed context"}],
        "session_info": {"usage": {"total": 42}},
    }
    monkeypatch.setattr(server, "_session_uses_compute_host", lambda _session: True)
    monkeypatch.setattr(server, "_send_compute_host_control", lambda *args, **kwargs: ack)

    try:
        resp = server.handle_request(
            {"id": "1", "method": "session.compress", "params": {"session_id": "sid"}}
        )
    finally:
        server._sessions.pop("sid", None)

    assert resp["result"] == {
        "status": "compressed",
        "turn_isolation": True,
        "host_ack": {key: value for key, value in ack.items() if key != "messages"},
        "info": {"usage": {"total": 42}},
        "messages": [{"role": "user", "text": "compressed context"}],
        "usage": {"total": 42},
    }


def test_session_compress_forwards_config_ceiling_budget_to_compute_host(monkeypatch):
    session = _session(agent=None, _compute_host_active=True)
    server._sessions["sid"] = session
    calls = []

    def send_control(*args, **kwargs):
        calls.append((args, kwargs))
        return {
            "type": "control.ack",
            "result": {
                "status": "compressed",
                "messages": [],
                "removed": 0,
                "summary": {"headline": "Already compressed", "noop": True},
            },
        }

    monkeypatch.setattr(server, "_session_uses_compute_host", lambda _session: True)
    monkeypatch.setattr(server, "_send_compute_host_control", send_control)
    monkeypatch.setattr(
        server, "_load_cfg", lambda: {"compression": {"context_total_ceiling_seconds": 300}}
    )

    try:
        resp = server.handle_request(
            {"id": "1", "method": "session.compress", "params": {"session_id": "sid"}}
        )
    finally:
        server._sessions.pop("sid", None)

    assert resp["result"]["status"] == "compressed"
    assert len(calls) == 1
    (sid_arg,), kwargs = calls[0]
    assert sid_arg == "sid"
    assert kwargs["route_name"] == "session.compress"
    assert kwargs["command"] == "/compress"
    assert kwargs["wait"] is True
    # #97948: the waiter follows compression.context_total_ceiling_seconds
    # (+30s slack) instead of a hard-coded 120s, and registers a late-ack
    # handler so a compress that outlives it is still adopted.
    assert kwargs["timeout"] == 330.0
    assert callable(kwargs["on_late_ack"])


def test_session_compress_preserves_compute_host_aborted_summary(monkeypatch):
    session = _session(agent=None, _compute_host_active=True)
    server._sessions["sid"] = session
    result = {
        "status": "aborted",
        "messages": [{"role": "user", "content": "preserved context"}],
        "removed": 0,
        "summary": {
            "aborted": True,
            "headline": "Compression aborted: 6 messages preserved",
            "note": "No compression provider is configured.",
        },
    }
    monkeypatch.setattr(server, "_session_uses_compute_host", lambda _session: True)
    monkeypatch.setattr(
        server,
        "_send_compute_host_control",
        lambda *args, **kwargs: {
            "type": "control.ack",
            "result": result,
            "session_key": "rotated-host-key",
            "history_version": 7,
            "message_count": 1,
            "session_info": {"model": "host-model"},
        },
    )

    try:
        resp = server.handle_request(
            {"id": "1", "method": "session.compress", "params": {"session_id": "sid"}}
        )
    finally:
        server._sessions.pop("sid", None)

    assert resp["result"] == {**result, "turn_isolation": True}
    assert session["session_key"] == "rotated-host-key"
    assert session["history_version"] == 7
    assert session["_metadata_message_count"] == 1
    assert session["_metadata_mirror"]["model"] == "host-model"


def test_session_compress_reports_aborted_summary_without_success(monkeypatch):
    compression_state = types.SimpleNamespace(
        _last_compress_aborted=True,
        _last_summary_fallback_used=False,
        _last_summary_error=(
            "Provider 'opencode-zen' is set in config.yaml but no API key was found."
        ),
    )
    agent = types.SimpleNamespace(
        context_compressor=compression_state,
        _cached_system_prompt="",
        tools=None,
    )
    history = [{"role": "user", "content": f"m{i}"} for i in range(6)]
    server._sessions["sid"] = _session(agent=agent, history=history)

    monkeypatch.setattr(
        server,
        "_compress_session_history",
        lambda session, focus_topic=None, **_kw: (0, {"total": 42}),
    )
    monkeypatch.setattr(server, "_session_info", lambda _agent, *a: {"model": "x"})

    try:
        with patch("hermes_runtime.server._emit"):
            resp = server.handle_request(
                {
                    "id": "1",
                    "method": "session.compress",
                    "params": {"session_id": "sid"},
                }
            )

        result = resp["result"]
        assert result["status"] == "aborted"
        assert result["removed"] == 0
        assert result["summary"]["aborted"] is True
        assert "no API key was found" in result["summary"]["note"]
        assert "Compressed:" not in result["summary"]["headline"]
    finally:
        server._sessions.pop("sid", None)


def test_session_compress_syncs_session_key_after_rotation(monkeypatch):
    """LCM notification follows the TUI's final session-key transition."""
    from agent.conversation_compression import (
        _queue_context_engine_compression_notification,
    )

    events = []
    agent = types.SimpleNamespace(
        session_id="rotated-id",
        context_compressor=types.SimpleNamespace(
            on_session_start=lambda *_args, **_kwargs: events.append("notify")
        ),
    )
    server._sessions["sid"] = _session(agent=agent)
    server._sessions["sid"]["session_key"] = "old-key"
    server._sessions["sid"]["pending_title"] = "stale title"

    def _compress(session, focus_topic=None, **_kw):
        _queue_context_engine_compression_notification(
            session["agent"],
            new_session_id="rotated-id",
            old_session_id="old-key",
        )
        return 2, {"total": 42}

    monkeypatch.setattr(server, "_compress_session_history", _compress)
    monkeypatch.setattr(server, "_session_info", lambda _agent, *a: {"model": "x"})
    restart_calls = []
    monkeypatch.setattr(
        server,
        "_restart_slash_worker",
        lambda sid, s: (restart_calls.append(s), events.append("sync")),
    )

    try:
        with patch("hermes_runtime.server._emit"):
            server.handle_request(
                {
                    "id": "1",
                    "method": "session.compress",
                    "params": {"session_id": "sid"},
                }
            )

        assert server._sessions["sid"]["session_key"] == "rotated-id"
        assert server._sessions["sid"]["pending_title"] is None
        assert len(restart_calls) == 1
        assert events == ["sync", "notify"]
    finally:
        server._sessions.pop("sid", None)


def test_session_compress_sync_failure_discards_lcm_notification(monkeypatch):
    from agent.conversation_compression import (
        _queue_context_engine_compression_notification,
    )

    events = []
    agent = types.SimpleNamespace(
        session_id="rotated-id",
        context_compressor=types.SimpleNamespace(
            on_session_start=lambda *_args, **_kwargs: events.append("notify")
        ),
    )
    server._sessions["sid"] = _session(agent=agent)
    server._sessions["sid"]["session_key"] = "old-key"

    def _compress(session, focus_topic=None, **_kw):
        _queue_context_engine_compression_notification(
            session["agent"],
            new_session_id="rotated-id",
            old_session_id="old-key",
        )
        return 2, {"total": 42}

    monkeypatch.setattr(server, "_compress_session_history", _compress)
    monkeypatch.setattr(
        server,
        "_session_info",
        lambda *_args: (_ for _ in ()).throw(RuntimeError("finalization failed")),
    )

    try:
        with patch("hermes_runtime.server._emit"):
            resp = server.handle_request(
                {
                    "id": "1",
                    "method": "session.compress",
                    "params": {"session_id": "sid"},
                }
            )
        assert resp["error"]["code"] == 5005
        assert events == []
    finally:
        server._sessions.pop("sid", None)


def test_slash_exec_r7_read_commands_use_metadata_mirror_flag_on(monkeypatch):
    class _ExplodingWorker:
        def __init__(self, *args, **kwargs):
            raise AssertionError("slash worker should not run for isolated read commands")

    history_from_db = [
        {"role": "user", "content": "live question from state db"},
        {"role": "assistant", "content": "live answer from state db"},
    ]

    class _DB:
        def get_session(self, key):
            assert key == "session-key"
            return {
                "title": "Live title",
                "started_at": 1_700_000_000,
                "updated_at": 1_700_000_060,
                "pinned": True,
            }

        def get_resume_conversations(self, session_id):
            return (
                self.get_messages_as_conversation(session_id, repair_alternation=True),
                self.get_messages_as_conversation(session_id, include_ancestors=True),
            )

        def get_ancestor_display_prefix(self, _sid):
            return []

        def get_messages_as_conversation(self, key, include_ancestors=True, repair_alternation=False, **_kwargs):
            assert key == "session-key"
            assert include_ancestors is True
            return list(history_from_db)

    server._sessions["sid"] = _session(
        agent=None,
        history=[{"role": "user", "content": "stale parent mirror"}],
        _compute_host_active=True,
        _metadata_mirror={
            "model": "host-model",
            "provider": "host-provider",
            "system_prompt": "host system prompt",
            "tools": {"core": ["terminal", "read_file"]},
            "usage": {
                "model": "host-model",
                "input": 100,
                "output": 20,
                "reasoning": 5,
                "prompt": 120,
                "completion": 20,
                "total": 140,
                "calls": 2,
                "context_used": 80,
                "context_max": 1000,
                "context_percent": 8,
                "compressions": 1,
            },
        },
        _metadata_message_count=2,
    )
    monkeypatch.setattr(server, "_SlashWorker", _ExplodingWorker)
    monkeypatch.setattr(server, "_get_db", lambda: _DB())
    monkeypatch.setattr(server, "_load_cfg", lambda: {"dashboard": {"turn_isolation": True}})

    cases = {
        "usage": "140",
        "history": "live question from state db",
        "status": "140",
        "context": "Context usage: 80 / 1,000 tokens",
        "tools": "terminal",
        "help": "/status",
    }

    try:
        for command, expected in cases.items():
            resp = server.handle_request(
                {
                    "id": command,
                    "method": "slash.exec",
                    "params": {"command": command, "session_id": "sid"},
                }
            )
            assert "result" in resp, (command, resp)
            assert expected in resp["result"]["output"]
            assert "stale parent mirror" not in resp["result"]["output"]
            assert "(._.)" not in resp["result"]["output"]
        mirrored_usage = server._sessions["sid"]["_metadata_mirror"]["usage"]
        for estimated in (True, False):
            mirrored_usage["context_estimated"] = estimated
            mirrored_usage["context_source"] = "local_estimate" if estimated else "provider_usage"
            response = server.handle_request({
                "id": "context-provenance", "method": "slash.exec",
                "params": {"command": "context", "session_id": "sid"},
            })
            mark = "~" if estimated else ""
            assert f"Context usage: {mark}80 / 1,000 tokens ({mark}8.0%)" in response["result"]["output"]
    finally:
        server._sessions.pop("sid", None)


def test_prompt_submit_sets_approval_session_key(monkeypatch):
    from tools.approval import get_current_session_key

    captured = {}

    class _Agent:
        def run_conversation(self, prompt, conversation_history=None, stream_callback=None, **_kwargs):
            captured["session_key"] = get_current_session_key(default="")
            return {
                "final_response": "ok",
                "messages": [{"role": "assistant", "content": "ok"}],
            }

    class _ImmediateThread:
        def __init__(self, target=None, daemon=None, **_thread_options):
            self._target = target

        def start(self):
            self._target()

    server._sessions["sid"] = _session(agent=_Agent())
    monkeypatch.setattr(server.threading, "Thread", _ImmediateThread)
    monkeypatch.setattr(server, "_emit", lambda *args, **kwargs: None)
    monkeypatch.setattr(server, "make_stream_renderer", lambda cols: None)
    monkeypatch.setattr(server, "render_message", lambda raw, cols: None)

    resp = server.handle_request(
        {
            "id": "1",
            "method": "prompt.submit",
            "params": {"session_id": "sid", "text": "ping"},
        }
    )

    assert resp["result"]["status"] == "streaming"
    assert captured["session_key"] == "session-key"


def test_prompt_submit_expands_context_refs(monkeypatch):
    captured = {}

    class _Agent:
        model = "test/model"
        base_url = ""
        api_key = ""

        def run_conversation(self, prompt, conversation_history=None, stream_callback=None, **_kwargs):
            captured["prompt"] = prompt
            return {
                "final_response": "ok",
                "messages": [{"role": "assistant", "content": "ok"}],
            }

    class _ImmediateThread:
        def __init__(self, target=None, daemon=None, **_thread_options):
            self._target = target

        def start(self):
            self._target()

    fake_ctx = types.ModuleType("agent.context_references")
    fake_ctx.preprocess_context_references = (
        lambda message, **kwargs: types.SimpleNamespace(
            blocked=False,
            message="expanded prompt",
            warnings=[],
            references=[],
            injected_tokens=0,
        )
    )
    fake_meta = types.ModuleType("agent.model_metadata")
    fake_meta.get_model_context_length = lambda *args, **kwargs: 100000

    server._sessions["sid"] = _session(agent=_Agent())
    monkeypatch.setattr(server.threading, "Thread", _ImmediateThread)
    monkeypatch.setattr(server, "_emit", lambda *args, **kwargs: None)
    monkeypatch.setattr(server, "make_stream_renderer", lambda cols: None)
    monkeypatch.setattr(server, "render_message", lambda raw, cols: None)
    monkeypatch.setitem(sys.modules, "agent.context_references", fake_ctx)
    monkeypatch.setitem(sys.modules, "agent.model_metadata", fake_meta)

    server.handle_request(
        {
            "id": "1",
            "method": "prompt.submit",
            "params": {"session_id": "sid", "text": "@diff"},
        }
    )

    assert captured["prompt"] == "expanded prompt"


def test_image_attach_appends_local_image(monkeypatch):
    fake_cli = types.ModuleType("cli")
    fake_cli._IMAGE_EXTENSIONS = {".png"}
    fake_cli._detect_file_drop = lambda raw: {
        "path": Path("/tmp/cat.png"),
        "is_image": True,
        "remainder": "",
    }
    fake_cli._split_path_input = lambda raw: (raw, "")
    fake_cli._resolve_attachment_path = lambda raw: Path("/tmp/cat.png")

    server._sessions["sid"] = _session()
    monkeypatch.setitem(sys.modules, "cli", fake_cli)

    resp = server.handle_request(
        {
            "id": "1",
            "method": "image.attach",
            "params": {"session_id": "sid", "path": "/tmp/cat.png"},
        }
    )

    assert resp["result"]["attached"] is True
    assert resp["result"]["name"] == "cat.png"
    assert len(server._sessions["sid"]["attached_images"]) == 1


def test_image_attach_accepts_unquoted_screenshot_path_with_spaces(monkeypatch):
    screenshot = Path("/tmp/Screenshot 2026-04-21 at 1.04.43 PM.png")
    fake_cli = types.ModuleType("cli")
    fake_cli._IMAGE_EXTENSIONS = {".png"}
    fake_cli._detect_file_drop = lambda raw: {
        "path": screenshot,
        "is_image": True,
        "remainder": "",
    }
    fake_cli._split_path_input = lambda raw: (
        "/tmp/Screenshot",
        "2026-04-21 at 1.04.43 PM.png",
    )
    fake_cli._resolve_attachment_path = lambda raw: None

    server._sessions["sid"] = _session()
    monkeypatch.setitem(sys.modules, "cli", fake_cli)

    resp = server.handle_request(
        {
            "id": "1",
            "method": "image.attach",
            "params": {"session_id": "sid", "path": str(screenshot)},
        }
    )

    assert resp["result"]["attached"] is True
    assert resp["result"]["path"] == str(screenshot)
    assert resp["result"]["remainder"] == ""
    assert len(server._sessions["sid"]["attached_images"]) == 1


def test_file_attach_uploads_remote_file_into_session_workspace(monkeypatch, tmp_path):
    """Remote case: client path doesn't exist on gateway → decode data_url bytes.

    Staged into the session home's ``attachments/`` dir (bind-mounted into
    container backends) rather than the workspace (#76577).
    """
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    home = tmp_path / "home"
    fake_cli = types.ModuleType("cli")
    fake_cli._detect_file_drop = lambda raw: None
    fake_cli._split_path_input = lambda raw: (raw, "")
    fake_cli._resolve_attachment_path = lambda raw: None

    server._sessions["sid"] = _session(cwd=str(workspace), profile_home=str(home))
    monkeypatch.setitem(sys.modules, "cli", fake_cli)

    try:
        resp = server.handle_request(
            {
                "id": "1",
                "method": "file.attach",
                "params": {
                    "session_id": "sid",
                    "path": "/Users/alice/Downloads/report.txt",
                    "name": "report.txt",
                    "data_url": "data:text/plain;base64,aGVsbG8gd29ybGQ=",
                },
            }
        )

        stored = home / "attachments" / "report.txt"
        assert resp["result"]["attached"] is True
        assert resp["result"]["uploaded"] is True
        assert resp["result"]["path"] == str(stored)
        assert resp["result"]["ref_text"] == f"@file:{stored}"
        assert stored.read_text(encoding="utf-8-sig") == "hello world"
    finally:
        server._sessions.pop("sid", None)


def test_file_attach_copies_gateway_visible_file_outside_workspace(monkeypatch, tmp_path):
    """Local case: gateway can see the file but it's outside the workspace → copy in."""
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    home = tmp_path / "home"
    source = tmp_path / "outside.txt"
    source.write_text("outside workspace", encoding="utf-8")
    fake_cli = types.ModuleType("cli")
    fake_cli._detect_file_drop = lambda raw: None
    fake_cli._split_path_input = lambda raw: (raw, "")
    fake_cli._resolve_attachment_path = lambda raw: source

    server._sessions["sid"] = _session(cwd=str(workspace), profile_home=str(home))
    monkeypatch.setitem(sys.modules, "cli", fake_cli)

    try:
        resp = server.handle_request(
            {
                "id": "1",
                "method": "file.attach",
                "params": {"session_id": "sid", "path": str(source)},
            }
        )

        stored = home / "attachments" / "outside.txt"
        assert resp["result"]["attached"] is True
        assert resp["result"]["uploaded"] is True
        assert resp["result"]["ref_text"] == f"@file:{stored}"
        assert stored.read_text(encoding="utf-8-sig") == "outside workspace"
    finally:
        server._sessions.pop("sid", None)


def test_file_attach_uses_in_workspace_file_without_copying(monkeypatch, tmp_path):
    """Local case: file already inside the workspace → ref it directly, no copy."""
    workspace = tmp_path / "workspace"
    (workspace / "data").mkdir(parents=True)
    source = workspace / "data" / "exam.csv"
    source.write_text("a,b,c\n1,2,3\n", encoding="utf-8")
    fake_cli = types.ModuleType("cli")
    fake_cli._detect_file_drop = lambda raw: None
    fake_cli._split_path_input = lambda raw: (raw, "")
    fake_cli._resolve_attachment_path = lambda raw: source

    server._sessions["sid"] = _session(cwd=str(workspace))
    monkeypatch.setitem(sys.modules, "cli", fake_cli)

    try:
        resp = server.handle_request(
            {
                "id": "1",
                "method": "file.attach",
                "params": {"session_id": "sid", "path": str(source)},
            }
        )

        assert resp["result"]["attached"] is True
        assert resp["result"]["uploaded"] is False
        assert resp["result"]["ref_text"] == "@file:data/exam.csv"
        # No copy: nothing staged under desktop-attachments or the home
        # attachments dir.
        assert not (workspace / ".hermes" / "desktop-attachments").exists()
        assert not (tmp_path / "home" / "attachments").exists()
    finally:
        server._sessions.pop("sid", None)


def test_file_attach_errors_when_unresolvable_and_no_bytes(monkeypatch, tmp_path):
    """Remote path not on gateway and no data_url → actionable error, not a stage."""
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    fake_cli = types.ModuleType("cli")
    fake_cli._detect_file_drop = lambda raw: None
    fake_cli._split_path_input = lambda raw: (raw, "")
    fake_cli._resolve_attachment_path = lambda raw: None

    server._sessions["sid"] = _session(cwd=str(workspace))
    monkeypatch.setitem(sys.modules, "cli", fake_cli)

    try:
        resp = server.handle_request(
            {
                "id": "1",
                "method": "file.attach",
                "params": {"session_id": "sid", "path": "/Users/alice/missing.txt"},
            }
        )

        assert "error" in resp
    finally:
        server._sessions.pop("sid", None)


def test_file_attach_quotes_ref_with_spaces(monkeypatch, tmp_path):
    """Staged names with spaces must be backtick-quoted so the @file: ref parses."""
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    fake_cli = types.ModuleType("cli")
    fake_cli._detect_file_drop = lambda raw: None
    fake_cli._split_path_input = lambda raw: (raw, "")
    fake_cli._resolve_attachment_path = lambda raw: None

    server._sessions["sid"] = _session(cwd=str(workspace), profile_home=str(tmp_path / "home"))
    monkeypatch.setitem(sys.modules, "cli", fake_cli)

    try:
        resp = server.handle_request(
            {
                "id": "1",
                "method": "file.attach",
                "params": {
                    "session_id": "sid",
                    "name": "my exam schedule.csv",
                    "data_url": "data:text/csv;base64,YSxiCg==",
                },
            }
        )

        stored = tmp_path / "home" / "attachments" / "my exam schedule.csv"
        assert resp["result"]["attached"] is True
        assert resp["result"]["ref_text"] == f"@file:`{stored}`"
        assert stored.read_text(encoding="utf-8-sig") == "a,b\n"
    finally:
        server._sessions.pop("sid", None)


def test_file_attach_workspace_storage_follows_the_owning_profiles_config(monkeypatch, tmp_path):
    """``attachments.storage: workspace`` is read from the SESSION's profile config.

    Opt-in profiles stage under ``<workspace>/.hermes/attachments`` (inside the
    allowed ref root, #110662); a profile without the opt-in keeps the
    bind-mounted ``<profile home>/attachments`` — one serve process, two homes,
    each session follows its own profile's config.
    """
    workspace_a = tmp_path / "work-a"
    workspace_a.mkdir()
    home_a = tmp_path / "home-a"
    home_a.mkdir()
    (home_a / "config.yaml").write_text("attachments:\n  storage: workspace\n", encoding="utf-8")

    def fake_resolve(raw):
        return None

    fake_cli = types.ModuleType("cli")
    fake_cli._detect_file_drop = lambda raw: None
    fake_cli._split_path_input = lambda raw: (raw, "")
    fake_cli._resolve_attachment_path = fake_resolve
    monkeypatch.setitem(sys.modules, "cli", fake_cli)

    def attach(sid: str, params: dict) -> dict:
        return server.handle_request(
            {"id": "1", "method": "file.attach", "params": {"session_id": sid, **params}}
        )

    try:
        server._sessions["sid-a"] = _session(cwd=str(workspace_a), profile_home=str(home_a))
        resp = attach(
            "sid-a",
            {
                "path": "/Users/alice/Downloads/report.txt",
                "name": "report.txt",
                "data_url": "data:text/plain;base64,aGVsbG8=",
            },
        )

        staged_a = workspace_a / ".hermes" / "attachments" / "report.txt"
        assert resp["result"]["attached"] is True
        assert resp["result"]["path"] == str(staged_a)
        assert staged_a.read_text(encoding="utf-8") == "hello"
        # Workspace staging lands inside the allowed ref root: the ref is
        # workspace-relative, not an absolute out-of-workspace path.
        assert resp["result"]["ref_text"] == "@file:.hermes/attachments/report.txt"
        assert not (home_a / "attachments").exists()

        # Same serve process, second profile WITHOUT the opt-in: staging stays
        # on that profile's hermes-home attachments dir.
        workspace_b = tmp_path / "work-b"
        workspace_b.mkdir()
        home_b = tmp_path / "home-b"
        home_b.mkdir()
        server._sessions["sid-b"] = _session(cwd=str(workspace_b), profile_home=str(home_b))
        resp = attach(
            "sid-b",
            {
                "path": "/Users/alice/Downloads/report.txt",
                "name": "report.txt",
                "data_url": "data:text/plain;base64,aGVsbG8=",
            },
        )

        staged_b = home_b / "attachments" / "report.txt"
        assert resp["result"]["path"] == str(staged_b)
        assert staged_b.read_text(encoding="utf-8") == "hello"
        assert not (workspace_b / ".hermes").exists()
    finally:
        server._sessions.pop("sid-a", None)
        server._sessions.pop("sid-b", None)



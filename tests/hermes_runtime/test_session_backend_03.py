

import json








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


def test_load_enabled_toolsets_honors_disabled_project_on_focus_path(monkeypatch):
    """#54433: focus/coding posture must not re-add `project` when disabled."""
    monkeypatch.delenv("HERMES_TUI_TOOLSETS", raising=False)

    import agent.coding_context as cc

    monkeypatch.setattr(cc, "coding_selection", lambda **_: ["coding", "figma"])
    monkeypatch.setattr(server, "_load_disabled_toolsets", lambda: ["project"])

    result = server._load_enabled_toolsets("tui")
    assert result == ["coding", "figma"]
    assert "project" not in result


def test_load_enabled_toolsets_honors_disabled_project_on_configured_fallback(
    monkeypatch,
):
    """#54433: configured/fallback path must not re-add disabled `project`."""
    monkeypatch.delenv("HERMES_TUI_TOOLSETS", raising=False)

    import agent.coding_context as cc
    import hermes_cli.tools_config as tools_config_mod

    monkeypatch.setattr(cc, "coding_selection", lambda **_: None)
    monkeypatch.setattr(
        tools_config_mod,
        "_get_platform_tools",
        lambda *_args, **_kwargs: {"memory", "web"},
    )
    monkeypatch.setattr(server, "_load_disabled_toolsets", lambda: ["project"])

    result = server._load_enabled_toolsets("tui")
    assert result == ["memory", "web"]
    assert "project" not in result


def test_with_session_toolsets_keeps_desktop_ui_when_project_disabled(monkeypatch):
    """#54433: a disabled name is subtracted from the client-surface fold-in, but
    ``desktop_ui`` — the client's own control surface — survives the subtraction."""
    monkeypatch.setattr(server, "_load_disabled_toolsets", lambda: ["project"])

    assert server._with_session_toolsets(["memory"], "desktop") == ["memory", "catalog", "desktop_ui"]
    # Nothing disabled: the fold-in keeps every client-surface toolset.
    monkeypatch.setattr(server, "_load_disabled_toolsets", lambda: None)
    assert server._with_session_toolsets(["memory"], "desktop") == ["memory", "catalog", "desktop_ui", "project"]


def test_load_enabled_toolsets_rejects_disabled_mcp_env(monkeypatch, capsys):
    monkeypatch.setenv("HERMES_TUI_TOOLSETS", "mcp-off")
    monkeypatch.setitem(
        sys.modules,
        "hermes_cli.plugins",
        types.SimpleNamespace(discover_plugins=lambda: None),
    )

    import hermes_cli.config as config_mod

    monkeypatch.setattr(
        config_mod,
        "read_raw_config",
        lambda: {"mcp_servers": {"mcp-off": {"enabled": False}}},
    )
    monkeypatch.setattr(
        config_mod, "load_config", lambda: {"platform_toolsets": {"cli": ["memory"]}}
    )

    # Sorted: ["memory", "project"]. `kanban` is a configurable opt-in and is
    # never recovered onto a saved list; `project` is GUI-only, folded in by
    # _load_enabled_toolsets. Toolsets inside their first release
    # (_RECENTLY_SHIPPED_TOOLSETS) are back-filled onto saved lists that never
    # offered them — allow those too.
    from hermes_cli.tools_config import _RECENTLY_SHIPPED_TOOLSETS

    result = server._load_enabled_toolsets()
    assert result is not None
    assert {"memory", "project"} <= set(result)
    assert "kanban" not in result
    assert set(result) - {"memory", "project"} <= _RECENTLY_SHIPPED_TOOLSETS
    err = capsys.readouterr().err
    assert "mcp-off" in err


def test_load_enabled_toolsets_falls_back_when_tui_env_invalid(monkeypatch, capsys):
    monkeypatch.setenv("HERMES_TUI_TOOLSETS", "nope")
    monkeypatch.setitem(
        sys.modules,
        "hermes_cli.plugins",
        types.SimpleNamespace(discover_plugins=lambda: None),
    )

    import hermes_cli.config as config_mod

    monkeypatch.setattr(
        config_mod, "load_config", lambda: {"platform_toolsets": {"cli": ["memory"]}}
    )

    from hermes_cli.tools_config import _RECENTLY_SHIPPED_TOOLSETS

    result = server._load_enabled_toolsets()
    assert result is not None
    assert {"memory", "project"} <= set(result)
    assert "kanban" not in result
    assert set(result) - {"memory", "project"} <= _RECENTLY_SHIPPED_TOOLSETS
    assert capsys.readouterr().err.strip()  # a fallback warning is printed


def test_load_enabled_toolsets_warns_when_config_fallback_fails(monkeypatch, capsys):
    monkeypatch.setenv("HERMES_TUI_TOOLSETS", "nope")
    monkeypatch.setitem(
        sys.modules,
        "hermes_cli.plugins",
        types.SimpleNamespace(discover_plugins=lambda: None),
    )

    import hermes_cli.config as config_mod

    monkeypatch.setattr(
        config_mod, "load_config", lambda: (_ for _ in ()).throw(RuntimeError("boom"))
    )

    assert server._load_enabled_toolsets() is None
    assert capsys.readouterr().err.strip()  # a warning is printed


def test_load_enabled_toolsets_honors_builtin_env_if_config_fails(monkeypatch):
    monkeypatch.setenv("HERMES_TUI_TOOLSETS", "web")

    import hermes_cli.config as config_mod

    monkeypatch.setattr(
        config_mod, "load_config", lambda: (_ for _ in ()).throw(RuntimeError("boom"))
    )

    assert server._load_enabled_toolsets() == ["web"]


def test_load_enabled_toolsets_all_env_means_all(monkeypatch):
    monkeypatch.setenv("HERMES_TUI_TOOLSETS", "all")

    assert server._load_enabled_toolsets() is None


def test_load_disabled_toolsets_reads_agent_config(monkeypatch):
    import hermes_cli.config as config_mod

    monkeypatch.setattr(
        config_mod,
        "load_config",
        lambda: {"agent": {"disabled_toolsets": ["browser"]}},
    )

    assert server._load_disabled_toolsets() == ["browser"]


def test_load_disabled_toolsets_none_when_unset_or_config_fails(monkeypatch):
    import hermes_cli.config as config_mod

    monkeypatch.setattr(config_mod, "load_config", lambda: {"agent": {"disabled_toolsets": []}})
    assert server._load_disabled_toolsets() is None

    monkeypatch.setattr(config_mod, "load_config", dict)
    assert server._load_disabled_toolsets() is None

    monkeypatch.setattr(
        config_mod, "load_config", lambda: (_ for _ in ()).throw(RuntimeError("boom"))
    )
    assert server._load_disabled_toolsets() is None


def test_load_enabled_toolsets_reports_disabled_mcp_separately(monkeypatch, capsys):
    monkeypatch.setenv("HERMES_TUI_TOOLSETS", "web,mcp-off,nope")
    monkeypatch.setitem(
        sys.modules,
        "hermes_cli.plugins",
        types.SimpleNamespace(discover_plugins=lambda: None),
    )

    import hermes_cli.config as config_mod

    monkeypatch.setattr(
        config_mod,
        "read_raw_config",
        lambda: {"mcp_servers": {"mcp-off": {"enabled": False}}},
    )

    assert server._load_enabled_toolsets() == ["web"]
    err = capsys.readouterr().err
    assert "nope" in err
    assert "mcp-off" in err


def test_history_to_messages_preserves_tool_calls_for_resume_display():
    history = [
        {"role": "user", "content": "first prompt"},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "id": "call_1",
                    "function": {
                        "name": "search_files",
                        "arguments": json.dumps({"pattern": "resume"}),
                    },
                }
            ],
        },
        {"role": "tool", "content": "{}", "tool_call_id": "call_1"},
        {"role": "assistant", "content": "first answer"},
        {"role": "user", "content": "second prompt"},
    ]

    assert server._history_to_messages(history) == [
        {"role": "user", "text": "first prompt"},
        {
            "args": {"pattern": "resume"},
            "context": "resume",
            "name": "search_files",
            "role": "tool",
            "tool_call_id": history[2]["tool_call_id"],
        },
        {"role": "assistant", "text": "first answer"},
        {"role": "user", "text": "second prompt"},
    ]


def test_history_to_messages_types_the_failed_turn_boundary_for_resume():
    """Desktop keys the failed-turn boundary on ``display_kind`` (a room poller must not post it
    as the member's reply); rows written before the closer typed it are typed on read."""
    from agent.turn_failure_copy import FAILED_TURN_DISPLAY_KIND, FAILED_TURN_NOTICE, PARTIAL_FAILED_TURN_NOTICE

    history = [
        {"role": "user", "content": "a"},
        {"role": "assistant", "content": FAILED_TURN_NOTICE, "display_kind": FAILED_TURN_DISPLAY_KIND},
        {"role": "user", "content": "b"},
        {"role": "assistant", "content": PARTIAL_FAILED_TURN_NOTICE},  # legacy untyped row
        {"role": "user", "content": "c"},
        {"role": "assistant", "content": f"Quoting Hermes: {FAILED_TURN_NOTICE}"},  # a real reply
    ]

    assert [m.get("display_kind") for m in server._history_to_messages(history)] == [
        None, FAILED_TURN_DISPLAY_KIND, None, FAILED_TURN_DISPLAY_KIND, None, None,
    ]


def test_history_to_messages_drops_pure_compaction_scaffolding():
    from agent.context_compressor import (
        HISTORICAL_TASK_HEADING,
        SUMMARY_PREFIX,
        _SUMMARY_END_MARKER,
    )

    summary = (
        f"{SUMMARY_PREFIX}\n\n"
        f"{HISTORICAL_TASK_HEADING}\nold work\n\n"
        f"{_SUMMARY_END_MARKER}"
    )

    assert server._history_to_messages(
        [
            {"role": "user", "content": summary},
            {"role": "assistant", "content": "real answer"},
        ]
    ) == [{"role": "assistant", "text": "real answer"}]


def test_history_to_messages_preserves_live_ask_without_compaction_scaffolding():
    from agent.context_compressor import (
        HISTORICAL_TASK_HEADING,
        SUMMARY_PREFIX,
        _SUMMARY_END_MARKER,
    )

    carrier = (
        f"{SUMMARY_PREFIX}\n\n"
        f"{HISTORICAL_TASK_HEADING}\nold work\n\n"
        f"{_SUMMARY_END_MARKER}\n\n"
        "test the browser controller"
    )

    assert server._history_to_messages(
        [
            {
                "role": "user",
                "content": carrier,
                "tool_calls": [{"id": "stale"}],
                "reasoning": "internal compaction reasoning",
            }
        ]
    ) == [{"role": "user", "text": "test the browser controller"}]


def test_history_to_messages_unwraps_merged_assistant_carrier():
    from agent.context_compressor import (
        HISTORICAL_TASK_HEADING,
        SUMMARY_PREFIX,
        _MERGED_PRIOR_CONTEXT_HEADER,
        _MERGED_SUMMARY_DELIMITER,
        _SUMMARY_END_MARKER,
    )

    carrier = (
        f"{_MERGED_PRIOR_CONTEXT_HEADER}\n"
        "real completed answer\n\n"
        f"{_MERGED_SUMMARY_DELIMITER}\n\n"
        f"{SUMMARY_PREFIX}\n\n"
        f"{HISTORICAL_TASK_HEADING}\nold work\n\n"
        f"{_SUMMARY_END_MARKER}"
    )

    assert server._history_to_messages(
        [
            {
                "role": "assistant",
                "content": carrier,
                "tool_calls": [{"id": "stale"}],
                "reasoning_details": [{"summary": "internal"}],
            }
        ]
    ) == [{"role": "assistant", "text": "real completed answer"}]


def test_history_to_messages_ships_full_tool_args():
    # This is the display projection. `context` is an 80-char preview for
    # collapsed row titles. A renderer that shows the full call (the expanded
    # `$` transcript in the desktop) rebuilds it from `args`. When the
    # projection dropped the args, the preview truncation was permanent.
    long_command = "echo " + "x" * 200
    history = [
        {"role": "user", "content": "run it"},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "id": "call_1",
                    "function": {
                        "name": "terminal",
                        "arguments": json.dumps({"command": long_command}),
                    },
                }
            ],
        },
        {"role": "tool", "content": "{}", "tool_call_id": "call_1"},
    ]

    rows = server._history_to_messages(history)
    assert rows[1]["args"] == {"command": long_command}
    # The preview stays alongside for the collapsed title.
    assert rows[1]["context"]

    # A tool row with no recorded args keeps the old small shape.
    argless = server._history_to_messages(
        [{"role": "tool", "content": "{}", "tool_call_id": "missing"}]
    )
    assert "args" not in argless[0]


def test_tool_start_ships_full_args(monkeypatch):
    # The desktop rebuilds the expanded row's `$` transcript from args. When
    # only the 80-char `context` preview shipped, the expanded command was
    # truncated until tool.complete. tool.complete already ships full args to
    # every client, so tool.start does too. There is no per-client gate.
    events: list[tuple[str, str, dict]] = []
    monkeypatch.setattr(
        server, "_emit", lambda event_type, sid, payload: events.append((event_type, sid, payload))
    )
    long_command = "echo " + "y" * 200
    monkeypatch.setitem(
        server._sessions,
        "args-test",
        {"source": "desktop", "tool_progress_mode": "all", "tool_started_at": {}},
    )

    server._on_tool_start("args-test", "tool-1", "terminal", {"command": long_command})
    server._on_tool_start("args-test", "tool-2", "terminal", {})

    assert events[0][2]["args"] == {"command": long_command}
    # Empty args stay omitted. Argless tools get no noise key.
    assert "args" not in events[1][2]


def test_history_to_messages_keeps_reasoning_only_assistant_turn():
    # A thinking-only assistant turn (reasoning present, no visible text) is
    # persisted and recallable, but was dropped from the resumed session view
    # as "empty" -- so it vanished while the agent could still recall it from
    # the transcript. Keep it (with reasoning) so the desktop "Thinking…"
    # disclosure renders. (#44022)
    history = [
        {"role": "user", "content": "think about this"},
        {"role": "assistant", "content": "", "reasoning": "step-by-step thoughts"},
        {"role": "assistant", "content": "here is the answer"},
    ]

    assert server._history_to_messages(history) == [
        {"role": "user", "text": "think about this"},
        {"role": "assistant", "text": "", "reasoning": "step-by-step thoughts"},
        {"role": "assistant", "text": "here is the answer"},
    ]


def test_history_to_messages_still_drops_empty_assistant_without_reasoning():
    # A genuinely empty assistant turn (no text, no reasoning, no tool calls)
    # remains filtered out -- the fix only spares reasoning-bearing turns.
    history = [
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": "", "reasoning": ""},
        {"role": "assistant", "content": "   "},
        {"role": "assistant", "content": "real reply"},
    ]

    assert server._history_to_messages(history) == [
        {"role": "user", "text": "hi"},
        {"role": "assistant", "text": "real reply"},
    ]


def test_history_to_messages_renders_multimodal_content():
    # bb/gui preserves image URLs in the resume payload so the desktop
    # renderer's extractEmbeddedImages can pull them back out and display
    # the actual image instead of a placeholder. This also keeps the
    # resume payload in sync with the cached message.
    history = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "look here"},
                {"type": "image_url", "image_url": {"url": "data:image/png;base64,abc"}},
            ],
        },
        {"role": "assistant", "content": "saw it"},
    ]

    assert server._history_to_messages(history) == [
        {"role": "user", "text": "look here\ndata:image/png;base64,abc"},
        {"role": "assistant", "text": "saw it"},
    ]


def test_history_to_messages_strips_legacy_discord_triggering_note():
    # Rows written before the gateway persisted the authored text carry the model-facing
    # routing note in user ``content``; this projection heals them for TUI/web resume
    # (the desktop hydration strip is the same rule). Reply pointer and assistant rows are kept.
    from gateway.run_inbound import discord_triggering_note

    note = discord_triggering_note("123")
    history = [
        {"role": "user", "content": f"{note}\n\n[Replying to: hi]\nwhat is up"},
        {"role": "assistant", "content": f"echo: {note}"},
    ]

    assert server._history_to_messages(history) == [
        {"role": "user", "text": "[Replying to: hi]\nwhat is up"},
        {"role": "assistant", "text": f"echo: {note}"},
    ]


def test_history_to_messages_hides_gateway_system_markers():
    # Model-switch / personality notices are persisted as role=user [System: …]
    # rows so strict providers accept them mid-history, but they are model-facing
    # metadata -- never a user turn. They must not render as a user bubble on any
    # surface, and dropping them from the display projection also stops the
    # stored marker from shifting the desktop's user-message ordinals and
    # duplicating the optimistic prompt (#67603).
    history = [
        {"role": "user", "content": "first question"},
        {"role": "assistant", "content": "first answer"},
        {
            "role": "user",
            "content": "[System: The active model for this chat has changed to k3.]",
        },
        {"role": "user", "content": "second question"},
        {"role": "assistant", "content": "second answer"},
        {
            "role": "user",
            "content": (
                "[System: The user has changed the assistant's personality. "
                "Adopt the new persona going forward.]"
            ),
        },
    ]

    assert server._history_to_messages(history) == [
        {"role": "user", "text": "first question"},
        {"role": "assistant", "text": "first answer"},
        {"role": "user", "text": "second question"},
        {"role": "assistant", "text": "second answer"},
    ]


def test_history_to_messages_drops_display_hidden_scaffolding():
    # A mid-stream steer persists an interrupted-turn checkpoint. When nothing
    # reached the screen the row carries only model-facing scaffolding and is
    # marked display_kind="hidden"; the scaffolded bytes live in the server-only
    # api_content sidecar for provider replay. This projection -- the single
    # display source every client reads -- must drop the row by its declared
    # display_kind, not just the "[System:" string convention, or the raw
    # "[This response was interrupted by a user correction.]" paints as an
    # assistant bubble (and api_content must never ship to a client).
    history = [
        {"role": "user", "content": "go"},
        {
            "role": "assistant",
            "content": "[This response was interrupted by a user correction.]",
            "api_content": "[This response was interrupted by a user correction.]",
            "display_kind": "hidden",
        },
        {"role": "user", "content": "i love you"},
        {
            "role": "assistant",
            "content": "Love you too",
            "api_content": (
                "[This response was interrupted by a user correction.]\n\n"
                "Visible response before the interruption:\n\nLove you too"
            ),
        },
    ]

    projected = server._history_to_messages(history)

    assert projected == [
        {"role": "user", "text": "go"},
        {"role": "user", "text": "i love you"},
        {"role": "assistant", "text": "Love you too"},
    ]
    # Server-only sidecar never crosses the wire.
    assert all("api_content" not in m for m in projected)


def test_history_to_messages_projects_a_skill_turn_to_its_invocation():
    # A /skill invocation is persisted EXPANDED: the activation note plus the
    # entire skill body. That payload is model-facing scaffolding -- this
    # projection is the single display source every client reads, so it must
    # hand back the invocation the user typed and never the body. Without it a
    # chat bubble renders the whole skill as if the user had written it.
    scaffolded = (
        '[IMPORTANT: The user has invoked the "work" skill, indicating they '
        "want you to follow its instructions. The full skill content is "
        "loaded below.]\n\n"
        "# /work\n\nSPIN UP A WORKTREE, never the primary checkout.\n\n"
        "The user has provided the following instruction alongside the skill "
        "invocation: fix the title leak"
    )

    history = [
        {"role": "user", "content": scaffolded},
        {"role": "assistant", "content": "on it"},
    ]

    assert server._history_to_messages(history) == [
        {
            "role": "user",
            "text": "/work fix the title leak",
            "display_kind": "skill_invocation",
        },
        {"role": "assistant", "text": "on it"},
    ]


def test_history_to_messages_projects_a_bare_skill_turn_to_the_command():
    scaffolded = (
        '[IMPORTANT: The user has invoked the "work" skill, indicating they '
        "want you to follow its instructions. The full skill content is "
        "loaded below.]\n\n# /work\n\nSPIN UP A WORKTREE."
    )

    assert server._history_to_messages([{"role": "user", "content": scaffolded}]) == [
        {"role": "user", "text": "/work", "display_kind": "skill_invocation"}
    ]


def test_expand_skill_invocation_for_replay_round_trips_the_projection(
    tmp_path, monkeypatch
):
    # Rewind/regenerate replays a turn from what the transcript SHOWS, and a
    # skill turn shows its invocation. Re-running that verbatim would send the
    # agent the literal "/work fix it" instead of the skill, so the server
    # re-expands it — the exact inverse of _skill_scaffold_projection, with the
    # body never leaving the server.
    from agent import skill_commands
    from agent import skill_utils
    from tools import skills_tool

    skills_dir = tmp_path / "skills"
    (skills_dir / "worktree-kickoff").mkdir(parents=True)
    (skills_dir / "worktree-kickoff" / "SKILL.md").write_text(
        "---\nname: worktree-kickoff\ndescription: Spin up a worktree\n---\n\n"
        "# kickoff\n\nSPIN UP A WORKTREE, never the primary checkout.\n"
    )
    monkeypatch.setattr(skills_tool, "SKILLS_DIR", skills_dir)
    monkeypatch.setattr(skill_utils, "get_external_skills_dirs", lambda *a, **k: [])
    monkeypatch.setattr(skill_commands, "_skill_commands_by_key", {})
    skill_commands.scan_skill_commands()

    expanded = server._expand_skill_invocation_for_replay(
        "/worktree-kickoff fix it", "task-1"
    )

    assert "SPIN UP A WORKTREE" in expanded
    assert server._skill_scaffold_projection(expanded) == "/worktree-kickoff fix it"


def test_expand_skill_invocation_for_replay_leaves_ordinary_text_alone(monkeypatch):
    from agent import skill_commands
    from agent import skill_utils

    monkeypatch.setattr(skill_utils, "get_external_skills_dirs", lambda *a, **k: [])
    monkeypatch.setattr(skill_commands, "_skill_commands_by_key", {})

    assert server._expand_skill_invocation_for_replay("just words", "t") == "just words"
    # A core slash command is not a skill — nothing to expand.
    assert server._expand_skill_invocation_for_replay("/status", "t") == "/status"


def test_command_dispatch_and_catalog_resolve_project_skills_from_the_session_cwd(tmp_path, monkeypatch):
    # TUI/desktop: the RPC thread has no session context and the terminal scope / process env resolve a
    # placeholder ``terminal.cwd`` to $HOME, so project skills of the session's repo never registered and
    # ``/<name>`` died with "not a quick/plugin/bundle/skill command". Two sessions in two trusted repos
    # in ONE process must each catalog and dispatch their own repo's skill (the cached registry is
    # keyed by project root, not just platform + home).
    from agent import skill_utils

    _two_repo_project_skill_sessions(tmp_path, monkeypatch)
    for sid, own, other in (("sid-a", "alpha-skill", "beta-skill"), ("sid-b", "beta-skill", "alpha-skill")):
        catalog = server._methods["commands.catalog"]("c", {"session_id": sid})["result"]
        assert f"/{own}" in catalog["skills"] and f"/{other}" not in catalog["skills"]
        res = server._methods["command.dispatch"]("d", {"name": own, "arg": "go", "session_id": sid})
        assert res["result"]["type"] == "skill", res
        assert f"BODY OF {own.upper()}" in res["result"]["message"]
        miss = server._methods["command.dispatch"]("m", {"name": other, "arg": "", "session_id": sid})
        assert miss["error"]["code"] == 4018
    # Nothing leaks past the RPC: the thread's logical cwd is unbound again.
    assert skill_utils.find_project_root() is None


def test_command_dispatch_reviews_staged_skill_writes(tmp_path, monkeypatch):
    # #98330/#118812: command.dispatch is the desktop/TUI FALLBACK stage when the slash
    # worker cannot start. With no /skills route there, a failed worker turned every
    # review command into the routing refusal, stranding staged writes with no surface
    # left. The review slice (pending/approve/reject/diff/approval) must answer; the
    # CLI-hub mutations must keep refusing.
    from tools import write_approval as wa

    home = tmp_path / "home"
    home.mkdir()
    (home / "config.yaml").write_text("skills:\n  write_approval: true\n", encoding="utf-8")
    monkeypatch.setenv("HERMES_HOME", str(home))
    rec = wa.stage_write(
        wa.SKILLS,
        {"action": "create", "name": "demo-review",
         "content": "---\nname: demo-review\ndescription: Use when testing staged review.\n---\n\nBody.\n"},
        summary="create demo-review", origin="background_review")
    server._sessions["sid-wa"] = _session(profile_home=str(home))
    try:
        pending = server._methods["command.dispatch"](
            "d1", {"name": "skills", "arg": "pending", "session_id": "sid-wa"})
        assert pending["result"]["type"] == "exec", pending
        assert rec["id"] in pending["result"]["output"]
        assert "demo-review" in pending["result"]["output"]

        diff = server._methods["command.dispatch"](
            "d2", {"name": "skills", "arg": f"diff {rec['id']}", "session_id": "sid-wa"})
        assert diff["result"]["type"] == "exec", diff
        assert "demo-review" in diff["result"]["output"]

        reject = server._methods["command.dispatch"](
            "d3", {"name": "skills", "arg": f"reject {rec['id']}", "session_id": "sid-wa"})
        assert "Rejected pending skills write" in reject["result"]["output"], reject
        assert wa.pending_count(wa.SKILLS) == 0

        # The hub mutations are NOT routed — the desktop allowlist mirrors this refusal.
        refused = server._methods["command.dispatch"](
            "d4", {"name": "skills", "arg": "install demo", "session_id": "sid-wa"})
        assert refused["error"]["code"] == 4018, refused
    finally:
        server._sessions.pop("sid-wa", None)


def test_command_dispatch_skills_gate_off_answers_instead_of_refusing(tmp_path, monkeypatch):
    # Gateway parity: the gate being off must not strand writes that are already staged,
    # and with an empty pile it points at the toggle instead of the routing refusal.
    home = tmp_path / "home"
    home.mkdir()
    (home / "config.yaml").write_text("skills:\n  write_approval: false\n", encoding="utf-8")
    monkeypatch.setenv("HERMES_HOME", str(home))
    server._sessions["sid-wa"] = _session(profile_home=str(home))
    try:
        res = server._methods["command.dispatch"](
            "d", {"name": "skills", "arg": "pending", "session_id": "sid-wa"})
        assert res["result"]["type"] == "exec", res
        assert "write_approval" in res["result"]["output"], res
    finally:
        server._sessions.pop("sid-wa", None)


def test_command_dispatch_memory_pending_reviews_staged_writes(tmp_path, monkeypatch):
    # Same fallback for /memory (#118812): the store must resolve under the session's
    # profile home, applying the write on approve instead of dropping it.
    from tools import write_approval as wa

    home = tmp_path / "home"
    home.mkdir()
    (home / "config.yaml").write_text("memory:\n  write_approval: true\n", encoding="utf-8")
    monkeypatch.setenv("HERMES_HOME", str(home))
    wa.stage_write(wa.MEMORY, {"action": "add", "target": "memory", "content": "dispatched fact"},
                   summary="add dispatched fact", origin="foreground")
    server._sessions["sid-wa"] = _session(profile_home=str(home))
    try:
        res = server._methods["command.dispatch"](
            "d1", {"name": "memory", "arg": "pending", "session_id": "sid-wa"})
        assert res["result"]["type"] == "exec", res
        assert "dispatched fact" in res["result"]["output"]

        approve = server._methods["command.dispatch"](
            "d2", {"name": "memory", "arg": "approve all", "session_id": "sid-wa"})
        assert "Approved 1" in approve["result"]["output"], approve
        assert wa.pending_count(wa.MEMORY) == 0
    finally:
        server._sessions.pop("sid-wa", None)


def test_complete_slash_and_skills_reload_are_bound_to_the_session_cwd(tmp_path, monkeypatch):
    # The '/' popup and /reload-skills ran the registry unbound: the popup never offered a project skill
    # ``command.dispatch`` accepts, and a rescan after that dispatch reported the session's project skills
    # as "Removed" and republished a registry without them.

    _two_repo_project_skill_sessions(tmp_path, monkeypatch)
    items = server._methods["complete.slash"]("s", {"text": "/alph", "session_id": "sid-a"})["result"]["items"]
    assert [i["text"] for i in items if i["kind"] == "skill"] == ["alpha-skill"]
    assert server._methods["command.dispatch"]("d", {"name": "alpha-skill", "arg": "", "session_id": "sid-a"})[
        "result"]["type"] == "skill"
    reload = server._methods["skills.reload"]("r", {"session_id": "sid-a"})["result"]
    assert reload["result"]["removed"] == [], reload["output"]
    # The session-bound registry still offers the project skill after the reload
    # (the multi-slot cache keeps each session-identity's view; the launch-env
    # identity outside any session legitimately sees none of them).
    items = server._methods["complete.slash"]("s", {"text": "/alph", "session_id": "sid-a"})["result"]["items"]
    assert [i["text"] for i in items if i["kind"] == "skill"] == ["alpha-skill"], reload["output"]
    # Another session's reload resolves ITS repo, not the launch env.
    other = server._methods["skills.reload"]("r", {"session_id": "sid-b"})["result"]
    assert {i["name"] for i in other["result"]["added"]} == {"beta-skill"}, other["output"]


def test_history_to_messages_types_a_legacy_auto_continue_row():
    # A crash-interrupted turn used to be typed only AFTER it finished, so a
    # turn killed a second time (or any row written before turn-start typing
    # landed) sits in the DB untyped and painted the raw recovery note as a
    # user bubble. The projection recognizes the synthetic note's fixed
    # prefix so those rows still read as a timeline event.
    history = [
        {"role": "user", "content": "keep going"},
        {"role": "user", "content": server._auto_continue_note("keep going")},
    ]

    projected = server._history_to_messages(history)

    assert projected == [
        {"role": "user", "text": "keep going"},
        {
            "role": "user",
            "text": server._auto_continue_note("keep going"),
            "display_kind": "auto_continue",
        },
    ]


def test_history_to_messages_keeps_real_user_bracket_text():
    # Only role=user rows whose text OPENS with the [System: marker sentinel are
    # bookkeeping notices. A genuine user turn that merely mentions the token is
    # a real message and stays visible.
    history = [
        {"role": "user", "content": "why does [System: ...] show up in my chat?"},
        {"role": "assistant", "content": "it should not"},
    ]

    assert server._history_to_messages(history) == [
        {"role": "user", "text": "why does [System: ...] show up in my chat?"},
        {"role": "assistant", "text": "it should not"},
    ]


@pytest.mark.parametrize("omit_messages", [False, True])
def test_session_resume_uses_parent_lineage_for_display(monkeypatch, omit_messages):
    captured = {}
    target = "tip-omit" if omit_messages else "tip-full"

    class FakeDB:
        def get_session(self, target):
            return {"id": target}

        def reopen_session(self, target):
            captured["reopened"] = target

        def get_resume_conversations(self, session_id):
            return (
                self.get_messages_as_conversation(session_id, repair_alternation=True),
                self.get_messages_as_conversation(session_id, include_ancestors=True),
            )

        def get_ancestor_display_prefix(self, _sid):
            return []

        def get_messages_as_conversation(
            self,
            target,
            include_ancestors=False,
            repair_alternation=False,
            include_row_ids=False,
            **_kwargs,
        ):
            captured.setdefault("history_calls", []).append(
                (target, include_ancestors, include_row_ids)
            )
            return (
                [
                    {"role": "user", "content": "root prompt"},
                    {"role": "assistant", "content": "root answer"},
                ]
                if include_ancestors
                else [{"role": "user", "content": "tip prompt"}]
            )

    monkeypatch.setattr(server, "_get_db", lambda: FakeDB())
    monkeypatch.setattr(server, "_enable_gateway_prompts", lambda: None)
    monkeypatch.setattr(server, "_set_session_context", lambda target: [])
    monkeypatch.setattr(server, "_clear_session_context", lambda tokens: None)
    monkeypatch.setattr(
        server,
        "_make_agent",
        lambda *args, **kwargs: types.SimpleNamespace(model="test"),
    )
    monkeypatch.setattr(
        server,
        "_session_info",
        lambda agent, *a: {"model": "test", "tools": {}, "skills": {}},
    )
    monkeypatch.setattr(
        server, "_init_session", lambda sid, key, agent, history, cols=80, **_kwargs: None
    )
    # The deferred pre-warm timer is neutered module-wide by the autouse
    # _neuter_agent_prewarm_timer fixture; this test only asserts the
    # returned display history.

    params = {"session_id": target}
    if omit_messages:
        params["omit_messages"] = True
    resp = server.handle_request(
        {"id": "1", "method": "session.resume", "params": params}
    )

    expected = [] if omit_messages else [
        {"role": "user", "text": "root prompt"},
        {"role": "assistant", "text": "root answer"},
    ]
    assert resp["result"]["messages"] == expected
    assert resp["result"]["message_count"] == (1 if omit_messages else 2)
    assert resp["result"]["messages_omitted"] is omit_messages


def test_live_visible_history_falls_back_without_db_or_key():
    in_memory = [{"role": "user", "content": "hi"}]
    # No DB handle available.
    assert server._live_visible_history({"session_key": "s"}, None, in_memory) == in_memory

    # DB available but the session has no persist key yet.
    class DB:
        def get_messages_as_conversation(self, *a, **k):  # pragma: no cover - not reached
            raise AssertionError("must not query without a session_key")

    assert server._live_visible_history({}, DB(), in_memory) == in_memory


def test_live_visible_history_falls_back_when_db_empty():
    """A brand-new live session whose first turn hasn't been flushed keeps its
    in-memory history rather than rendering empty."""
    in_memory = [{"role": "user", "content": "fresh turn not flushed yet"}]

    class EmptyDB:
        def get_messages_as_conversation(self, *a, **k):
            return []

    assert server._live_visible_history({"session_key": "s"}, EmptyDB(), in_memory) == in_memory


def test_live_visible_history_falls_back_when_db_raises():
    in_memory = [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "yo"}]

    class BrokenDB:
        def get_messages_as_conversation(self, *a, **k):
            raise RuntimeError("db exploded")

    assert server._live_visible_history({"session_key": "s"}, BrokenDB(), in_memory) == in_memory


def test_live_visible_history_keeps_candidate_and_fresh_tail():
    """The hard case: the persisted candidate (missing from in-memory) AND a
    not-yet-flushed live turn (missing from the DB) must BOTH survive."""
    # Persisted display: has the verification candidate, lags the newest turn.
    db_display = [
        {"role": "user", "content": "turn 1"},
        {"role": "assistant", "content": "long substantive answer",
         "finish_reason": "verification_required"},
        {"role": "assistant", "content": "terse verified reply"},
    ]
    # In-memory model history: candidate collapsed out, but has a fresh turn 2.
    in_memory = [
        {"role": "user", "content": "turn 1"},
        {"role": "assistant", "content": "terse verified reply"},
        {"role": "user", "content": "turn 2 not flushed"},
        {"role": "assistant", "content": "turn 2 reply not flushed"},
    ]

    class DB:
        def get_messages_as_conversation(self, key, include_ancestors=False, repair_alternation=False, **_kwargs):
            return list(db_display)

    result = server._live_visible_history({"session_key": "s1"}, DB(), in_memory)
    assert result == [
        {"role": "user", "content": "turn 1"},
        {"role": "assistant", "content": "long substantive answer",
         "finish_reason": "verification_required"},
        {"role": "assistant", "content": "terse verified reply"},
        {"role": "user", "content": "turn 2 not flushed"},
        {"role": "assistant", "content": "turn 2 reply not flushed"},
    ]


def test_reconcile_display_with_live_trusts_db_when_tail_absent():
    """If the DB tail isn't in memory (DB ahead / diverged), don't duplicate —
    serve the persisted display."""
    db_display = [
        {"role": "user", "content": "a"},
        {"role": "assistant", "content": "b"},
    ]
    in_memory = [{"role": "user", "content": "unrelated"}]
    assert server._reconcile_display_with_live(db_display, in_memory) == db_display
    assert server._reconcile_display_with_live([], in_memory) == in_memory
    assert server._reconcile_display_with_live(db_display, []) == db_display


def test_live_visible_history_keeps_an_unflushed_repeat_of_the_last_turn_real_db(tmp_path):
    """The live tail is anchored on the last DB row's durable id, not its text: sending the same words again
    ("ok" -> "done") before the flush must not fold the new turn into the persisted one."""
    from hermes_state import SessionDB

    db = SessionDB(db_path=tmp_path / "state.db")
    db.create_session("s1", source="tui")
    db.append_message("s1", role="user", content="ok")
    db.append_message("s1", role="assistant", content="done")
    live = db.get_messages_as_conversation("s1") + [
        {"role": "user", "content": "ok"}, {"role": "assistant", "content": "done"}]

    merged = server._live_visible_history({"session_key": "s1"}, db, live)
    assert [m["content"] for m in merged] == ["ok", "done", "ok", "done"]
    # Fully flushed: the same four rows, never doubled.
    for msg in live[2:]:
        db.append_message("s1", role=msg["role"], content=msg["content"])
    flushed = db.get_messages_as_conversation("s1")
    assert [m["content"] for m in server._live_visible_history({"session_key": "s1"}, db, flushed)] == [
        "ok", "done", "ok", "done"]


def test_live_visible_history_matches_eager_resume_with_real_db(tmp_path):
    """E2E cross-builder consistency against a real SessionDB.

    A persisted verification candidate (finish_reason=verification_required)
    is collapsed out of the model history by repair_message_sequence but kept
    in the display lineage (#65919). The warm/live projection
    (_live_visible_history) must equal the eager session.resume display
    projection — both keeping the candidate — so switching to a live session
    shows the same substantive answer a cold resume would.
    """
    from hermes_state import SessionDB

    db = SessionDB(db_path=tmp_path / "state.db")
    db.create_session("s1", source="tui")
    db.append_message("s1", role="user", content="do the thing")
    db.append_message(
        "s1", role="assistant", content="long substantive answer",
        finish_reason="verification_required",
    )
    db.append_message(
        "s1", role="assistant", content="terse verified reply", finish_reason="stop",
    )

    model_history, display_history = db.get_resume_conversations("s1")

    # The divergence #65919 introduced: candidate absent from the model
    # projection, present in the display projection.
    assert not any("long substantive" in (m.get("content") or "") for m in model_history)
    assert any("long substantive" in (m.get("content") or "") for m in display_history)

    # Eager session.resume serves the display projection.
    eager_messages = server._history_to_messages(display_history)
    # Warm/live reuse: in-memory history is the collapsed model projection.
    live_history = server._live_visible_history({"session_key": "s1"}, db, list(model_history))
    # They must agree — the candidate survives the warm switch.
    assert server._history_to_messages(live_history) == eager_messages
    assert any(m.get("text") == "long substantive answer" for m in eager_messages)


def test_live_visible_history_keeps_candidate_and_new_flushed_turn_real_db(tmp_path):
    """Real-DB variant of the combined case: a candidate from turn 1 AND a
    fully-flushed turn 2 both appear once."""
    from hermes_state import SessionDB

    db = SessionDB(db_path=tmp_path / "state.db")
    db.create_session("s1", source="tui")
    db.append_message("s1", role="user", content="turn 1")
    db.append_message(
        "s1", role="assistant", content="candidate answer",
        finish_reason="verification_required",
    )
    db.append_message("s1", role="assistant", content="verified reply", finish_reason="stop")
    db.append_message("s1", role="user", content="turn 2")
    db.append_message("s1", role="assistant", content="turn 2 reply", finish_reason="stop")

    model_history, _display_history = db.get_resume_conversations("s1")
    live_history = server._live_visible_history({"session_key": "s1"}, db, list(model_history))
    texts = [m.get("text") for m in server._history_to_messages(live_history)]

    assert texts == [
        "turn 1",
        "candidate answer",
        "verified reply",
        "turn 2",
        "turn 2 reply",
    ]


def test_live_session_payload_reads_profile_db_not_launch_db(monkeypatch, tmp_path):
    """Warm/live reuse for a non-launch profile session must open that
    profile's state.db, not the process launch DB.

    App-global remote mode stores verification candidates in the resumed
    profile's DB. ``_live_session_payload`` previously hard-coded
    ``_get_db()`` (launch), so the display projection missed those rows and
    fell back to collapsed in-memory model history — while eager
    ``session.resume`` against the same profile still showed them.
    """
    from hermes_state import SessionDB

    launch_home = tmp_path / "launch"
    profile_home = tmp_path / "profile"
    launch_home.mkdir()
    profile_home.mkdir()

    launch_db = SessionDB(db_path=launch_home / "state.db")
    profile_db = SessionDB(db_path=profile_home / "state.db")
    profile_db.create_session("s-profile", source="tui")
    profile_db.append_message("s-profile", role="user", content="do the thing")
    profile_db.append_message(
        "s-profile",
        role="assistant",
        content="long substantive answer",
        finish_reason="verification_required",
    )
    profile_db.append_message(
        "s-profile",
        role="assistant",
        content="terse verified reply",
        finish_reason="stop",
    )
    model_history, display_history = profile_db.get_resume_conversations("s-profile")
    assert not any("long substantive" in (m.get("content") or "") for m in model_history)
    assert any("long substantive" in (m.get("content") or "") for m in display_history)

    session = {
        "session_key": "s-profile",
        "profile_home": str(profile_home),
        "agent": None,
        "history": list(model_history),
        "display_history_prefix": [],
        "history_lock": threading.Lock(),
        "created_at": 1.0,
        "last_active": 1.0,
        "running": False,
    }
    # Launch DB has no row for this session — the pre-fix path would miss
    # candidates and fall back to collapsed in-memory history.
    monkeypatch.setattr(server, "_get_db", lambda: launch_db)

    payload = server._live_session_payload("live1", session, touch=False)
    texts = [m.get("text") for m in payload.get("messages") or []]

    assert "long substantive answer" in texts
    assert texts == [m.get("text") for m in server._history_to_messages(display_history)]


def test_lazy_child_watch_resume_serves_candidate_inclusive_display(monkeypatch, tmp_path):
    """The delegated-child watch-window cold resume (lazy=True) must serve the
    verbatim display projection so a persisted verification candidate is not
    collapsed out of the watch window (#65919 sibling of the warm-payload fix).
    """
    from hermes_state import SessionDB

    db = SessionDB(db_path=tmp_path / "state.db")
    db.create_session("child1", source="tui")
    db.append_message("child1", role="user", content="child prompt")
    db.append_message(
        "child1", role="assistant", content="child substantive answer",
        finish_reason="verification_required",
    )
    db.append_message(
        "child1", role="assistant", content="child terse reply", finish_reason="stop",
    )

    lease = types.SimpleNamespace(session_id="child1", release=lambda: None)

    monkeypatch.setattr(server, "_get_db", lambda: db)
    monkeypatch.setattr(server, "_enable_gateway_prompts", lambda: None)
    monkeypatch.setattr(
        server, "_claim_active_session_slot", lambda *a, **k: (lease, None)
    )
    monkeypatch.setattr(
        server, "_deferred_session_record", lambda *a, **k: {"created_at": 123.0}
    )
    monkeypatch.setattr(server, "_claim_or_reuse_live", lambda *a, **k: None)
    monkeypatch.setattr(server, "_child_run_active", lambda *a, **k: False)
    monkeypatch.setattr(server, "_schedule_session_cap_enforcement", lambda *a, **k: None)

    resp = server.handle_request(
        {
            "id": "1",
            "method": "session.resume",
            "params": {"session_id": "child1", "lazy": True},
        }
    )

    assert "error" not in resp, resp
    texts = [m.get("text") for m in resp["result"]["messages"]]
    assert "child substantive answer" in texts
    assert texts == ["child prompt", "child substantive answer", "child terse reply"]



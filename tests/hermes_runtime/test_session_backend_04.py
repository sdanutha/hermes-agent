

import json








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


def test_session_resume_deferred_history_acknowledges_and_reuses(monkeypatch):
    history_started = threading.Event()
    release_history = threading.Event()
    build_started = threading.Event()
    history_calls = []
    auto_continue_calls = []
    ancestor = {"role": "assistant", "content": "ancestor"}
    loaded = {"role": "user", "content": "loaded"}

    class FakeDB:
        def get_session(self, target):
            return {"id": target, "message_count": 1200}

        def resolve_resume_session_id(self, target):
            return target

        def reopen_session(self, target):
            assert target == "large-session"

        def get_resume_conversations(self, target):
            history_calls.append(("resume", target))
            history_started.set()
            assert release_history.wait(timeout=2.0)
            return [loaded], [ancestor, loaded]

        def get_ancestor_display_prefix(self, target):
            history_calls.append(("prefix", target))
            return [ancestor]

    monkeypatch.setattr(server, "_get_db", lambda: FakeDB())
    monkeypatch.setattr(server, "_enable_gateway_prompts", lambda: None)
    monkeypatch.setattr(server, "_schedule_session_cap_enforcement", lambda: None)
    monkeypatch.setattr(
        server,
        "_start_agent_build",
        lambda _sid, _session: build_started.set(),
    )
    monkeypatch.setattr(
        server,
        "_maybe_schedule_auto_continue",
        lambda sid, session, stored_id: auto_continue_calls.append(
            (sid, session, stored_id)
        ),
    )

    try:
        first = server._methods["session.resume"](
            "r1",
            {
                "session_id": "large-session",
                "source": "desktop",
                "defer_history": True,
            },
        )

        assert first["result"]["hydrating"] is True
        assert first["result"]["messages"] == []
        assert first["result"]["message_count"] == 1200
        assert history_started.wait(timeout=1.0)

        second = server._methods["session.resume"](
            "r2",
            {"session_id": "large-session", "defer_history": True},
        )
        assert second["result"]["session_id"] == first["result"]["session_id"]
        assert second["result"]["hydrating"] is True
        assert second["result"]["messages"] == []

        release_history.set()
        sid = first["result"]["session_id"]
        assert server._sessions[sid]["resume_history_ready"].wait(timeout=1.0)
        assert build_started.wait(timeout=1.0)
        assert history_calls == [
            ("resume", "large-session"),
            ("prefix", "large-session"),
        ]
        assert server._sessions[sid]["history"] == [loaded]
        assert server._sessions[sid]["display_history_prefix"] == [ancestor]
        assert server._sessions[sid]["resume_message_count"] == 2
        assert auto_continue_calls == [(sid, server._sessions[sid], "large-session")]
    finally:
        release_history.set()
        for sid, session in list(server._sessions.items()):
            if session.get("session_key") == "large-session":
                lease = session.get("active_session_lease")
                if lease is not None:
                    lease.release()
                server._sessions.pop(sid, None)


def test_session_resume_deferred_history_failure_can_retry(monkeypatch):
    first_released = threading.Event()
    build_started = threading.Event()
    attempts = 0

    class FakeDB:
        def get_session(self, target):
            return {"id": target, "message_count": 1}

        def resolve_resume_session_id(self, target):
            return target

        def reopen_session(self, _target):
            pass

        def get_resume_conversations(self, _target):
            nonlocal attempts
            attempts += 1
            if attempts == 1:
                first_released.set()
                raise RuntimeError("sqlite read failed")
            loaded = [{"role": "user", "content": "retry loaded"}]
            return loaded, loaded

        def get_ancestor_display_prefix(self, _target):
            return []

    monkeypatch.setattr(server, "_get_db", lambda: FakeDB())
    monkeypatch.setattr(
        server,
        "_claim_active_session_slot",
        lambda *_args, **_kwargs: pytest.fail(
            "resume must not claim a session slot before the first prompt"
        ),
    )
    monkeypatch.setattr(server, "_enable_gateway_prompts", lambda: None)
    monkeypatch.setattr(server, "_schedule_session_cap_enforcement", lambda: None)
    monkeypatch.setattr(
        server,
        "_start_agent_build",
        lambda _sid, _session: build_started.set(),
    )

    try:
        first = server._methods["session.resume"](
            "r1",
            {"session_id": "retry-session", "defer_history": True},
        )
        first_sid = first["result"]["session_id"]
        assert first_released.wait(timeout=1.0)
        assert first_sid not in server._sessions

        second = server._methods["session.resume"](
            "r2",
            {"session_id": "retry-session", "defer_history": True},
        )
        second_sid = second["result"]["session_id"]
        assert second_sid != first_sid
        assert server._sessions[second_sid]["resume_history_ready"].wait(timeout=1.0)
        assert build_started.wait(timeout=1.0)
    finally:
        for sid, session in list(server._sessions.items()):
            if session.get("session_key") == "retry-session":
                lease = session.get("active_session_lease")
                if lease is not None:
                    lease.release()
                server._sessions.pop(sid, None)


def test_session_resume_deferred_history_close_cancels_build(monkeypatch):
    history_started = threading.Event()
    release_history = threading.Event()
    build_started = threading.Event()

    class FakeDB:
        def get_session(self, target):
            return {"id": target, "message_count": 1}

        def resolve_resume_session_id(self, target):
            return target

        def reopen_session(self, _target):
            pass

        def get_resume_conversations(self, _target):
            history_started.set()
            assert release_history.wait(timeout=2.0)
            loaded = [{"role": "user", "content": "late"}]
            return loaded, loaded

        def get_ancestor_display_prefix(self, _target):
            return []

    monkeypatch.setattr(server, "_get_db", lambda: FakeDB())
    monkeypatch.setattr(server, "_enable_gateway_prompts", lambda: None)
    monkeypatch.setattr(server, "_schedule_session_cap_enforcement", lambda: None)
    monkeypatch.setattr(
        server,
        "_start_agent_build",
        lambda _sid, _session: build_started.set(),
    )

    response = {}
    try:
        response = server._methods["session.resume"](
            "r1",
            {"session_id": "cancel-session", "defer_history": True},
        )
        sid = response["result"]["session_id"]
        session = server._sessions[sid]
        assert history_started.wait(timeout=1.0)

        assert server._close_session_by_id(sid, end_reason="tui_close") is True
        assert session["resume_history_ready"].is_set()
        assert session["resume_history_error"] == "session resume cancelled"

        release_history.set()
        time.sleep(0.05)
        assert not build_started.is_set()
        assert sid not in server._sessions
    finally:
        release_history.set()
        server._sessions.pop(response.get("result", {}).get("session_id", ""), None)


def test_session_resume_follows_compression_tip(monkeypatch, tmp_path):
    """Resuming a rotated-out parent id must load the continuation's messages.

    Regression for the desktop "I came back and the reply isn't there" report:
    auto-compression ends the live session and forks a continuation child, so a
    resume on the parent id (the desktop's routed id when the chat was opened
    before it rotated) used to reload the pre-compression transcript and drop
    the response generated after compression. session.resume must follow the
    compression tip via resolve_resume_session_id.
    """
    from hermes_state import SessionDB

    db = SessionDB(db_path=tmp_path / "state.db")
    base = int(time.time()) - 10_000
    db.create_session("parent_root", source="tui")
    db.append_message(
        "parent_root", role="user", content="pre-compression turn",
        timestamp=base + 10,
    )
    db.end_session("parent_root", "compression")
    db.create_session("cont_tip", source="tui", parent_session_id="parent_root")
    db.append_message(
        "cont_tip", role="assistant", content="post-compression reply",
        timestamp=base + 110,
    )
    conn = db._conn
    assert conn is not None
    conn.execute(
        "UPDATE sessions SET started_at = ?, ended_at = ? WHERE id = 'parent_root'",
        (base, base + 50),
    )
    conn.execute("UPDATE sessions SET started_at = ? WHERE id = 'cont_tip'", (base + 100,))
    conn.commit()

    captured = {}

    def fake_make_agent(sid, key, session_id=None, session_db=None, **kwargs):
        # Record only the FIRST (synchronous, eager) build. A stray background
        # build leaked from an earlier test's deferred resume could otherwise
        # overwrite this with its own session_id and corrupt the assertion.
        captured.setdefault("agent_session_id", session_id)
        return types.SimpleNamespace(model="test", provider="test")

    monkeypatch.setattr(server, "_get_db", lambda: db)
    monkeypatch.setattr(server, "_enable_gateway_prompts", lambda: None)
    monkeypatch.setattr(server, "_set_session_context", lambda target, cwd=None: [])
    monkeypatch.setattr(server, "_clear_session_context", lambda tokens: None)
    monkeypatch.setattr(server, "_make_agent", fake_make_agent)
    monkeypatch.setattr(
        server, "_session_info", lambda agent, *a: {"model": "test", "tools": {}, "skills": {}}
    )
    monkeypatch.setattr(
        server, "_init_session", lambda sid, key, agent, history, cols=80, **_kwargs: None
    )

    try:
        # eager_build: this asserts the synchronously-built agent binds to the
        # resolved tip (captured["agent_session_id"]); the compression-tip
        # resolution itself runs before the build and is mode-agnostic.
        resp = server.handle_request(
            {"id": "1", "method": "session.resume", "params": {"session_id": "parent_root", "eager_build": True}}
        )
    finally:
        db.close()

    # The agent must bind to the continuation tip, and the returned transcript
    # must include the post-compression reply (which lives only in the tip).
    assert resp["result"]["session_key"] == "cont_tip"
    assert captured["agent_session_id"] == "cont_tip"
    texts = [m.get("text") for m in resp["result"]["messages"]]
    assert "post-compression reply" in texts


def test_session_resume_passes_stored_runtime_to_agent(monkeypatch):
    captured = {}

    class FakeDB:
        def get_session(self, target):
            return {
                "id": target,
                "model": "gpt-5.4",
                "billing_provider": "openai-codex",
                "model_config": '{"reasoning_config":{"enabled":true,"effort":"high"},"service_tier":"priority","base_url":"https://custom.example/v1","api_mode":"chat_completions"}',
            }

        def reopen_session(self, target):
            pass

        def get_resume_conversations(self, session_id):
            return (
                self.get_messages_as_conversation(session_id, repair_alternation=True),
                self.get_messages_as_conversation(session_id, include_ancestors=True),
            )

        def get_ancestor_display_prefix(self, _sid):
            return []

        def get_messages_as_conversation(self, target, include_ancestors=False, repair_alternation=False, **_kwargs):
            return [{"role": "user", "content": "hello"}]

    def fake_make_agent(sid, key, session_id=None, session_db=None, **kwargs):
        captured.update(kwargs)
        return types.SimpleNamespace(model="gpt-5.4", provider="openai-codex")

    monkeypatch.setattr(server, "_get_db", lambda: FakeDB())
    monkeypatch.setattr(server, "_enable_gateway_prompts", lambda: None)
    monkeypatch.setattr(server, "_set_session_context", lambda target, cwd=None: [])
    monkeypatch.setattr(server, "_clear_session_context", lambda tokens: None)
    monkeypatch.setattr(server, "_make_agent", fake_make_agent)
    monkeypatch.setattr(server, "_session_info", lambda agent, *a: {"model": agent.model, "provider": agent.provider})

    def fake_init_session(sid, key, agent, history, cols=80, **_kwargs):
        server._sessions[sid] = {"agent": agent, "session_key": key}

    monkeypatch.setattr(server, "_init_session", fake_init_session)

    # eager_build: this asserts the synchronous build contract (stored runtime
    # overrides reach _make_agent, info comes from _session_info). The deferred
    # default restores the same overrides via _start_agent_build off-thread.
    resp = server.handle_request(
        {"id": "1", "method": "session.resume", "params": {"session_id": "stored-session", "eager_build": True}}
    )

    assert captured["model_override"] == {
        "model": "gpt-5.4",
        "provider": "openai-codex",
        "base_url": "https://custom.example/v1",
        "api_mode": "chat_completions",
    }
    assert captured["provider_override"] == "openai-codex"
    assert captured["reasoning_config_override"] == {"enabled": True, "effort": "high"}
    assert captured["service_tier_override"] == "priority"
    runtime_sid = resp["result"]["session_id"]
    assert server._sessions[runtime_sid]["model_override"] == captured["model_override"]


def test_session_resume_profile_uses_profile_db_cwd(monkeypatch, tmp_path):
    target = "stored-profile-session"
    launch_cwd = tmp_path / "launch"
    profile_cwd = tmp_path / "worker"
    profile_home = tmp_path / "profiles" / "worker"
    launch_cwd.mkdir()
    profile_cwd.mkdir()
    profile_home.mkdir(parents=True)
    captured = {}

    class ProfileDB:
        def get_session(self, _target):
            return {"id": target, "cwd": str(profile_cwd)}

        def get_session_by_title(self, _target):
            return None

        def reopen_session(self, _target):
            captured["reopened"] = _target

        def get_resume_conversations(self, session_id):
            return (
                self.get_messages_as_conversation(session_id, repair_alternation=True),
                self.get_messages_as_conversation(session_id, include_ancestors=True),
            )

        def get_ancestor_display_prefix(self, _sid):
            return []

        def get_messages_as_conversation(self, _target, include_ancestors=False, repair_alternation=False, **_kwargs):
            return [{"role": "user", "content": "hello"}]

        def update_session_cwd(self, *_args):
            raise AssertionError("profile row already has cwd")

    class LaunchDB:
        def get_session(self, _target):
            return {"id": target, "cwd": str(launch_cwd)}

        def update_session_cwd(self, *_args):
            captured["launch_update"] = True

    profile_db = ProfileDB()
    launch_db = LaunchDB()

    class FakeWorker:
        def __init__(self, *_args, **_kwargs):
            pass

        def close(self):
            pass

    def fake_make_agent(sid, key, session_id=None, session_db=None, **kwargs):
        captured["agent_db"] = session_db
        captured["agent_cwd"] = kwargs.get("cwd_override")
        return types.SimpleNamespace(model="test/model")

    monkeypatch.setenv("TERMINAL_CWD", str(launch_cwd))
    monkeypatch.setattr(server, "_profile_home", lambda _profile: profile_home)
    monkeypatch.setattr("hermes_state_registry.acquire", lambda db_path=None: profile_db)
    monkeypatch.setattr(server, "_get_db", lambda: launch_db)
    monkeypatch.setattr(server, "_enable_gateway_prompts", lambda: None)
    monkeypatch.setattr(
        server,
        "_set_session_context",
        lambda target, cwd=None: captured.setdefault("context_cwd", cwd) or [],
    )
    monkeypatch.setattr(server, "_clear_session_context", lambda tokens: None)
    monkeypatch.setattr(server, "_make_agent", fake_make_agent)
    monkeypatch.setattr(server, "_SlashWorker", FakeWorker)
    monkeypatch.setattr(server, "_wire_callbacks", lambda _sid: None)
    monkeypatch.setattr(server, "_emit", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        server,
        "_session_info",
        lambda _agent, session=None: {"cwd": session.get("cwd") if session else ""},
    )

    from tools import approval

    monkeypatch.setattr(approval, "register_gateway_notify", lambda key, cb: None)
    monkeypatch.setattr(approval, "load_permanent_allowlist", lambda: None)

    try:
        # eager_build: asserts the synchronous build receives the profile's db
        # (the deferred default builds with the same db via _start_agent_build).
        resp = server.handle_request(
            {
                "id": "1",
                "method": "session.resume",
                "params": {"session_id": target, "profile": "worker", "eager_build": True},
            }
        )

        assert "error" not in resp
        sid = resp["result"]["session_id"]
        assert captured["agent_db"] is profile_db
        assert captured["context_cwd"] == str(profile_cwd)
        assert captured["agent_cwd"] == str(profile_cwd)
        assert server._sessions[sid]["cwd"] == str(profile_cwd)
        assert resp["result"]["info"]["cwd"] == str(profile_cwd)
        assert "launch_update" not in captured
    finally:
        server._sessions.clear()


def test_session_cwd_set_profile_session_updates_profile_db(monkeypatch, tmp_path):
    target = "stored-profile-session"
    profile_home = tmp_path / "profiles" / "worker"
    profile_home.mkdir(parents=True)
    new_cwd = tmp_path / "new-workspace"
    new_cwd.mkdir()
    captured = {}

    class ProfileDB:
        def update_session_cwd(self, session_id, cwd, git_branch=None, git_repo_root=None):
            captured["profile_update"] = (session_id, cwd)

        def close(self):
            captured["profile_closed"] = True

    class LaunchDB:
        def update_session_cwd(self, *_args):
            captured["launch_update"] = True

    profile_db = ProfileDB()

    from tools import terminal_tool_lifecycle

    monkeypatch.setattr("hermes_state_registry.acquire", lambda db_path=None: profile_db)
    monkeypatch.setattr(server, "_get_db", lambda: LaunchDB())
    monkeypatch.setattr(terminal_tool_lifecycle, "cleanup_vm", lambda _key: None)
    monkeypatch.setattr(server, "_register_session_cwd", lambda _session: None)

    session = {"session_key": target, "profile_home": str(profile_home)}
    assert server._set_session_cwd(session, str(new_cwd)) == str(new_cwd)
    assert session["cwd"] == str(new_cwd)
    assert session["explicit_cwd"] is True
    assert captured["profile_update"] == (target, str(new_cwd))
    assert captured["profile_closed"] is True
    assert "launch_update" not in captured


def test_stored_session_runtime_overrides_skips_bare_billing_provider(monkeypatch):
    """A bare billing bucket ("custom"/"auto") must not be restored as the provider
    identity on resume. A custom endpoint that never used `/model` persists only
    `billing_provider="custom"`; restoring that broke `session.resume` with "No LLM provider
    configured" (agent_init treats it as non-routable). ``"openrouter"`` is NOT a bare bucket
    — it is a fully routable provider; see #57588. A real provider, or an explicit
    `model_config.provider`, is still restored.
    """
    # Bare "custom" bucket, no explicit model_config.provider: no provider override restored.
    ov = server._stored_session_runtime_overrides({"model": "my-model", "billing_provider": "custom"})
    assert "provider_override" not in ov
    assert ov["model_override"]["provider"] is None

    for bare in ("auto", "custom"):
        ov = server._stored_session_runtime_overrides({"model": "m", "billing_provider": bare})
        assert "provider_override" not in ov

    # A real provider in billing_provider is still restored.
    ov = server._stored_session_runtime_overrides({"model": "m", "billing_provider": "anthropic"})
    assert ov["provider_override"] == "anthropic"
    assert ov["model_override"]["provider"] == "anthropic"

    # An explicit ROUTABLE provider in model_config wins over the bare billing
    # bucket. It must actually resolve in the registry — a stale/renamed
    # provider is dropped (see TestStaleProviderNameFallsBack).
    cfg = {
        "custom_providers": [
            {
                "name": "myendpoint",
                "base_url": "https://myendpoint.invalid/v1",
                "api_key": "sk-test",
                "model": "m",
            }
        ]
    }
    import hermes_cli.runtime_provider as rp

    monkeypatch.setattr(rp, "load_config", lambda: cfg)
    monkeypatch.setattr("hermes_cli.config.load_config", lambda: cfg)
    ov = server._stored_session_runtime_overrides(
        {"model": "m", "billing_provider": "custom", "model_config": {"provider": "custom:myendpoint"}}
    )
    assert ov["provider_override"] == "custom:myendpoint"
    assert ov["model_override"]["provider"] == "custom:myendpoint"


def test_stored_session_runtime_overrides_restores_explicit_normal_tier():
    overrides = server._stored_session_runtime_overrides(
        {
            "model": "gpt-5.4",
            "model_config": {"service_tier": "normal"},
        }
    )

    assert "service_tier_override" in overrides
    assert overrides["service_tier_override"] == ""


def test_openrouter_session_resume_restores_provider():
    """OpenRouter is a fully routable provider — sessions that used OpenRouter must
    restore the "openrouter" provider override on resume, not fall through to whatever
    the current global model is.  (#57588)
    """
    # OpenRouter session with no explicit model_config.provider (the common case
    # for sessions that never used /model): billing_provider="openrouter" should
    # be restored as the provider override.
    ov = server._stored_session_runtime_overrides(
        {"model": "anthropic/claude-opus-4.8", "billing_provider": "openrouter"}
    )
    assert ov["provider_override"] == "openrouter"
    assert ov["model_override"]["provider"] == "openrouter"
    assert ov["model_override"]["model"] == "anthropic/claude-opus-4.8"

    # When an explicit model_config.provider exists, it takes precedence over
    # billing_provider (this path was already correct).
    ov = server._stored_session_runtime_overrides(
        {
            "model": "anthropic/claude-opus-4.8",
            "billing_provider": "openrouter",
            "model_config": {"provider": "openrouter", "base_url": "https://openrouter.ai/api/v1"},
        }
    )
    assert ov["provider_override"] == "openrouter"


def test_persist_live_session_runtime_preserves_resume_metadata(monkeypatch):
    updates = {}

    class FakeDB:
        def get_session(self, session_id):
            assert session_id == "stored-session"
            return {"model_config": '{"_branched_from":"root"}'}

        def update_session_meta(self, session_id, model_config_json, model=None):
            updates["meta"] = (session_id, json.loads(model_config_json), model)

    agent = types.SimpleNamespace(
        model="gpt-5.4",
        provider="openai-codex",
        base_url="https://custom.example/v1",
        api_mode="chat_completions",
        reasoning_config={"enabled": True, "effort": "high"},
        service_tier="priority",
        _session_db=FakeDB(),
    )

    server._persist_live_session_runtime({"agent": agent, "session_key": "stored-session"})

    assert "model" not in updates
    assert updates["meta"] == (
        "stored-session",
        {
            "_branched_from": "root",
            "model": "gpt-5.4",
            "provider": "openai-codex",
            "base_url": "https://custom.example/v1",
            "api_mode": "chat_completions",
            "reasoning_config": {"enabled": True, "effort": "high"},
            "service_tier": "priority",
        },
        "gpt-5.4",
    )


def test_persist_live_session_runtime_preserves_explicit_normal_tier():
    updates = {}

    class FakeDB:
        def get_session(self, _session_id):
            return {"model_config": '{"service_tier":"priority"}'}

        def update_session_meta(self, _session_id, model_config_json, model=None):
            updates["config"] = json.loads(model_config_json)

    agent = types.SimpleNamespace(
        model="gpt-5.4",
        provider="openai-codex",
        base_url=None,
        api_mode=None,
        reasoning_config=None,
        service_tier="",
        _session_db=FakeDB(),
    )

    server._persist_live_session_runtime(
        {
            "agent": agent,
            "session_key": "stored-session",
            "create_service_tier_override": "",
        }
    )

    assert updates["config"]["service_tier"] == "normal"


def test_status_callback_emits_kind_and_text():
    with patch("hermes_runtime.server._emit") as emit:
        cb = server._agent_cbs("sid")["status_callback"]
        cb("context_pressure", "85% to compaction")

    emit.assert_called_once_with(
        "status.update",
        "sid",
        {"kind": "context_pressure", "text": "85% to compaction"},
    )


def test_status_callback_accepts_single_message_argument():
    with patch("hermes_runtime.server._emit") as emit:
        cb = server._agent_cbs("sid")["status_callback"]
        cb("thinking...")

    emit.assert_called_once_with(
        "status.update",
        "sid",
        {"kind": "status", "text": "thinking..."},
    )


def test_resolve_model_uses_inference_model_env(monkeypatch):
    monkeypatch.delenv("HERMES_MODEL", raising=False)
    monkeypatch.setenv("HERMES_INFERENCE_MODEL", " anthropic/claude-sonnet-4.6\n")

    assert server._resolve_model() == "anthropic/claude-sonnet-4.6"


def test_resolve_model_strips_config_model(monkeypatch):
    monkeypatch.delenv("HERMES_MODEL", raising=False)
    monkeypatch.delenv("HERMES_INFERENCE_MODEL", raising=False)
    monkeypatch.setattr(
        server, "_load_cfg", lambda: {"model": {"default": " nous/hermes-test "}}
    )

    assert server._resolve_model() == "nous/hermes-test"


def test_config_sync_switches_unpinned_session(monkeypatch):
    _patch_config_model(monkeypatch, "new/model", provider="nous")
    session = _sync_test_session(config_model_seen=("old/model", "nous"))
    calls = []
    monkeypatch.setattr(
        server,
        "_apply_model_switch",
        lambda sid, sess, raw, **kw: calls.append((sid, raw, kw)),
    )

    server._sync_agent_model_with_config("sid", session)

    assert calls == [
        (
            "sid",
            "new/model --provider nous",
            {
                "confirm_expensive_model": True,
                "pin_session_override": False,
                "persist_override": False,
                "count_switch": False,
            },
        )
    ]
    assert session["config_model_seen"] == ("new/model", "nous")


def test_config_sync_treats_auto_provider_as_unset(monkeypatch):
    _patch_config_model(monkeypatch, "new/model", provider="auto")
    session = _sync_test_session(config_model_seen=("old/model", ""))
    calls = []
    monkeypatch.setattr(
        server,
        "_apply_model_switch",
        lambda sid, sess, raw, **kw: calls.append(raw),
    )

    server._sync_agent_model_with_config("sid", session)

    assert calls == ["new/model"]


def test_config_sync_skips_session_pinned_by_model_command(monkeypatch):
    _patch_config_model(monkeypatch, "new/model")
    session = _sync_test_session(
        config_model_seen=("old/model", ""),
        model_override={"model": "pinned/model"},
    )
    monkeypatch.setattr(
        server,
        "_apply_model_switch",
        lambda *a, **k: pytest.fail("pinned session must not be switched"),
    )

    server._sync_agent_model_with_config("sid", session)


def test_config_sync_noop_when_config_unchanged(monkeypatch):
    _patch_config_model(monkeypatch, "old/model")
    session = _sync_test_session(config_model_seen=("old/model", ""))
    monkeypatch.setattr(
        server,
        "_apply_model_switch",
        lambda *a, **k: pytest.fail("unchanged config must not switch"),
    )

    server._sync_agent_model_with_config("sid", session)


def test_config_sync_adopts_baseline_when_agent_already_on_target(monkeypatch):
    # Branched/resumed sessions reach their first sync with no snapshot but
    # an agent already built from config; that must not trigger a switch.
    _patch_config_model(monkeypatch, "old/model")
    session = _sync_test_session()
    monkeypatch.setattr(
        server,
        "_apply_model_switch",
        lambda *a, **k: pytest.fail("agent already on target must not switch"),
    )

    server._sync_agent_model_with_config("sid", session)

    assert session["config_model_seen"] == ("old/model", "")


def test_config_sync_switches_when_only_provider_differs(monkeypatch):
    _patch_config_model(monkeypatch, "old/model", provider="nous")
    session = _sync_test_session(config_model_seen=("old/model", ""))
    calls = []
    monkeypatch.setattr(
        server,
        "_apply_model_switch",
        lambda sid, sess, raw, **kw: calls.append(raw),
    )

    server._sync_agent_model_with_config("sid", session)

    assert calls == ["old/model --provider nous"]


def test_config_sync_failure_emits_error_once_per_edit(monkeypatch):
    _patch_config_model(monkeypatch, "broken/model")
    session = _sync_test_session(config_model_seen=("old/model", ""))

    def boom(*a, **k):
        raise ValueError("no such model")

    monkeypatch.setattr(server, "_apply_model_switch", boom)
    emits = []
    monkeypatch.setattr(
        server, "_emit", lambda ev, sid, payload: emits.append((ev, payload))
    )

    server._sync_agent_model_with_config("sid", session)
    server._sync_agent_model_with_config("sid", session)

    assert len(emits) == 1
    assert emits[0][0] == "error"
    assert "broken/model" in emits[0][1]["message"]


def test_config_sync_config_wins_over_env_seed(monkeypatch):
    # Hosted instances set HERMES_INFERENCE_MODEL as a provision-time seed;
    # the per-turn sync must follow config.yaml edits, not stay pinned to it.
    monkeypatch.setenv("HERMES_INFERENCE_MODEL", "seed/model")
    monkeypatch.delenv("HERMES_MODEL", raising=False)
    monkeypatch.setattr(server, "_load_cfg", lambda: {"model": {"default": "new/model"}})
    session = _sync_test_session(config_model_seen=("seed/model", ""))
    calls = []
    monkeypatch.setattr(
        server,
        "_apply_model_switch",
        lambda sid, sess, raw, **kw: calls.append(raw),
    )

    server._sync_agent_model_with_config("sid", session)

    assert calls == ["new/model"]
    assert session["config_model_seen"] == ("new/model", "")


def test_config_sync_ignores_env_seed_without_config_model(monkeypatch):
    # `hermes --tui -m <model>` sets HERMES_MODEL/HERMES_INFERENCE_MODEL as a
    # launch-scoped seed. When config.yaml has NO model.default (typical
    # custom-provider-only setup), the sync must NOT adopt the env seed as a
    # config target — doing so replayed the -m flag as a /model switch and
    # (with persist_switch_by_default=True) wrote it into config.yaml
    # permanently.
    monkeypatch.setenv("HERMES_MODEL", "one-shot/model")
    monkeypatch.setenv("HERMES_INFERENCE_MODEL", "one-shot/model")
    monkeypatch.setattr(
        server, "_load_cfg", lambda: {"model": {"provider": "custom:mylocal"}}
    )
    session = _sync_test_session()
    monkeypatch.setattr(
        server,
        "_apply_model_switch",
        lambda *a, **k: pytest.fail("env seed must not trigger a config sync switch"),
    )

    server._sync_agent_model_with_config("sid", session)


def test_apply_model_switch_persist_override_false_never_persists(monkeypatch):
    # Internal callers (config sync, /moa one-shot + restore) pass
    # persist_override=False; even with persist_switch_by_default=True the
    # switch must not write config.yaml.
    import types as _types

    result = _types.SimpleNamespace(
        success=True,
        new_model="new/model",
        target_provider="nous",
        base_url="",
        api_key="key",
        api_mode="chat_completions",
        warning_message="",
        model_info=None,
        error_message="",
    )
    monkeypatch.setattr(
        "hermes_cli.model_switch.switch_model", lambda **kw: result
    )
    monkeypatch.setattr(
        "hermes_cli.model_switch.resolve_persist_behavior",
        lambda *a: pytest.fail("persist_override must bypass resolve_persist_behavior"),
    )
    monkeypatch.setattr(
        "hermes_cli.model_switch.persist_model_selection",
        lambda _r: pytest.fail("persist_override=False must not persist"),
    )
    monkeypatch.setattr(
        "hermes_cli.model_cost_guard.expensive_model_warning",
        lambda *a, **k: None,
    )
    session = {"agent": None}

    out = server._apply_model_switch(
        "sid", session, "new/model --provider nous", persist_override=False
    )

    assert out["value"] == "new/model"
    assert session["model_override"]["model"] == "new/model"


def test_startup_runtime_uses_tui_provider_env(monkeypatch):
    monkeypatch.setenv("HERMES_MODEL", "nous/hermes-test")
    monkeypatch.setenv("HERMES_TUI_PROVIDER", "nous")
    monkeypatch.delenv("HERMES_INFERENCE_PROVIDER", raising=False)

    assert server._resolve_startup_runtime() == ("nous/hermes-test", "nous")


def test_startup_runtime_does_not_treat_inference_provider_as_explicit(monkeypatch):
    monkeypatch.setenv("HERMES_MODEL", "nous/hermes-test")
    monkeypatch.delenv("HERMES_TUI_PROVIDER", raising=False)
    monkeypatch.setenv("HERMES_INFERENCE_PROVIDER", "nous")
    monkeypatch.setattr(
        "hermes_cli.models.detect_static_provider_for_model",
        lambda model, provider: None,
    )

    assert server._resolve_startup_runtime() == ("nous/hermes-test", None)


def test_load_fallback_model_merges_chain_providers_first(monkeypatch):
    # Parity with HermesCLI / gateway: fallback_providers stays first and keeps
    # its order, with any distinct legacy fallback_model entry merged in after
    # (deduped on provider/model/base_url).
    fallback_chain = [
        {"provider": "openrouter", "model": "openai/gpt-5.5"},
        {"provider": "anthropic", "model": "claude-sonnet-4-6"},
    ]
    monkeypatch.setattr(
        server,
        "_load_cfg",
        lambda: {
            "fallback_model": {"provider": "legacy", "model": "legacy-model"},
            "fallback_providers": fallback_chain,
        },
    )

    assert server._load_fallback_model() == [
        {"provider": "openrouter", "model": "openai/gpt-5.5"},
        {"provider": "anthropic", "model": "claude-sonnet-4-6"},
        {"provider": "legacy", "model": "legacy-model"},
    ]


def test_make_agent_passes_configured_fallback_chain(monkeypatch):
    captured = {}
    fallback_chain = [
        {"provider": "openrouter", "model": "openai/gpt-5.5"},
    ]

    def fake_agent(**kwargs):
        captured.update(kwargs)
        return types.SimpleNamespace(model=kwargs.get("model"))

    monkeypatch.delenv("HERMES_MODEL", raising=False)
    monkeypatch.delenv("HERMES_INFERENCE_MODEL", raising=False)
    monkeypatch.delenv("HERMES_TUI_PROVIDER", raising=False)
    monkeypatch.delenv("HERMES_DESKTOP", raising=False)
    monkeypatch.delenv("HERMES_DESKTOP_TERMINAL", raising=False)
    monkeypatch.setattr(
        server,
        "_load_cfg",
        lambda: {
            "model": {"default": "gpt-5.5", "provider": "openai-codex"},
            "fallback_providers": fallback_chain,
        },
    )
    monkeypatch.setattr(
        "hermes_cli.runtime_provider.resolve_runtime_provider",
        lambda requested=None, target_model=None: {
            "provider": "openai-codex",
            "base_url": "https://chatgpt.com/backend-api/codex",
            "api_key": "token",
            "api_mode": "codex_responses",
            "credential_pool": None,
        },
    )
    monkeypatch.setattr("run_agent.AIAgent", fake_agent)
    monkeypatch.setattr(server, "_load_enabled_toolsets", lambda *_a, **_kw: ["file"])
    monkeypatch.setattr(server, "_get_db", lambda: None)

    agent = server._make_agent("sid", "session-key")

    assert agent.model == "gpt-5.5"
    assert captured["fallback_model"] == fallback_chain
    assert captured["platform"] == "tui"


def test_make_agent_forwards_agent_disabled_toolsets(monkeypatch):
    """``agent.disabled_toolsets`` must reach AIAgent in gateway sessions too: only the AIAgent
    filter strips a toolset out of composite defaults (``hermes-cli``), which the gateway's
    enabled-list resolver can't reach into (#44499)."""
    captured = _capture_make_agent_kwargs(monkeypatch)
    monkeypatch.setattr(server, "_load_enabled_toolsets", lambda *_a, **_kw: ["file"])
    monkeypatch.setattr(server, "_load_disabled_toolsets", lambda: ["browser"])

    server._make_agent("sid", "session-key")

    assert captured["disabled_toolsets"] == ["browser"]


def test_make_agent_disabled_toolsets_none_by_default(monkeypatch):
    captured = _capture_make_agent_kwargs(monkeypatch)
    monkeypatch.setattr(server, "_load_enabled_toolsets", lambda *_a, **_kw: ["file"])
    monkeypatch.setattr(server, "_load_disabled_toolsets", lambda: None)

    server._make_agent("sid", "session-key")

    assert captured["disabled_toolsets"] is None


@pytest.mark.parametrize("identity, user_id", [
    ({"user_id": "oidc|abc123", "provider": "oidc"}, "oidc:oidc|abc123"),
    ({"user_id": "alice", "provider": "basic"}, "basic:alice"),
    ({"user_id": "alice", "provider": "oidc"}, "oidc:alice"),
    (None, None),
    ({"user_id": "server-internal", "provider": "server-internal"}, None),
    ({"user_id": "", "provider": "oidc"}, None),
    ({"user_id": "abc", "provider": ""}, None),
])
def test_make_agent_passes_the_authenticated_dashboard_user_as_user_id(monkeypatch, identity, user_id):
    """WSTransport.auth_identity reaches the agent as ``user_id`` prefixed with the login provider, so a
    basic-auth ``alice`` and an OIDC ``alice`` stay two memory peers. The legacy token, stdio and the PTY
    child's server-internal credential name no human, so the agent gets no user id for them."""
    captured = _capture_make_agent_kwargs(monkeypatch)
    transport = types.SimpleNamespace(auth_identity=identity)
    monkeypatch.setitem(server._sessions, "sid-auth", {"session_key": "k", "transport": transport})

    server._make_agent("sid-auth", "k")

    assert captured["user_id"] == user_id


def test_make_agent_passes_no_user_id_for_an_unknown_session(monkeypatch):
    captured = _capture_make_agent_kwargs(monkeypatch)
    server._sessions.pop("sid-missing", None)

    server._make_agent("sid-missing", "k")

    assert captured["user_id"] is None


def test_make_agent_keeps_the_login_after_a_second_window_attaches_and_detaches(monkeypatch):
    """A pop-out turns the transport slot into a FanoutTransport, which names no login. The rebuild must read the
    login the record was created with, before and after the pop-out goes away."""
    captured = _capture_make_agent_kwargs(monkeypatch)
    first, popout = _LoginSocket(), _LoginSocket()
    record = _login_session(monkeypatch, "sid-popout", "k", first)

    assert server._attach_session_transport(record, popout)
    server._make_agent("sid-popout", "k")
    assert captured["user_id"] == "basic:alice"

    server._detach_session_transport(record, popout)
    server._make_agent("sid-popout", "k")
    assert captured["user_id"] == "basic:alice"


def test_build_branch_agent_carries_the_parent_login(monkeypatch, tmp_path):
    """The branch agent is built before its record exists, so the parent's login is passed to the build and
    copied onto the new record."""
    captured = _capture_make_agent_kwargs(monkeypatch)
    socket = _LoginSocket()
    parent = _login_session(monkeypatch, "sid-parent", "parent-key", socket, cwd=str(tmp_path), source="desktop")

    def fake_init_session(sid, key, agent, history, **kwargs):
        monkeypatch.setitem(server._sessions, sid, {"session_key": key, "transport": server._stdio_transport})

    monkeypatch.setattr(server, "_set_session_context", lambda key, cwd=None: [])
    monkeypatch.setattr(server, "_clear_session_context", lambda tokens: None)
    monkeypatch.setattr(server, "_init_session", fake_init_session)
    monkeypatch.setattr(server, "_transfer_db_to_agent", lambda *args: False)
    token = bind_transport(socket)
    try:
        server._build_branch_agent(parent, "sid-branch", "branch-key", [], "desktop")
    finally:
        reset_transport(token)

    assert captured["user_id"] == "basic:alice"
    assert server._sessions["sid-branch"]["auth_user_id"] == "basic:alice"


def test_deferred_session_record_stamps_the_creating_login():
    token = bind_transport(_LoginSocket("carol", "oidc"))
    try:
        record = server._deferred_session_record(
            "deferred-key", cols=80, cwd="/tmp", history=[], lease=None)
    finally:
        reset_transport(token)

    assert record["auth_user_id"] == "oidc:carol"
    assert server._session_auth_user_id(record) == "oidc:carol"


def test_compute_host_turn_frame_carries_the_session_login(monkeypatch):
    record = _login_session(monkeypatch, "sid-host", "host-key", _LoginSocket(), history=[],
                            history_lock=threading.Lock(), cwd="/tmp", cols=80)
    monkeypatch.setattr(server, "_session_cwd", lambda session: "/tmp")

    frame = server._compute_host_turn_frame("rid", "sid-host", record, "hello")

    assert frame["auth_user_id"] == "basic:alice"


def test_attaching_a_different_login_keeps_the_creator_and_warns_once(monkeypatch):
    """Ownership is not enforced. The record keeps the creator's login, and the attach is logged once with both
    ids rather than on every prompt the second client sends."""
    creator, other = _LoginSocket("alice"), _LoginSocket("bob")
    record = _login_session(monkeypatch, "sid-shared", "shared-key", creator)

    assert server._attach_session_transport(record, other)
    assert server._attach_session_transport(record, other)

    assert server._session_auth_user_id(record) == "basic:alice"


def test_background_agent_kwargs_preserves_full_fallback_chain(monkeypatch):
    chain = [
        {"provider": "openrouter", "model": "openai/gpt-5.5"},
        {"provider": "anthropic", "model": "claude-sonnet-4-6"},
    ]
    agent = types.SimpleNamespace(
        model="gpt-5.5",
        provider="openai-codex",
        _fallback_chain=chain,
    )
    monkeypatch.setattr(server, "_load_cfg", lambda: {"max_turns": 25})
    monkeypatch.setattr(server, "_load_enabled_toolsets", lambda *_a, **_kw: ["file"])
    monkeypatch.setattr(server, "_get_db", lambda: None)

    kwargs = server._background_agent_kwargs(agent, "task-id")

    assert kwargs["fallback_model"] == chain


def test_background_agent_kwargs_preserves_empty_fallback_chain(monkeypatch):
    agent = types.SimpleNamespace(
        model="gpt-5.5",
        provider="anthropic",
        _fallback_chain=[],
    )
    monkeypatch.setattr(
        server,
        "_load_cfg",
        lambda: {
            "max_turns": 25,
            "fallback_providers": [
                {"provider": "openrouter", "model": "openai/gpt-5.5"},
            ],
        },
    )
    monkeypatch.setattr(server, "_load_enabled_toolsets", lambda *_a, **_kw: ["file"])
    monkeypatch.setattr(server, "_get_db", lambda: None)

    kwargs = server._background_agent_kwargs(agent, "task-id")

    assert kwargs["fallback_model"] == []


def test_background_agent_kwargs_forwards_agent_disabled_toolsets(monkeypatch):
    agent = types.SimpleNamespace(
        model="gpt-5.5",
        provider="anthropic",
        _fallback_chain=[],
        disabled_toolsets=["browser"],
    )
    monkeypatch.setattr(server, "_load_cfg", lambda: {"max_turns": 25})
    monkeypatch.setattr(server, "_load_enabled_toolsets", lambda *_a, **_kw: ["file"])
    monkeypatch.setattr(server, "_get_db", lambda: None)

    kwargs = server._background_agent_kwargs(agent, "task-id")

    assert kwargs["disabled_toolsets"] == ["browser"]



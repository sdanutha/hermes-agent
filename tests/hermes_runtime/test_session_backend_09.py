

import json




import os




import sys


import threading


import time


import types




from pathlib import Path


from unittest.mock import Mock


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


def test_setup_runtime_check_agrees_with_session_fallback_chain(monkeypatch):
    """#111775: with the primary blocked and a complete fallback entry, the probe answers what
    ``_make_agent`` would build (fallback provider + model); an explicit ``provider`` stays strict
    so another provider's fallback cannot mask a failed connection."""
    from hermes_cli.auth import AuthError
    monkeypatch.setattr("hermes_cli.main._has_any_provider_configured", lambda **_kw: True)
    monkeypatch.setattr(server, "_resolve_startup_runtime", lambda: ("claude-sonnet-4-5", None))
    monkeypatch.setattr(server, "_load_fallback_model",
                        lambda: [{"provider": "openrouter", "model": "openai/gpt-4.1-mini", "api_key": "sk-or-fb"}])

    def fake_resolve(*, requested=None, target_model=None, explicit_api_key=None, **_kw):
        if requested == "openrouter":
            return {"provider": "openrouter", "api_key": explicit_api_key, "source": "explicit"}
        raise AuthError("No Anthropic credentials found.", provider="anthropic")

    monkeypatch.setattr("hermes_cli.runtime_provider.resolve_runtime_provider", fake_resolve)

    default = server.handle_request({"id": "1", "method": "setup.runtime_check", "params": {}})
    assert default["result"]["ok"] is True
    assert (default["result"]["provider"], default["result"]["model"]) == ("openrouter", "openai/gpt-4.1-mini")

    strict = server.handle_request(
        {"id": "2", "method": "setup.runtime_check", "params": {"provider": "anthropic"}})
    assert strict["result"]["ok"] is False
    assert "Anthropic" in strict["result"]["error"]


def test_setup_runtime_check_reports_target_model_on_credential_failure(monkeypatch):
    """#111775: the probe names the model session creation would use, never ``model: null``."""
    monkeypatch.setattr("hermes_cli.main._has_any_provider_configured", lambda **_kw: True)
    monkeypatch.setattr(server, "_resolve_startup_runtime", lambda: ("z-ai/glm-5.2", None))
    monkeypatch.setattr(
        "hermes_cli.runtime_provider.resolve_runtime_provider",
        lambda *, requested=None, target_model=None: {
            "provider": "zai", "api_key": "", "source": "env/config"
        },
    )

    resp = server.handle_request({"id": "1", "method": "setup.runtime_check", "params": {}})

    assert resp["result"]["ok"] is False
    assert resp["result"]["model"] == "z-ai/glm-5.2"


def test_setup_runtime_check_scopes_launch_profile_in_multiplex_backend(monkeypatch, tmp_path):
    """The launch profile needs a scope too when its Codex route reads an override."""
    from agent import secret_scope
    from hermes_runtime import launch_profile_policy

    launch_home = tmp_path / ".hermes"
    launch_home.mkdir()
    monkeypatch.setenv("HERMES_CODEX_BASE_URL", "https://codex.launch.test/v1")
    monkeypatch.setattr(server, "_hermes_home", launch_home)
    monkeypatch.setattr(launch_profile_policy, "_snapshot", None)
    monkeypatch.setattr("hermes_cli.main._has_any_provider_configured", lambda **_kw: True)
    monkeypatch.setattr(server, "_resolve_startup_runtime", lambda: ("gpt-5.3-codex", None))

    def resolve_codex(requested=None, **_kwargs):
        assert requested == "openai-codex"
        return {
            "provider": "openai-codex",
            "api_key": "codex-oauth-token",
            "base_url": secret_scope.get_secret("HERMES_CODEX_BASE_URL"),
            "source": "credential-pool",
        }

    monkeypatch.setattr("hermes_cli.runtime_provider.resolve_runtime_provider", resolve_codex)
    secret_scope.set_multiplex_active(True)
    try:
        response = server.handle_request(
            {"id": "1", "method": "setup.runtime_check", "params": {"provider": "openai-codex"}}
        )
    finally:
        secret_scope.set_multiplex_active(False)

    assert response["result"] == {
        "ok": True,
        "provider": "openai-codex",
        "model": "gpt-5.3-codex",
        "source": "credential-pool",
        "free_tier_route": False,
    }


def test_setup_readiness_scopes_to_requested_profile(monkeypatch, tmp_path):
    """#94071: the Desktop preflights a freshly created bot on its target
    backend. ``profile`` binds THAT profile's home + .env — launch-process
    credentials must not make an unconfigured bot look ready, and the bot's
    own .env must be what the strict check sees."""
    from agent import secret_scope
    from hermes_constants import get_hermes_home

    bot_home = tmp_path / "profiles" / "bot"
    bot_home.mkdir(parents=True)
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-launch-profile-secret-0000")
    monkeypatch.setattr("hermes_cli.profiles.profile_exists", lambda name: name == "bot")
    monkeypatch.setattr(server, "_profile_home", lambda profile: bot_home if profile == "bot" else None)
    seen = {}

    def fake_resolve(requested=None, **kwargs):
        seen["home"] = Path(str(get_hermes_home())).resolve()
        seen["secret"] = secret_scope.get_secret("OPENROUTER_API_KEY")
        return {"provider": "openrouter", "api_key": seen["secret"] or "", "source": "env"}

    monkeypatch.setattr("hermes_cli.runtime_provider.resolve_runtime_provider", fake_resolve)

    secret_scope.set_multiplex_active(True)
    try:
        status = server.handle_request(
            {"id": "1", "method": "setup.status", "params": {"profile": "bot"}}
        )
        assert status["result"] == {"provider_configured": False, "profile": "bot"}

        (bot_home / ".env").write_text("OPENROUTER_API_KEY=sk-or-bot-profile-secret-00001\n", encoding="utf-8")
        status = server.handle_request(
            {"id": "2", "method": "setup.status", "params": {"profile": "bot"}}
        )
        runtime = server.handle_request(
            {"id": "3", "method": "setup.runtime_check", "params": {"profile": "bot"}}
        )
    finally:
        secret_scope.set_multiplex_active(False)

    assert status["result"] == {"provider_configured": True, "profile": "bot"}
    assert runtime["result"]["ok"] is True
    assert runtime["result"]["profile"] == "bot"
    assert seen == {"home": bot_home.resolve(), "secret": "sk-or-bot-profile-secret-00001"}
    assert Path(str(get_hermes_home())).resolve() != bot_home.resolve()


def test_setup_readiness_unknown_profile_never_answers_for_launch_profile(monkeypatch):
    monkeypatch.setattr("hermes_cli.main._has_any_provider_configured", lambda **_kw: True)
    monkeypatch.setattr(
        "hermes_cli.runtime_provider.resolve_runtime_provider",
        lambda requested=None, **kw: {"provider": "openrouter", "api_key": "sk-or-launch-0000000000", "source": "env"},
    )
    monkeypatch.setattr("hermes_cli.profiles.profile_exists", lambda name: False)

    for method in ("setup.status", "setup.runtime_check"):
        resp = server.handle_request({"id": "1", "method": method, "params": {"profile": "ghost"}})
        assert resp["result"]["ok"] is False
        assert resp["result"]["profile"] == "ghost"


def test_complete_slash_returns_plain_string_fields():
    # prompt_toolkit hands us FormattedText (a list subclass) for
    # display/display_meta; the TUI's CompletionItem contract is plain
    # strings, and shipping the raw list trips Ink's row layout into
    # 1-char truncation of the next column (/goal → /goa).
    resp = server.handle_request(
        {"id": "1", "method": "complete.slash", "params": {"text": "/g"}}
    )

    items = resp["result"]["items"]
    goal = next((it for it in items if it["text"] == "goal"), None)
    assert goal is not None
    assert isinstance(goal["display"], str), goal["display"]
    assert isinstance(goal["meta"], str), goal["meta"]
    assert goal["display"] == "/goal"
    for item in items:
        assert isinstance(item["display"], str), item
        assert isinstance(item["meta"], str), item


def test_complete_slash_details_args():
    resp_root = server.handle_request(
        {"id": "0", "method": "complete.slash", "params": {"text": "/details"}}
    )
    resp_section = server.handle_request(
        {"id": "1", "method": "complete.slash", "params": {"text": "/details t"}}
    )
    resp_mode = server.handle_request(
        {
            "id": "2",
            "method": "complete.slash",
            "params": {"text": "/details thinking e"},
        }
    )

    assert resp_root["result"]["replace_from"] == len("/details")
    assert any(item["text"] == " thinking" for item in resp_root["result"]["items"])
    assert any(item["text"] == "thinking" for item in resp_section["result"]["items"])
    assert any(item["text"] == "expanded" for item in resp_mode["result"]["items"])


def test_complete_slash_offers_skills_alongside_commands(monkeypatch):
    """A bare `/` must reach the skills, not just the registry.

    The completer emits every registry command before the first skill, so one
    flat cap spent every row on commands and no skill was reachable at all.
    """
    _slash_skill_fixtures(monkeypatch)

    kinds = {item["kind"] for item in _slash_completions("/")}

    assert kinds == {"command", "skill"}


def test_complete_slash_ranks_skills_by_recorded_usage(monkeypatch):
    """The skills someone actually invokes lead the ones they never opened."""
    _slash_skill_fixtures(monkeypatch)

    skills = [
        item["text"].strip() for item in _slash_completions("/") if item["kind"] == "skill"
    ]

    assert skills[:3] == ["work", "research", "clean"]


def test_complete_slash_prunes_unused_builtins_only_while_browsing(monkeypatch):
    """A bare `/` is browsing and may prune; a typed query is a search.

    A search that hides a match is broken, so the never-opened bundled skill
    disappears from `/` and comes straight back the moment it is typed for.
    """
    _slash_skill_fixtures(monkeypatch)

    browsing = {item["text"].strip() for item in _slash_completions("/")}
    searching = {item["text"].strip() for item in _slash_completions("/unused")}

    assert "unused-bundled" not in browsing
    assert "unused-bundled" in searching


def test_complete_slash_leaves_argument_stages_alone(monkeypatch):
    """Ranking applies to the command token, never to a command's own args.

    `/details c` completes that command's modes; a skill named /clean also
    starts with a `c` and must not be offered as one of them.
    """
    _slash_skill_fixtures(monkeypatch)

    items = _slash_completions("/details c")

    assert [item["text"] for item in items] == ["collapsed", "cycle"]


def test_config_get_reasoning_renders_dict_form_custom_tier(tmp_path, monkeypatch):
    """`agent.reasoning_effort: {enabled: true, effort: thinking}` (a provider's bespoke tier)
    must read back as the tier name, not `str(dict)`."""
    monkeypatch.setattr(server, "_hermes_home", tmp_path)
    (tmp_path / "config.yaml").write_text(
        "agent:\n  reasoning_effort:\n    enabled: true\n    effort: thinking\n", encoding="utf-8"
    )

    resp = server.handle_request({"id": "1", "method": "config.get", "params": {"key": "reasoning"}})

    assert resp["result"]["value"] == "thinking"


def test_config_set_reasoning_updates_live_session_and_agent(tmp_path, monkeypatch):
    monkeypatch.setattr(server, "_hermes_home", tmp_path)
    (tmp_path / "config.yaml").write_text("agent:\n  reasoning_effort: medium\n", encoding="utf-8")
    agent = types.SimpleNamespace(reasoning_config=None)
    server._sessions["sid"] = _session(agent=agent)

    resp_effort = server.handle_request(
        {
            "id": "1",
            "method": "config.set",
            "params": {
                "session_id": "sid",
                "key": "reasoning",
                "value": "low",
            },
        }
    )
    assert resp_effort["result"]["value"] == "low"
    assert agent.reasoning_config == {"enabled": True, "effort": "low"}
    assert server._sessions["sid"]["create_reasoning_override"] == {"enabled": True, "effort": "low"}
    assert server._load_cfg()["agent"]["reasoning_effort"] == "medium"

    resp_status = server.handle_request(
        {
            "id": "5",
            "method": "config.get",
            "params": {"session_id": "sid", "key": "reasoning"},
        }
    )
    assert resp_status["result"]["value"] == "low"

    resp_global_status = server.handle_request(
        {"id": "6", "method": "config.get", "params": {"key": "reasoning"}}
    )
    assert resp_global_status["result"]["value"] == "medium"

    del server._sessions["sid"]["create_reasoning_override"]
    agent.reasoning_config = {"enabled": True, "effort": "high"}
    resp_agent_status = server.handle_request(
        {
            "id": "7",
            "method": "config.get",
            "params": {"session_id": "sid", "key": "reasoning"},
        }
    )
    assert resp_agent_status["result"]["value"] == "high"

    resp_show = server.handle_request(
        {
            "id": "2",
            "method": "config.set",
            "params": {"session_id": "sid", "key": "reasoning", "value": "show"},
        }
    )
    assert resp_show["result"]["value"] == "show"
    assert server._sessions["sid"]["show_reasoning"] is True
    assert server._load_cfg()["display"]["sections"]["thinking"] == "expanded"

    resp_hide = server.handle_request(
        {
            "id": "3",
            "method": "config.set",
            "params": {"session_id": "sid", "key": "reasoning", "value": "hide"},
        }
    )
    assert resp_hide["result"]["value"] == "hide"
    assert server._sessions["sid"]["show_reasoning"] is False
    assert server._load_cfg()["display"]["sections"]["thinking"] == "hidden"

    # /reasoning full | clamp — parity with the classic CLI reasoning_full
    # toggle. In the TUI these map to the thinking section's expand/collapse
    # rendering (no fixed 10-line recap exists here).
    resp_full = server.handle_request(
        {
            "id": "4",
            "method": "config.set",
            "params": {"session_id": "sid", "key": "reasoning", "value": "full"},
        }
    )
    assert resp_full["result"]["value"] == "full"
    cfg_full = server._load_cfg()
    assert cfg_full["display"]["reasoning_full"] is True
    assert cfg_full["display"]["sections"]["thinking"] == "expanded"

    resp_clamp = server.handle_request(
        {
            "id": "5",
            "method": "config.set",
            "params": {"session_id": "sid", "key": "reasoning", "value": "clamp"},
        }
    )
    assert resp_clamp["result"]["value"] == "clamp"
    cfg_clamp = server._load_cfg()
    assert cfg_clamp["display"]["reasoning_full"] is False
    assert cfg_clamp["display"]["sections"]["thinking"] == "collapsed"


def test_config_set_reasoning_global_scope_clears_session_override(tmp_path, monkeypatch):
    monkeypatch.setattr(server, "_hermes_home", tmp_path)
    (tmp_path / "config.yaml").write_text("agent:\n  reasoning_effort: medium\n", encoding="utf-8")
    agent = types.SimpleNamespace(reasoning_config=None)
    server._sessions["sid"] = _session(agent=agent)
    server._sessions["sid"]["create_reasoning_override"] = {"enabled": True, "effort": "low"}

    resp = server.handle_request(
        {
            "id": "1",
            "method": "config.set",
            "params": {
                "session_id": "sid",
                "key": "reasoning",
                "value": "high",
                "scope": "global",
            },
        }
    )

    assert resp["result"]["value"] == "high"
    assert server._load_cfg()["agent"]["reasoning_effort"] == "high"
    assert "create_reasoning_override" not in server._sessions["sid"]

    status = server.handle_request(
        {"id": "2", "method": "config.get", "params": {"session_id": "sid", "key": "reasoning"}}
    )
    assert status["result"]["value"] == "high"


def test_config_set_verbose_updates_session_mode_and_agent(tmp_path, monkeypatch):
    monkeypatch.setattr(server, "_hermes_home", tmp_path)
    agent = types.SimpleNamespace(verbose_logging=False)
    server._sessions["sid"] = _session(agent=agent)

    resp = server.handle_request(
        {
            "id": "1",
            "method": "config.set",
            "params": {"session_id": "sid", "key": "verbose", "value": "cycle"},
        }
    )

    assert resp["result"]["value"] == "verbose"
    assert server._sessions["sid"]["tool_progress_mode"] == "verbose"
    assert agent.verbose_logging is True


def test_config_set_model_waits_for_lazy_agent_before_switch(monkeypatch):
    """A model switch against a lazy-created live session must apply to the
    real agent, not just process env, before the prompt is dispatched.
    """

    agent_ready = threading.Event()
    agent = types.SimpleNamespace(model="old/model", provider="old-provider")
    session = _session(agent=agent)
    session["agent"] = None
    session["agent_ready"] = agent_ready
    server._sessions["sid"] = session
    calls = []

    def fake_start(sid, target):
        calls.append(("start", sid))
        target["agent"] = agent
        agent_ready.set()

    def fake_apply(sid, target, raw, **kwargs):
        calls.append(("apply", sid, target.get("agent"), raw))
        if target.get("agent") is not agent:
            raise AssertionError("model switch ran before lazy agent was ready")
        return {"value": "new/model", "warning": ""}

    monkeypatch.setattr(server, "_start_agent_build", fake_start)
    monkeypatch.setattr(server, "_apply_model_switch", fake_apply)

    try:
        resp = server.handle_request(
            {
                "id": "1",
                "method": "config.set",
                "params": {"session_id": "sid", "key": "model", "value": "new/model"},
            }
        )

        assert resp["result"]["value"] == "new/model"
        assert calls == [("start", "sid"), ("apply", "sid", agent, "new/model")]
    finally:
        server._sessions.pop("sid", None)


def test_config_set_model_requires_confirmation_for_expensive_model(monkeypatch):
    class _Agent:
        provider = "openrouter"
        model = "old/model"
        base_url = ""
        api_key = "sk-or"
        switched = False

        def switch_model(self, **_kwargs):
            self.switched = True

    result = types.SimpleNamespace(
        success=True,
        new_model="openai/gpt-5.5-pro",
        target_provider="openrouter",
        api_key="sk-or",
        base_url="https://openrouter.ai/api/v1",
        api_mode="chat_completions",
        warning_message="",
        model_info=types.SimpleNamespace(
            has_cost_data=lambda: True,
            cost_input=25.0,
            cost_output=125.0,
        ),
    )

    agent = _Agent()
    server._sessions["sid"] = _session(agent=agent)
    monkeypatch.setattr(
        "hermes_cli.model_switch.switch_model", lambda **_kwargs: result
    )
    monkeypatch.setattr(server, "_restart_slash_worker", lambda sid, session: None)
    monkeypatch.setattr(server, "_emit", lambda *args, **kwargs: None)

    resp = server.handle_request(
        {
            "id": "1",
            "method": "config.set",
            "params": {
                "session_id": "sid",
                "key": "model",
                "value": "openai/gpt-5.5-pro --provider openrouter",
            },
        }
    )

    assert resp["result"]["confirm_required"] is True
    assert agent.switched is False

    confirmed = server.handle_request(
        {
            "id": "2",
            "method": "config.set",
            "params": {
                "session_id": "sid",
                "key": "model",
                "value": "openai/gpt-5.5-pro --provider openrouter",
                "confirm_expensive_model": True,
            },
        }
    )

    assert confirmed["result"]["confirm_required"] is False
    assert confirmed["result"]["value"] == "openai/gpt-5.5-pro"
    assert agent.switched is True


def test_config_set_model_global_persists(monkeypatch):
    from hermes_cli.config import get_config_path, read_user_config_raw

    class _Agent:
        provider = "openrouter"
        model = "old/model"
        base_url = ""
        api_key = "sk-old"

        def switch_model(self, **kwargs):
            return None

    result = types.SimpleNamespace(
        success=True,
        new_model="anthropic/claude-sonnet-4.6",
        target_provider="anthropic",
        api_key="sk-new",
        base_url="https://api.anthropic.com",
        api_mode="anthropic_messages",
        warning_message="",
    )
    switch = Mock(return_value=result)
    config_path = get_config_path()

    server._sessions["sid"] = _session(agent=_Agent())
    monkeypatch.setattr("hermes_cli.model_switch.switch_model", switch)
    monkeypatch.setattr(server, "_restart_slash_worker", lambda sid, session: None)
    monkeypatch.setattr(server, "_emit", lambda *args, **kwargs: None)
    config_path.write_text("model:\n  model_slots:\n    fast: keep\n", encoding="utf-8")

    resp = server.handle_request(
        {
            "id": "1",
            "method": "config.set",
            "params": {
                "session_id": "sid",
                "key": "model",
                "value": "anthropic/claude-sonnet-4.6 --global",
            },
        }
    )

    assert resp["result"]["value"] == "anthropic/claude-sonnet-4.6"
    assert switch.call_args.kwargs["is_global"] is True
    saved_values = read_user_config_raw(config_path)["model"]
    assert saved_values["default"] == "anthropic/claude-sonnet-4.6"
    assert saved_values["provider"] == "anthropic"
    assert saved_values["base_url"] == "https://api.anthropic.com"
    assert saved_values["model_slots"] == {"fast": "keep"}


def test_config_set_model_explicit_provider_skips_broken_default_init(monkeypatch):
    seen = {"build": 0, "wait": 0, "requested": []}
    session = _session()
    session["agent"] = None
    server._sessions["sid"] = session
    monkeypatch.setattr(server, "_load_cfg", lambda: {"model": {"default": "broken/model", "provider": "openrouter"}})
    monkeypatch.setattr(server, "_start_agent_build", lambda *_args: seen.__setitem__("build", seen["build"] + 1))
    monkeypatch.setattr(server, "_wait_agent", lambda *_args: seen.__setitem__("wait", seen["wait"] + 1))
    monkeypatch.setattr(server, "_emit", lambda *args, **kwargs: None)
    monkeypatch.setattr(server, "_restart_slash_worker", lambda *args, **kwargs: None)

    def fake_runtime_provider(*, requested=None, target_model=None, **_kwargs):
        seen["requested"].append((requested, target_model))
        if requested is None:
            raise RuntimeError("broken default provider should not be initialized")
        if requested == "anthropic":
            return {
                "api_key": "sk-anthropic",
                "api_mode": "anthropic_messages",
                "base_url": "https://api.anthropic.com",
            }
        raise RuntimeError(f"unexpected provider {requested}")

    monkeypatch.setattr("hermes_cli.runtime_provider.resolve_runtime_provider", fake_runtime_provider)

    try:
        resp = server.handle_request(
            {
                "id": "1",
                "method": "config.set",
                "params": {
                    "session_id": "sid",
                    "key": "model",
                    "value": "claude-sonnet-4.6 --provider anthropic",
                },
            }
        )

        assert resp["result"]["value"] == "claude-sonnet-4-6"
        assert seen["build"] == 0
        assert seen["wait"] == 0
        assert seen["requested"] == [("anthropic", "claude-sonnet-4.6")]
        assert session["model_override"]["provider"] == "anthropic"
        assert session["model_override"]["model"] == "claude-sonnet-4-6"
    finally:
        server._sessions.pop("sid", None)


@pytest.mark.parametrize(
    ("provider_flag", "failure_text"),
    [
        (" --provider custom:new-provider", "Unknown provider 'removed-provider'"),
        ("", ""),
    ],
)
def test_config_set_model_recovers_failed_profile_resume_after_build_completes(
    monkeypatch, tmp_path, provider_flag, failure_text
):
    """Recovery waits for the real failed build and uses its owning profile.

    Both the failed and replacement generations cross the real deferred-build
    boundary. Provider resolution is the only model-switch leaf replaced.
    """
    from agent.secret_scope import current_secret_scope
    from hermes_constants import get_hermes_home

    launch_url = "https://launch.example/v1"
    profile_url = "https://profile.example/v1"
    launch_home = tmp_path / "launch"
    profile_home = tmp_path / "profiles" / "work"
    launch_home.mkdir()
    profile_home.mkdir(parents=True)
    (launch_home / "config.yaml").write_text(
        "model:\n"
        "  default: launch/model\n"
        "  provider: custom:new-provider\n"
        "providers:\n"
        "  new-provider:\n"
        f"    base_url: {launch_url}\n"
        "    key_env: LAUNCH_API_KEY\n",
        encoding="utf-8",
    )
    (launch_home / ".env").write_text(
        "LAUNCH_API_KEY=launch-secret\n", encoding="utf-8"
    )
    (profile_home / "config.yaml").write_text(
        "model:\n"
        "  default: old/model\n"
        "  provider: custom:new-provider\n"
        "providers:\n"
        "  new-provider:\n"
        f"    base_url: {profile_url}\n"
        "    key_env: PROFILE_API_KEY\n",
        encoding="utf-8",
    )
    (profile_home / ".env").write_text(
        "PROFILE_API_KEY=profile-secret\n", encoding="utf-8"
    )
    monkeypatch.setenv("HERMES_HOME", str(launch_home))
    monkeypatch.delenv("HERMES_MODEL", raising=False)
    monkeypatch.delenv("HERMES_INFERENCE_MODEL", raising=False)

    class ControlledReady:
        def __init__(self):
            self._event = threading.Event()
            self.wait_entered = threading.Event()

        def wait(self, timeout=None):
            self.wait_entered.set()
            return self._event.wait(timeout)

        def is_set(self):
            return self._event.is_set()

        def set(self):
            self._event.set()

    old_ready = ControlledReady()
    reasoning = {"effort": "high"}
    old_override = {"model": "old/model", "provider": "removed-provider"}
    session = _session(
        agent_ready=old_ready,
        agent_error=None,
        model_override=old_override,
        resume_runtime_overrides={
            "model_override": old_override,
            "provider_override": "removed-provider",
            "reasoning_config_override": reasoning,
        },
        profile_home=str(profile_home),
    )
    session["agent"] = None
    server._sessions["sid"] = session
    old_finally_entered = threading.Event()
    release_old_finally = threading.Event()
    switch_called = threading.Event()
    seen = {"switch": None, "build": None, "persisted": []}
    make_calls = 0

    def fake_switch_model(**kwargs):
        provider = kwargs["user_providers"]["new-provider"]
        secrets = dict(current_secret_scope() or {})
        api_key = secrets[provider["key_env"]]
        seen["switch"] = {
            "home": get_hermes_home(),
            "secrets": secrets,
            "base_url": provider["base_url"],
            "api_key": api_key,
            "current_provider": kwargs["current_provider"],
            "current_api_key": kwargs["current_api_key"],
        }
        switch_called.set()
        return types.SimpleNamespace(
            success=True,
            new_model="new/model",
            target_provider="custom:new-provider",
            api_key=api_key,
            base_url=provider["base_url"],
            api_mode="chat_completions",
            warning_message="",
            model_info=None,
            error_message="",
        )

    class FakeDb:
        def __init__(self, *_args, **_kwargs):
            pass

        def get_session(self, _key):
            return {"model_config": {}}

        def update_session_meta(self, key, model_config, model):
            seen["persisted"].append(
                {
                    "key": key,
                    "model": model,
                    "config": json.loads(model_config),
                }
            )

        def close(self):
            pass

    def fake_make_agent(_sid, _key, **kwargs):
        nonlocal make_calls
        make_calls += 1
        if make_calls == 1:
            raise RuntimeError(failure_text)
        override = kwargs["model_override"]
        seen["build"] = {
            "home": get_hermes_home(),
            "secrets": dict(current_secret_scope() or {}),
            "overrides": kwargs,
        }
        return types.SimpleNamespace(
            model=override["model"],
            provider="custom",
            base_url=override["base_url"],
            api_key=override["api_key"],
            api_mode=override["api_mode"],
            reasoning_config=kwargs.get("reasoning_config_override"),
            service_tier=None,
            _session_db=kwargs.get("session_db"),
        )

    real_transfer = server._transfer_db_to_agent

    def barrier_transfer(agent, db):
        if agent is None and not old_finally_entered.is_set():
            old_finally_entered.set()
            assert release_old_finally.wait(timeout=10)
        return real_transfer(agent, db)

    monkeypatch.setattr("hermes_cli.model_switch.switch_model", fake_switch_model)
    monkeypatch.setattr(
        "hermes_cli.model_selection_guards.combined_selection_warning",
        lambda *args, **kwargs: None,
    )
    monkeypatch.setattr("hermes_state_registry.acquire", FakeDb)
    monkeypatch.setattr(server, "_make_agent", fake_make_agent)
    monkeypatch.setattr(server, "_transfer_db_to_agent", barrier_transfer)
    monkeypatch.setattr(
        "hermes_runtime.entry.ensure_mcp_discovery_started", lambda: None
    )
    monkeypatch.setattr(server, "_wire_callbacks", lambda _sid: None)
    monkeypatch.setattr(server, "_start_notification_poller", lambda *a, **k: None)
    monkeypatch.setattr(server, "_notify_session_boundary", lambda *a, **k: None)
    monkeypatch.setattr(server, "_session_info", lambda *a, **k: {})
    monkeypatch.setattr(server, "_probe_config_health", lambda *_args: None)
    monkeypatch.setattr(server, "_schedule_mcp_late_refresh", lambda *a, **k: None)
    monkeypatch.setattr(server, "_emit", lambda *a, **k: None)

    response = {}

    def run_request():
        try:
            response["value"] = server.handle_request(
                {
                    "id": "1",
                    "method": "config.set",
                    "params": {
                        "session_id": "sid",
                        "key": "model",
                        "value": f"new/model{provider_flag}",
                    },
                }
            )
        except BaseException as exc:
            response["error"] = exc

    request_thread = threading.Thread(target=run_request)
    old_build_thread = None
    try:
        server._start_agent_build("sid", session)
        old_build_thread = session["_agent_build_thread"]
        assert old_finally_entered.wait(timeout=10)
        assert session["agent_error"] == failure_text
        assert not old_ready.is_set()

        request_thread.start()
        assert old_ready.wait_entered.wait(timeout=2), (
            "model recovery did not wait for the failed build generation"
        )
        assert not switch_called.is_set()
        release_old_finally.set()
        request_thread.join(timeout=10)

        assert not request_thread.is_alive()
        assert "error" not in response
        assert response["value"]["result"]["value"] == "new/model"
        assert make_calls == 2
        assert seen["switch"] == {
            "home": profile_home,
            "secrets": {"PROFILE_API_KEY": "profile-secret"},
            "base_url": profile_url,
            "api_key": "profile-secret",
            "current_provider": (
                "custom:new-provider" if provider_flag else "custom"
            ),
            "current_api_key": "profile-secret" if not provider_flag else "",
        }
        assert seen["build"]["home"] == profile_home
        assert seen["build"]["secrets"] == {
            "PROFILE_API_KEY": "profile-secret"
        }
        overrides = seen["build"]["overrides"]
        assert overrides["model_override"] == session["model_override"]
        assert overrides["provider_override"] == "custom:new-provider"
        assert overrides["reasoning_config_override"] == reasoning
        assert session["agent_error"] is None
        assert session["agent"].model == "new/model"
        assert session["agent"].base_url == profile_url
        assert session["agent"].api_key == "profile-secret"
        assert seen["persisted"] == [
            {
                "key": "session-key",
                "model": "new/model",
                "config": {
                    "model": "new/model",
                    "provider": "custom:new-provider",
                    "base_url": profile_url,
                    "api_mode": "chat_completions",
                    "reasoning_config": reasoning,
                },
            }
        ]
    finally:
        release_old_finally.set()
        old_ready.set()
        request_thread.join(timeout=10)
        if old_build_thread is not None:
            old_build_thread.join(timeout=10)
        new_build_thread = session.get("_agent_build_thread")
        if new_build_thread is not None:
            new_build_thread.join(timeout=10)
        server._sessions.pop("sid", None)


def test_config_set_model_explicit_provider_surfaces_selected_provider_errors(monkeypatch):
    seen = {"build": 0, "wait": 0}
    session = _session()
    session["agent"] = None
    server._sessions["sid"] = session
    monkeypatch.setattr(server, "_load_cfg", lambda: {"model": {"default": "broken/model", "provider": "openrouter"}})
    monkeypatch.setattr(server, "_start_agent_build", lambda *_args: seen.__setitem__("build", seen["build"] + 1))
    monkeypatch.setattr(server, "_wait_agent", lambda *_args: seen.__setitem__("wait", seen["wait"] + 1))

    def fake_runtime_provider(*, requested=None, **_kwargs):
        if requested is None:
            raise RuntimeError("broken default provider should not be initialized")
        if requested == "anthropic":
            raise RuntimeError("missing anthropic API key")
        raise RuntimeError(f"unexpected provider {requested}")

    monkeypatch.setattr("hermes_cli.runtime_provider.resolve_runtime_provider", fake_runtime_provider)

    try:
        resp = server.handle_request(
            {
                "id": "1",
                "method": "config.set",
                "params": {
                    "session_id": "sid",
                    "key": "model",
                    "value": "claude-sonnet-4.6 --provider anthropic",
                },
            }
        )

        assert resp["error"]["code"] == 5001
        assert "anthropic" in resp["error"]["message"].lower()
        assert "missing anthropic api key" in resp["error"]["message"].lower()
        assert seen["build"] == 0
        assert seen["wait"] == 0
    finally:
        server._sessions.pop("sid", None)


def test_config_set_model_does_not_leak_inference_provider_env(monkeypatch):
    """A /model switch must NOT mutate process-global env vars. The desktop /
    dashboard hermes_runtime backend hosts every same-profile session in one
    process; writing HERMES_INFERENCE_PROVIDER on a switch leaked the new
    provider into every other live session's next agent rebuild. The switch
    must instead record a per-session override and leave shared env untouched.

    (Was test_config_set_model_syncs_inference_provider_env, which asserted the
    leaky env-sync contract that caused the cross-session contamination bug.)
    """

    class _Agent:
        provider = "openrouter"
        model = "old/model"
        base_url = ""
        api_key = "sk-or"

        def switch_model(self, **_kwargs):
            return None

    result = types.SimpleNamespace(
        success=True,
        new_model="claude-sonnet-4.6",
        target_provider="anthropic",
        api_key="sk-ant",
        base_url="https://api.anthropic.com",
        api_mode="anthropic_messages",
        warning_message="",
    )

    session = _session(agent=_Agent())
    server._sessions["sid"] = session
    monkeypatch.setenv("HERMES_INFERENCE_PROVIDER", "openrouter")
    monkeypatch.setattr(
        "hermes_cli.model_switch.switch_model", lambda **_kwargs: result
    )
    monkeypatch.setattr(server, "_restart_slash_worker", lambda sid, session: None)
    monkeypatch.setattr(server, "_emit", lambda *args, **kwargs: None)

    try:
        server.handle_request(
            {
                "id": "1",
                "method": "config.set",
                "params": {
                    "session_id": "sid",
                    "key": "model",
                    "value": "claude-sonnet-4.6 --provider anthropic",
                },
            }
        )

        # Shared process env is UNCHANGED (the contamination vector is gone).
        assert os.environ["HERMES_INFERENCE_PROVIDER"] == "openrouter"
        # The switch was recorded as a per-session override instead.
        assert session["model_override"]["provider"] == "anthropic"
        assert session["model_override"]["model"] == "claude-sonnet-4.6"
    finally:
        server._sessions.clear()


def test_config_set_model_records_per_session_override_not_env(monkeypatch):
    """Regression for #16857 via the per-session override (not env vars):
    /model must record the user's explicit provider on the session so a later
    /new (which rebuilds via _make_agent honoring model_override) honours that
    choice — WITHOUT writing process-global env vars that would leak into
    sibling sessions.

    (Was test_config_set_model_syncs_tui_provider_unconditionally.)
    """

    class _Agent:
        provider = "openrouter"
        model = "old/model"
        base_url = ""
        api_key = "sk-or"

        def switch_model(self, **_kwargs):
            return None

    result = types.SimpleNamespace(
        success=True,
        new_model="deepseek-v4-pro",
        target_provider="custom:xuanji",
        api_key="sk-xuanji",
        base_url="https://xuanji.example/v1",
        api_mode="chat_completions",
        warning_message="",
    )

    session = _session(agent=_Agent())
    server._sessions["sid"] = session
    monkeypatch.delenv("HERMES_TUI_PROVIDER", raising=False)
    monkeypatch.delenv("HERMES_INFERENCE_PROVIDER", raising=False)
    monkeypatch.setattr(
        "hermes_cli.model_switch.switch_model", lambda **_kwargs: result
    )
    monkeypatch.setattr(server, "_restart_slash_worker", lambda sid, session: None)
    monkeypatch.setattr(server, "_emit", lambda *args, **kwargs: None)

    try:
        server.handle_request(
            {
                "id": "1",
                "method": "config.set",
                "params": {
                    "session_id": "sid",
                    "key": "model",
                    "value": "deepseek-v4-pro --provider custom:xuanji",
                },
            }
        )

        # No process-global env mutation.
        assert "HERMES_TUI_PROVIDER" not in os.environ
        assert "HERMES_INFERENCE_PROVIDER" not in os.environ
        # The user's explicit provider + resolved endpoint live on the session,
        # carried into the next /new rebuild by _make_agent.
        override = session["model_override"]
        assert override["provider"] == "custom:xuanji"
        assert override["model"] == "deepseek-v4-pro"
        assert override["base_url"] == "https://xuanji.example/v1"
        assert override["api_key"] == "sk-xuanji"
        assert override["api_mode"] == "chat_completions"
    finally:
        server._sessions.clear()


def test_config_set_model_switches_agent_without_touching_env(monkeypatch):
    """A /model switch mutates the target session's agent in place and records
    a per-session override; it does NOT write HERMES_MODEL / HERMES_TUI_PROVIDER
    etc. into the shared process environment.

    (Was test_config_set_model_syncs_tui_provider_env.)
    """

    class Agent:
        model = "gpt-5.3-codex"
        provider = "openai-codex"
        base_url = ""
        api_key = ""
        session_id = "sid"
        _cached_system_prompt = "Model: gpt-5.3-codex\nProvider: openai-codex"

        def switch_model(self, **kwargs):
            self.model = kwargs["new_model"]
            self.provider = kwargs["new_provider"]

        def _build_system_prompt(self, _system_message=None):
            return f"Model: {self.model}\nProvider: {self.provider}"

    class SessionDB:
        def __init__(self):
            self.model_config = None
            self.system_prompt = None
            self.messages = []

        def get_session(self, _session_id):
            return {"model_config": self.model_config}

        def update_session_meta(self, _session_id, model_config_json, _model=None):
            self.model_config = model_config_json

        def update_system_prompt(self, _session_id, system_prompt):
            self.system_prompt = system_prompt

        def append_message(self, session_id, role, content=None, **_kwargs):
            self.messages.append(
                {"session_id": session_id, "role": role, "content": content}
            )

        def deactivate_messages_by_display_kind(self, _session_id, _display_kind):
            return 0

    agent = Agent()
    db = SessionDB()
    agent._session_db = db
    session = _session(agent=agent)
    server._sessions["sid"] = session
    monkeypatch.setenv("HERMES_TUI_PROVIDER", "openai-codex")
    monkeypatch.delenv("HERMES_MODEL", raising=False)
    monkeypatch.delenv("HERMES_INFERENCE_MODEL", raising=False)
    monkeypatch.setattr(server, "_restart_slash_worker", lambda sid, session: None)
    monkeypatch.setattr(server, "_emit", lambda *args, **kwargs: None)

    def fake_switch_model(**kwargs):
        return types.SimpleNamespace(
            success=True,
            new_model="anthropic/claude-sonnet-4.6",
            target_provider="anthropic",
            api_key="key",
            base_url="https://api.anthropic.com",
            api_mode="anthropic_messages",
            warning_message="",
        )

    monkeypatch.setattr("hermes_cli.model_switch.switch_model", fake_switch_model)

    try:
        resp = server.handle_request(
            {
                "id": "1",
                "method": "config.set",
                "params": {
                    "session_id": "sid",
                    "key": "model",
                    "value": "anthropic/claude-sonnet-4.6 --provider anthropic",
                },
            }
        )

        assert resp["result"]["value"] == "anthropic/claude-sonnet-4.6"
        # Agent switched in place...
        assert agent.model == "anthropic/claude-sonnet-4.6"
        assert agent.provider == "anthropic"
        # ...override recorded on the session...
        assert session["model_override"]["model"] == "anthropic/claude-sonnet-4.6"
        assert session["model_override"]["provider"] == "anthropic"
        # ...the persisted prompt snapshot tracks the new runtime identity too.
        # Without this, the next turn restored the old system prompt from the DB:
        # API calls went to the new model, but "what model are you?" still read
        # "Model: old/model" from the stored prompt.
        assert db.system_prompt == (
            "Model: anthropic/claude-sonnet-4.6\nProvider: anthropic"
        )
        assert agent._cached_system_prompt == db.system_prompt
        assert session["history"][-1]["role"] == "user"
        assert "changed to anthropic/claude-sonnet-4.6" in session["history"][-1]["content"]
        assert db.messages[-1] == {
            # The agent's own db handle belongs to the agent's live session id, not the gateway's
            # session_key — the same `getattr(agent, "session_id", None) or session_key` the sibling
            # system-prompt persist uses one screen up (#123545: these two must not diverge).
            "session_id": agent.session_id,
            "role": "user",
            "content": session["history"][-1]["content"],
        }
        # ...and the shared process env was NOT touched.
        assert os.environ["HERMES_TUI_PROVIDER"] == "openai-codex"
        assert "HERMES_MODEL" not in os.environ
        assert "HERMES_INFERENCE_MODEL" not in os.environ
    finally:
        server._sessions.clear()



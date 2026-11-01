





import os




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


def test_full_duplex_stop_phrase_mid_generation_ends_voice_chat(monkeypatch, tmp_path):
    """Bare 'stop' during generation = interrupt the turn AND end the voice
    chat ('stop everything'), emitted as the explicit stop_phrase signal."""
    monkeypatch.setenv("HERMES_VOICE", "1")
    monkeypatch.setattr(server, "_load_cfg", lambda: {"voice": {"barge_in": True}})
    events: list = []
    monkeypatch.setattr(
        server, "_voice_emit", lambda event, payload=None: events.append((event, payload))
    )

    wav = tmp_path / "stop.wav"
    wav.write_bytes(b"RIFF")

    interrupted = threading.Event()
    fake_agent = types.SimpleNamespace(interrupt=lambda: interrupted.set())
    monkeypatch.setattr(
        server, "_sessions", {"sid-fd": {"running": True, "agent": fake_agent}}
    )

    def fake_listen(should_stop, is_playing=None, on_trigger=None, **_kw):
        on_trigger("generation")
        return str(wav)

    _fake_tts_modules(
        monkeypatch,
        listen=fake_listen,
        transcribe=lambda path, model=None: {"success": True, "transcript": "stop"},
    )
    # is_voice_stop_phrase lives in the faked tools.voice_mode namespace.
    sys.modules["tools.voice_mode"].is_voice_stop_phrase = (
        lambda text: text.strip().lower() == "stop"
    )
    monkeypatch.setitem(
        sys.modules,
        "hermes_cli.voice",
        types.SimpleNamespace(stop_continuous=lambda **_kw: None, speak_text=lambda *a, **k: None),
    )

    server._arm_full_duplex_listener()

    deadline = time.monotonic() + 5.0
    while time.monotonic() < deadline and wav.exists():
        time.sleep(0.01)
    assert interrupted.is_set()
    assert ("voice.transcript", {"stop_phrase": True, "text": "stop"}) in events
    assert os.environ.get("HERMES_VOICE") == "0"  # voice chat ended


def test_speak_text_with_barge_arms_monitor_and_cuts_playback(monkeypatch, tmp_path):
    """The fallback whole-reply speak path (streaming pipeline couldn't
    start) and the voice.tts RPC must be barge-able too: speaking over the
    reply cuts playback and the captured interruption is emitted as
    voice.transcript — previously these paths called speak_text bare and
    were uninterruptible by voice."""
    import tools.tts_streaming as ts

    ts._interrupted_at = None
    monkeypatch.setenv("HERMES_VOICE", "1")
    monkeypatch.setenv("HERMES_VOICE_TTS", "1")
    monkeypatch.setattr(
        server,
        "_load_cfg",
        lambda: {"voice": {"barge_in": True, "barge_in_grace_seconds": 0}},
    )
    events: list = []
    monkeypatch.setattr(
        server, "_voice_emit", lambda event, payload=None: events.append((event, payload))
    )

    wav = tmp_path / "barge.wav"
    wav.write_bytes(b"RIFF")

    speak_calls = {}
    speak_started = threading.Event()
    release_speak = threading.Event()

    def fake_speak_text(text, stop_event=None):
        speak_calls["text"] = text
        speak_calls["stop_event"] = stop_event
        speak_started.set()
        release_speak.wait(5)

    monkeypatch.setitem(
        sys.modules,
        "hermes_cli.voice",
        types.SimpleNamespace(speak_text=fake_speak_text),
    )

    def fake_listen(should_stop, is_playing=None, on_trigger=None, **_kw):
        speak_started.wait(5)
        on_trigger("playback")  # user talks over the reply → cut now
        return str(wav)

    _fake_tts_modules(
        monkeypatch,
        listen=fake_listen,
        transcribe=lambda path, model=None: {"success": True, "transcript": "hang on"},
    )

    server._speak_text_with_barge("a long spoken reply")

    deadline = time.monotonic() + 5.0
    while time.monotonic() < deadline and ("voice.transcript", {"text": "hang on"}) not in events:
        time.sleep(0.01)
    release_speak.set()

    assert speak_calls["text"] == "a long spoken reply"
    # The pipeline stop event is shared with speak_text so a streaming
    # dispatch inside it is cut too.
    assert speak_calls["stop_event"] is not None
    assert speak_calls["stop_event"].is_set()
    assert ("voice.interrupted", None) in events
    assert ("voice.transcript", {"text": "hang on"}) in events
    assert ts.take_speech_interrupted() is True


def test_speak_text_with_barge_no_monitor_when_voice_mode_off(monkeypatch):
    """Auto-speak with voice mode off (no mic loop) must not open the mic."""
    monkeypatch.setenv("HERMES_VOICE", "0")
    monkeypatch.setenv("HERMES_VOICE_TTS", "1")
    monkeypatch.setattr(server, "_load_cfg", lambda: {"voice": {"barge_in": True}})

    listened = threading.Event()

    def fake_listen(should_stop, capture=False, on_trigger=None, **_kw):
        listened.set()

    done_speaking = threading.Event()
    monkeypatch.setitem(
        sys.modules,
        "hermes_cli.voice",
        types.SimpleNamespace(
            speak_text=lambda text, stop_event=None: done_speaking.set()
        ),
    )
    _fake_tts_modules(monkeypatch, listen=fake_listen)

    server._speak_text_with_barge("quiet reply")
    assert done_speaking.wait(5)
    time.sleep(0.1)
    assert not listened.is_set()


def test_clarify_callback_uses_configured_timeout(monkeypatch):
    """The TUI/desktop clarify bridge sends a ``clarify`` server request with the canonical clarify timeout
    (via _clarify_timeout_seconds), and returns the response's ``answers`` and ``outcome``."""
    monkeypatch.setattr(server, "_clarify_timeout_seconds", lambda: 42)
    reply = {"answers": {"q0": "a"}, "outcome": "submitted"}
    captured = _capture_server_request(monkeypatch, reply)
    questions = [{"qid": "q0", "question": "Pick one", "choices": ["a", "b"], "multi_select": False}]

    result = server._agent_cbs("sid-1")["clarify_callback"](questions)

    assert result == reply
    assert captured["method"] == "clarify" and captured["sid"] == "sid-1"
    assert captured["timeout"] == 42
    assert captured["params"] == {"questions": questions}
    assert captured["qids"] == ["q0"]


@pytest.mark.parametrize(
    ("configured", "expected"),
    [(0, None), (-1, None), (42, 42)],
)
def test_clarify_timeout_seconds_maps_non_positive_to_unlimited(monkeypatch, configured, expected):
    """A ``<= 0`` clarify timeout means unlimited and reaches the server request as None
    (wait(None) waits forever) rather than an immediate wait(0) skip."""
    monkeypatch.setattr("tools.clarify_gateway.get_clarify_timeout", lambda: configured)

    assert server._clarify_timeout_seconds() == expected


def test_build_persist_message_with_image_refs_without_images_returns_text(monkeypatch):
    """#70720: when no images are attached the persisted message is the raw
    prompt — no @image directive prefix is introduced."""
    assert server._build_persist_message_with_image_refs("what is this?", []) == "what is this?"
    assert server._build_persist_message_with_image_refs("", []) == ""


def test_build_persist_message_with_image_refs_appends_existing_paths(monkeypatch, tmp_path):
    """Attached images that still exist on disk are persisted as trailing
    ``@image:<path>`` directive lines so the desktop renders them after a
    restart (instead of the vision-only enrichment that silently breaks)."""
    img = tmp_path / "cat.png"
    img.write_bytes(b"\x89PNG")

    result = server._build_persist_message_with_image_refs("what is in this photo?", [str(img)])

    assert result == f"what is in this photo?\n@image:{img}"


def test_build_persist_message_with_image_refs_skips_missing_paths(monkeypatch, tmp_path):
    """Only paths that still exist are persisted; a missing file must not
    inject a dangling @image ref into the transcript."""
    existing = tmp_path / "a.png"
    existing.write_bytes(b"png")
    missing = str(tmp_path / "gone.png")

    result = server._build_persist_message_with_image_refs("compare them", [str(existing), missing])

    assert result == f"compare them\n@image:{existing}"


def test_build_persist_message_with_image_refs_without_text_is_refs_only(monkeypatch, tmp_path):
    """A stand-alone attachment (no caption) persists as just the directive
    line, so a bare image survives in history and is not dropped as empty."""
    img = tmp_path / "only.png"
    img.write_bytes(b"png")

    assert server._build_persist_message_with_image_refs("", [str(img)]) == f"@image:{img}"


def test_build_persist_message_quotes_paths_containing_spaces(tmp_path):
    """The unquoted alternative in the directive pattern is ``\\S+``, so a path
    with a space parses as a truncated ref with the tail left as loose text.
    Desktop composer images live in the app's userData dir, which on macOS is
    ``~/Library/Application Support/...`` — a space every time."""
    img_dir = tmp_path / "Application Support" / "Hermes" / "composer-images"
    img_dir.mkdir(parents=True)
    img = img_dir / "cat.png"
    img.write_bytes(b"png")

    result = server._build_persist_message_with_image_refs("what is this?", [str(img)])

    assert result == f"what is this?\n@image:`{img}`"


def test_persist_user_message_mirrors_the_shape_sent_to_the_model(tmp_path):
    """A native-vision turn sends ``content`` as a parts list, and the session
    store ignores a plain-string override for a list payload. The override must
    mirror the list shape (ref text + the original image parts) or it is
    silently dropped and the attachment never reaches history."""
    img = tmp_path / "cat.png"
    img.write_bytes(b"png")
    image_part = {"type": "image_url", "image_url": {"url": "data:image/png;base64,AAAA"}}
    native_parts = [{"type": "text", "text": "api-only text"}, image_part]

    override = server._build_persist_user_message("what is this?", [str(img)], native_parts)

    assert override == [{"type": "text", "text": f"what is this?\n@image:{img}"}, image_part]


def test_persist_user_message_stays_a_string_for_text_mode(tmp_path):
    """Text-mode (vision-preprocessed) turns send a string, so the override
    stays a string — the shape the session store rewrites directly."""
    img = tmp_path / "cat.png"
    img.write_bytes(b"png")

    override = server._build_persist_user_message("what is this?", [str(img)], "enriched api-only text")

    assert override == f"what is this?\n@image:{img}"


def test_native_vision_turn_persists_a_renderable_image_ref(tmp_path):
    """End to end through the real session-store flush: whichever image input
    mode the turn used, the durable row carries an ``@image:`` ref the desktop
    can render after a restart."""
    from unittest.mock import MagicMock

    from agent.image_routing import build_native_content_parts
    from run_agent import AIAgent

    img_dir = tmp_path / "Application Support" / "composer-images"
    img_dir.mkdir(parents=True)
    img = img_dir / "cat.png"
    img.write_bytes(
        bytes.fromhex(
            "89504e470d0a1a0a0000000d494844520000000100000001080600000"
            "01f15c4890000000a49444154789c6360000002000100ffff0300000600"
            "0557bfabd40000000049454e44ae426082"
        )
    )
    native_parts, skipped = build_native_content_parts("what is in this photo?", [str(img)])
    assert not skipped

    agent = AIAgent.__new__(AIAgent)
    agent._session_db = MagicMock()
    agent._session_db_created = True
    agent.session_id = "s-1"
    agent._last_flushed_db_idx = 0
    agent._persist_disabled = False
    agent._flushed_db_message_ids = set()
    agent._flushed_db_message_session_id = None
    agent._pending_cli_user_message = None
    agent._persist_user_message_timestamp = None
    agent._persist_user_message_idx = 0
    agent._persist_user_message_override = server._build_persist_user_message(
        "what is in this photo?", [str(img)], native_parts
    )

    agent._flush_messages_to_session_db([{"role": "user", "content": native_parts}], [])

    written = agent._session_db.append_messages_batch.call_args.kwargs["messages"][0]["content"]
    assert f"@image:`{img}`" in written
    assert "what is in this photo?" in written
    # The model keeps the pixels for the rest of the session.
    assert any(part.get("type") == "image_url" for part in agent._persist_user_message_override)


def test_prompt_submit_passes_persist_user_message_to_agent(monkeypatch):
    """#70720: _run_prompt_submit must forward the (image-ref-aware) persisted
    user message to run_conversation via persist_user_message, so the gateway
    stores the UI-recognizable form instead of the vision enrichment."""
    captured = {}

    class _Agent:
        def run_conversation(self, prompt, conversation_history=None, stream_callback=None, **_kwargs):
            captured["persist_user_message"] = _kwargs.get("persist_user_message")
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
    try:
        monkeypatch.setattr(server.threading, "Thread", _ImmediateThread)
        monkeypatch.setattr(server, "_get_usage", lambda _a: {})
        monkeypatch.setattr(server, "render_message", lambda _t, _c: "")
        monkeypatch.setattr(server, "_emit", lambda *a: None)

        resp = server.handle_request(
            {
                "id": "1",
                "method": "prompt.submit",
                "params": {"session_id": "sid", "text": "hi"},
            }
        )
        assert resp.get("result")

        # Without attachments the persist form equals the raw prompt.
        assert captured.get("persist_user_message") == "hi"
    finally:
        server._sessions.pop("sid", None)


def test_fallback_session_info_reports_session_cwd_not_launch_dir(monkeypatch):
    """A lazily-resumed session must report ITS workspace, not the gateway's.

    ``_fallback_session_info`` used ``_default_session_cwd()`` — the directory the
    gateway process happened to start in — so the desktop Files pane painted the
    wrong project for any session resumed without a built agent (#71254).
    """
    monkeypatch.setattr(server, "_default_session_cwd", lambda: "/gateway/launch/dir")
    monkeypatch.setattr(server.git_probe, "branch", lambda cwd: "bb/feature")
    monkeypatch.setattr(server, "_project_info_for_cwd", lambda cwd: None)
    monkeypatch.setattr(server, "_resolve_model", lambda: "test-model")

    info = server._fallback_session_info({"cwd": "/projects/session-own-repo"})

    assert info["cwd"] == "/projects/session-own-repo"
    assert info["branch"] == "bb/feature"


def test_fallback_session_info_always_emits_branch(monkeypatch):
    """``branch`` is always present so a client can CLEAR a stale label.

    Omitting the key left the desktop showing the previous conversation's branch
    after switching into a non-git session.
    """
    monkeypatch.setattr(server, "_default_session_cwd", lambda: "/gateway/launch/dir")
    monkeypatch.setattr(server.git_probe, "branch", lambda cwd: "")
    monkeypatch.setattr(server, "_project_info_for_cwd", lambda cwd: None)
    monkeypatch.setattr(server, "_resolve_model", lambda: "test-model")

    info = server._fallback_session_info({"cwd": "/plain/folder"})

    assert "branch" in info
    assert info["branch"] == ""


def test_persist_branch_seed_keeps_reasoning_fields(monkeypatch, tmp_path):
    """The seed write must carry the parent's reasoning fields.

    A branch is a draft until its first submit, so this is the only write that
    ever persists the copied transcript. Persisting role/content alone left the
    branch resuming without the parent's reasoning, preserved thinking blocks or
    Codex encrypted-reasoning/message-item continuation state.
    """
    from hermes_state import SessionDB

    db = SessionDB(db_path=tmp_path / "state.db")
    session = _session(
        session_key="branch-key",
        parent_session_id="parent-key",
        history=_branch_history(),
        seeded=True,  # stamped by session.create: this history exists only in memory
    )
    try:
        db.create_session("branch-key", source="tui")
        monkeypatch.setattr(server, "_get_db", lambda: db)

        server._persist_branch_seed(session)

        assistant = _branched_assistant(db, "branch-key")
        assert assistant["reasoning"] == BRANCH_REASONING
        assert assistant["reasoning_content"] == BRANCH_REASONING_CONTENT
        assert assistant["reasoning_details"] == BRANCH_REASONING_DETAILS
        assert assistant["codex_reasoning_items"] == BRANCH_CODEX_REASONING_ITEMS
        assert assistant["codex_message_items"] == BRANCH_CODEX_MESSAGE_ITEMS
        marker = _branched_marker(db, "branch-key")
        assert marker is not None, (
            "the branch seed dropped display_kind: the marker re-entered the "
            "truncate ordinal address space as a phantom user turn (#82756)"
        )
        assert session["_branch_seed_persisted"] is True
    finally:
        db.close()


def test_session_branch_keeps_reasoning_fields(monkeypatch, tmp_path):
    """session.branch copies the live transcript with its reasoning fields.

    Same drop as the seed path: the copy loop wrote only role/content, so the
    new session row replayed without the reasoning context the parent had.
    """
    from hermes_state import SessionDB

    db = SessionDB(db_path=tmp_path / "state.db")
    server._sessions["sid"] = _session(history=_branch_history())
    try:
        db.create_session("session-key", source="tui")
        monkeypatch.setattr(server, "_get_db", lambda: db)
        monkeypatch.setattr(server, "_new_session_key", lambda: "branch-key")
        monkeypatch.setattr(server, "_start_agent_build", lambda sid, session: None)
        monkeypatch.setattr(server, "_wait_agent", lambda session, rid: None)
        monkeypatch.setattr(
            server, "_claim_active_session_slot", lambda *args, **kwargs: (None, None)
        )
        monkeypatch.setattr(server, "_resolve_model", lambda: "test-model")
        monkeypatch.setattr(server, "_session_cwd", lambda session: str(tmp_path))
        monkeypatch.setattr(
            server, "_make_agent", lambda *args, **kwargs: types.SimpleNamespace()
        )
        monkeypatch.setattr(server, "_init_session", lambda *args, **kwargs: None)

        resp = server.handle_request(
            {"id": "1", "method": "session.branch", "params": {"session_id": "sid"}}
        )

        assert resp.get("result"), f"got error: {resp.get('error')}"
        assert db.get_session_title_source("branch-key") == SessionDB.TITLE_SOURCE_DERIVED
        assistant = _branched_assistant(db, "branch-key")
        assert assistant["reasoning"] == BRANCH_REASONING
        assert assistant["reasoning_content"] == BRANCH_REASONING_CONTENT
        assert assistant["reasoning_details"] == BRANCH_REASONING_DETAILS
        assert assistant["codex_reasoning_items"] == BRANCH_CODEX_REASONING_ITEMS
        assert assistant["codex_message_items"] == BRANCH_CODEX_MESSAGE_ITEMS
        marker = _branched_marker(db, "branch-key")
        assert marker is not None, (
            "session.branch dropped display_kind: the marker re-entered the "
            "truncate ordinal address space as a phantom user turn (#82756)"
        )
    finally:
        server._sessions.pop("sid", None)
        db.close()


def test_save_cfg_preserves_user_comments(tmp_path, monkeypatch):
    """The TUI gateway must not strip user-edited comments on setting writes."""
    cfg_path = tmp_path / "config.yaml"
    cfg_path.write_text(
        "# top of file note\n"
        "model:\n"
        "  # provider rationale\n"
        "  default: claude-opus-4-7\n"
        "display:\n"
        "  skin: default  # trailing skin note\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(server, "_hermes_home", tmp_path)

    server._save_cfg(
        {
            "model": {"default": "claude-opus-4-7"},
            "display": {"skin": "mono"},
        }
    )

    text = cfg_path.read_text(encoding="utf-8-sig")
    assert "# top of file note" in text
    assert "# provider rationale" in text
    assert "# trailing skin note" in text

    import hermes_yaml as _yaml

    parsed = _yaml.safe_load(text)
    assert parsed["display"]["skin"] == "mono"


def test_save_cfg_preserves_top_level_key_order(tmp_path, monkeypatch):
    """Top-level keys must keep the file's hand-edited ordering instead of
    being rewritten alphabetically by the underlying YAML dumper."""
    cfg_path = tmp_path / "config.yaml"
    cfg_path.write_text(
        "model:\n"
        "  default: claude-opus-4-7\n"
        "toolsets:\n"
        "  - hermes-cli\n"
        "agent:\n"
        "  max_turns: 90\n"
        "display:\n"
        "  skin: default\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(server, "_hermes_home", tmp_path)

    # Caller's dict iteration order is intentionally alphabetical to confirm
    # the helper consults disk order, not caller order.
    server._save_cfg(
        {
            "agent": {"max_turns": 90},
            "display": {"skin": "mono"},
            "model": {"default": "claude-opus-4-7"},
            "toolsets": ["hermes-cli"],
        }
    )

    text = cfg_path.read_text(encoding="utf-8-sig")
    top_keys = [
        line.split(":", 1)[0]
        for line in text.splitlines()
        if line and not line.startswith(" ") and not line.startswith("-")
        and not line.startswith("#")
    ]
    assert top_keys == ["model", "toolsets", "agent", "display"]


def test_save_cfg_keeps_unicode_personalities_readable(tmp_path, monkeypatch):
    """The catgirl/kawaii personality prompts must stay readable on disk
    instead of being \\uXXXX-escaped on every unrelated setting write."""
    cfg_path = tmp_path / "config.yaml"
    cfg_path.write_text(
        "agent:\n"
        "  personalities:\n"
        "    catgirl: \"nya (=^･ω･^=) 你好\"\n"
        "display:\n"
        "  skin: default\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(server, "_hermes_home", tmp_path)

    # Simulate an unrelated /skin write — must not corrupt the catgirl
    # personality string sitting in agent.personalities.
    server._save_cfg(
        {
            "agent": {"personalities": {"catgirl": "nya (=^･ω･^=) 你好"}},
            "display": {"skin": "mono"},
        }
    )

    text = cfg_path.read_text(encoding="utf-8-sig")
    assert "你好" in text
    assert "(=^･ω･^=)" in text
    assert "\\u4f60" not in text


def test_personality_marker_does_not_shift_truncate_ordinal(monkeypatch):
    """A personality pivot must not occupy a slot in the ordinal address space.

    ``_apply_personality_to_session`` injects its pivot as ``role=user`` so
    strict OpenAI-compatible providers accept it mid-conversation. Untagged, the
    ordinal filter (``role == "user" and not display_kind``) counted it as a
    real user turn while no client renders it as one, so every rewind issued
    after a personality change resolved one turn too early and
    ``replace_messages()`` hard-deleted the extra span (#82756, third occurrence
    after #70516 / #80763).

    The sibling ``test_prompt_submit_truncate_ordinal_skips_display_kind_rows``
    pins the filter itself; this one pins the producer, which is where the
    invariant was actually broken.
    """

    class _Agent:
        def run_conversation(self, prompt, conversation_history=None, stream_callback=None, **_kwargs):
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

    session = _session(
        agent=_Agent(),
        history=[
            {"role": "user", "content": "first"},
            {"role": "assistant", "content": "first reply"},
        ],
    )
    server._sessions["personality-ordinal-sid"] = session
    stub_db = _StubDb()

    try:
        monkeypatch.setattr(server, "_session_info", lambda *a, **k: {})
        monkeypatch.setattr(server, "_emit", lambda *a: None)

        # Real production injection point — not a hand-written marker dict.
        server._apply_personality_to_session(
            "personality-ordinal-sid", session, "talk like a pirate", "pirate"
        )

        marker = session["history"][-1]
        assert marker["role"] == "user", "provider compatibility: pivot rides as a user turn"

        # Two more real turns land after the personality change.
        session["history"].extend(
            [
                {"role": "user", "content": "second"},
                {"role": "assistant", "content": "second reply"},
                {"role": "user", "content": "third"},
                {"role": "assistant", "content": "third reply"},
            ]
        )
        history_before = list(session["history"])
        third_index = history_before.index({"role": "user", "content": "third"})

        monkeypatch.setattr(server.threading, "Thread", _ImmediateThread)
        monkeypatch.setattr(server, "_get_usage", lambda _a: {})
        monkeypatch.setattr(server, "render_message", lambda _t, _c: "")
        monkeypatch.setattr(server, "_get_db", lambda: stub_db)

        # The client counts three user bubbles (first=0, second=1, third=2) —
        # it never sees the pivot. Rewinding to "third" must cut exactly there.
        resp = server.handle_request(
            {
                "id": "1",
                "method": "prompt.submit",
                "params": {
                    "session_id": "personality-ordinal-sid",
                    "text": "third, reworded",
                    "truncate_before_user_ordinal": 2,
                    "confirm_truncate": True,
                },
            }
        )
        assert resp.get("result"), f"got error: {resp.get('error')}"

        expected = history_before[:third_index]
        assert stub_db.replaced == [("session-key", expected)], (
            "the pivot shifted the ordinal: the cut landed at "
            f"{len(stub_db.replaced[0][1]) if stub_db.replaced else None} instead of {third_index}"
        )
        # The turn before the target must survive — that is the span the three
        # reported incidents lost.
        assert {"role": "user", "content": "second"} in expected
        # And the mechanism that keeps it out of the address space, so a future
        # producer cannot regress this by dropping the tag.
        assert marker.get("display_kind"), (
            "an untagged role=user pivot silently consumes a truncate ordinal slot"
        )
    finally:
        server._sessions.pop("personality-ordinal-sid", None)


def test_prompt_submit_truncation_archives_instead_of_deleting(monkeypatch):
    """The rewind write must be recoverable, not a hard DELETE.

    #70516 / #80763 / #82756 all ended at this one call, and all three were
    unrecoverable for the same reason: `replace_messages` DELETEs the rows and
    the FTS entry goes with them. Guarding the *aim* of a rewind still leaves
    every other way of aiming it wrong terminal, so the write itself has to
    stop destroying. The storage-layer contract this relies on is pinned in
    tests/hermes_state/test_replace_messages_archive_siblings.py.
    """

    captured = {}

    class _Agent:
        def run_conversation(self, prompt, conversation_history=None, stream_callback=None, **_kwargs):
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

    class _StubDb:
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
            captured["active_only"] = active_only
            captured["archive_dropped"] = archive_dropped
            captured["reject_active_turn_lease"] = reject_active_turn_lease

    server._sessions["archive-trunc-sid"] = _session(
        agent=_Agent(),
        history=[
            {"role": "user", "content": "first"},
            {"role": "assistant", "content": "first reply"},
            {"role": "user", "content": "second"},
            {"role": "assistant", "content": "second reply"},
        ],
    )

    try:
        monkeypatch.setattr(server.threading, "Thread", _ImmediateThread)
        monkeypatch.setattr(server, "_session_info", lambda *a, **k: {})
        monkeypatch.setattr(server, "_emit", lambda *a: None)
        monkeypatch.setattr(server, "_get_usage", lambda _a: {})
        monkeypatch.setattr(server, "render_message", lambda _t, _c: "")
        monkeypatch.setattr(server, "_get_db", lambda: _StubDb())

        resp = server.handle_request(
            {
                "id": "1",
                "method": "prompt.submit",
                "params": {
                    "session_id": "archive-trunc-sid",
                    "text": "second, reworded",
                    "truncate_before_user_ordinal": 1,
                    "confirm_truncate": True,
                },
            }
        )

        assert resp.get("result"), f"got error: {resp.get('error')}"
        assert captured.get("archive_dropped") is True, (
            "a rewind must soft-archive the turns it drops, not DELETE them"
        )
        # #80216: still must not touch rows archived by an earlier compaction.
        assert captured.get("active_only") is True
        assert captured.get("reject_active_turn_lease") is True
    finally:
        server._sessions.pop("archive-trunc-sid", None)


def test_prompt_submit_unmatched_row_id_refuses_even_with_ordinal(monkeypatch):
    """#82959: Unknown row_id must refuse (4018), not fall back to a client ordinal."""
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
            replaced.append((key, list(messages)))

    history = [
        {"_row_id": 101, "role": "user", "content": "first"},
        {"_row_id": 102, "role": "assistant", "content": "reply 1"},
        {"_row_id": 103, "role": "user", "content": "second"},
        {"_row_id": 104, "role": "assistant", "content": "reply 2"},
    ]
    sess = _session(history=list(history))
    server._sessions["fallback-row-id-sid"] = sess
    monkeypatch.setattr(server, "_get_db", lambda: _FakeDB())
    monkeypatch.setattr(
        server, "_start_agent_build", lambda *a, **k: pytest.fail("must not start a turn")
    )

    try:
        # Stale row_id 999 not in history — even with a valid ordinal, refuse.
        resp = server.handle_request(
            {
                "id": "1",
                "method": "prompt.submit",
                "params": {
                    "session_id": "fallback-row-id-sid",
                    "text": "new turn",
                    "truncate_before_row_id": 999,
                    "truncate_before_user_ordinal": 1,
                    "confirm_truncate": True,
                },
            }
        )
        assert resp.get("error") is not None
        assert resp["error"]["code"] == 4018
        assert replaced == []
        assert len(sess["history"]) == 4
    finally:
        server._sessions.pop("fallback-row-id-sid", None)


def test_prompt_submit_unmatched_message_id_refuses_even_with_ordinal(monkeypatch):
    """#82959: Unknown message_id must refuse; no silent ordinal degradation."""
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
            replaced.append((key, list(messages)))

    # Production-shaped history: no renderer "id" keys on user dicts.
    history = [
        {"_row_id": 201, "role": "user", "content": "first"},
        {"_row_id": 202, "role": "assistant", "content": "reply 1"},
        {"_row_id": 203, "role": "user", "content": "second"},
        {"_row_id": 204, "role": "assistant", "content": "reply 2"},
    ]
    sess = _session(history=list(history))
    server._sessions["synthetic-msg-id-sid"] = sess
    monkeypatch.setattr(server, "_get_db", lambda: _FakeDB())
    monkeypatch.setattr(
        server, "_start_agent_build", lambda *a, **k: pytest.fail("must not start a turn")
    )

    try:
        resp = server.handle_request(
            {
                "id": "1",
                "method": "prompt.submit",
                "params": {
                    "session_id": "synthetic-msg-id-sid",
                    "text": "fresh start",
                    "truncate_before_message_id": "user-1723456789-0",
                    "truncate_before_user_ordinal": 0,
                    "confirm_truncate": True,
                    "confirm_empty_truncate": True,
                },
            }
        )
        assert resp.get("error") is not None
        assert resp["error"]["code"] == 4018
        assert replaced == []
        assert len(sess["history"]) == 4
    finally:
        server._sessions.pop("synthetic-msg-id-sid", None)


def test_prompt_submit_row_id_resolves_via_db_when_memory_lacks_stamps(monkeypatch):
    """#82959: After turn rewrite strips _row_id, resolve against durable DB history."""
    replaced = []
    # Live memory after turn completion: provider-format, no _row_id stamps.
    live_history = [
        {"role": "user", "content": "first"},
        {"role": "assistant", "content": "reply 1"},
        {"role": "user", "content": "second"},
        {"role": "assistant", "content": "reply 2"},
    ]
    durable_history = [
        {"_row_id": 501, "role": "user", "content": "first"},
        {"_row_id": 502, "role": "assistant", "content": "reply 1"},
        {"_row_id": 503, "role": "user", "content": "second"},
        {"_row_id": 504, "role": "assistant", "content": "reply 2"},
    ]

    class _FakeDB:
        def replace_messages(
            self,
            key,
            messages,
            active_only=False,
            archive_dropped=False,
            reject_active_turn_lease=False,
        ):
            replaced.append((key, list(messages)))

        def get_messages_as_conversation(self, key, repair_alternation=False, include_row_ids=False):
            assert include_row_ids is True
            return list(durable_history)

    sess = _session(history=list(live_history), session_key="db-row-key")
    server._sessions["db-row-resolve-sid"] = sess
    monkeypatch.setattr(server, "_get_db", lambda: _FakeDB())
    monkeypatch.setattr(server, "_start_agent_build", lambda *a, **k: None)

    try:
        resp = server.handle_request(
            {
                "id": "1",
                "method": "prompt.submit",
                "params": {
                    "session_id": "db-row-resolve-sid",
                    "text": "new turn",
                    "truncate_before_row_id": 503,
                    "confirm_truncate": True,
                },
            }
        )
        assert resp.get("error") is None, resp
        assert len(sess["history"]) == 2
        assert sess["history"][-1]["content"] == "reply 1"
        assert len(replaced) == 1
        # Healing: live list should now carry stamps for subsequent rewinds.
        assert sess["history"][0].get("_row_id") == 501
    finally:
        server._sessions.pop("db-row-resolve-sid", None)


def test_prompt_submit_row_id_real_sessiondb_resolve_without_memory_stamps(
    monkeypatch, tmp_path
):
    """#82959 production path: real SessionDB insert → live history without
    _row_id (turn rewrite) → truncate_before_row_id cuts durable + memory.

    No hand-seeded ids and no MagicMock state manager — the contract that
    #82766 review said unit fixtures must exercise.
    """
    from hermes_state import SessionDB

    db = SessionDB(db_path=tmp_path / "rowid-trunc.db")
    session_key = "real-db-row-trunc"
    db.create_session(session_key, "cli")
    msgs = [
        {"role": "user", "content": "first"},
        {"role": "assistant", "content": "reply 1"},
        {"role": "user", "content": "second"},
        {"role": "assistant", "content": "reply 2"},
        {"role": "user", "content": "third"},
        {"role": "assistant", "content": "reply 3"},
    ]
    with db._lock:
        db._insert_message_rows(db._conn, session_key, msgs)
        db._conn.commit()
    row_ids = [m["_row_id"] for m in msgs]
    assert all(isinstance(r, int) and r > 0 for r in row_ids)

    # After turn completion gateway rewrites history as provider-format dicts
    # without _row_id — production-shaped live memory.
    live_history = [{"role": m["role"], "content": m["content"]} for m in msgs]
    assert all("_row_id" not in m for m in live_history)

    sess = _session(history=list(live_history), session_key=session_key)
    sid = "real-db-row-trunc-sid"
    server._sessions[sid] = sess
    monkeypatch.setattr(server, "_get_db", lambda: db)
    monkeypatch.setattr(server, "_start_agent_build", lambda *a, **k: None)
    monkeypatch.setattr(server, "_start_inflight_turn", lambda *a, **k: None)

    try:
        # Cut before second user turn (row_ids[2]) — leave first exchange only.
        resp = server.handle_request(
            {
                "id": "1",
                "method": "prompt.submit",
                "params": {
                    "session_id": sid,
                    "text": "rewound second",
                    "truncate_before_row_id": row_ids[2],
                    "truncate_before_user_ordinal": 1,
                    "confirm_truncate": True,
                    "confirm_deep_truncate": True,
                },
            }
        )
        assert resp.get("error") is None, resp
        assert len(sess["history"]) == 2
        assert sess["history"][0]["content"] == "first"
        assert sess["history"][1]["content"] == "reply 1"
        # Durable active transcript matches the cut plus the prompt just sent, which is durable at
        # submit (#111868) — before the turn runs (archive_dropped keeps inactive rows;
        # get_messages_as_conversation returns active only).
        active = db.get_messages_as_conversation(session_key)
        assert [m["content"] for m in active] == ["first", "reply 1", "rewound second"]
        # Heal stamps for subsequent rewinds when memory lined up with DB.
        assert sess["history"][0].get("_row_id") is not None
    finally:
        server._sessions.pop(sid, None)


def test_prompt_submit_row_id_real_sessiondb_unknown_refuses_despite_ordinal(
    monkeypatch, tmp_path
):
    """#82959 fail-closed: unknown row_id + valid ordinal must not truncate
    real SessionDB (the mass-delete class when durable id cannot resolve).
    """
    from hermes_state import SessionDB

    db = SessionDB(db_path=tmp_path / "rowid-refuse.db")
    session_key = "real-db-row-refuse"
    db.create_session(session_key, "cli")
    msgs = [
        {"role": "user", "content": "first"},
        {"role": "assistant", "content": "reply 1"},
        {"role": "user", "content": "second"},
        {"role": "assistant", "content": "reply 2"},
    ]
    with db._lock:
        db._insert_message_rows(db._conn, session_key, msgs)
        db._conn.commit()

    live_history = [{"role": m["role"], "content": m["content"]} for m in msgs]
    sess = _session(history=list(live_history), session_key=session_key)
    sid = "real-db-row-refuse-sid"
    server._sessions[sid] = sess
    monkeypatch.setattr(server, "_get_db", lambda: db)
    monkeypatch.setattr(
        server, "_start_agent_build", lambda *a, **k: pytest.fail("must not start a turn")
    )
    monkeypatch.setattr(
        server, "_start_inflight_turn", lambda *a, **k: pytest.fail("must not start a turn")
    )

    n_before = len(db.get_messages_as_conversation(session_key))
    try:
        resp = server.handle_request(
            {
                "id": "1",
                "method": "prompt.submit",
                "params": {
                    "session_id": sid,
                    "text": "stale durable id",
                    "truncate_before_row_id": 999_999_999,
                    "truncate_before_user_ordinal": 0,
                    "confirm_truncate": True,
                    "confirm_empty_truncate": True,
                },
            }
        )
        assert resp.get("error") is not None
        assert resp["error"]["code"] == 4018
        assert len(sess["history"]) == 4
        assert len(db.get_messages_as_conversation(session_key)) == n_before
    finally:
        server._sessions.pop(sid, None)


def test_prompt_submit_row_id_misaligned_memory_refuses_content_swap(
    monkeypatch, tmp_path
):
    """#82959 heal-path guard: equal-length but content-misaligned live memory
    must refuse (4018), not zip-stamp durable ids positionally and cut the
    wrong turn. Probe 4a from the PR #83202 review.
    """
    from hermes_state import SessionDB

    db = SessionDB(db_path=tmp_path / "rowid-misalign-content.db")
    session_key = "real-db-row-misalign-content"
    db.create_session(session_key, "cli")
    msgs = [
        {"role": "user", "content": "A"},
        {"role": "assistant", "content": "ra"},
        {"role": "user", "content": "B"},
        {"role": "assistant", "content": "rb"},
    ]
    with db._lock:
        db._insert_message_rows(db._conn, session_key, msgs)
        db._conn.commit()
    rid_b = msgs[2]["_row_id"]
    original_row_ids = [message["_row_id"] for message in msgs]

    # Same length + same role pattern, but content positions swapped: a
    # positional stamp would mark live "B" with durable A's row id and the
    # cut would keep the very turn the user rewound past.
    live_history = [
        {"role": "user", "content": "B"},
        {"role": "assistant", "content": "rb"},
        {"role": "user", "content": "A"},
        {"role": "assistant", "content": "ra"},
    ]
    sess = _session(history=list(live_history), session_key=session_key)
    sid = "misalign-content-sid"
    server._sessions[sid] = sess
    monkeypatch.setattr(server, "_get_db", lambda: db)
    monkeypatch.setattr(
        server, "_start_agent_build", lambda *a, **k: pytest.fail("must not start a turn")
    )
    monkeypatch.setattr(
        server, "_start_inflight_turn", lambda *a, **k: pytest.fail("must not start a turn")
    )

    n_before = len(db.get_messages_as_conversation(session_key))
    try:
        resp = server.handle_request(
            {
                "id": "1",
                "method": "prompt.submit",
                "params": {
                    "session_id": sid,
                    "text": "rewind B",
                    "truncate_before_row_id": rid_b,
                    "rebind_survivor_row_ids": [*original_row_ids, 999_999],
                    "confirm_truncate": True,
                },
            }
        )
        assert resp.get("error") is not None
        assert resp["error"]["code"] == 4018
        # Fail-closed: nothing stamped onto the misaligned dicts, nothing cut.
        assert all("_row_id" not in m for m in sess["history"])
        assert len(sess["history"]) == 4
        assert len(db.get_messages_as_conversation(session_key)) == n_before
    finally:
        server._sessions.pop(sid, None)



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


def test_prompt_submit_resolves_row_id_swallowed_by_marker_merge(monkeypatch):
    """#94486 (live-session shape): model-switch markers persist as role=user
    (#48338), so the just-sent prompt forms a user;user run that
    repair_message_sequence merges into the FIRST row — the target's _row_id
    vanishes from the repaired view and truncate_before_row_id fails closed
    (4018) even though the row is physically present. Identity resolution must
    read the un-repaired durable transcript, the same discipline the rebind
    path already applies ('repair can merge a physical user;user pair while
    preserving only the first row id').
    """
    import copy

    from agent.agent_runtime_helpers import repair_message_sequence

    verbatim = [
        {"_row_id": 201, "role": "user", "content": "first"},
        {"_row_id": 202, "role": "assistant", "content": "reply 1"},
        {
            "_row_id": 203,
            "role": "user",
            "display_kind": "model_switch",
            "content": "[System: The active model for this chat has changed to ds4.]",
        },
        {
            "_row_id": 204,
            "role": "user",
            "display_kind": "model_switch",
            "content": "[System: The active model for this chat has changed to kk3.]",
        },
        {"_row_id": 205, "role": "user", "content": "the interrupted prompt"},
    ]
    repaired = copy.deepcopy(verbatim)
    assert repair_message_sequence(None, repaired) == 2  # 3-row run, two merges
    # On main the merge buries the plain row's id inside the marker.
    assert len(repaired) == 3

    replaced = []
    fake_db = _mk_merge_fake_db(verbatim, repaired, replaced)

    @contextlib.contextmanager
    def _fake_session_db(_session):
        yield fake_db

    # Live history as a running session holds it after a turn rewrite: every
    # physical row present, row-id stamps lost (provider-format rewrite), so
    # resolution must go through the durable fallback.
    live = [
        {k: v for k, v in message.items() if k != "_row_id"} for message in verbatim
    ]
    server._sessions["marker-merge-sid"] = _session(history=list(live))
    monkeypatch.setattr(server, "_session_db", _fake_session_db)
    monkeypatch.setattr(server, "_get_db", lambda: fake_db)
    monkeypatch.setattr(server, "_start_agent_build", lambda *a, **k: None)
    monkeypatch.setattr(server, "_start_inflight_turn", lambda *a, **k: None)

    try:
        resp = server.handle_request(
            {
                "id": "1",
                "method": "prompt.submit",
                "params": {
                    "session_id": "marker-merge-sid",
                    "text": "edited prompt",
                    "truncate_before_row_id": 205,
                    "confirm_truncate": True,
                },
            }
        )
        assert isinstance(resp, dict)
        err = resp.get("error")
        assert err is None, err
        # The cut lands before row 205: the markers and everything earlier
        # survive (durable rows healed their stamps onto the live list).
        assert len(replaced) == 1
        assert replaced[0][0] == "session-key"
        assert [m["content"] for m in replaced[0][1]] == [
            "first",
            "reply 1",
            verbatim[2]["content"],
            verbatim[3]["content"],
        ]
        assert len(server._sessions["marker-merge-sid"]["history"]) == 4
    finally:
        server._sessions.pop("marker-merge-sid", None)


def test_prompt_submit_resolves_row_id_swallowed_by_plain_user_merge(monkeypatch):
    """Same swallowed-id class with NO display marker: an interrupted turn
    persists no assistant row, the user's resend lands as a second consecutive
    plain user row, and repair merges it into the earlier one — the resend's
    row id is unresolvable on every later rewind/edit. Repair-side address
    preservation keyed on display_kind cannot reach this shape; reading the
    un-repaired transcript for identity can.
    """
    import copy

    from agent.agent_runtime_helpers import repair_message_sequence

    verbatim = [
        {"_row_id": 301, "role": "user", "content": "first"},
        {"_row_id": 302, "role": "assistant", "content": "reply 1"},
        {"_row_id": 303, "role": "user", "content": "prompt whose turn was cut off"},
        {"_row_id": 304, "role": "user", "content": "the resent prompt"},
    ]
    repaired = copy.deepcopy(verbatim)
    assert repair_message_sequence(None, repaired) == 1
    assert len(repaired) == 3
    assert repaired[2]["_row_id"] == 303  # merge keeps only the FIRST row id

    replaced = []
    fake_db = _mk_merge_fake_db(verbatim, repaired, replaced)

    @contextlib.contextmanager
    def _fake_session_db(_session):
        yield fake_db

    # Production cold resume materializes the live history from the DB with
    # repair_alternation=True (server._load_resume_transcript), stamps kept:
    # rows 303/304 collapse into ONE live carrier at 303, so 304 has no live
    # user ordinal. The un-repaired DB read finds 304 as physical ordinal 2,
    # but the live list only has user ordinals 0/1 — resolution must map the
    # DB row onto the repaired live carrier instead of trusting the same
    # ordinal against a list that cannot have it.
    live = copy.deepcopy(repaired)
    assert live[2]["_row_id"] == 303
    server._sessions["plain-merge-sid"] = _session(history=list(live))
    monkeypatch.setattr(server, "_session_db", _fake_session_db)
    monkeypatch.setattr(server, "_get_db", lambda: fake_db)
    monkeypatch.setattr(server, "_start_agent_build", lambda *a, **k: None)
    monkeypatch.setattr(server, "_start_inflight_turn", lambda *a, **k: None)

    try:
        resp = server.handle_request(
            {
                "id": "1",
                "method": "prompt.submit",
                "params": {
                    "session_id": "plain-merge-sid",
                    "text": "edited resend",
                    "truncate_before_row_id": 304,
                    "confirm_truncate": True,
                },
            }
        )
        assert isinstance(resp, dict)
        err = resp.get("error")
        assert err is None, err
        assert len(replaced) == 1
        # The cut retains 303: deriving it from the physical durable prefix
        # keeps the inner row boundary instead of collapsing the pair.
        assert [m["content"] for m in replaced[0][1]] == [
            "first",
            "reply 1",
            "prompt whose turn was cut off",
        ]
        assert len(server._sessions["plain-merge-sid"]["history"]) == 3
    finally:
        server._sessions.pop("plain-merge-sid", None)


def test_prompt_submit_row_id_ignores_platform_id_fallback(monkeypatch):
    """truncate_before_row_id must not match string platform IDs."""
    history = [
        {"id": "999", "role": "user", "content": "first"},
        {"role": "assistant", "content": "reply 1"},
    ]
    server._sessions["string-id-sid"] = _session(history=list(history))
    try:
        resp = server.handle_request({
            "id": "1",
            "method": "prompt.submit",
            "params": {
                "session_id": "string-id-sid",
                "text": "new turn",
                "truncate_before_row_id": 999,
                "confirm_truncate": True,
            }
        })
        assert resp.get("error") is not None
        assert resp["error"]["code"] == 4018
    finally:
        server._sessions.pop("string-id-sid", None)


def test_prompt_submit_refuses_empty_truncation_without_confirm(monkeypatch):
    """A confirmed rewind still must not wipe a non-empty transcript by accident.

    Ordinal 0 cuts at the first user message (history[:0] == []) and
    replace_messages() would DELETE every durable row. Even a submit that
    declares rewind intent needs the second opt-in for that edge.
    """
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
        {"_row_id": 102, "role": "assistant", "content": "ok"},
        {"_row_id": 103, "role": "user", "content": "second"},
        {"_row_id": 104, "role": "assistant", "content": "done"},
    ]
    server._sessions["empty-trunc-sid"] = _session(history=list(history))
    monkeypatch.setattr(server, "_get_db", lambda: _FakeDB())
    monkeypatch.setattr(
        server, "_start_agent_build", lambda *a, **k: pytest.fail("must not start a turn")
    )
    monkeypatch.setattr(
        server, "_start_inflight_turn", lambda *a, **k: pytest.fail("must not start a turn")
    )

    try:
        # Missing confirm → refuse.
        resp = server.handle_request(
            {
                "id": "1",
                "method": "prompt.submit",
                "params": {
                    "session_id": "empty-trunc-sid",
                    "text": "fresh typed message",
                    "truncate_before_row_id": 101,
                    "truncate_before_user_ordinal": 0,
                    "confirm_truncate": True,
                },
            }
        )
        assert resp["error"]["code"] == 4028
        assert "confirm_empty_truncate" in resp["error"]["message"]
        # Explicit falsey values must not satisfy the opt-in either.
        for falsey in (False, 0, "", "false", "no"):
            resp = server.handle_request(
                {
                    "id": "1",
                    "method": "prompt.submit",
                    "params": {
                        "session_id": "empty-trunc-sid",
                        "text": "fresh typed message",
                        "truncate_before_row_id": 101,
                        "truncate_before_user_ordinal": 0,
                        "confirm_truncate": True,
                        "confirm_empty_truncate": falsey,
                    },
                }
            )
            assert resp["error"]["code"] == 4028, falsey
        assert server._sessions["empty-trunc-sid"]["history"] == history
        assert server._sessions["empty-trunc-sid"]["running"] is False
        assert server._sessions["empty-trunc-sid"]["history_version"] == 0
        assert replaced == []
    finally:
        server._sessions.pop("empty-trunc-sid", None)


def test_notification_poller_live_loop_requeues_foreign_completion_for_owner(
    monkeypatch,
):
    """A foreign live-loop dequeue is handed back to its proven owner."""
    import queue as _queue_mod

    from tools.process_registry import process_registry

    delivered = {"a": [], "b": []}
    emitted = []
    session_a = _session(session_key="session-a-live-handoff")
    session_b = _session(session_key="session-b-live-handoff")
    event = {
        "type": "completion",
        "session_id": "proc-live-handoff",
        "session_key": "session-a-live-handoff",
        "command": "echo owner",
        "exit_code": 0,
        "output": "owner",
    }
    isolated_queue: _queue_mod.Queue = _queue_mod.Queue()
    isolated_queue.put(event)
    monkeypatch.setattr(process_registry, "completion_queue", isolated_queue)
    monkeypatch.setattr(server, "_get_db", lambda: None)
    monkeypatch.setattr(server, "_emit", lambda *args, **_kwargs: emitted.append(args))

    def _deliver(_rid, sid, session, text, **_kw):
        delivered["a" if sid == "sid-a-live-handoff" else "b"].append(text)
        session["running"] = False
        return True  # the execution gate accepted the turn; None would re-queue for redelivery

    monkeypatch.setattr(server, "_run_prompt_submit", _deliver)
    server._sessions.update(
        {
            "sid-a-live-handoff": session_a,
            "sid-b-live-handoff": session_b,
        }
    )
    process_registry._completion_consumed.discard(event["session_id"])

    try:
        server._notification_poller_loop(
            _StopAfterOneNotificationPoll(), "sid-b-live-handoff", session_b
        )

        assert delivered["b"] == []
        assert emitted == []
        assert isolated_queue.qsize() == 1
        assert isolated_queue.queue[0] is event

        server._notification_poller_loop(
            _StopAfterOneNotificationPoll(), "sid-a-live-handoff", session_a
        )

        assert len(delivered["a"]) == 1
        assert "proc-live-handoff" in delivered["a"][0]
        assert delivered["b"] == []
        assert isolated_queue.empty()
    finally:
        server._sessions.pop("sid-a-live-handoff", None)
        server._sessions.pop("sid-b-live-handoff", None)
        process_registry._completion_consumed.discard(event["session_id"])
        while not isolated_queue.empty():
            isolated_queue.get_nowait()


def test_completion_ownership_lineage_lookup_failure_fails_closed(monkeypatch):
    """A provenance lookup failure cannot turn an addressed event into ours."""
    import queue as _queue_mod

    from tools.process_registry import process_registry

    class _BrokenDB:
        def resolve_resume_session_id(self, _session_key):
            raise RuntimeError("lineage database unavailable")

    session = _session(session_key="unrelated-live-session")
    event = {
        "type": "completion",
        "session_id": "proc-unknown-lineage",
        "session_key": "unknown-parent",
        "command": "echo unknown",
        "exit_code": 0,
        "output": "unknown",
    }
    isolated_queue: _queue_mod.Queue = _queue_mod.Queue()
    isolated_queue.put(event)
    monkeypatch.setattr(process_registry, "completion_queue", isolated_queue)
    monkeypatch.setattr(server, "_get_db", lambda: _BrokenDB())

    drained = process_registry.drain_notifications(
        session_key="unrelated-live-session",
        owns_event=lambda candidate: server._session_owns_notification_event(
            "sid-unrelated-live", session, candidate
        ),
    )

    assert drained == []
    assert isolated_queue.qsize() == 1
    assert isolated_queue.get_nowait() is event


@pytest.mark.parametrize(
    "routing",
    [
        {"session_key": "missing-owner-key"},
        {"origin_ui_session_id": "missing-owner-sid"},
    ],
)
def test_notification_poller_live_loop_drops_addressed_orphan(
    monkeypatch, routing
):
    """A live poll never injects an addressed event whose owner is gone."""
    import queue as _queue_mod

    from tools.process_registry import process_registry

    delivered = []
    emitted = []
    session = _session(session_key="unrelated-live-key")
    event = {
        "type": "completion",
        "session_id": "proc-live-orphan",
        "command": "echo orphan",
        "exit_code": 0,
        "output": "orphan",
        **routing,
    }
    isolated_queue: _queue_mod.Queue = _queue_mod.Queue()
    isolated_queue.put(event)
    monkeypatch.setattr(process_registry, "completion_queue", isolated_queue)
    monkeypatch.setattr(server, "_get_db", lambda: None)
    monkeypatch.setattr(server, "_emit", lambda *args, **_kwargs: emitted.append(args))
    monkeypatch.setattr(
        server,
        "_run_prompt_submit",
        lambda _rid, _sid, _session, text, **_kw: delivered.append(text),
    )
    server._sessions["sid-live-orphan"] = session
    process_registry._completion_consumed.discard(event["session_id"])

    try:
        server._notification_poller_loop(
            _StopAfterOneNotificationPoll(), "sid-live-orphan", session
        )

        assert delivered == []
        assert emitted == []
        assert isolated_queue.empty()
    finally:
        server._sessions.pop("sid-live-orphan", None)
        process_registry._completion_consumed.discard(event["session_id"])
        while not isolated_queue.empty():
            isolated_queue.get_nowait()


@pytest.mark.parametrize(
    "routing",
    [
        {"session_key": "session-b"},
        {"origin_ui_session_id": "sid_gone"},
    ],
)
def test_notification_poller_drops_orphaned_events(monkeypatch, routing):
    """Addressed completions whose owner is gone are dropped, not hijacked."""
    import queue as _queue_mod

    from tools.process_registry import process_registry

    emitted = []
    delivered = []
    sess = _session(session_key="session-a")
    server._sessions["sid_a"] = sess
    monkeypatch.setattr(server, "_emit", lambda *a, **kw: emitted.append(a))
    monkeypatch.setattr(
        server,
        "_run_prompt_submit",
        lambda _rid, _sid, _session, text, **_kw: delivered.append(text),
    )
    monkeypatch.setattr(server, "_get_db", lambda: None)

    isolated_queue: _queue_mod.Queue = _queue_mod.Queue()
    monkeypatch.setattr(process_registry, "completion_queue", isolated_queue)
    process_registry._completion_consumed.discard("proc_ghost")
    isolated_queue.put(
        {
            "type": "completion",
            "session_id": "proc_ghost",
            "command": "echo from ghost",
            "exit_code": 0,
            "output": "ghost output",
            **routing,
        }
    )

    stop = threading.Event()
    stop.set()

    try:
        server._notification_poller_loop(stop, "sid_a", sess)

        assert [a for a in emitted if a[0] == "status.update"] == []
        assert delivered == []
    finally:
        server._sessions.pop("sid_a", None)
        while not process_registry.completion_queue.empty():
            process_registry.completion_queue.get_nowait()


@pytest.mark.parametrize(
    ("routing", "resolved_key"),
    [
        ({"session_key": "session-a"}, None),
        (
            {
                "session_key": "stale-durable-key",
                "origin_ui_session_id": "sid_a",
            },
            None,
        ),
        ({"session_key": "old-parent-key"}, "session-a"),
    ],
)
def test_notification_poller_delivers_owned_events(
    monkeypatch, routing, resolved_key
):
    """Direct, UI-origin, and compression-lineage owners are delivered."""
    import queue as _queue_mod

    from tools.process_registry import process_registry

    class _CompressionDB:
        def resolve_resume_session_id(self, key):
            return resolved_key if key == "old-parent-key" and resolved_key else key

    delivered = []
    emitted = []
    sess = _session(session_key="session-a")
    server._sessions["sid_a"] = sess
    monkeypatch.setattr(server, "_emit", lambda *a, **kw: emitted.append(a))
    monkeypatch.setattr(
        server,
        "_run_prompt_submit",
        lambda _rid, _sid, _session, text, **_kw: delivered.append(text),
    )
    monkeypatch.setattr(server, "_get_db", lambda: _CompressionDB())

    isolated_queue: _queue_mod.Queue = _queue_mod.Queue()
    monkeypatch.setattr(process_registry, "completion_queue", isolated_queue)
    process_registry._completion_consumed.discard("proc_mine")
    isolated_queue.put(
        {
            "type": "completion",
            "session_id": "proc_mine",
            "command": "echo mine",
            "exit_code": 0,
            "output": "mine",
            **routing,
        }
    )

    stop = threading.Event()
    stop.set()

    try:
        server._notification_poller_loop(stop, "sid_a", sess)

        status_calls = [a for a in emitted if a[0] == "status.update"]
        assert len(status_calls) == 1
        assert status_calls[0][2]["kind"] == "process"
        assert len(delivered) == 1
        assert "proc_mine" in delivered[0]
    finally:
        server._sessions.pop("sid_a", None)
        while not process_registry.completion_queue.empty():
            process_registry.completion_queue.get_nowait()


def test_run_prompt_submit_binds_exact_steer_authority_and_resets_contextvars(
    monkeypatch, tmp_path
):
    """The turn thread commissions children with this session generation only."""
    from tools.delegate_tool import _capture_gateway_steer_authority
    from hermes_runtime.transport import (
        bind_transport,
        current_transport,
        reset_transport,
    )

    class _Transport:
        def write(self, _obj):
            return True

        def close(self):
            return None

    observed = {}
    owner_transport = _Transport()
    previous_transport = _Transport()
    previous_record = {"session_key": "previous-generation"}

    class _CapturingAgent(_RecordingAgent):
        def run_conversation(self, prompt, **kwargs):
            authority = _capture_gateway_steer_authority("sid-owner")
            observed["transport"] = authority[0]
            observed["record"] = authority[1]
            return super().run_conversation(prompt, **kwargs)

    _configure_immediate_prompt_run(monkeypatch, tmp_path)
    session = _session(
        session_key="session-owner",
        agent=_CapturingAgent([]),
        running=True,
        transport=owner_transport,
    )
    server._sessions["sid-owner"] = session
    transport_token = bind_transport(previous_transport)
    record_token = server._current_runtime_session_record.set(previous_record)
    try:
        server._run_prompt_submit("rid-owner", "sid-owner", session, "commission")

        assert observed == {"transport": owner_transport, "record": session}
        assert current_transport() is previous_transport
        assert server._current_runtime_session_record.get() is previous_record
    finally:
        server._current_runtime_session_record.reset(record_token)
        reset_transport(transport_token)
        server._sessions.pop("sid-owner", None)


def test_run_prompt_submit_rejects_worker_when_close_wins_publication(
    monkeypatch, tmp_path
):
    """A close claimed during message.start must prevent the worker from running."""
    _configure_immediate_prompt_run(monkeypatch, tmp_path, immediate_threads=False)
    emit_entered = threading.Event()
    release_emit = threading.Event()
    dispatch_results = []
    turns = []
    popped = []
    sid = "close-wins-publication"
    session = _session(
        session_key="close-wins-publication-key",
        agent=_RecordingAgent(turns),
        running=True,
    )

    def _blocking_emit(event, *_args, **_kwargs):
        if event == "message.start":
            emit_entered.set()
            assert release_emit.wait(timeout=2.0)

    monkeypatch.setattr(server, "_emit", _blocking_emit)
    server._sessions[sid] = session
    dispatch_thread = threading.Thread(
        target=lambda: dispatch_results.append(
            server._run_prompt_submit("rid", sid, session, "turn")
        )
    )

    try:
        dispatch_thread.start()
        assert emit_entered.wait(timeout=1.0)
        popped.append(server._pop_session_by_id(sid))
        assert popped == [session]
        release_emit.set()
        dispatch_thread.join(timeout=2.0)
    finally:
        release_emit.set()
        dispatch_thread.join(timeout=2.0)
        run_thread = session.get("_run_thread")
        if run_thread is not None and run_thread.is_alive():
            run_thread.join(timeout=2.0)
        server._sessions.pop(sid, None)

    assert dispatch_results == [False]
    assert session["running"] is False
    assert turns == []


@pytest.mark.parametrize("exit_code", [0, 7])
def test_run_prompt_submit_requeues_foreign_completion(
    monkeypatch, tmp_path, exit_code
):
    import queue as _queue_mod

    from tools.process_registry import process_registry

    _configure_immediate_prompt_run(monkeypatch, tmp_path)
    turns = []
    session_a = _session(session_key="session-a")
    session_b = _session(
        session_key="session-b",
        agent=_RecordingAgent(turns),
        running=True,
    )
    event = {
        "type": "completion",
        "session_id": f"proc_foreign_{exit_code}",
        "session_key": "session-a",
        "command": "safe-test-command",
        "exit_code": exit_code,
        "output": "foreign",
    }
    isolated_queue: _queue_mod.Queue = _queue_mod.Queue()
    isolated_queue.put(event)
    monkeypatch.setattr(process_registry, "completion_queue", isolated_queue)
    server._sessions["sid_a"] = session_a
    server._sessions["sid_b"] = session_b

    try:
        server._run_prompt_submit("rid-b", "sid_b", session_b, "session-b-turn")

        assert turns == ["session-b-turn"]
        assert isolated_queue.get_nowait() == event
        assert isolated_queue.empty()
    finally:
        server._sessions.pop("sid_a", None)
        server._sessions.pop("sid_b", None)
        process_registry._completion_consumed.discard(event["session_id"])


def test_run_prompt_submit_delivers_completion_observed_by_poll(monkeypatch, tmp_path):
    import queue as _queue_mod

    from tools.process_registry import process_registry

    _configure_immediate_prompt_run(monkeypatch, tmp_path)
    turns = []
    session = _session(
        session_key="session-a",
        agent=_RecordingAgent(turns),
        running=True,
    )
    event = {
        "type": "completion",
        "session_id": "proc_polled",
        "session_key": "session-a",
        "command": "safe-test-command",
        "exit_code": 0,
        "output": "observed but not consumed",
    }
    isolated_queue: _queue_mod.Queue = _queue_mod.Queue()
    isolated_queue.put(event)
    monkeypatch.setattr(process_registry, "completion_queue", isolated_queue)
    process_registry._completion_consumed.discard(event["session_id"])
    process_registry._poll_observed.add(event["session_id"])
    server._sessions["sid_a"] = session

    try:
        server._run_prompt_submit("rid-a", "sid_a", session, "session-a-turn")

        assert turns[0] == "session-a-turn"
        assert len(turns) == 2
        assert "proc_polled" in turns[1]
        assert isolated_queue.empty()
    finally:
        server._sessions.pop("sid_a", None)
        process_registry._completion_consumed.discard(event["session_id"])
        process_registry._poll_observed.discard(event["session_id"])


def test_run_prompt_submit_requeues_all_unstarted_notifications_with_real_threading(
    monkeypatch, tmp_path
):
    import queue as _queue_mod

    from tools.process_registry import process_registry

    _configure_immediate_prompt_run(
        monkeypatch, tmp_path, immediate_threads=False
    )
    real_thread_class = threading.Thread
    threads = []
    nested_started = threading.Event()
    release_nested = threading.Event()
    turns = []

    def _recording_thread(*args, **kwargs):
        thread = real_thread_class(*args, **kwargs)
        threads.append(thread)
        return thread

    class _BlockingNotificationAgent(_RecordingAgent):
        def run_conversation(self, prompt, conversation_history=None, stream_callback=None, **_kwargs):
            turns.append(prompt)
            if "proc_batch_1" in prompt:
                nested_started.set()
                if not release_nested.wait(timeout=5):
                    raise TimeoutError("notification turn was not released")
            return {"final_response": "", "messages": []}

    monkeypatch.setattr(server.threading, "Thread", _recording_thread)
    session = _session(
        session_key="session-a",
        agent=_BlockingNotificationAgent(turns),
        running=True,
    )
    events = [
        {
            "type": "completion",
            "session_id": f"proc_batch_{index}",
            "session_key": "session-a",
            "command": "safe-test-command",
            "exit_code": 0,
            "output": f"owned-{index}",
        }
        for index in range(1, 4)
    ]
    # Consecutive completions share one turn (#104671); a watch_match is a turn
    # barrier, so it is the in-flight turn behind which batch_2/batch_3 must survive.
    events[0].update(type="watch_match", pattern="owned-1")
    isolated_queue: _queue_mod.Queue = _queue_mod.Queue()
    for event in events:
        isolated_queue.put(event)
    monkeypatch.setattr(process_registry, "completion_queue", isolated_queue)
    server._sessions["sid_a"] = session

    try:
        server._run_prompt_submit("rid-a", "sid_a", session, "session-a-turn")

        assert nested_started.wait(timeout=5)
        threads[0].join(timeout=5)
        assert not threads[0].is_alive()
        # Membership, not order: the completion_queue is process-global, and
        # notification pollers leaked by earlier session.init tests in this
        # file legitimately steal-and-requeue foreign-session events (see
        # _notification_poller_loop's belongs-elsewhere branch), rotating the
        # queue. The requeue contract is that batch_2 and batch_3 both remain
        # queued (never consumed) while batch_1's turn is in flight — so drain
        # with a deadline (an event may be transiently held by a poller
        # mid-cycle) and assert exactly {batch_2, batch_3} come back.
        queued: dict = {}
        deadline = time.time() + 5.0
        while time.time() < deadline and set(queued) != {
            "proc_batch_2",
            "proc_batch_3",
        }:
            try:
                evt = isolated_queue.get(timeout=0.1)
            except _queue_mod.Empty:
                continue
            queued[evt["session_id"]] = evt
        assert set(queued) == {"proc_batch_2", "proc_batch_3"}
    finally:
        release_nested.set()
        for thread in threads:
            thread.join(timeout=5)
        server._sessions.pop("sid_a", None)
        while not isolated_queue.empty():
            isolated_queue.get_nowait()
        for event in events:
            process_registry._completion_consumed.discard(event["session_id"])
            process_registry._poll_observed.discard(event["session_id"])


def test_run_prompt_submit_delivers_completion_owned_through_compression_lineage(
    monkeypatch, tmp_path
):
    import queue as _queue_mod

    from tools.process_registry import process_registry

    class _CompressionDB:
        def resolve_resume_session_id(self, key):
            return "new-child-key" if key == "old-parent-key" else key

    _configure_immediate_prompt_run(monkeypatch, tmp_path)
    monkeypatch.setattr(server, "_get_db", lambda: _CompressionDB())
    ownership_checks = []
    original_owns_event = server._session_owns_notification_event

    def _record_ownership_check(sid, checked_session, checked_event):
        ownership_checks.append(checked_event["session_id"])
        return original_owns_event(sid, checked_session, checked_event)

    monkeypatch.setattr(
        server, "_session_owns_notification_event", _record_ownership_check
    )
    turns = []
    session = _session(
        session_key="new-child-key",
        agent=_RecordingAgent(turns),
        running=True,
    )
    event = {
        "type": "completion",
        "session_id": "proc_precompression",
        "session_key": "old-parent-key",
        "command": "safe-test-command",
        "exit_code": 0,
        "output": "owned",
    }
    isolated_queue: _queue_mod.Queue = _queue_mod.Queue()
    isolated_queue.put(event)
    monkeypatch.setattr(process_registry, "completion_queue", isolated_queue)
    server._sessions["sid_b"] = session

    try:
        server._run_prompt_submit("rid-b", "sid_b", session, "session-b-turn")

        assert turns[0] == "session-b-turn"
        assert len(turns) == 2
        assert "proc_precompression" in turns[1]
        assert ownership_checks == ["proc_precompression"]
        assert isolated_queue.empty()
    finally:
        server._sessions.pop("sid_b", None)
        process_registry._completion_consumed.discard(event["session_id"])


def test_run_prompt_submit_prefers_origin_ui_session_id(monkeypatch, tmp_path):
    import queue as _queue_mod

    from tools.process_registry import process_registry

    _configure_immediate_prompt_run(monkeypatch, tmp_path)
    ownership_checks = []
    original_owns_event = server._session_owns_notification_event

    def _record_ownership_check(sid, checked_session, checked_event):
        ownership_checks.append(checked_event["session_id"])
        return original_owns_event(sid, checked_session, checked_event)

    monkeypatch.setattr(
        server, "_session_owns_notification_event", _record_ownership_check
    )
    turns = []
    session = _session(
        session_key="current-key",
        agent=_RecordingAgent(turns),
        running=True,
    )
    event = {
        "type": "completion",
        "session_id": "proc_origin_owned",
        "session_key": "stale-durable-key",
        "origin_ui_session_id": "sid_b",
        "command": "safe-test-command",
        "exit_code": 0,
        "output": "owned",
    }
    isolated_queue: _queue_mod.Queue = _queue_mod.Queue()
    isolated_queue.put(event)
    monkeypatch.setattr(process_registry, "completion_queue", isolated_queue)
    server._sessions["sid_b"] = session

    try:
        server._run_prompt_submit("rid-b", "sid_b", session, "session-b-turn")

        assert turns[0] == "session-b-turn"
        assert len(turns) == 2
        assert "proc_origin_owned" in turns[1]
        assert ownership_checks == ["proc_origin_owned"]
        assert isolated_queue.empty()
    finally:
        server._sessions.pop("sid_b", None)
        process_registry._completion_consumed.discard(event["session_id"])


    """session.create must NOT eagerly write a DB row.

    Every TUI/desktop launch opens a session here just to paint the composer;
    eagerly creating a row left an empty "Untitled" session behind for every
    launch the user never typed into. The row is created lazily on first prompt.
    """
    created = []

    class _FakeDB:
        def create_session(self, *args, **kwargs):
            created.append((args, kwargs))

    monkeypatch.setattr(server, "_get_db", lambda: _FakeDB())
    monkeypatch.setattr(server, "_start_agent_build", lambda *a, **k: None)
    monkeypatch.setattr(
        server.threading,
        "Timer",
        lambda *a, **k: types.SimpleNamespace(daemon=False, start=lambda: None),
    )

    resp = server.handle_request(
        {"id": "1", "method": "session.create", "params": {"cols": 80}}
    )
    sid = resp["result"]["session_id"]
    try:
        assert resp["result"]["stored_session_id"]
        assert created == [], "session.create should not persist an empty DB row"
    finally:
        server._sessions.pop(sid, None)


def test_ensure_session_db_row_persists_explicit_cwd(monkeypatch, tmp_path):
    """An explicitly chosen workspace is persisted as the session cwd."""
    created = []

    class _FakeDB:
        def create_session(self, key, source=None, model=None, model_config=None, parent_session_id=None, cwd=None, profile_name=None, user_id=None):
            created.append(
                {"key": key, "source": source, "model": model, "model_config": model_config, "cwd": cwd}
            )

    monkeypatch.setattr(server, "_get_db", lambda: _FakeDB())
    monkeypatch.setattr(server, "_resolve_model", lambda: "test-model")
    monkeypatch.delenv("HERMES_DESKTOP", raising=False)
    monkeypatch.delenv("HERMES_DESKTOP_TERMINAL", raising=False)

    server._ensure_session_db_row({"session_key": "k1", "cwd": str(tmp_path), "explicit_cwd": True})

    assert created == [
        {"key": "k1", "source": "tui", "model": "test-model", "model_config": None, "cwd": str(tmp_path)}
    ]


def test_ensure_session_db_row_persists_session_source(monkeypatch):
    created = []

    class _FakeDB:
        def create_session(self, key, source=None, model=None, model_config=None, parent_session_id=None, cwd=None, profile_name=None, user_id=None):
            created.append(
                {"key": key, "source": source, "model": model, "model_config": model_config, "cwd": cwd}
            )

    monkeypatch.setattr(server, "_get_db", lambda: _FakeDB())
    monkeypatch.setattr(server, "_resolve_model", lambda: "test-model")

    server._ensure_session_db_row({"session_key": "k1", "source": "tool"})

    assert created == [
        {"key": "k1", "source": "tool", "model": "test-model", "model_config": None, "cwd": None}
    ]


def test_ensure_session_db_row_records_a_terminal_workspace(monkeypatch, tmp_path):
    """A terminal session's directory IS its workspace, so the row records it.

    The user cd'd there before running hermes. Leaving it null stranded the row
    with no cwd and no git_repo_root, so the sidebar could never place the
    session under its project.
    """
    created = []

    class _FakeDB:
        def create_session(self, key, source=None, model=None, model_config=None, parent_session_id=None, cwd=None, profile_name=None, user_id=None):
            created.append(
                {"key": key, "source": source, "model": model, "model_config": model_config, "cwd": cwd}
            )

    monkeypatch.setattr(server, "_get_db", lambda: _FakeDB())
    monkeypatch.setattr(server, "_resolve_model", lambda: "test-model")
    monkeypatch.delenv("HERMES_DESKTOP", raising=False)
    monkeypatch.delenv("HERMES_DESKTOP_TERMINAL", raising=False)

    server._ensure_session_db_row({"session_key": "k1", "cwd": str(tmp_path)})

    assert created == [
        {"key": "k1", "source": "tui", "model": "test-model", "model_config": None, "cwd": str(tmp_path)}
    ]


def test_ensure_session_db_row_defaults_desktop_to_no_workspace(monkeypatch, tmp_path):
    """The desktop launches from wherever the bundle was opened, so an unpicked
    cwd is an artifact — those chats stay null and group under "No workspace"."""
    created = []

    class _FakeDB:
        def create_session(self, key, source=None, model=None, model_config=None, parent_session_id=None, cwd=None, profile_name=None, user_id=None):
            created.append(
                {"key": key, "source": source, "model": model, "model_config": model_config, "cwd": cwd}
            )

    monkeypatch.setattr(server, "_get_db", lambda: _FakeDB())
    monkeypatch.setattr(server, "_resolve_model", lambda: "test-model")

    server._ensure_session_db_row({"session_key": "k1", "source": "desktop", "cwd": str(tmp_path)})

    assert created == [
        {"key": "k1", "source": "desktop", "model": "test-model", "model_config": None, "cwd": None}
    ]


def test_ensure_session_db_row_persists_session_model_override(monkeypatch):
    """The session's composer pick (model + effort + fast) must own the DB row.

    Regression for the "switched to gpt-5.5, reconnect snapped back to opus"
    bug: the row was created with the global default and won the INSERT-OR-IGNORE
    race, so resume rebuilt from the global model and silently reverted the
    chat. The override model + a model_config carrying provider/reasoning/
    service_tier must be persisted so session.resume restores all three.
    """
    created = []

    class _FakeDB:
        def create_session(self, key, source=None, model=None, model_config=None, parent_session_id=None, cwd=None, profile_name=None, user_id=None):
            created.append(
                {"key": key, "model": model, "model_config": model_config, "cwd": cwd}
            )

    monkeypatch.setattr(server, "_get_db", lambda: _FakeDB())
    monkeypatch.setattr(server, "_resolve_model", lambda: "global/default")

    server._ensure_session_db_row(
        {
            "session_key": "k1",
            "model_override": {"model": "openai/gpt-5.5", "provider": "openrouter"},
            "create_reasoning_override": {"effort": "high"},
            "create_service_tier_override": "priority",
        }
    )

    assert len(created) == 1
    row = created[0]
    assert row["model"] == "openai/gpt-5.5"
    assert row["model_config"]["model"] == "openai/gpt-5.5"
    assert row["model_config"]["provider"] == "openrouter"
    assert row["model_config"]["reasoning_config"] == {"effort": "high"}
    assert row["model_config"]["service_tier"] == "priority"


def test_ensure_session_db_row_no_override_uses_global(monkeypatch):
    """A chat that made no explicit pick falls back to the global model and
    writes no model_config (so it tracks the profile default)."""
    created = []

    class _FakeDB:
        def create_session(self, key, source=None, model=None, model_config=None, parent_session_id=None, cwd=None, profile_name=None, user_id=None):
            created.append({"model": model, "model_config": model_config})

    monkeypatch.setattr(server, "_get_db", lambda: _FakeDB())
    monkeypatch.setattr(server, "_resolve_model", lambda: "global/default")

    server._ensure_session_db_row({"session_key": "k1", "model_override": None})

    assert created == [{"model": "global/default", "model_config": None}]


def test_ensure_session_db_row_stamps_profile_name(monkeypatch, tmp_path):
    """A profile session's row carries its owning profile_name, so unified
    multi-profile aggregation never has to guess from which state.db file the
    row happened to be read (the cross-profile session-jump bug)."""
    profile_home = tmp_path / "profiles" / "mlperf"
    profile_home.mkdir(parents=True)
    created = []

    class _ProfileDB:
        def __init__(self, db_path=None):
            created.append({"db_path": db_path})

        def create_session(self, key, **kwargs):
            created[-1].update({"key": key, "profile_name": kwargs.get("profile_name")})

        def close(self):
            pass

    monkeypatch.setattr("hermes_state_registry.acquire", _ProfileDB)
    monkeypatch.setattr(server, "_resolve_model", lambda: "test-model")

    server._ensure_session_db_row(
        {"session_key": "k1", "profile_home": str(profile_home)}
    )

    assert created and created[0]["key"] == "k1"
    assert created[0]["profile_name"] == "mlperf"
    assert created[0]["db_path"] == profile_home / "state.db"


def test_ensure_session_db_row_stamps_launch_profile_name(monkeypatch):
    """A launch-profile session row is stamped with the ACTUAL profile name,
    never NULL. NULL-as-launch-profile rows vanish from the desktop sidebar
    (profile-keyed matching) and break @session:<profile>/<id> deep links, and
    the #94724 one-shot backfill cannot keep repairing rows minted after it
    ran (#99222)."""
    created = []

    class _FakeDB:
        def create_session(self, key, **kwargs):
            created.append({"key": key, "profile_name": kwargs.get("profile_name")})

    monkeypatch.setattr(server, "_get_db", lambda: _FakeDB())
    monkeypatch.setattr(server, "_resolve_model", lambda: "test-model")
    monkeypatch.setattr(server, "_current_profile_name", lambda: "default")

    server._ensure_session_db_row({"session_key": "k1"})

    assert created and created[0]["key"] == "k1"
    assert created[0]["profile_name"] == "default"



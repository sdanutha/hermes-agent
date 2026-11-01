









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


def test_file_attach_workspace_storage_falls_back_for_non_local_workspace(monkeypatch, tmp_path):
    """Workspace storage only applies to a workspace on THIS host.

    A session bound to a remote cwd (ssh profile) can't be written from the
    gateway; those attachments keep the bind-mounted ``<profile home>/attachments``
    so container/remote backends still receive them (#76577).
    """
    home = tmp_path / "home"
    home.mkdir()
    (home / "config.yaml").write_text("attachments:\n  storage: workspace\n", encoding="utf-8")
    remote_cwd = str(tmp_path / "remote-host" / "agent-a")  # does not exist here
    fake_cli = types.ModuleType("cli")
    fake_cli._detect_file_drop = lambda raw: None
    fake_cli._split_path_input = lambda raw: (raw, "")
    fake_cli._resolve_attachment_path = lambda raw: None
    monkeypatch.setitem(sys.modules, "cli", fake_cli)

    server._sessions["sid"] = _session(cwd=remote_cwd, profile_home=str(home))

    try:
        resp = server.handle_request(
            {
                "id": "1",
                "method": "file.attach",
                "params": {
                    "session_id": "sid",
                    "path": "/Users/alice/Downloads/report.txt",
                    "name": "report.txt",
                    "data_url": "data:text/plain;base64,aGVsbG8=",
                },
            }
        )

        stored = home / "attachments" / "report.txt"
        assert resp["result"]["attached"] is True
        assert resp["result"]["path"] == str(stored)
        assert stored.read_text(encoding="utf-8") == "hello"
    finally:
        server._sessions.pop("sid", None)


def test_commands_catalog_surfaces_quick_commands(monkeypatch):
    monkeypatch.setattr(
        server,
        "_load_cfg",
        lambda: {
            "quick_commands": {
                "build": {"type": "exec", "command": "npm run build"},
                "git": {"type": "alias", "target": "/shell git"},
                "notes": {
                    "type": "exec",
                    "command": "cat NOTES.md",
                    "description": "Open design notes",
                },
            }
        },
    )

    resp = server.handle_request(
        {"id": "1", "method": "commands.catalog", "params": {}}
    )

    pairs = dict(resp["result"]["pairs"])
    assert "npm run build" in pairs["/build"]
    assert pairs["/git"].startswith("alias →")
    assert pairs["/notes"] == "Open design notes"

    user_cat = next(
        c for c in resp["result"]["categories"] if c["name"] == "User commands"
    )
    user_pairs = dict(user_cat["pairs"])
    assert set(user_pairs) == {"/build", "/git", "/notes"}

    assert resp["result"]["canon"]["/build"] == "/build"
    assert resp["result"]["canon"]["/notes"] == "/notes"


def test_commands_catalog_ranks_skill_commands_by_recorded_usage(monkeypatch):
    """Skill entries carry the usage + origin the `/` menu ranks on.

    Without it the menu is alphabetical, so a bundled skill the user has never
    opened outranks the one they invoke daily.
    """
    monkeypatch.setattr(
        "hermes_runtime.command_discovery._skill_usage_lookup",
        lambda: (
            lambda name: {"research": 60, "work": 172}.get(name, 0),
            lambda name: "bundled" if name == "research-paper-writing" else "local",
        ),
    )
    monkeypatch.setattr(
        "agent.skill_commands.scan_skill_commands",
        lambda: {
            "/research": {"name": "research", "description": "Look it up"},
            "/research-paper-writing": {
                "name": "research-paper-writing",
                "description": "Write a paper",
            },
            "/work": {"name": "work", "description": "Fresh worktree"},
        },
    )

    resp = server.handle_request(
        {"id": "1", "method": "commands.catalog", "params": {}}
    )

    skills = resp["result"]["skills"]
    assert skills["/work"] == {"usage": 172, "origin": "local"}
    assert skills["/research"] == {"usage": 60, "origin": "local"}
    assert skills["/research-paper-writing"] == {"usage": 0, "origin": "bundled"}

    # Every advertised skill command is rankable — a missing entry silently
    # sorts that skill to the bottom of the menu.
    advertised = {name for name, _ in resp["result"]["pairs"]}
    assert set(skills) <= advertised
    assert resp["result"]["skill_count"] == len(skills)


def test_commands_catalog_survives_an_unreadable_usage_sidecar(monkeypatch):
    """A broken/absent .usage.json degrades to no ranking, never a broken menu."""
    monkeypatch.setattr(
        "tools.skill_usage.load_usage",
        lambda: (_ for _ in ()).throw(OSError("sidecar is gone")),
    )

    resp = server.handle_request(
        {"id": "1", "method": "commands.catalog", "params": {}}
    )

    assert "error" not in resp
    assert all(
        entry == {"usage": 0, "origin": "local"}
        for entry in resp["result"]["skills"].values()
    )


def test_commands_catalog_has_no_duplicate_or_alias_colliding_names():
    """No command may be advertised twice, and no advertised command may
    shadow an alias of a different command (e.g. the historical /compact
    collision where the registry aliased compact -> compress while the TUI
    also registered its own /compact display toggle; see #57133)."""
    resp = server.handle_request(
        {"id": "1", "method": "commands.catalog", "params": {}}
    )

    names = [name for name, _ in resp["result"]["pairs"]]
    dupes = {n for n in names if names.count(n) > 1}
    assert not dupes, f"duplicate commands advertised in catalog: {sorted(dupes)}"

    canon = resp["result"]["canon"]
    colliding = {
        name
        for name in names
        if canon.get(name.lower(), name) != name
    }
    assert not colliding, (
        f"catalog commands shadow aliases of other commands: {sorted(colliding)}"
    )


def test_commands_catalog_filters_gateway_only_commands_and_keeps_status_visible():
    resp = server.handle_request(
        {"id": "1", "method": "commands.catalog", "params": {}}
    )

    pairs = dict(resp["result"]["pairs"])
    canon = resp["result"]["canon"]

    assert "/status" in pairs
    assert canon["/status"] == "/status"
    assert "/approvals" in pairs

    assert "/topic" not in pairs
    assert "/approve" not in pairs
    assert "/deny" not in pairs
    assert "/sethome" not in pairs

    assert "/update" in pairs
    assert canon["/update"] == "/update"

    assert "/topic" not in canon
    assert "/approve" not in canon
    assert "/deny" not in canon
    assert "/set-home" not in canon


def test_commands_catalog_includes_desktop_meta_without_skills():
    resp = server.handle_request(
        {"id": "1", "method": "commands.catalog", "params": {}}
    )

    commands = resp["result"]["commands"]
    assert commands["/compact"]["argument_mode"] == commands["/compress"]["argument_mode"]

    for skill in resp["result"]["skills"]:
        assert skill not in commands


def test_commands_catalog_includes_plugin_commands(monkeypatch):
    monkeypatch.setattr(
        "hermes_cli.plugins.get_plugin_commands",
        lambda: {
            "lcm": {
                "description": "Latent consistency",
                "args_hint": "<prompt>",
                "argument_mode": "text",
            }
        },
    )

    resp = server.handle_request(
        {"id": "1", "method": "commands.catalog", "params": {}}
    )

    assert resp["result"]["commands"]["/lcm"] == {
        "argument_mode": "text",
        "desktop": None,
    }
    pairs = dict(resp["result"]["pairs"])
    assert "/lcm" in pairs
    plugin_cat = next(
        c for c in resp["result"]["categories"] if c["name"] == "Plugin commands"
    )
    assert "/lcm" in dict(plugin_cat["pairs"])


def test_plugin_slash_command_runs_under_the_session_env(monkeypatch):
    # TUI/Desktop sibling of #108698: command.dispatch and slash.exec ran plugin handlers on the RPC
    # thread with no HERMES_SESSION_* binding, so a handler reading get_session_env() saw "" (or the
    # launch process's inherited values) instead of the session it was invoked from.
    from gateway.session_context import get_session_env

    seen = {}

    def handler(arg):
        seen["key"] = get_session_env("HERMES_SESSION_KEY")
        return f"ok:{arg}"

    monkeypatch.setattr("hermes_cli.plugins.get_plugin_command_handler",
                        lambda name: handler if name == "whoami" else None)
    monkeypatch.setattr(server, "_sessions", {"sid-p": {"session_key": "agent:tui:key-p", "cwd": ""}})

    res = server._methods["command.dispatch"]("d", {"name": "whoami", "arg": "x", "session_id": "sid-p"})

    assert res["result"] == {"type": "plugin", "output": "ok:x"}
    assert seen["key"] == "agent:tui:key-p"
    # Nothing leaks past the RPC.
    assert get_session_env("HERMES_SESSION_KEY") in ("", None)


def test_session_status_reads_live_gateway_agent(monkeypatch):
    agent = types.SimpleNamespace(
        model="live-model",
        provider="live-provider",
        session_total_tokens=1234,
    )
    server._sessions["sid"] = _session(agent=agent, running=True)

    class _DB:
        def get_session(self, key):
            assert key == "session-key"
            return {
                "title": "Live TUI",
                "started_at": 1_700_000_000,
                "updated_at": 1_700_000_060,
            }

    monkeypatch.setattr(server, "_get_db", lambda: _DB())
    try:
        resp = server.handle_request(
            {"id": "1", "method": "session.status", "params": {"session_id": "sid"}}
        )
    finally:
        server._sessions.pop("sid", None)

    out = resp["result"]["output"]
    for value in ("session-key", "Live TUI", "live-model", "live-provider", "1,234"):
        assert value in out


def test_session_status_reads_live_compute_host_metadata(monkeypatch):
    agent = types.SimpleNamespace(
        model="stale-gateway-model",
        provider="stale-gateway-provider",
    )
    server._sessions["sid"] = _session(
        agent=agent,
        _compute_host_active=True,
        _metadata_mirror={
            "model": "live-host-model",
            "provider": "live-host-provider",
        },
    )
    monkeypatch.setattr(server, "_get_db", lambda: None)

    try:
        resp = server.handle_request(
            {"id": "1", "method": "session.status", "params": {"session_id": "sid"}}
        )
    finally:
        server._sessions.pop("sid", None)

    assert "Model: live-host-model (live-host-provider)" in resp["result"]["output"]


def test_session_status_falls_back_to_agent_before_first_host_frame(monkeypatch):
    agent = types.SimpleNamespace(model="live-model", provider="live-provider")
    server._sessions["sid"] = _session(agent=agent, _compute_host_active=True)
    monkeypatch.setattr(server, "_get_db", lambda: None)

    try:
        resp = server.handle_request(
            {"id": "1", "method": "session.status", "params": {"session_id": "sid"}}
        )
    finally:
        server._sessions.pop("sid", None)

    assert "Model: live-model (live-provider)" in resp["result"]["output"]


def test_skills_reload_runs_in_gateway_process(monkeypatch):
    from agent import skill_commands

    called = {}
    monkeypatch.setattr(
        skill_commands,
        "reload_skills",
        lambda: called.setdefault(
            "result",
            {
                "added": [{"name": "new-skill", "description": "demo"}],
                "removed": [],
                "total": 42,
            },
        ),
    )

    resp = server.handle_request({"id": "1", "method": "skills.reload", "params": {}})

    assert "new-skill" in resp["result"]["output"]


def test_snapshot_restore_is_blocked_from_tui_worker():
    server._sessions["sid"] = _session()
    try:
        worker_resp = server.handle_request(
            {
                "id": "1",
                "method": "slash.exec",
                "params": {"command": "snapshot restore latest", "session_id": "sid"},
            }
        )
        dispatch_resp = server.handle_request(
            {
                "id": "2",
                "method": "command.dispatch",
                "params": {
                    "arg": "restore latest",
                    "name": "snapshot",
                    "session_id": "sid",
                },
            }
        )
    finally:
        server._sessions.pop("sid", None)

    assert worker_resp["error"]["code"] == 4018
    assert (
        "snapshot restore mutates live config/state" in worker_resp["error"]["message"]
    )
    assert dispatch_resp["result"]["type"] == "exec"
    assert (
        "/snapshot restore is blocked in the TUI" in dispatch_resp["result"]["output"]
    )


def test_command_dispatch_exec_nonzero_surfaces_error(monkeypatch):
    monkeypatch.setattr(
        server,
        "_load_cfg",
        lambda: {"quick_commands": {"boom": {"type": "exec", "command": "boom"}}},
    )
    monkeypatch.setattr(
        server.subprocess,
        "run",
        lambda *args, **kwargs: types.SimpleNamespace(
            returncode=1, stdout="", stderr="failed"
        ),
    )

    resp = server.handle_request(
        {"id": "1", "method": "command.dispatch", "params": {"name": "boom"}}
    )

    assert "error" in resp
    assert "failed" in resp["error"]["message"]


def test_input_detect_drop_path_with_spaces(tmp_path):
    """input.detect_drop correctly handles image paths containing spaces."""
    # Create a minimal PNG file with a space in its name
    img = tmp_path / "screenshot with spaces.png"
    img.write_bytes(b"\x89PNG\r\n\x1a\n")  # valid PNG header

    server._sessions["sid"] = _session()

    resp = server.handle_request(
        {
            "id": "2",
            "method": "input.detect_drop",
            "params": {"session_id": "sid", "text": str(img)},
        }
    )

    assert resp["result"]["matched"] is True
    assert resp["result"]["is_image"] is True
    assert resp["result"]["path"] == str(img)
    assert resp["result"]["text"] == f"[User attached image: {img.name}]"
    # Verify attachment was recorded in the session
    assert len(server._sessions["sid"]["attached_images"]) == 1
    assert server._sessions["sid"]["attached_images"][0] == str(img)


def test_input_detect_drop_path_with_spaces_and_remainder(tmp_path):
    """input.detect_drop splits remainder when path contains spaces."""
    img = tmp_path / "photo with space.jpg"
    img.write_bytes(b"\xff\xd8\xff" + b"fakejpeg")  # minimal-ish JPEG header

    server._sessions["sid"] = _session()

    user_input = f"{img} describe this image"
    resp = server.handle_request(
        {
            "id": "3",
            "method": "input.detect_drop",
            "params": {"session_id": "sid", "text": user_input},
        }
    )

    assert resp["result"]["matched"] is True
    assert resp["result"]["is_image"] is True
    assert resp["result"]["path"] == str(img)
    # Remainder becomes the text sent to the model
    assert resp["result"]["text"] == "describe this image"
    assert server._sessions["sid"]["attached_images"][0] == str(img)


def test_rollback_restore_resolves_number_and_file_path():
    calls = {}

    class _Mgr:
        enabled = True
        unsupported_backend_reason = staticmethod(lambda task_id="default": None)

        def list_checkpoints(self, cwd):
            return [{"hash": "aaa111"}, {"hash": "bbb222"}]

        def restore(self, cwd, target, file_path=None, safe=False):
            calls["args"] = (cwd, target, file_path)
            return {"success": True, "message": "done"}

    server._sessions["sid"] = _session(
        agent=types.SimpleNamespace(_checkpoint_mgr=_Mgr()), history=[]
    )
    resp = server.handle_request(
        {
            "id": "1",
            "method": "rollback.restore",
            "params": {"session_id": "sid", "hash": "2", "file_path": "src/app.tsx"},
        }
    )

    assert resp["result"]["success"] is True
    assert calls["args"][1] == "bbb222"
    assert calls["args"][2] == "src/app.tsx"


def test_rollback_list_carries_checkpoint_reason_as_message():
    reason = "before write_file: a.py [nested git repos not captured: tool]"

    class _Mgr:
        enabled = True

        def list_checkpoints(self, cwd):
            return [{"hash": "aaa111", "short_hash": "aaa", "timestamp": "2026-09-29T10:00:00+00:00",
                     "reason": reason, "files_changed": 1}]

    server._sessions["sid"] = _session(
        agent=types.SimpleNamespace(_checkpoint_mgr=_Mgr()), history=[]
    )
    resp = server.handle_request(
        {"id": "1", "method": "rollback.list", "params": {"session_id": "sid"}}
    )

    row = resp["result"]["checkpoints"][0]
    assert row["hash"] == "aaa111"
    assert row["message"] == reason


def test_rollback_restore_truncates_from_real_user_turn_not_marker(monkeypatch):
    """rollback.restore must truncate from the last *real* user turn,
    not a display_kind timeline marker (same bug class as /undo).
    """

    class _Mgr:
        enabled = True
        unsupported_backend_reason = staticmethod(lambda task_id="default": None)

        def list_checkpoints(self, cwd):
            return [{"hash": "abc123"}]

        def restore(self, cwd, target, file_path=None, safe=False):
            return {"success": True, "message": "restored"}

    history = [
        {"role": "user", "content": "first question"},
        {"role": "assistant", "content": "first answer"},
        {"role": "user", "content": "second question"},
        {"role": "assistant", "content": "second answer"},
        {
            "role": "user",
            "content": "background agent finished",
            "display_kind": "async_delegation_complete",
        },
    ]
    server._sessions["sid"] = _session(
        agent=types.SimpleNamespace(_checkpoint_mgr=_Mgr()),
        history=list(history),
        session_key="",
    )
    try:
        resp = server.handle_request(
            {
                "id": "1",
                "method": "rollback.restore",
                "params": {"session_id": "sid", "hash": "abc123"},
            }
        )

        assert resp["result"]["success"] is True
        assert resp["result"]["history_removed"] == 3  # q2 + a2 + marker
        # Only first exchange remains
        remaining = server._sessions["sid"]["history"]
        assert [m["content"] for m in remaining] == ["first question", "first answer"]
    finally:
        server._sessions.pop("sid", None)


def test_rollback_restore_skips_legacy_compaction_handoff(monkeypatch):
    """rollback.restore must not truncate from a legacy standalone compaction
    handoff — a durable role=user row persisted pre-#80622 with NO
    display_kind. Same bug class as the display_kind marker above, caught
    only by the is_user_originated_turn predicate.
    """
    from agent.context_compressor import (
        COMPRESSED_SUMMARY_METADATA_KEY,
        HISTORICAL_TASK_HEADING,
        SUMMARY_PREFIX,
        _SUMMARY_END_MARKER,
    )

    class _Mgr:
        enabled = True
        unsupported_backend_reason = staticmethod(lambda task_id="default": None)

        def list_checkpoints(self, cwd):
            return [{"hash": "abc123"}]

        def restore(self, cwd, target, file_path=None, safe=False):
            return {"success": True, "message": "restored"}

    handoff = {
        "role": "user",
        "content": (
            f"{SUMMARY_PREFIX}\n{HISTORICAL_TASK_HEADING}\n"
            f"User asked: 'old task'\n\n{_SUMMARY_END_MARKER}"
        ),
        COMPRESSED_SUMMARY_METADATA_KEY: True,
        # NOTE: no display_kind — the legacy-persistence shape (#80622).
    }
    history = [
        {"role": "user", "content": "first question"},
        {"role": "assistant", "content": "first answer"},
        {"role": "user", "content": "second question"},
        {"role": "assistant", "content": "second answer"},
        handoff,
    ]
    server._sessions["sid"] = _session(
        agent=types.SimpleNamespace(_checkpoint_mgr=_Mgr()),
        history=list(history),
        session_key="",
    )
    try:
        resp = server.handle_request(
            {
                "id": "1",
                "method": "rollback.restore",
                "params": {"session_id": "sid", "hash": "abc123"},
            }
        )

        assert resp["result"]["success"] is True
        # Truncation lands on "second question", not the handoff row.
        assert resp["result"]["history_removed"] == 3  # q2 + a2 + handoff
        remaining = server._sessions["sid"]["history"]
        assert [m["content"] for m in remaining] == ["first question", "first answer"]
    finally:
        server._sessions.pop("sid", None)


def test_rollback_restore_preserves_composite_carrier_scaffold(monkeypatch, tmp_path):
    """A checkpoint restore drops the live ask but keeps compacted context."""
    from agent.context_compressor import (
        HISTORICAL_TASK_HEADING,
        SUMMARY_PREFIX,
        _SUMMARY_END_MARKER,
    )
    from hermes_state import SessionDB

    class _Mgr:
        enabled = True
        unsupported_backend_reason = staticmethod(lambda task_id="default": None)

        def list_checkpoints(self, cwd):
            return [{"hash": "abc123"}]

        def restore(self, cwd, target, file_path=None, safe=False):
            return {"success": True, "message": "restored"}

    carrier = {
        "role": "user",
        "content": (
            f"{SUMMARY_PREFIX}\n{HISTORICAL_TASK_HEADING}\nold task\n\n"
            f"{_SUMMARY_END_MARKER}\n\nREAL ASK"
        ),
    }
    db = SessionDB(db_path=tmp_path / "state.db")
    db.create_session("rollback-carrier", source="tui")
    db.append_message("rollback-carrier", "user", carrier["content"])
    db.append_message("rollback-carrier", "assistant", "answer")
    durable = db.get_messages_as_conversation("rollback-carrier")
    agent = types.SimpleNamespace(
        _checkpoint_mgr=_Mgr(),
        _session_messages=list(durable),
        _last_flushed_db_idx=len(durable),
        _db_flush_scan_prefix=list(durable),
    )
    server._sessions["sid"] = _session(
        agent=agent,
        history=list(durable),
        session_key="rollback-carrier",
    )
    monkeypatch.setattr(server, "_get_db", lambda: db)
    try:
        resp = server.handle_request(
            {
                "id": "1",
                "method": "rollback.restore",
                "params": {"session_id": "sid", "hash": "abc123"},
            }
        )

        assert "result" in resp, resp
        assert resp["result"]["success"] is True
        assert resp["result"]["history_removed"] == 2
        remaining = server._sessions["sid"]["history"]
        assert len(remaining) == 1
        assert remaining[0]["display_kind"] == "hidden"
        assert SUMMARY_PREFIX in remaining[0]["content"]
        assert "REAL ASK" not in remaining[0]["content"]
        cold = db.get_messages_as_conversation(
            "rollback-carrier", include_row_ids=True
        )
        assert len(cold) == 1
        assert cold[0]["content"] == remaining[0]["content"]
        assert cold[0]["display_kind"] == "hidden"
        assert cold[0]["_row_id"] == remaining[0]["_row_id"]
        assert agent._session_messages == remaining
        assert agent._last_flushed_db_idx == 1
        assert agent._db_flush_scan_prefix == remaining
        inactive = db.get_messages_as_conversation(
            "rollback-carrier", include_inactive=True
        )
        assert any("REAL ASK" in str(message.get("content")) for message in inactive)
        assert any(message.get("content") == "answer" for message in inactive)
    finally:
        server._sessions.pop("sid", None)
        db.close()


def test_session_steer_calls_agent_steer_when_agent_supports_it():
    """The TUI RPC method must call agent.steer(text) and return a
    queued status without touching interrupt state.
    """
    calls = {}

    class _Agent:
        def steer(self, text):
            calls["steer_text"] = text
            return True

        def interrupt(self, *args, **kwargs):
            calls["interrupt_called"] = True

    server._sessions["sid"] = _session(agent=_Agent(), running=True)
    try:
        resp = server.handle_request(
            {
                "id": "1",
                "method": "session.steer",
                "params": {"session_id": "sid", "text": "also check auth.log"},
            }
        )
    finally:
        server._sessions.pop("sid", None)

    assert "result" in resp, resp
    assert resp["result"]["status"] == "queued"
    assert resp["result"]["text"] == "also check auth.log"
    assert calls["steer_text"] == "also check auth.log"
    assert "interrupt_called" not in calls  # must NOT interrupt


def test_session_steer_on_idle_session_is_rejected_not_parked():
    """#64578: with no live turn a steer has no tool call to ride. Accepting it parked the text in
    the agent's pending-steer slot, where the next turn's pre-API drain spliced it after an OLD tool
    row. It must come back 'rejected' (clients then send it as a normal prompt) and never reach
    agent.steer()."""
    agent = _RecordingSteerAgent()
    server._sessions["sid"] = _session(agent=agent, running=False)
    try:
        resp = server.handle_request(
            {"id": "1", "method": "session.steer", "params": {"session_id": "sid", "text": "check the logs"}}
        )
    finally:
        server._sessions.pop("sid", None)

    assert resp["result"]["status"] == "rejected", resp
    assert agent.steered == []


def test_steer_slash_on_idle_session_sends_as_next_turn():
    """#64578: idle `/steer <text>` via command.dispatch must go out as a normal next-turn message
    with a notice saying so, not claim "Steer queued" while stashing the text on the agent."""
    agent = _RecordingSteerAgent()
    server._sessions["sid"] = _session(agent=agent, running=False)
    try:
        res = server._methods["command.dispatch"](
            "1", {"name": "steer", "arg": "check the logs", "session_id": "sid"})
    finally:
        server._sessions.pop("sid", None)

    result = res["result"]
    assert result["type"] == "send"
    assert result["message"] == "check the logs"
    assert result.get("notice")
    assert agent.steered == []


def test_steer_slash_during_live_turn_still_steers():
    agent = _RecordingSteerAgent()
    server._sessions["sid"] = _session(agent=agent, running=True)
    try:
        res = server._methods["command.dispatch"](
            "1", {"name": "steer", "arg": "check the logs", "session_id": "sid"})
    finally:
        server._sessions.pop("sid", None)

    assert res["result"]["type"] == "exec"
    assert agent.steered == ["check the logs"]


def test_session_steer_rejects_empty_text():
    server._sessions["sid"] = _session(
        agent=types.SimpleNamespace(steer=lambda t: True)
    )
    try:
        resp = server.handle_request(
            {
                "id": "1",
                "method": "session.steer",
                "params": {"session_id": "sid", "text": "   "},
            }
        )
    finally:
        server._sessions.pop("sid", None)

    assert "error" in resp, resp
    assert resp["error"]["code"] == 4002


def test_session_steer_errors_when_agent_has_no_steer_method():
    server._sessions["sid"] = _session(agent=types.SimpleNamespace())  # no steer()
    try:
        resp = server.handle_request(
            {
                "id": "1",
                "method": "session.steer",
                "params": {"session_id": "sid", "text": "hi"},
            }
        )
    finally:
        server._sessions.pop("sid", None)

    assert "error" in resp, resp
    assert resp["error"]["code"] == 4010


def test_session_redirect_calls_capable_core_agent(monkeypatch):
    calls = []
    agent = types.SimpleNamespace(
        _supports_active_turn_redirect=True,
        redirect=lambda text: calls.append(text) or True,
    )
    session = _session(agent=agent)
    session["inflight_turn"] = {"user": "original request", "assistant": "partial reply"}
    server._sessions["sid"] = session
    try:
        before = session.get("last_active")
        resp = server.handle_request(
            {
                "id": "1",
                "method": "session.redirect",
                "params": {"session_id": "sid", "text": "use Postgres"},
            }
        )
    finally:
        server._sessions.pop("sid", None)

    assert resp["result"] == {
        "status": "redirected",
        "text": "use Postgres",
    }
    assert calls == ["use Postgres"]
    # The correction is recorded alongside the prompt that started the turn,
    # never over it — resume must be able to rebuild both bubbles.
    assert session["inflight_turn"]["user"] == "original request"
    assert session["inflight_turn"]["corrections"] == ["use Postgres"]
    assert session.get("last_active") is not None
    assert before is None or session["last_active"] >= before


def test_session_redirect_rpc_drops_queued_duplicate_of_inflight_user():
    """#84417: Desktop ``session.redirect`` must purge stale self-duplicates.

    Production path: renderer steers via ``session.redirect`` (not
    ``prompt.submit``). A self-copy of the live original user text already in
    the server queue must not survive a successful redirect — otherwise
    post-turn ``_drain_queued_prompt`` restarts prompt P after Q is handled.
    Unrelated next-turn envelopes stay.
    """
    original = "deepseek released a new flash model — I changed all settings to flash"
    agent = types.SimpleNamespace(
        _supports_active_turn_redirect=True,
        redirect=lambda text: True,
    )
    session = _session(agent=agent, running=True)
    session["inflight_turn"] = {
        "user": original,
        "assistant": "partial",
        "streaming": True,
        "error": "",
    }
    session["queued_prompt"] = {"text": original, "transport": "ws-1"}
    session["queued_prompts"] = [
        {"text": original, "transport": "ws-1"},
        {"text": "unrelated later task", "transport": "ws-1"},
    ]
    server._sessions["sid"] = session
    try:
        resp = server.handle_request(
            {
                "id": "1",
                "method": "session.redirect",
                "params": {
                    "session_id": "sid",
                    "text": "what about the pricing instead?",
                },
            }
        )
    finally:
        server._sessions.pop("sid", None)

    assert resp["result"]["status"] == "redirected"
    assert session["inflight_turn"]["user"] == original
    assert session["inflight_turn"]["corrections"] == [
        "what about the pricing instead?"
    ]
    # Self-duplicates of the live original are gone; legitimate follow-up kept.
    assert session.get("queued_prompt") == {
        "text": "unrelated later task",
        "transport": "ws-1",
    }
    assert not session.get("queued_prompts")


def test_session_redirect_build_window_scrubs_stale_p_when_queuing_q():
    """#84417: build-window queue of Q must not leave P ahead of Q."""
    original = "live original P"
    session = _session(running=True)
    session["agent"] = None  # async agent build window
    session["inflight_turn"] = {
        "user": original,
        "assistant": "",
        "streaming": True,
        "error": "",
    }
    session["queued_prompt"] = {"text": original, "transport": "ws-1"}
    server._sessions["sid"] = session
    try:
        resp = server.handle_request(
            {
                "id": "1",
                "method": "session.redirect",
                "params": {"session_id": "sid", "text": "correction Q"},
            }
        )
    finally:
        server._sessions.pop("sid", None)

    assert resp["result"] == {"status": "queued", "text": "correction Q"}
    assert session["queued_prompt"]["text"] == "correction Q"
    assert not session.get("queued_prompts")


def test_session_redirect_records_correction_without_erasing_prompt():
    """A redirect must not overwrite the turn's original user text.

    The inflight snapshot is the only thing session.resume can replay, so
    overwriting ``user`` erased the prompt that started the turn and the
    client repainted the thread with the user's message missing.
    """
    session = {}
    server._start_inflight_turn(session, "remove the session counts")
    server._append_inflight_delta(session, "Moving.")
    server._record_inflight_correction(session, "hurry up")
    server._record_inflight_correction(session, "and the worktree ones")

    snapshot = server._inflight_snapshot(session)
    assert snapshot is not None

    assert snapshot["user"] == "remove the session counts"
    assert snapshot["corrections"] == ["hurry up", "and the worktree ones"]


def test_inflight_snapshot_carries_arrival_order_offsets():
    """Each correction records how much assistant text had already streamed.

    Resuming clients rebuild ARRIVAL order from these boundaries: the
    correction bubble lands after the output the user had already seen and
    before the output it redirected (#73793), instead of above the whole
    reply.
    """
    session = {}
    server._start_inflight_turn(session, "remove the session counts")
    server._append_inflight_delta(session, "Moving.")
    server._record_inflight_correction(session, "hurry up")
    server._append_inflight_delta(session, "Still.")
    server._record_inflight_correction(session, "and the worktree ones")
    server._append_inflight_delta(session, "Done soon.")

    snapshot = server._inflight_snapshot(session)
    assert snapshot is not None

    assert snapshot["corrections"] == ["hurry up", "and the worktree ones"]
    assert snapshot["correction_offsets"] == [len("Moving."), len("Moving.Still.")]


def test_turn_admission_carries_synthetic_display_metadata_into_inflight_snapshot(monkeypatch):
    """A reconnect must retain the typed synthetic user bubble (#112144)."""
    monkeypatch.setattr(server, "_ensure_active_session_slot", lambda *_args: None)
    agent = Mock()
    session = {"agent": agent, "attached_images": [], "history_lock": threading.RLock()}
    display_metadata = {"display_text": "Finished syncing the workspace"}

    assert server._admit_prompt_turn(
        "sid", session, "process completed", None, None, "process_complete", display_metadata,
    ) == ([], agent)

    snapshot = server._inflight_snapshot(session)

    assert snapshot is not None
    assert snapshot["display_kind"] == "process_complete"
    assert snapshot["display_metadata"] == {"display_text": "Finished syncing the workspace"}


def test_inflight_snapshot_omits_offsets_when_not_fully_recorded():
    """A pre-upgrade in-memory turn may carry corrections without offsets.

    The parallel list is only sent when every correction has one, so clients
    can trust the pairing and older snapshots degrade to the no-offset path.
    """
    session = {}
    server._start_inflight_turn(session, "prompt")
    turn = session["inflight_turn"]
    turn["corrections"] = ["legacy correction"]

    snapshot = server._inflight_snapshot(session)
    assert snapshot is not None

    assert snapshot["corrections"] == ["legacy correction"]
    assert "correction_offsets" not in snapshot


def test_inflight_snapshot_omits_corrections_when_none_recorded():
    session = {}
    server._start_inflight_turn(session, "just the prompt")

    snapshot = server._inflight_snapshot(session)
    assert snapshot is not None
    assert "corrections" not in snapshot


def test_new_turn_does_not_inherit_prior_turn_corrections():
    session = {}
    server._start_inflight_turn(session, "first prompt")
    server._record_inflight_correction(session, "first correction")
    server._start_inflight_turn(session, "second prompt")

    snapshot = server._inflight_snapshot(session)
    assert snapshot is not None

    assert snapshot["user"] == "second prompt"
    assert "corrections" not in snapshot


def test_session_redirect_queues_during_agent_build_window(monkeypatch):
    # A fresh turn flips running=True and builds the agent asynchronously, so
    # session["agent"] is briefly None. A correction landing here must queue
    # (lossless, reaches the model next turn), not hard-reject as unsupported.
    session = _session(running=True)
    session["agent"] = None
    server._sessions["sid"] = session
    try:
        resp = server.handle_request(
            {
                "id": "1",
                "method": "session.redirect",
                "params": {"session_id": "sid", "text": "wait, use SQLite"},
            }
        )
    finally:
        server._sessions.pop("sid", None)

    assert resp["result"] == {"status": "queued", "text": "wait, use SQLite"}
    assert session["queued_prompt"]["text"] == "wait, use SQLite"


def test_session_redirect_rejects_when_idle_without_agent(monkeypatch):
    # No live turn and no agent: nothing to redirect, and we must not queue a
    # phantom turn — keep the explicit unsupported rejection.
    session = _session(running=False)
    session["agent"] = None
    server._sessions["sid"] = session
    try:
        resp = server.handle_request(
            {
                "id": "1",
                "method": "session.redirect",
                "params": {"session_id": "sid", "text": "hi"},
            }
        )
    finally:
        server._sessions.pop("sid", None)

    assert resp["error"]["code"] == 4010
    assert session.get("queued_prompt") is None


def test_session_info_includes_session_title(monkeypatch):
    class _FakeDB:
        def get_session_title(self, key):
            assert key == "session-key"
            return "Dashboard title"

    monkeypatch.setattr(server, "_get_db", lambda: _FakeDB())

    info = server._session_info(
        types.SimpleNamespace(tools=[], model="test/model", provider="openai-codex"),
        {"session_key": "session-key", "history": []},
    )

    assert info["title"] == "Dashboard title"


def test_session_info_reports_pending_model_switch(monkeypatch):
    """A model queued mid-turn shows as the session's model in session.info, so
    the end-of-turn settle doesn't blip the UI back to the still-live old model
    before the switch applies at the next turn start."""
    agent = types.SimpleNamespace(tools=[], model="old/model", provider="openai")
    session = {
        "session_key": "",
        "history": [],
        "pending_model_switch": {
            "raw": "new/model --provider anthropic",
            "display_model": "new/model",
            "display_provider": "anthropic",
        },
    }

    info = server._session_info(agent, session)
    assert info["model"] == "new/model"
    assert info["provider"] == "anthropic"

    # With nothing queued the live agent model wins, as before.
    session.pop("pending_model_switch")
    assert server._session_info(agent, session)["model"] == "old/model"


def test_session_info_includes_turn_started_at():
    agent = types.SimpleNamespace(tools=[], model="", provider="")
    session = {
        "history": [],
        "inflight_turn": {"started_at": 1_700_000_123.5},
        "running": True,
    }

    assert server._session_info(agent, session)["turn_started_at"] == 1_700_000_123.5

    session["inflight_turn"] = None
    session["running"] = False
    assert server._session_info(agent, session)["turn_started_at"] is None


def test_session_undo_rejects_while_running():
    """Fix for TUI silent-drop #1: /undo must not mutate history
    while the agent is mid-turn — would either clobber the undo or
    cause prompt.submit to silently drop the agent's response."""
    server._sessions["sid"] = _session(
        running=True,
        history=[
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "hello"},
        ],
    )
    try:
        resp = server.handle_request(
            {"id": "1", "method": "session.undo", "params": {"session_id": "sid"}}
        )
        assert resp.get("error"), "session.undo should reject while running"
        assert resp["error"]["code"] == 4009
        # History must be unchanged
        assert len(server._sessions["sid"]["history"]) == 2
    finally:
        server._sessions.pop("sid", None)


def test_session_undo_allowed_when_idle():
    """Regression guard: when not running, /undo still works."""
    server._sessions["sid"] = _session(
        running=False,
        session_key="",
        history=[
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "hello"},
        ],
    )
    try:
        resp = server.handle_request(
            {"id": "1", "method": "session.undo", "params": {"session_id": "sid"}}
        )
        assert resp.get("result"), f"got error: {resp.get('error')}"
        assert resp["result"]["removed"] == 2
        assert server._sessions["sid"]["history"] == []
    finally:
        server._sessions.pop("sid", None)


def test_session_compress_rejects_while_running(monkeypatch):
    server._sessions["sid"] = _session(running=True)
    try:
        resp = server.handle_request(
            {"id": "1", "method": "session.compress", "params": {"session_id": "sid"}}
        )
        assert resp.get("error")
        assert resp["error"]["code"] == 4009
    finally:
        server._sessions.pop("sid", None)



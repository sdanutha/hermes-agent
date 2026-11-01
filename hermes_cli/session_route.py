"""Stored model-route restoration without an interactive model picker."""
from __future__ import annotations

def stored_session_route(session_meta, *, current_model, current_provider):
    """The route a resumed session should run on, or ``None`` when the stored one is absent or
    already current. Returns ``(model, provider, base_url, api_mode, provider_changed)``; the
    canonical row reader is ``SessionDB.session_gateway_runtime`` (``model_config.gateway_runtime``,
    else the TUI's top-level keys). Bare ``custom`` is healed because the CLI resolve path
    hard-fails on it (the TUI gateway keeps it when a base_url exists)."""
    stored_model = str((session_meta or {}).get("model") or "").strip()
    if not stored_model:
        return None
    from hermes_state import SessionDB as _SessionDB
    runtime = _SessionDB.session_gateway_runtime(session_meta)
    base_url = runtime.get("base_url") or None
    api_mode = runtime.get("api_mode") or None
    stored_provider = runtime.get("provider") or None
    provider = _heal_bare_custom_provider(stored_provider, base_url=base_url, model=stored_model)
    if provider != stored_provider:
        # A healed entry resolves its own current endpoint. A route that didn't heal is gone with its
        # endpoint and wire, which must not ride on the ambient provider.
        base_url = None
        if provider is None:
            api_mode = None
    provider_changed = bool(provider) and provider != current_provider
    if stored_model == current_model and not provider_changed:
        return None
    from hermes_cli.runtime_provider import is_foreign_provider_endpoint
    if is_foreign_provider_endpoint(provider, base_url):
        # The endpoint and its wire belong to the provider this chat left; resolve the stored one's own.
        base_url = api_mode = None
    # A row's api_mode/base_url were written for whichever model the session last ran. Providers that
    # pick the wire per model (OpenCode Zen/Go, Copilot, Nous) re-derive both from the stored model, or a
    # resumed opencode-go session keeps a MiniMax-era anthropic_messages route for a chat_completions
    # model (#96066) — the CLI/oneshot twin of hermes_runtime's _rederive_per_model_route.
    from hermes_cli.model_switch import model_derived_api_mode
    derived = model_derived_api_mode(provider or "", stored_model)
    if derived is not None:
        from hermes_cli.models import normalize_opencode_base_url
        api_mode = derived
        base_url = normalize_opencode_base_url(provider, api_mode, base_url) or None
    return stored_model, provider, base_url, api_mode, provider_changed

def _heal_bare_custom_provider(provider, *, base_url, model):
    """Bare ``custom`` is a billing class, not a routable identity: persisting/restoring it makes a
    later resume hard-fail once the config default leaves the custom endpoint. Recover the durable
    ``custom:<name>`` menu key from the endpoint, else drop the provider (None)."""
    if str(provider or "").strip().lower() != "custom":
        return provider
    try:
        # Heal bare "custom" persisted by older builds / gateway turns: it's the resolved billing class, not
        # a routable identity. (Stricter than the TUI gateway's recovery, which keeps bare "custom" when a
        # base_url exists — the CLI's resolve path would hard-fail on it, #14676.)
        from hermes_cli.runtime_provider import canonical_custom_identity
        return canonical_custom_identity(base_url=base_url or None, model=model or None) or None
    except Exception:
        return None

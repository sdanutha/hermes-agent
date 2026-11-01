"""Private ticket-authenticated session transport for the headless runtime.

Only the owning OS user can mint tickets through the private control socket.
The optional HTTP API adapter provides bearer-authenticated remote access.
"""
import asyncio
from dataclasses import dataclass
import socket
from typing import Any
import logging

logger = logging.getLogger(__name__)
_ACTIVE_RUNNER = None


@dataclass
class GatewayAPIHandle:
    api_origin: str
    server: Any
    task: asyncio.Task
    app: Any
    socket: socket.socket


async def start_gateway_api(runner, *, host="127.0.0.1", port=0):
    """Bind the private session listener; runtime bootstrap owns signals and shutdown."""
    global _ACTIVE_RUNNER
    import uvicorn
    from starlette.applications import Starlette
    from starlette.responses import JSONResponse
    from starlette.routing import Route, WebSocketRoute

    if host not in {"127.0.0.1", "::1"}:
        raise ValueError("Private session transport must bind loopback; use the authenticated API adapter remotely")
    if _ACTIVE_RUNNER is not None:
        raise RuntimeError("gateway API already started")

    async def health(request):
        ready = (runner.session_runtime_descriptor.get("state") == "ready" and not runner._draining)
        return JSONResponse({"status": "ready" if ready else "unavailable"}, status_code=200 if ready else 503)

    async def refuse(ws):
        await ws.close(code=4403)

    app = Starlette(routes=[Route("/health", health), WebSocketRoute("/api/ws", refuse)])
    app.state.bound_host = host
    app.state.bound_port = None
    app.state.auth_required = False
    app.state.trusted_public_hosts = frozenset()
    app.state.gateway_runner = runner
    app.state.session_authority = runner.session_authority
    config = uvicorn.Config(GatewayRuntimeAPI(app, runner, app), host=host, port=port,
        log_level="warning", proxy_headers=False, ws_ping_interval=None, ws_ping_timeout=None,
        ws_max_size=8 * 1024 * 1024, timeout_graceful_shutdown=5, lifespan="off")
    server = uvicorn.Server(config)
    family, kind, proto, _, address = socket.getaddrinfo(host, port, type=socket.SOCK_STREAM)[0]
    listener = socket.socket(family, kind, proto)
    _ACTIVE_RUNNER = runner
    try:
        listener.bind(address)
        listener.setblocking(False)
        app.state.bound_port = listener.getsockname()[1]
        config.load()
        server.lifespan = config.lifespan_class(config)
        await server.startup(sockets=[listener])
        if not server.started or server.should_exit:
            raise RuntimeError("gateway API startup failed")
    except BaseException:
        listener.close()
        _ACTIVE_RUNNER = None
        raise

    async def serve():
        global _ACTIVE_RUNNER
        try:
            await server.main_loop()
        finally:
            try:
                await server.shutdown(sockets=[listener])
            finally:
                listener.close()
                _ACTIVE_RUNNER = None

    task = asyncio.create_task(serve(), name="gateway-api")
    origin_host = f"[{host}]" if ":" in host else host
    return GatewayAPIHandle(f"http://{origin_host}:{app.state.bound_port}", server, task, app, listener)


async def stop_gateway_api(handle):
    handle.server.should_exit = True
    await asyncio.shield(asyncio.gather(handle.task, return_exceptions=True))


def _restart_wait_read(runner, scope) -> bool:
    from gateway.drain_control import restart_wait_read
    return scope['type'] == 'http' and restart_wait_read(runner, scope.get('method'), scope.get('path'))


class GatewayRuntimeAPI:
    """Redeem private local tickets at the existing WS subprotocol boundary.

    Other credentials and HTTP routes retain the complete dashboard gate. Local
    bootstrap never grants an exposure/worker ticket interactive permissions.
    *web_app* is the dashboard mount (or any object whose ``state`` carries the listener's
    ``bound_host`` / ``auth_required`` / ``trusted_public_hosts``).
    """
    def __init__(self, app, runner, web_app):
        self.app, self.runner, self.web_app = app, runner, web_app

    async def __call__(self, scope, receive, send):
        if scope['type'] != 'websocket' or scope.get('path') != '/api/ws':
            return await self._serve(scope, receive, send)
        started = accepted = False

        async def send_watched(message):
            nonlocal started, accepted
            started = True
            accepted = accepted or message.get('type') == 'websocket.accept'
            await send(message)

        try:
            return await self._serve(scope, receive, send_watched)
        except Exception as exc:
            # Uvicorn answers an upgrade that raised before accept with HTTP 500 and logs it only to
            # its own logger, never gateway.log: the dialing cron worker or managed worker sees a bare
            # InvalidStatus (#139922). Name the cause here; uvicorn still sends the 500, but only when
            # nothing was sent yet: a handler that had closed the upgrade already got its 403 out.
            if not accepted:
                logger.error('Gateway refused a session WebSocket upgrade from %s %s: %s: %s',
                             scope.get('hermes.gateway_socket_peer') or scope.get('client'),
                             'after its handler had refused it' if started else 'with HTTP 500',
                             type(exc).__name__, exc, exc_info=True)
            raise

    async def _serve(self, scope, receive, send):
        descriptor = getattr(self.runner, 'session_runtime_descriptor', None)
        if scope['type'] not in {'http', 'websocket'} or descriptor is None:
            return await self.app(scope, receive, send)
        draining_socket = (descriptor['state'] in {'ready', 'draining'}
            and (self.runner._draining or descriptor['state'] == 'draining')
            and scope['type'] == 'websocket' and scope['path'] == '/api/ws')
        if ((descriptor['state'] != 'ready' or self.runner._draining) and not draining_socket
                and not _restart_wait_read(self.runner, scope)):
            if scope['type'] == 'websocket':
                await send({'type': 'websocket.close', 'code': 1013})
            else:
                from starlette.responses import JSONResponse
                await JSONResponse({'error': 'gateway_not_ready', 'state': descriptor['state']},
                                   status_code=503)(scope, receive, send)
            return
        # Capture before Uvicorn's proxy middleware rewrites scope.client (native grants, both kinds).
        scope['hermes.gateway_socket_peer'] = scope.get('client')
        from gateway.run_idle_exit import attached_client, note_client_activity
        if scope['type'] != 'websocket' or scope['path'] != '/api/ws':
            note_client_activity(self.runner)  # dashboard/API traffic keeps an idle-exit gateway up
            return await self.app(scope, receive, send)
        original_receive = receive
        allow_draining = False
        admitted_in_wait = False
        from gateway.drain_control import restart_wait_open

        async def receive_admitted():
            message = await original_receive()
            # Only a socket admitted INTO the restart wait (to answer a parked prompt) outlives it. One
            # opened before the wait is cut as before: its client reconnects to the next owner, and a
            # held one (a kanban worker polling its card) would otherwise pin the drain to its cap.
            if ((descriptor['state'] != 'ready' or self.runner._draining) and not allow_draining
                    and not (admitted_in_wait and restart_wait_open(self.runner))):
                return {'type': 'websocket.disconnect', 'code': 1013}
            return message

        receive = receive_admitted
        from starlette.websockets import WebSocket
        from gateway.runtime_boundary import (
            _gateway_ws_ticket_from_subprotocol, ws_client_reason, ws_host_origin_reason,
        )
        state = getattr(self.web_app, 'state', None)
        scope['app'] = self.web_app
        ws = WebSocket(scope, receive, send)
        ticket, reason = _gateway_ws_ticket_from_subprotocol(ws)
        if reason == 'none' or ws.headers.get('origin'):
            if draining_socket:
                await ws.close(code=1013)
                return
            return await self.app(scope, receive, send)
        if (reason != 'ok' or ws_host_origin_reason(ws, state) is not None or ws_client_reason(ws, state) is not None
                or ws.headers.get('origin')
                or not ws.client or ws.client.host not in {'127.0.0.1', '::1'}):
            await ws.close(code=4403)
            return
        operator = True
        try:
            grant = self.runner.session_ticket_store.redeem(ticket, profile_id=None, purpose='interactive')
        except PermissionError:
            operator = False
            try:
                grant = self.runner.session_ticket_store.redeem(ticket, profile_id=None, purpose='worker-adoption')
            except PermissionError:
                # Browser/OAuth tickets have a separate issuer.
                if draining_socket:
                    await ws.close(code=1013)
                    return
                return await self.app(scope, receive, send)
        allow_draining = not operator
        admitted_in_wait = restart_wait_open(self.runner)
        # A restart's after-turn wait still owns its turns: an interactive client attaches to answer
        # a parked approval/clarify (#139041); stop() closes this window.
        if ((descriptor['state'] != 'ready' or self.runner._draining) and not allow_draining
                and not admitted_in_wait):
            await ws.close(code=1013)
            return
        from gateway.session_authorities import authority_for_profile_id
        authority = authority_for_profile_id(self.runner, grant['profile_id'])
        if authority is None:
            await ws.close(code=4403)
            return
        # The ticket names the served profile; this connection binds to that home's authority.
        scope['hermes.session_authority'] = authority
        from gateway.session_contract import CANONICAL_GATEWAY_PROTOCOL
        from hermes_runtime.ws import handle_ws
        from gateway.session_authorities import owner_scope
        with owner_scope(authority), attached_client(self.runner):
            await handle_ws(ws, auth_identity={'user_id': grant['subject'], 'provider': 'local',
                                              'profile_id': grant['profile_id'],
                                              'instance_id': grant['instance_id'],
                                              'capabilities': grant['capabilities'], 'native_bootstrap': True,
                                              'profile_scope': grant.get('scope', 'profile')},
                            subprotocol=CANONICAL_GATEWAY_PROTOCOL, operator=operator)

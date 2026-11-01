"""Headless entry point through the real owner, model loop, tools and session store."""
import asyncio
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import os
from pathlib import Path
import subprocess
import sqlite3
import socket
from urllib.request import Request, urlopen
import sys
import threading

import pytest

from tests.gateway.test_normal_runtime_boot import control
from tests.gateway.fixtures.local_recovery_probe import rpc, websocket
from tests.waiting import async_wait_until, wait_until


class Model(BaseHTTPRequestHandler):
    def log_message(self, *args):
        pass

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers['Content-Length'])))
        messages = body.get('messages') or []
        last = messages[-1] if messages else {}
        if last.get('content') == 'BLOCK':
            self.server.blocked.set()
            self.server.release.wait(30)
        call = last.get('role') == 'user' and last.get('content') == 'READ'
        message = ({'role': 'assistant', 'content': None, 'tool_calls': [
            {'id': 'read', 'index': 0, 'type': 'function', 'function': {
                'name': 'read_file', 'arguments': json.dumps({'path': str(self.server.target)})}}]}
            if call else {'role': 'assistant', 'content': 'CORE_OK'})
        choice = {'index': 0, 'message': message, 'finish_reason': 'tool_calls' if call else 'stop'}
        frame = {'id': 'core', 'model': 'fixture', 'choices': [choice],
                 'usage': {'prompt_tokens': 10, 'completion_tokens': 2, 'total_tokens': 12}}
        if body.get('stream'):
            choice['delta'] = choice.pop('message')
            payload = ('data: ' + json.dumps(frame) + '\n\ndata: [DONE]\n\n').encode()
        else:
            payload = json.dumps(frame).encode()
        self.send_response(200)
        self.send_header('Content-Type', 'text/event-stream' if body.get('stream') else 'application/json')
        self.send_header('Content-Length', str(len(payload)))
        self.end_headers()
        try:
            self.wfile.write(payload)
        except (BrokenPipeError, ConnectionResetError):
            pass


async def exercise(home, descriptor, peer):
    async with websocket(home, descriptor) as ws:
        created = await rpc(ws, 'session.create', request_id='core-create', source='cli',
                            cwd=str(home), toolsets=['file'], ignore_rules=True)
        assert 'result' in created, created
        sid = created['result']['session_id']
        initial = (await rpc(ws, 'session.resume', session_id=sid))['result']
        assert 'result' in await rpc(ws, 'prompt.submit', session_id=sid, input_id='read', text='READ')

        def completed():
            with sqlite3.connect(home / 'state.db') as db:
                return db.execute("SELECT status FROM session_admissions WHERE request_id='read'").fetchone() == ('terminal',)

        await async_wait_until(completed, timeout=60, what='real tool turn')
        snapshot = (await rpc(ws, 'session.resume', session_id=sid))['result']
        assert any(m['role'] == 'tool' and 'core-file-proof' in str(m['content']) for m in snapshot['messages'])
        replay = (await rpc(ws, 'session.events.since', session_id=sid,
                            replay_epoch=initial['replay_epoch'], last_sequence=initial['last_sequence']))['result']
        assert any(event['type'] == 'tool.complete' for event in replay['events'])
        await rpc(ws, 'prompt.submit', session_id=sid, input_id='block', text='BLOCK')
        assert await asyncio.to_thread(peer.blocked.wait, 30)
        running = (await rpc(ws, 'session.resume', session_id=sid))['result']
        stopped = await rpc(ws, 'session.interrupt', session_id=sid,
                            execution_generation=running['execution_generation'])
        assert 'result' in stopped, stopped
        peer.release.set()
        return sid


def exercise_http(root, env, peer, port):
    url = f'http://127.0.0.1:{port}'
    command = [sys.executable, str(root / 'examples/core_client.py'), '--url', url]
    completed = subprocess.run([*command, 'SAY_OK'], env=env, capture_output=True,
                               text=True, timeout=45, check=False)
    assert completed.returncode == 0, completed.stdout + completed.stderr
    assert 'CORE_OK' in completed.stdout
    peer.release.clear()
    peer.blocked.clear()
    request = Request(url + '/v1/runs', data=json.dumps({'input': 'BLOCK'}).encode(),
                      headers={'Authorization': 'Bearer ' + env['API_SERVER_KEY'],
                               'Content-Type': 'application/json'})
    with urlopen(request, timeout=15) as response:
        run = json.load(response)
    assert peer.blocked.wait(30)
    stopped = subprocess.run([*command, '--stop', run['run_id']], env=env, capture_output=True,
                             text=True, timeout=20, check=False)
    peer.release.set()
    assert stopped.returncode == 0, stopped.stdout + stopped.stderr


@pytest.mark.platforms('posix')
def test_headless_execution_cancellation_persistence_and_shutdown(tmp_path):
    root = Path(__file__).resolve().parents[1]
    home, user = tmp_path / 'state', tmp_path / 'user'
    home.mkdir(mode=0o700)
    user.mkdir()
    target = home / 'proof.txt'
    target.write_text('core-file-proof')
    peer = ThreadingHTTPServer(('127.0.0.1', 0), Model)
    peer.target = target
    peer.blocked, peer.release = threading.Event(), threading.Event()
    thread = threading.Thread(target=peer.serve_forever, daemon=True)
    thread.start()
    url = f'http://127.0.0.1:{peer.server_port}/v1'
    with socket.socket() as reservation:
        reservation.bind(('127.0.0.1', 0))
        api_port = reservation.getsockname()[1]
    (home / 'config.yaml').write_text(json.dumps({
        'gateway': {'multiplex_profiles': False},
        'platforms': {'api_server': {'enabled': True, 'extra': {'host': '127.0.0.1', 'port': api_port}}},
        'model': {'provider': 'custom', 'default': 'fixture', 'base_url': url},
        'auxiliary': {'title_generation': {'enabled': False}},
        'platform_toolsets': {'cli': ['file']}}))
    (home / '.env').write_text('API_SERVER_KEY=core-fixture-credential-0123456789abcdef\nOPENAI_API_KEY=local-fixture-only\n')
    env = {k: os.environ[k] for k in ('PATH', 'LANG', 'TZ') if k in os.environ}
    env.update(HOME=str(user), HERMES_HOME=str(home), HERMES_RUNTIME_DIR=str(tmp_path / 'runtime'),
               PYTHONPATH=str(root), OPENAI_API_KEY='local-fixture-only', OPENAI_BASE_URL=url,
               API_SERVER_KEY='core-fixture-credential-0123456789abcdef')
    with (tmp_path / 'owner.log').open('w+') as log:
        proc = subprocess.Popen([sys.executable, '-m', 'hermes_core', 'run'], env=env, cwd=root,
                                stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT)
        def ready():
            assert proc.poll() is None, (log.seek(0), log.read())
            try:
                descriptor = control(home, 'identify')
                return descriptor if descriptor.get('state') == 'ready' else None
            except (OSError, ValueError):
                return None
        try:
            descriptor = wait_until(ready, timeout=60, what='core readiness')
            checked = subprocess.run([sys.executable, '-m', 'hermes_core', 'health'], env=env,
                                     capture_output=True, text=True, timeout=15, check=False)
            assert checked.returncode == 0, checked.stdout + checked.stderr
            try:
                sid = asyncio.run(exercise(home, descriptor, peer))
            except Exception:
                log.seek(0)
                print(log.read())
                raise
            exercise_http(root, env, peer, api_port)
            chat = subprocess.run([sys.executable, '-m', 'hermes_core', 'chat', '-q', 'SAY_OK',
                                   '--provider', 'custom', '--model', 'fixture', '--base-url', url],
                                  env=env, cwd=root, capture_output=True, text=True, timeout=45, check=False)
            assert chat.returncode == 0, chat.stdout + chat.stderr
            assert 'CORE_OK' in chat.stdout

            proc.terminate()
            # The existing owner treats unmarked SIGTERM as an external stop (exit 1),
            # after graceful teardown, so supervisors can distinguish planned stops.
            assert proc.wait(timeout=40) == 1
            assert not (home / "gateway.pid").exists()
        finally:
            peer.release.set()
            if proc.poll() is None:
                proc.kill()
                proc.wait(timeout=10)
            peer.shutdown()
            peer.server_close()
    from hermes_state import SessionDB
    with SessionDB(home / 'state.db', read_only=True) as db:
        assert any(m['role'] == 'tool' for m in db.get_messages(sid))

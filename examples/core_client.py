"""Bearer-authenticated HTTP integration: submit, follow events, resume, or stop a run.

python examples/core_client.py 'Read README.md and summarize it'
python examples/core_client.py --session SESSION_ID 'Continue'
python examples/core_client.py --stop RUN_ID
"""
import argparse
import json
import os
from urllib.request import Request, urlopen


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('message', nargs='?')
    parser.add_argument('--url', default='http://127.0.0.1:8642')
    parser.add_argument('--session')
    parser.add_argument('--stop', metavar='RUN_ID')
    parser.add_argument('--health', action='store_true')
    args = parser.parse_args()
    key = os.environ.get('API_SERVER_KEY')
    if not key:
        parser.error('Set API_SERVER_KEY to the runtime listener credential')

    def request(path, body=None):
        return urlopen(Request(args.url.rstrip('/') + path,
            data=json.dumps(body).encode() if body is not None else None,
            headers={'Authorization': 'Bearer ' + key, 'Content-Type': 'application/json'}), timeout=300)

    if args.health:
        with request('/health/detailed') as response:
            print(response.read().decode())
        return
    if args.stop:
        with request(f'/v1/runs/{args.stop}/stop', {}) as response:
            print(response.read().decode())
        return
    if not args.message:
        parser.error('Provide a message, --health, or --stop RUN_ID')
    body = {'input': args.message}
    if args.session:
        body['session_id'] = args.session
    with request('/v1/runs', body) as response:
        run = json.load(response)
    print(json.dumps(run), flush=True)
    run_id = run['run_id']
    with request(f'/v1/runs/{run_id}/events') as response:
        for line in response:
            print(line.decode().rstrip(), flush=True)
    with request(f'/v1/runs/{run_id}') as response:
        print(response.read().decode())


if __name__ == '__main__':
    main()

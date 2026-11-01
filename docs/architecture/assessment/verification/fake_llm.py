"""Stdlib fake OpenAI chat-completions server: call 1 -> read_file tool call, call 2 -> final text."""
import json, sys, time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
LOG = sys.argv[2]
state = {"n": 0}
def reply(req):
    msgs = req.get("messages", [])
    has_tool_result = any(m.get("role") == "tool" for m in msgs)
    if not has_tool_result:
        return {"role": "assistant", "content": None, "tool_calls": [{"id": "call_1", "type": "function",
                "function": {"name": "read_file", "arguments": json.dumps({"path": sys.argv[3]})}}]}, "tool_calls"
    tool = [m for m in msgs if m.get("role") == "tool"][-1]
    return {"role": "assistant", "content": "DONE: " + str(tool.get("content"))[:80]}, "stop"
class H(BaseHTTPRequestHandler):
    def log_message(self, *a): pass
    def do_GET(self):
        self.send_response(200); self.send_header("Content-Type","application/json"); self.end_headers()
        self.wfile.write(json.dumps({"object":"list","data":[{"id":"fake-model","object":"model","context_length":128000}]}).encode())
    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))) or b"{}")
        state["n"] += 1
        with open(LOG, "a") as f:
            f.write(json.dumps({"path": self.path, "n": state["n"], "stream": body.get("stream"),
                                "tools": sorted(t["function"]["name"] for t in body.get("tools", []) or []),
                                "roles": [m.get("role") for m in body.get("messages", [])],
                                "sys_len": len(str((body.get("messages") or [{}])[0].get("content","")))}) + "\n")
        msg, fr = reply(body)
        usage = {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15}
        if body.get("stream"):
            self.send_response(200); self.send_header("Content-Type", "text/event-stream"); self.end_headers()
            delta = {"role": "assistant"}
            if msg.get("content"): delta["content"] = msg["content"]
            if msg.get("tool_calls"):
                delta["tool_calls"] = [dict(tc, index=i) for i, tc in enumerate(msg["tool_calls"])]
            for chunk in ({"choices":[{"index":0,"delta":delta,"finish_reason":None}]},
                          {"choices":[{"index":0,"delta":{},"finish_reason":fr}],"usage":usage}):
                chunk.update(id="c1", object="chat.completion.chunk", created=int(time.time()), model="fake-model")
                self.wfile.write(b"data: " + json.dumps(chunk).encode() + b"\n\n"); self.wfile.flush()
            self.wfile.write(b"data: [DONE]\n\n"); self.wfile.flush()
        else:
            out = {"id":"c1","object":"chat.completion","created":int(time.time()),"model":"fake-model",
                   "choices":[{"index":0,"message":msg,"finish_reason":fr}],"usage":usage}
            data = json.dumps(out).encode()
            self.send_response(200); self.send_header("Content-Type","application/json"); self.send_header("Content-Length", str(len(data))); self.end_headers(); self.wfile.write(data)
ThreadingHTTPServer(("127.0.0.1", int(sys.argv[1])), H).serve_forever()

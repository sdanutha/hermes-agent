import sys, os, time, json
sys.path.insert(0, os.getcwd())
t0 = time.perf_counter()
from run_agent import AIAgent
t1 = time.perf_counter()
agent = AIAgent(base_url="http://127.0.0.1:%s/v1" % sys.argv[1], api_key="sk-fake", provider="custom",
                model="fake-model", enabled_toolsets=["file", "terminal"], quiet_mode=True,
                skip_memory=True, skip_context_files=True, max_iterations=6)
t2 = time.perf_counter()
res = agent.run_conversation("Read the probe file and report it.")
t3 = time.perf_counter()
print(json.dumps({"import_s": round(t1-t0,2), "init_s": round(t2-t1,2), "turn_s": round(t3-t2,2),
  "tools": sorted(t["function"]["name"] for t in (agent.tools or [])),
  "final": res.get("final_response"), "n_messages": len(res.get("messages", [])),
  "roles": [m.get("role") for m in res.get("messages", [])], "modules": len(sys.modules),
  "session_id": agent.session_id}, indent=1))
import collections
c = collections.Counter()
for m in list(sys.modules):
    p = m.split('.')
    c[p[0] if p[0] not in ('plugins','hermes_plugins') else '.'.join(p[:2])] += 1
with open(sys.argv[2], 'w') as f:
    for k, v in c.most_common(45): f.write(f"{k} {v}\n")

import sys, time, json, os, collections
sys.path.insert(0, os.getcwd())
import model_tools
from tools.registry import registry
import toolsets
ents = registry.get_all_entries()
by=collections.defaultdict(list)
for e in ents:
    n = getattr(e,'name',None); ts=getattr(e,'toolset',None)
    by[ts].append(n)
print("total_tools", len(ents))
for ts in sorted(by, key=lambda k: str(k)):
    print(f"{ts}: {len(by[ts])} {sorted(by[ts])}")
core = toolsets._HERMES_CORE_TOOLS
sz = 0; miss=[]
for n in core:
    s = registry.get_schema(n)
    if s is None: miss.append(n); continue
    sz += len(json.dumps(s))
print("core_count", len(core), "core_schema_chars", sz, "missing", miss)
allsz = sum(len(json.dumps(registry.get_schema(getattr(e,'name')) or {})) for e in ents)
print("all_schema_chars", allsz)
print("TOOLSETS keys", len(toolsets.TOOLSETS), sorted(toolsets.TOOLSETS)[:200])

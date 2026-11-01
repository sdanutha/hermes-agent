import sys, json, os, collections
sys.path.insert(0, os.getcwd())
import model_tools, toolsets
from tools.registry import registry
c = collections.Counter(); n = collections.Counter()
for t in toolsets._HERMES_CORE_TOOLS:
    ts = registry.get_toolset_for_tool(t); s = registry.get_schema(t)
    c[ts] += len(json.dumps(s)); n[ts] += 1
for ts, v in c.most_common(): print(f"{ts}: {n[ts]} tools, {v} chars")

import sys, json, os
sys.path.insert(0, os.getcwd())
import model_tools
for ts in (["hermes-cli"], ["safe"], ["file","terminal"]):
    d = model_tools.get_tool_definitions(enabled_toolsets=ts, quiet_mode=True)
    print(ts, len(d), len(json.dumps(d)), sorted(x["function"]["name"] for x in d))

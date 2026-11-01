import sys, builtins, os
sys.path.insert(0, os.getcwd())
targets = set(sys.argv[2:]); first={}
orig = builtins.__import__
stack=[]
def imp(name, globals=None, locals=None, fromlist=(), level=0):
    caller = (globals or {}).get('__name__','?')
    full = name
    if level and globals and globals.get('__package__'):
        pkg = globals['__package__'].rsplit('.', level-1)[0] if level>1 else globals['__package__']
        full = pkg + ('.'+name if name else '')
    for t in targets:
        if (full==t or full.startswith(t+'.')) and t not in first and t not in sys.modules:
            first[t]=caller
    return orig(name, globals, locals, fromlist, level)
builtins.__import__=imp
__import__(sys.argv[1])
for t,c in first.items(): print(t, "<-", c)

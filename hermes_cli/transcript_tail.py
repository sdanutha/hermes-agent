"""Bounded transcript reads for subagent diagnostics."""
from agent.i18n import t

def read_tail(path):
    if not path:
        return t('cli.subagents.transcript_unavailable')
    try:
        with open(path, 'rb') as stream:
            stream.seek(0, 2)
            stream.seek(max(0, stream.tell() - 32768))
            text = stream.read(32768).decode('utf-8', errors='replace')
        return ''.join(c for c in text if c.isprintable() or c in '\n\t')
    except OSError:
        return t('cli.subagents.transcript_unavailable')

"""Durable busy wall-time budget shared by local inference processes.

Locks protect short state transactions only. In-flight calls drain before the
full cooldown begins; a natural idle interval also satisfies the required rest.
"""
from __future__ import annotations

from contextlib import contextmanager
import json
import math
import os
from pathlib import Path
import threading
import time
import uuid

RUN_SECONDS = 7200
REST_SECONDS = 600
ROOT = Path(__file__).resolve().parents[2] / 'build' / 'inference-budget'
_THREAD_LOCK = threading.RLock()


def _process_identity(pid):
    """Include creation identity so PID reuse/reboot cannot leave a stale lease."""
    try:
        boot = Path('/proc/sys/kernel/random/boot_id').read_text().strip()
        start = Path(f'/proc/{pid}/stat').read_text().rsplit(')', 1)[1].split()[19]
        return f'{boot}:{start}'
    except FileNotFoundError:
        return None


class InferenceBudget:
    def __init__(self, directory: Path = ROOT, *, clock=time.time):
        if os.name != 'posix' or not Path('/proc/sys/kernel/random/boot_id').is_file():
            raise ValueError('Run local model inference inside WSL/Linux to share the inference budget safely')
        self.directory = directory
        self.clock = clock

    @contextmanager
    def transaction(self):
        self.directory.mkdir(parents=True, exist_ok=True)
        with _THREAD_LOCK, (self.directory / 'state.lock').open('a+b') as stream:
            stream.seek(0, os.SEEK_END)
            if not stream.tell():
                stream.write(b'0')
                stream.flush()
            stream.seek(0)
            import fcntl
            fcntl.flock(stream.fileno(), fcntl.LOCK_EX)
            try:
                path = self.directory / 'state.json'
                now = self.clock()
                if path.exists():
                    state = json.loads(path.read_text(encoding='utf-8'))
                    required = {'version', 'busy', 'updated', 'idle_since', 'cooldown_until', 'active'}
                    if (not isinstance(state, dict) or set(state) != required or state['version'] != 1
                            or not isinstance(state['active'], dict)
                            or any(type(state[key]) not in (int, float) or not math.isfinite(state[key])
                                   or state[key] < 0 for key in required - {'version', 'active'})
                            or any(not isinstance(lease, dict) or set(lease) != {'pid', 'identity'}
                                   or type(lease['pid']) is not int or lease['pid'] <= 0
                                   or not isinstance(lease['identity'], str)
                                   for lease in state['active'].values())):
                        raise ValueError('Invalid inference budget state; preserve it for inspection')
                else:
                    state = dict(version=1, busy=0.0, updated=now, idle_since=now,
                                 cooldown_until=0.0, active={})
                # A backwards wall clock must not erase accumulated work.
                now = max(now, state['updated'])
                if state['active']:
                    state['busy'] += now - state['updated']
                    for token, lease in list(state['active'].items()):
                        if lease['identity'].startswith('windows:'):
                            raise ValueError('A Windows inference lease remains; inspect that task before resuming in WSL')
                        if _process_identity(lease['pid']) != lease['identity']:
                            del state['active'][token]
                    if not state['active']:
                        state['idle_since'] = now
                if not state['active']:
                    if state['cooldown_until'] and now >= state['cooldown_until']:
                        state.update(busy=0.0, cooldown_until=0.0)
                    elif not state['cooldown_until'] and now - state['idle_since'] >= REST_SECONDS:
                        state['busy'] = 0.0
                    if state['busy'] >= RUN_SECONDS and not state['cooldown_until']:
                        state['cooldown_until'] = now + REST_SECONDS
                state['updated'] = now
                yield state, now
                temporary = path.with_suffix('.tmp')
                with temporary.open('w', encoding='utf-8') as output:
                    json.dump(state, output, sort_keys=True)
                    output.flush()
                    os.fsync(output.fileno())
                os.replace(temporary, path)
            finally:
                stream.seek(0)
                fcntl.flock(stream.fileno(), fcntl.LOCK_UN)

    def reserve(self):
        """Return a request token or a bounded wait before retrying admission."""
        with self.transaction() as (state, now):
            if state['cooldown_until'] > now:
                return None, min(30.0, state['cooldown_until'] - now)
            if state['busy'] >= RUN_SECONDS:
                return None, 1.0  # Drain existing calls before starting cooldown.
            token = uuid.uuid4().hex
            identity = _process_identity(os.getpid())
            if identity is None:
                raise RuntimeError('Cannot identify inference process')
            state['active'][token] = {'pid': os.getpid(), 'identity': identity}
            return token, 0.0

    def release(self, token):
        with self.transaction() as (state, now):
            state['active'].pop(token, None)
            if not state['active']:
                state['idle_since'] = now
                if state['busy'] >= RUN_SECONDS and not state['cooldown_until']:
                    state['cooldown_until'] = now + REST_SECONDS

    @contextmanager
    def request(self):
        announced = False
        while True:
            token, delay = self.reserve()
            if token is not None:
                break
            if not announced:
                print('Local inference budget reached; draining requests and resting for 10 minutes.', flush=True)
                announced = True
            time.sleep(delay)
        try:
            yield
        finally:
            self.release(token)

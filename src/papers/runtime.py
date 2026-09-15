"""WSL job owner: model lifecycle, daily priority, batch checkpoints and Git publishing."""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from datetime import datetime
import json
import os
from pathlib import Path
import signal
import shutil
import subprocess
import sys
import time
import urllib.request
from zoneinfo import ZoneInfo

from papers import paths
from papers.model_runtime import DEFAULT_MODEL, DEFAULT_MODEL_TIMEOUT_SECONDS, DEFAULT_MODEL_WORKERS, MAX_MODEL_WORKERS

PUBLIC = ('content/papers/archive.json', 'content/papers/arxiv-candidates.json',
          'content/papers/conference-library.json',
          'content/papers/paper-annotations.json', 'docs/notes/', 'docs/index.html',
          'docs/search-index.json', 'docs/togos-papers.json')
PRIVATE = paths.ROOT / 'build/paper-summaries'
ZONE = ZoneInfo('Asia/Shanghai')
BACKFILL_BATCH_SIZE = 100
GIT_PUSH_TIMEOUT_SECONDS = 5 * 60
GIT_PUSH_RETRY_SECONDS = 60
GIT_SSH_COMMAND = 'ssh -o ServerAliveInterval=15 -o ServerAliveCountMax=4'
PENDING_PUSH = PRIVATE / 'pending-runtime-push.json'


class RetryableGitPush(RuntimeError):
    """A transient push failure that must not discard completed publication work."""


def command(*args, capture=False, check=True, timeout=None):
    return subprocess.run(args, cwd=paths.ROOT, check=check, text=True,
                          stdout=subprocess.PIPE if capture else None, timeout=timeout)


def git(*args, capture=False):
    result = command('git', *args, capture=capture)
    return result.stdout.strip() if capture else ''


def _origin_is_ancestor():
    return command('git', 'merge-base', '--is-ancestor', 'origin/main', 'HEAD',
                   check=False).returncode == 0


def _push_main_once(commit, *, timeout=GIT_PUSH_TIMEOUT_SECONDS):
    environment = os.environ.copy()
    environment['GIT_SSH_COMMAND'] = GIT_SSH_COMMAND
    process = subprocess.Popen(
        ('git', 'push', 'origin', f'{commit}:refs/heads/main'), cwd=paths.ROOT, env=environment,
        start_new_session=True,
    )
    try:
        return_code = process.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        process.wait()
        raise RetryableGitPush(
            f'git push timed out after {timeout:g}s; completed commit retained for retry'
        ) from None
    if return_code:
        raise RetryableGitPush(
            f'git push failed: exit={return_code}; completed commit retained for retry'
        )


def _push_main():
    if (PRIVATE / 'recheck/local-only.json').exists():
        raise RuntimeError('local recheck review is active; remote publication is disabled')
    receipt = json.loads(PENDING_PUSH.read_text(encoding='utf-8'))
    commit = receipt.get('commit')
    base = receipt.get('base')
    if (not commit or not base or git('rev-parse', 'HEAD', capture=True) != commit
            or git('rev-parse', f'{commit}^', capture=True) != base):
        raise RuntimeError('Publication history changed; preserve concurrent task commits for review')
    _push_main_once(commit)
    git('fetch', 'origin', 'main')
    if commit != git('rev-parse', 'origin/main', capture=True):
        raise RetryableGitPush('git push was not confirmed by origin/main; completed commit retained for retry')
    PENDING_PUSH.unlink(missing_ok=True)


def clean_pull():
    if (PRIVATE / 'recheck/local-only.json').exists():
        return False
    if git('branch', '--show-current', capture=True) != 'main':
        raise RuntimeError('runtime requires main; integrate the reviewed migration first')
    if git('status', '--porcelain', capture=True):
        raise RuntimeError('worktree is not clean; preserve changes and resolve before retry')
    git('pull', '--ff-only', 'origin', 'main')
    recovered = False
    if git('rev-parse', 'HEAD', capture=True) != git('rev-parse', 'origin/main', capture=True):
        if not _origin_is_ancestor():
            raise RuntimeError('local main and origin/main diverged; preserve both histories for review')
        receipt = json.loads(PENDING_PUSH.read_text(encoding='utf-8')) if PENDING_PUSH.exists() else {}
        if receipt.get('commit') != git('rev-parse', 'HEAD', capture=True):
            raise RuntimeError('Local commits await explicit publication; another maintenance job must not push them')
        print('Recovering this runtime publication from its pending push receipt', flush=True)
        _push_main()
        recovered = True
    git('var', 'GIT_AUTHOR_IDENT', capture=True)
    return recovered


def allowed_path(path):
    return any((path.startswith(item) and path.endswith('.html') and '/' not in path[len(item):])
               if item.endswith('/') else path == item for item in PUBLIC)


def publish(mode, *, expected_head):
    if (PRIVATE / 'recheck/local-only.json').exists():
        command(sys.executable, '-m', 'papers', 'build')
        print('Local review: built results; Git publication remains disabled', flush=True)
        return
    if git('branch', '--show-current', capture=True) != 'main' or git('rev-parse', 'HEAD', capture=True) != expected_head:
        raise RuntimeError('Another task changed the branch or committed during inference; preserve both results for review')
    # No untracked or unrelated file can hitchhike in an automatic commit.
    names = git('diff', '--name-only', capture=True).splitlines()
    staged = git('diff', '--cached', '--name-only', capture=True).splitlines()
    untracked = git('ls-files', '--others', '--exclude-standard', capture=True).splitlines()
    if any(not allowed_path(name) for name in names + staged + untracked):
        raise RuntimeError('unexpected public changes; preserve worktree for review')
    command(sys.executable, '-m', 'papers', 'build')
    git('diff', '--check')
    for path in (paths.DOCS / 'assets/js').glob('*.js'):
        node = shutil.which('node') or shutil.which('node.exe')
        if node is None:
            raise RuntimeError('Node.js is required for public JavaScript validation')
        subprocess.run([node, '--check'], input=path.read_text(encoding='utf-8'), text=True, check=True)
    names = git('diff', '--name-only', capture=True).splitlines()
    untracked = git('ls-files', '--others', '--exclude-standard', capture=True).splitlines()
    for name in names + untracked:
        if not allowed_path(name):
            raise RuntimeError('build touched unrelated output')
        path = paths.ROOT / name
        if path.is_file() and any(value in path.read_text(encoding='utf-8') for value in
                                  ('/mnt/g/share', '/home/zyf', 'G:\\share', 'build/paper-summaries', 'vllm-paper.service')):
            raise RuntimeError('public output contains private runtime information')
    git('add', '--', *PUBLIC)
    git('diff', '--cached', '--check')
    if not git('diff', '--cached', '--name-only', capture=True):
        print('No public changes', flush=True)
        return
    if git('branch', '--show-current', capture=True) != 'main' or git('rev-parse', 'HEAD', capture=True) != expected_head:
        raise RuntimeError('Publication base changed during build; preserve the staged results for review')
    git('commit', '-m', f'Publish {mode} paper results for {datetime.now(ZONE):%Y-%m-%d}')
    from papers.candidate_ledger import atomic_write_json
    commit = git('rev-parse', 'HEAD', capture=True)
    if git('rev-parse', f'{commit}^', capture=True) != expected_head:
        raise RuntimeError('Concurrent commit detected; local results retained without pushing')
    atomic_write_json(PENDING_PUSH, {'commit': commit, 'base': expected_head})
    _push_main()


@contextmanager
def lock(name, *, blocking=True):
    import fcntl
    PRIVATE.mkdir(parents=True, exist_ok=True)
    with (PRIVATE / name).open('a+b') as stream:
        try:
            fcntl.flock(stream, fcntl.LOCK_EX | (0 if blocking else fcntl.LOCK_NB))
        except BlockingIOError:
            yield False
            return
        try:
            yield True
        finally:
            fcntl.flock(stream, fcntl.LOCK_UN)


def daily_waiting():
    with lock('daily-request.lock', blocking=False) as available:
        return not available


@contextmanager
def model_service(service):
    started = False
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))

    def registered_model():
        try:
            with opener.open('http://127.0.0.1:8000/v1/models', timeout=5) as response:
                model = json.load(response)['data'][0]['id']
            return model if isinstance(model, str) and model else None
        except (OSError, ValueError, KeyError, IndexError):
            return None

    try:
        model = registered_model()
        if model is None and command('systemctl', 'is-active', '--quiet', service, check=False).returncode:
            command('sudo', '-n', '/usr/bin/systemctl', 'start', service)
            started = True
        deadline = time.monotonic() + DEFAULT_MODEL_TIMEOUT_SECONDS
        while True:
            model = registered_model()
            if model is None:
                if command('systemctl', 'is-failed', '--quiet', service, check=False).returncode == 0:
                    raise RuntimeError('model service failed during startup; inspect its systemd journal')
                if time.monotonic() >= deadline:
                    raise RuntimeError('model readiness timed out') from None
                time.sleep(5)
                continue
            break
        if model != os.environ.get('TOGOS_WSL_LLM_MODEL', DEFAULT_MODEL):
            raise RuntimeError('active model differs from the configured paper model; preserve the existing service and inspect its configuration')
        yield model
    finally:
        if started:
            command('sudo', '-n', '/usr/bin/systemctl', 'stop', service)


def in_weekend_window(now=None):
    now = now or datetime.now(ZONE)
    return ((now.weekday() == 5 and (now.hour, now.minute) >= (9, 30))
            or (now.weekday() == 6 and (now.hour, now.minute) < (23, 30)))


def execute(mode, args):
    recovered_push = clean_pull()
    expected_head = git('rev-parse', 'origin/main', capture=True)
    if mode == 'backfill' and recovered_push:
        print('Recovered prior publication push', flush=True)
        return 0
    with model_service(args.service) as model:
        common = ['--model', model, '--workers', str(args.workers), '--timeout', str(args.timeout)]
        if mode == 'daily':
            result = command(sys.executable, '-m', 'papers', 'daily', *common, '--limit', str(args.limit), check=False)
        else:
            from datetime import timedelta
            now = datetime.now(ZONE)
            # Release runtime.lock after each checkpoint batch. Publication
            # batching never imposes a paper-count-based inference rest.
            batch_count = 1
            command_args = [sys.executable, '-m', 'papers', 'batch',
                            '--batch-size', str(BACKFILL_BATCH_SIZE),
                            '--max-batches', str(batch_count), '--batch-pause', '0', *common]
            if mode == 'weekend':
                end = (now + timedelta(days=6 - now.weekday())).replace(
                    hour=23, minute=30, second=0, microsecond=0)
                result = command(*command_args, check=False,
                                 timeout=max(1, (end - now).total_seconds()))
            else:
                result = command(*command_args, check=False)
            if result.returncode in (0, 3):
                command(sys.executable, '-m', 'papers', 'publish-offline')
        if result.returncode not in (0, 3):
            raise RuntimeError(f'{mode} failed: exit={result.returncode}; preserve checkpoint and worktree')
        if mode == 'daily':
            from papers.conference_library import LIBRARY
            if LIBRARY.exists():
                from papers.conference_intake import summarize, rules
                # The daily owner already holds runtime.lock; finish a small queue slice.
                conference_result = summarize(limit=rules()['conference_intake']['summary_batch_size'],
                                              timeout=args.timeout, model=model, runtime_owned=True)
                if conference_result.get('counts', {}).get('failed'):
                    result.returncode = 3
                command(sys.executable, '-m', 'papers', 'build')
        publish(mode, expected_head=expected_head)
        return result.returncode


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=['daily', 'weekend', 'backfill'])
    parser.add_argument('--service', default='vllm-paper.service')
    parser.add_argument('--workers', type=int, default=DEFAULT_MODEL_WORKERS)
    parser.add_argument('--timeout', type=float, default=DEFAULT_MODEL_TIMEOUT_SECONDS)
    parser.add_argument('--limit', type=int, default=100)
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args(argv)
    effective_mode = 'backfill' if args.mode == 'weekend' else args.mode
    if (not 1 <= args.workers <= MAX_MODEL_WORKERS or args.limit < 1 or args.timeout <= 0
            or args.service != 'vllm-paper.service'):
        parser.error(f'workers 1-{MAX_MODEL_WORKERS}, positive limit/timeout and configured model service required')
    if args.dry_run:
        print(json.dumps({'mode': effective_mode, 'workers': args.workers, 'timeout': args.timeout,
                          'limit': args.limit, 'weekend_window': in_weekend_window(),
                          'checkpoint_batch_size': BACKFILL_BATCH_SIZE,
                          'inference_run_seconds': 7200, 'inference_rest_seconds': 600,
                          'public_paths': PUBLIC}))
        return 0
    if sys.platform != 'linux':
        parser.error('run this command inside WSL')
    signal.signal(signal.SIGTERM, lambda *_: (_ for _ in ()).throw(KeyboardInterrupt()))
    try:
        if effective_mode == 'daily':
            with lock('daily-request.lock', blocking=False) as acquired:
                if not acquired:
                    print('Daily job already queued or running')
                    return 0
                with lock('runtime.lock'):
                    return execute('daily', args)
        # A single durable weekend owner; daily can take runtime.lock between batches.
        with lock('weekend-owner.lock', blocking=False) as acquired:
            if not acquired:
                print('Weekend job already running')
                return 0
            while True:
                if daily_waiting():
                    time.sleep(5)
                    continue
                try:
                    with lock('runtime.lock'):
                        if daily_waiting():
                            continue
                        result = execute('backfill', args)
                except RetryableGitPush as error:
                    print(f'{error}; retrying in {GIT_PUSH_RETRY_SECONDS}s', file=sys.stderr, flush=True)
                    time.sleep(GIT_PUSH_RETRY_SECONDS)
                    continue
                from papers.batch.cycle import load_state, recovery_cycle_pending
                state = load_state()
                if state is None:
                    if recovery_cycle_pending():
                        continue
                    return result
                if state.get('network_paused') or state['phase'] == 'summarize':
                    return 3
        return 0
    except KeyboardInterrupt:
        print('Stopped; completed caches and batch checkpoint retained', flush=True)
        return 130
    except subprocess.TimeoutExpired:
        print('Weekend window ended; cached results and checkpoint retained', flush=True)
        return 130
    except (RuntimeError, subprocess.CalledProcessError) as error:
        print(str(error), file=sys.stderr, flush=True)
        return 2


if __name__ == '__main__':
    raise SystemExit(main())
